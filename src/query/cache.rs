//! The cached columns and pin observations behind `Counter` and `Evaluation`:
//! observation validation, deferred ancestor refresh and the cached read.

use crate::{Tdd, Engine, OperationError};
use crate::vtree::{VarId, VtreeIdx};
use crate::limits::PollGate;
use crate::value::Retention;
use super::fold::{Column, LevelFold, fold_level, fold_bottom_up};

/// A leaf observation and membership in a pending refresh.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PinState {
    pub(crate) value: Option<bool>,
    pub(crate) dirty: bool,
}

#[derive(Debug)]
pub(crate) struct Observations {
    pub(crate) pins: Vec<PinState>,
    pub(crate) changed: Vec<VtreeIdx>,
    pub(crate) retention: Retention,
    pub(crate) evaluated: bool,
}

impl Observations {
    pub(crate) fn new(eng: &Engine, pin_slots: usize, retention: Retention) -> Result<Self, OperationError> {
        let mut pins = Vec::new();
        eng.limits().try_resize(&mut pins, pin_slots, PinState::default())?;
        let mut changed = Vec::new();
        if pin_slots != 0 && retention == Retention::All { eng.limits().reserve_exact(&mut changed, pin_slots)?; }
        Ok(Self { pins, changed, retention, evaluated: false })
    }

    pub(crate) fn set_pin(&mut self, tdd: &Tdd, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        let leaf = self.validate_pin(tdd, var)?;
        self.set_leaf_pin(leaf, val);
        Ok(())
    }

    pub(crate) fn observe<L: crate::LiteralInput>(&mut self, tdd: &Tdd, literals: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.apply_pins(tdd, literals.as_ref().iter().map(|&input| {
            input.literal().map(|literal| (literal.var, Some(literal.sign)))
        }))
    }

    pub(crate) fn set_pins(&mut self, tdd: &Tdd, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.apply_pins(tdd, pins.iter().map(|&pin| Ok(pin)))
    }

    /// Validate every pin in order, then apply them in order, so an invalid
    /// one leaves every observation unchanged.
    fn apply_pins(
        &mut self, tdd: &Tdd,
        pins: impl Iterator<Item = Result<(VarId, Option<bool>), OperationError>> + Clone,
    ) -> Result<(), OperationError> {
        for pin in pins.clone() { self.validate_pin(tdd, pin?.0)?; }
        for pin in pins {
            let (var, val) = pin.expect("validated pin");
            let leaf = tdd.vtree.leaf_of(var).expect("validated pin variable");
            self.set_leaf_pin(leaf, val);
        }
        Ok(())
    }

    pub(crate) fn clear_pins(&mut self) {
        for leaf in 0..self.pins.len() {
            if self.pins[leaf].value.is_some() { self.set_leaf_pin(VtreeIdx(leaf as u32), None); }
        }
    }

    /// Resolve a variable to its structural leaf without changing counter state.
    fn validate_pin(&self, tdd: &Tdd, var: VarId) -> Result<VtreeIdx, OperationError> {
        let leaf = tdd.vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        // An implicit integer leaf can remain below a marginal parent.
        for level in std::iter::once(leaf).chain(tdd.vtree.node(leaf).parent()) {
            if tdd.levels[level.idx()].is_marginal() {
                return Err(OperationError::MarginalLevel(level));
            }
        }
        Ok(leaf)
    }

    /// Update a validated leaf's pin and record its deferred refresh once.
    fn set_leaf_pin(&mut self, leaf: VtreeIdx, val: Option<bool>) {
        if self.retention == Retention::Frontier {
            if self.pins[leaf.idx()].value != val {
                self.pins[leaf.idx()].value = val;
                self.evaluated = false;
            }
            return;
        }
        if !self.evaluated {
            self.clear_changed();
            self.pins[leaf.idx()].value = val;
            return;
        }
        if self.pins[leaf.idx()].value == val { return; }
        self.pins[leaf.idx()].value = val;
        if !self.pins[leaf.idx()].dirty {
            self.changed.push(leaf);
            self.pins[leaf.idx()].dirty = true;
        }
    }

    /// Extend the changed leaves with every ancestor, once each, and order
    /// them children first.
    fn add_ancestors(&mut self, eng: &Engine, tdd: &Tdd, gate: &mut PollGate) -> Result<(), OperationError> {
        eng.limits().try_resize(&mut self.pins, tdd.vtree.num_nodes(), PinState::default())?;
        let leaves = self.changed.len();
        for i in 0..leaves {
            let mut current = self.changed[i];
            while let Some(parent) = tdd.vtree.node(current).parent() {
                if self.pins[parent.idx()].dirty { break; }
                eng.limits().try_push(&mut self.changed, parent)?;
                self.pins[parent.idx()].dirty = true;
                gate.poll(1)?;
                current = parent;
            }
            gate.poll(1)?;
        }
        tdd.vtree.sort_bottom_up(&mut self.changed);
        Ok(())
    }

    pub(crate) fn invalidate(&mut self) {
        self.evaluated = false;
        self.clear_changed();
    }

    /// Clear pending traversal membership while retaining its allocated storage.
    pub(crate) fn clear_changed(&mut self) {
        for &level in &self.changed { self.pins[level.idx()].dirty = false; }
        self.changed.clear();
    }
}

/// What a retained query folds and returns: the per-node fold under the
/// current pins, the answer for the constant-false diagram, and the read of
/// the output node's value.
pub(crate) trait CachedQuery {
    /// One level's cached values.
    type Col: Column;
    /// The value a read returns.
    type Output;
    /// The fold that recomputes a column, borrowing the query and the pins.
    type Fold<'a>: LevelFold<Col = Self::Col> where Self: 'a;

    /// Refuse a diagram this query cannot read.
    fn admit(tdd: &Tdd) -> Result<(), OperationError>;
    /// The fold under the current pins.
    fn fold<'a>(&'a self, pins: &'a [PinState]) -> Self::Fold<'a>;
    /// The answer for the constant-false diagram, which has no output column.
    fn false_value(&self) -> Self::Output;
    /// Read the output node's value from its level's column.
    fn output(&self, col: &Self::Col, i: usize) -> Self::Output;
}

/// The columns and observations a retained query keeps between reads.
///
/// Reads refresh only what the observations changed and return the output
/// node's value. A failed read invalidates the columns, so the next read
/// recomputes them.
pub(crate) struct QueryCache<Q: CachedQuery> {
    query: Q,
    cols: Vec<Q::Col>,
    pub(super) observations: Observations,
}

impl<Q: CachedQuery> QueryCache<Q> {
    /// Admit `tdd`, reserve one empty column per vtree node and `pin_slots`
    /// pin slots. Columns are allocated on the first read.
    pub(super) fn new(eng: &Engine, tdd: &Tdd, query: Q, pin_slots: usize, retention: Retention) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.enter()?;
        Q::admit(tdd)?;
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
        cols.resize_with(tdd.vtree.num_nodes(), Q::Col::default);
        let observations = Observations::new(eng, pin_slots, retention)?;
        lim.check_stop()?;
        Ok(Self { query, cols, observations })
    }

    /// Refresh pending observations and return the output node's value under
    /// `eng`'s limits. Stops are checked even for a cached or constant answer.
    pub(super) fn read(&mut self, eng: &Engine, tdd: &Tdd) -> Result<Q::Output, OperationError> {
        let lim = eng.limits();
        let _op = lim.enter()?;
        let result = (|| {
            let mut gate = lim.gate();
            let value = if tdd.is_zero() { self.query.false_value() } else {
                self.refresh(eng, tdd, &mut gate)?;
                let out = tdd.output;
                self.query.output(&self.cols[out.vtree.idx()], out.local.idx())
            };
            gate.finish()?;
            Ok(value)
        })();
        if result.is_err() { self.observations.invalidate(); }
        result
    }

    /// Recompute the ancestors of changed pins, or every column after an
    /// invalidation, allocating only on a full pass.
    ///
    /// The cache is marked invalid before the fold runs user arithmetic, so an
    /// error or unwinding cannot leave partly refreshed columns as cached answers.
    pub(super) fn refresh(&mut self, eng: &Engine, tdd: &Tdd, gate: &mut PollGate) -> Result<(), OperationError> {
        let observations = &mut self.observations;
        if observations.evaluated && observations.changed.is_empty() { return Ok(()); }
        let incremental = observations.evaluated && observations.retention == Retention::All;
        observations.evaluated = false;
        if incremental {
            observations.add_ancestors(eng, tdd, gate)?;
            let fold = self.query.fold(&observations.pins);
            for &level in &observations.changed { fold_level(&fold, eng, tdd, &mut self.cols, level, gate)?; }
        } else {
            let retention = observations.retention;
            if retention == Retention::Frontier {
                for col in self.cols.iter_mut() { *col = Q::Col::default(); }
            }
            fold_bottom_up(&self.query.fold(&observations.pins), eng, tdd, &mut self.cols, retention, gate)?;
        }
        observations.clear_changed();
        observations.evaluated = true;
        Ok(())
    }

    /// Swap in a new query, invalidating every cached column; observations stay.
    pub(super) fn replace_query(&mut self, query: Q) -> Q {
        self.observations.invalidate();
        std::mem::replace(&mut self.query, query)
    }

    /// The columns, as the last refresh left them.
    pub(super) fn into_columns(self) -> Vec<Q::Col> {
        self.cols
    }
}

/// Owned query state or a temporary engine binding of a persistent query.
pub(crate) enum BoundState<'a, T> { Owned(T), Borrowed(&'a mut T) }
impl<T> BoundState<'_, T> {
    pub(crate) fn get(&self) -> &T {
        match self { Self::Owned(value) => value, Self::Borrowed(value) => value }
    }

    pub(crate) fn get_mut(&mut self) -> &mut T {
        match self { Self::Owned(value) => value, Self::Borrowed(value) => value }
    }
}
