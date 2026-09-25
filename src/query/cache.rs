//! Observation validation and deferred ancestor refresh shared by numeric queries.

use crate::{Tdd, Engine, OperationError};
use crate::vtree::{VarId, VtreeIdx};
use crate::limits::PollGate;
use crate::value::Retention;
use super::fold::{LevelFold, fold_level, fold_bottom_up};

/// A leaf observation and membership in a pending refresh.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PinState {
    pub(crate) value: Option<bool>,
    pub(crate) dirty: bool,
}

/// The columns a read must refresh.
pub(crate) enum Refresh<'a> { All(Retention), Changed(&'a [VtreeIdx]) }

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

    /// Mark invalid before entering user arithmetic, so errors and unwinding
    /// cannot leave partially refreshed columns available as cached answers.
    pub(crate) fn refresh(
        &mut self, eng: &Engine, tdd: &Tdd, gate: &mut PollGate,
        mut compute: impl FnMut(&[PinState], Refresh<'_>, &mut PollGate) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        if self.evaluated && self.changed.is_empty() { return Ok(()); }
        let incremental = self.evaluated && self.retention == Retention::All;
        self.evaluated = false;
        if incremental {
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
        }
        let plan = if incremental { Refresh::Changed(&self.changed) } else { Refresh::All(self.retention) };
        compute(&self.pins, plan, gate)?;
        self.clear_changed();
        self.evaluated = true;
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

/// Refresh existing columns, allocating only on a full pass. The caller's
/// fold keeps its specialized arithmetic, overflow handling and polling stride.
pub(crate) fn refresh_columns<F: LevelFold>(
    fold: &F, eng: &Engine, tdd: &Tdd, cols: &mut [F::Col],
    plan: Refresh<'_>, gate: &mut PollGate,
) -> Result<(), OperationError> {
    match plan {
        Refresh::Changed(changed) => {
            for &level in changed { fold_level(fold, eng, tdd, cols, level, gate)?; }
            Ok(())
        }
        Refresh::All(retention) => {
            if retention == Retention::Frontier {
                for col in cols.iter_mut() { *col = F::Col::default(); }
            }
            fold_bottom_up(fold, eng, tdd, cols, retention, gate)
        }
    }
}

/// Owned query state or a temporary engine binding of a persistent query.
pub(crate) enum BoundState<'a, T> { Owned(T), Borrowed(&'a mut T) }
impl<T> BoundState<'_, T> {
    pub(crate) fn get_mut(&mut self) -> &mut T {
        match self { Self::Owned(value) => value, Self::Borrowed(value) => value }
    }
}
