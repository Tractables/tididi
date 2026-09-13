//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{EncodedChildRef, ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, PinSemantics};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::limits::PollGate;
use crate::limits::OperationError;
use crate::diagram::PairsIter;
use crate::value::{ColumnRetention, Count, CountRead, CountVec, IntFold};
use crate::limits::ApplyBudget;
use crate::diagram::*;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use std::marker::PhantomData;

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(super) struct OverflowingCounts<'a> {
    pub(super) vtree: &'a Vtree,
    pub(super) pins: &'a [Option<bool>],
    pub(super) convention: PinSemantics,
}

impl LevelFold for OverflowingCounts<'_> {
    type Value = Count;
    type Col = CountVec<ApplyBudget>;

    fn alloc(&self, eng: &Engine, width: usize) -> Result<CountVec<ApplyBudget>, OperationError> {
        CountVec::try_with_width(eng, width)
    }

    fn release(&self, eng: &Engine, col: &mut Self::Col) {
        let bytes = col.buffer_bytes();
        *col = CountVec::default();
        eng.limits().release_bytes(bytes);
    }

    fn set(&self, eng: &Engine, col: &mut CountVec<ApplyBudget>, i: usize, v: Count) -> Result<(), OperationError> {
        col.set(eng, i, v)
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> Count {
        let leaf = self.vtree.leaf_of(var).expect("the fold visits a vtree leaf");
        let pin = self.pins.get(leaf.idx()).copied().flatten();
        Count::from_u128(leaf_seed(label, pin, self.convention))
    }

    /// A marginal level's counts are pin-independent — summed out before any pin
    /// existed — so they are read across verbatim.
    fn marginal_column(
        &self,
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        col: &mut CountVec<ApplyBudget>,
    ) -> Result<(), OperationError> {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a marginal level carries counts");
        let big = level.marginal_counts_big();
        for i in 0..counts.len() {
            col.set(eng, i, CountRead::from_slot(counts, big, i).to_count())?;
        }
        Ok(())
    }

    /// The shared two-pass integer fold, with this query's child readers.
    ///
    /// Reading a child is the only thing that differs from any other integer
    /// fold: a pinned counter resolves through a [`ChildDecoder`], which knows
    /// whether the ref is a node index or a marginal-side value.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, CountVec<ApplyBudget>>,
        right: Side<'_, CountVec<ApplyBudget>>,
    ) -> Count {
        IntFold::fold(pairs, |k| read_side(left, k), |k| read_side(right, k))
    }
}

/// Resolve one child ref of a pinned counter to a count read.
///
/// The sentinel ⟺ big-slot invariant, the exact-max promotion, and the
/// stale-overflow clear on recompute (a node may stop overflowing when pins
/// change) are all owned by [`CountVec::set`] / [`Count::from_u128`].
#[inline]
fn read_side<'a>(side: Side<'a, CountVec<ApplyBudget>>, k: EncodedChildRef) -> CountRead<'a> {
    let idx = match side.view.child(k) {
        ChildRef::Value(ValueRef::Inline(c)) => return CountRead::Fast(c as u128),
        ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => idx as usize,
    };
    side.col.get(idx)
}

/// The column retention used by a counter: [`KeepAllColumns`] or [`KeepFrontier`].
pub trait Retention: sealed::Sealed {
    /// The runtime policy the shared walk takes.
    const RETAIN: ColumnRetention;
}

/// Keep every column so reads after pin changes need only their ancestor cone.
#[derive(Debug, Clone, Copy)]
pub struct KeepAllColumns;

/// Keep only the walk frontier: each child column is freed as its parent's
/// completes. Peak is the frontier rather than the whole diagram, and the root
/// count is the only read.
#[derive(Debug, Clone, Copy)]
pub struct KeepFrontier;

impl Retention for KeepAllColumns {
    const RETAIN: ColumnRetention = ColumnRetention::All;
}

impl Retention for KeepFrontier {
    const RETAIN: ColumnRetention = ColumnRetention::Frontier;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::KeepAllColumns {}
    impl Sealed for super::KeepFrontier {}
}

/// A pinned model counter borrowing the diagram whose columns it caches.
///
/// [`try_model_count`](Self::try_model_count) refreshes changed pins before reading:
/// [`KeepAllColumns`] recomputes their ancestor cone, and [`KeepFrontier`]
/// performs a full fold while freeing completed child columns. The diagram
/// need not be canonical. Count-marginal levels keep their stored values;
/// [`set_pin`](Self::set_pin) rejects variables already summed out.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{ModelCounter, KeepAllColumns, PinSemantics};
/// use tididi::vtree::{VarId, Vtree};
/// let engine = Engine::new();
/// let tree = Arc::new(Vtree::balanced(4));
/// let f = Tdd::clause(&tree, [1, -2]);
/// let mut counter = ModelCounter::<KeepAllColumns>::try_new(&engine, &f, PinSemantics::Evidence)?;
/// assert_eq!(counter.try_model_count(&engine)?, f.model_count());
/// counter.set_pin(VarId(0), Some(true))?;
/// assert!(counter.try_model_count(&engine)? <= f.model_count());
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// The diagram cannot change while a counter borrowing it remains in use:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{ModelCounter, KeepAllColumns, PinSemantics};
/// use tididi::vtree::Vtree;
/// let eng = Engine::new();
/// let tree = Arc::new(Vtree::balanced(2));
/// let mut f = Tdd::clause(&tree, [1]);
/// let mut counter = ModelCounter::<KeepAllColumns>::new(&eng, &f, PinSemantics::Evidence);
/// tididi::reduce::minimize(&mut f);
/// counter.model_count(&eng);
/// ```
pub struct ModelCounter<'a, R: Retention> {
    tdd: &'a Tdd,
    cols: Vec<CountVec<ApplyBudget>>,
    pins: Vec<Option<bool>>,
    changed: Vec<VtreeIdx>,
    convention: PinSemantics,
    evaluated: bool,
    _marker: PhantomData<R>,
}

impl<R: Retention> std::fmt::Debug for ModelCounter<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelCounter")
            .field("retention", &R::RETAIN)
            .field("evaluated", &self.evaluated)
            .field("pins", &self.pins)
            .field("changed_since_pass", &self.changed.len())
            .finish()
    }
}

impl<'a, R: Retention> ModelCounter<'a, R> {
    /// Allocate a counter, panicking on an error from [`Self::try_new`].
    ///
    /// # Panics
    ///
    /// Panics on a weighted marginal level, a reservation refusal or an armed stop.
    pub fn new(eng: &Engine, tdd: &'a Tdd, convention: PinSemantics) -> Self {
        Self::try_new(eng, tdd, convention).expect("ModelCounter::new: operation refused")
    }

    /// Create an initially unpinned counter over the diagram's vtree under the engine's limits.
    ///
    /// Pin storage is proportional to the tree's size, including for sparse
    /// variable IDs. Value columns are allocated on the first count, and
    /// pin changes require no further allocation.
    ///
    /// # Errors
    ///
    /// [`OperationError::IncompatibleWeights`] for a weighted marginal level,
    /// [`OperationError::OverBudget`] for a refused buffer reservation, or
    /// [`OperationError::Stopped`] for an armed stop.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::query::{ModelCounter, KeepAllColumns, PinSemantics};
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&tree, [1, 2]);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = ModelCounter::<KeepAllColumns>::try_new(
    ///     &engine, &f, PinSemantics::Evidence)?;
    /// counter.set_pin(VarId(0), Some(false))?;
    /// assert_eq!(counter.try_model_count(&engine)?, 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn try_new(eng: &Engine, tdd: &'a Tdd, convention: PinSemantics) -> Result<Self, OperationError> {
        Self::allocate(eng, tdd, tdd.vtree.num_leaves() as usize, convention)
    }

    /// Allocate leaf-indexed pin slots, or zero slots for an internal unpinned query.
    pub(super) fn allocate(eng: &Engine, tdd: &'a Tdd, pin_slots: usize, convention: PinSemantics) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        if tdd.levels.iter().any(|level| level.is_weight_marginal()) {
            return Err(OperationError::IncompatibleWeights);
        }
        if lim.should_stop() { return Err(OperationError::Stopped); }
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
        cols.resize_with(tdd.vtree.num_nodes(), CountVec::default);
        let mut pins = Vec::new();
        lim.try_resize(&mut pins, pin_slots, None)?;
        let mut changed = Vec::new();
        if pin_slots != 0 { lim.reserve_exact(&mut changed, pin_slots)?; }
        if lim.should_stop() { return Err(OperationError::Stopped); }
        Ok(Self { tdd, cols, pins, changed, convention, evaluated: false, _marker: PhantomData })
    }

    /// Set or clear a vtree variable's pin, deferring affected counts until the next read.
    ///
    /// `Some(value)` pins the variable and `None` removes its pin. A variable
    /// absent from the function's support can still be pinned if the vtree
    /// carries it. Repeating a pin leaves the cached counts valid.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// or [`OperationError::MarginalLevel`] for a variable already summed out.
    /// An error leaves pins and cached counts unchanged, including when clearing a pin.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, OperationError, Tdd, Vtree};
    /// use tididi::query::{KeepAllColumns, ModelCounter, PinSemantics};
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::leaf(VarId(7)));
    /// let f = Tdd::one(&tree);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = ModelCounter::<KeepAllColumns>::try_new(&engine, &f, PinSemantics::Evidence)?;
    /// counter.set_pin(VarId(7), Some(true))?;
    /// assert_eq!(counter.try_model_count(&engine)?, 1u32.into());
    /// assert_eq!(counter.set_pin(VarId(0), Some(true)), Err(OperationError::VariableNotInVtree(VarId(0))));
    /// counter.set_pin(VarId(7), None)?;
    /// assert_eq!(counter.try_model_count(&engine)?, 2u32.into());
    /// # Ok::<(), OperationError>(())
    /// ```
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        let leaf = self.tdd.vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        // An implicit integer leaf can remain below a marginal parent.
        for level in std::iter::once(leaf).chain(self.tdd.vtree.node(leaf).parent()) {
            if self.tdd.levels[level.idx()].is_marginal() {
                return Err(OperationError::MarginalLevel(level));
            }
        }
        if self.pins[leaf.idx()] == val { return Ok(()); }
        self.pins[leaf.idx()] = val;
        if !self.changed.contains(&leaf) { self.changed.push(leaf); }
        Ok(())
    }

    /// Count under the engine's limits, panicking on an error from [`Self::try_model_count`].
    ///
    /// # Panics
    ///
    /// Panics on a reservation refusal or an armed stop.
    pub fn model_count(&mut self, eng: &Engine) -> BigUint {
        self.try_model_count(eng).expect("ModelCounter::model_count: operation refused")
    }

    /// Refresh the current pins and count under the engine's allocation and stop rules.
    ///
    /// Stops are checked even for a cached or constant answer, at amortized node
    /// boundaries during a refresh, and before return. A refusal invalidates the
    /// cached result; a later call recomputes before reading it. Pins are retained.
    /// The best-effort byte budget covers buffer growth during this call, not
    /// retained columns or allocations inside big-integer arithmetic.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] for a refused scratch or overflow-table
    /// reservation, or [`OperationError::Stopped`] for an armed stop.
    pub fn try_model_count(&mut self, eng: &Engine) -> Result<BigUint, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        let result = (|| {
            if lim.should_stop() { return Err(OperationError::Stopped); }
            let mut gate = PollGate::new(lim.reduce_poll_stride());
            let count = if self.tdd.is_zero() {
                BigUint::ZERO
            } else {
                self.refresh(eng, &mut gate)?;
                let out = self.tdd.output;
                match self.cols[out.vtree.idx()].get(out.local.idx()) {
                    CountRead::Fast(value) => BigUint::from(value),
                    CountRead::Big(value) => value.clone(),
                }
            };
            lim.poll(&mut gate, 1)?;
            lim.flush_poll(&mut gate)?;
            Ok(count)
        })();
        if result.is_err() { self.evaluated = false; }
        result
    }

    /// Refresh all columns or the dirty ancestor cone, leaving a failed pass invalidated.
    fn refresh(&mut self, eng: &Engine, gate: &mut PollGate) -> Result<(), OperationError> {
        if self.evaluated && self.changed.is_empty() { return Ok(()); }
        let incremental = self.evaluated && R::RETAIN == ColumnRetention::All;
        self.evaluated = false;
        let tdd = self.tdd;
        let fold = OverflowingCounts { vtree: &tdd.vtree, pins: &self.pins, convention: self.convention };
        if incremental {
            let mut in_cone = Vec::new();
            eng.limits().try_resize(&mut in_cone, tdd.vtree.num_nodes(), false)?;
            let mut cone = Vec::new();
            for &leaf in &self.changed {
                let mut t = leaf;
                while !in_cone[t.idx()] {
                    in_cone[t.idx()] = true;
                    eng.limits().try_push(&mut cone, t)?;
                    eng.limits().poll(gate, 1)?;
                    match tdd.vtree.node(t).parent() {
                        Some(parent) => t = parent,
                        None => break,
                    }
                }
            }
            for &t in tdd.vtree.bottom_up_subset(cone).levels() {
                fold_level(&fold, eng, tdd, &mut self.cols, t, Some(gate))?;
            }
        } else {
            if R::RETAIN == ColumnRetention::Frontier {
                for col in &mut self.cols { *col = CountVec::default(); }
            }
            fold_bottom_up(&fold, eng, tdd, &mut self.cols, R::RETAIN, Some(gate), |cols, ti| {
                let width = tdd.reference_slot_count(VtreeIdx(ti as u32));
                if cols[ti].len() != width { cols[ti] = fold.alloc(eng, width)?; }
                Ok(())
            })?;
        }
        self.changed.clear();
        self.evaluated = true;
        Ok(())
    }
}

impl ModelCounter<'_, KeepAllColumns> {
    /// Compute and return every fast count slot, preserving overflow sentinels.
    pub(crate) fn into_fast_counts(mut self, eng: &Engine) -> Vec<Vec<u128>> {
        let _op = eng.limits().begin_operation();
        let mut gate = PollGate::new(eng.limits().reduce_poll_stride());
        self.refresh(eng, &mut gate).expect("node_counts_u128: operation refused");
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
