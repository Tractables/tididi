//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, SeedConvention};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::limits::PollGate;
use crate::limits::ApplyError;
use crate::diagram::PairsIter;
use crate::value::{ColumnRetention, Count, CountRead, CountVec, IntFold};
use crate::limits::RecoveryPanic;
use crate::diagram::*;
use crate::vtree::{VarId, VtreeIdx};
use std::marker::PhantomData;

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(super) struct OverflowingCounts<'a> {
    pub(super) pins: &'a [Option<bool>],
    pub(super) convention: SeedConvention,
}

impl LevelFold for OverflowingCounts<'_> {
    type Value = Count;
    type Col = CountVec<RecoveryPanic>;

    fn alloc(&self, eng: &Engine, width: usize) -> CountVec<RecoveryPanic> {
        CountVec::with_width(eng, width)
    }

    fn set(&self, eng: &Engine, col: &mut CountVec<RecoveryPanic>, i: usize, v: Count) {
        col.set_i(eng, i, v);
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> Count {
        let pin = self.pins.get(var.idx()).copied().flatten();
        Count::from_u128(leaf_seed(label, pin, self.convention))
    }

    /// A marginal level's counts are pin-independent — summed out before any pin
    /// existed — so they are read across verbatim.
    fn marginal_column(
        &self,
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        col: &mut CountVec<RecoveryPanic>,
    ) {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a marginal level carries counts");
        let big = level.marginal_counts_big();
        for i in 0..counts.len() {
            col.set_i(eng, i, CountRead::from_slot(counts, big, i).to_count());
        }
    }

    /// The shared two-pass integer fold, with this query's child readers.
    ///
    /// Reading a child is the only thing that differs from any other integer
    /// fold: a pinned counter resolves through a [`SideView`], which knows
    /// whether the ref is a node index or a marginal-side value.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, CountVec<RecoveryPanic>>,
        right: Side<'_, CountVec<RecoveryPanic>>,
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
fn read_side<'a>(side: Side<'a, CountVec<RecoveryPanic>>, k: usize) -> CountRead<'a> {
    let idx = match side.view.child(NodeIdx(k as u32)) {
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
/// [`output_count`](Self::output_count) refreshes changed pins before reading:
/// [`KeepAllColumns`] recomputes their ancestor cone, and [`KeepFrontier`]
/// performs a full fold while freeing completed child columns. The diagram
/// need not be canonical. Count-marginal levels keep their stored values;
/// pins do not reach a region already summed out.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{IncrementalCounter, KeepAllColumns, SeedConvention};
/// use tididi::vtree::{VarId, Vtree};
/// let engine = Engine::new();
/// let tree = Arc::new(Vtree::balanced(4));
/// let f = Tdd::clause(&tree, [1, -2]);
/// let mut counter = IncrementalCounter::<KeepAllColumns>::new(&engine, &f, 4, SeedConvention::Fixed);
/// assert_eq!(counter.output_count(&engine), f.model_count());
/// counter.set_pin(VarId(0), Some(true));
/// assert!(counter.output_count(&engine) <= f.model_count());
/// ```
///
/// The diagram cannot change while a counter borrowing it remains in use:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{IncrementalCounter, KeepAllColumns, SeedConvention};
/// use tididi::vtree::Vtree;
/// let eng = Engine::new();
/// let tree = Arc::new(Vtree::balanced(2));
/// let mut f = Tdd::clause(&tree, [1]);
/// let mut counter = IncrementalCounter::<KeepAllColumns>::new(&eng, &f, 2, SeedConvention::Fixed);
/// tididi::reduce::minimize(&mut f);
/// counter.output_count(&eng);
/// ```
pub struct IncrementalCounter<'a, R: Retention> {
    tdd: &'a Tdd,
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    changed: Vec<VarId>,
    convention: SeedConvention,
    evaluated: bool,
    _marker: PhantomData<R>,
}

impl<R: Retention> std::fmt::Debug for IncrementalCounter<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalCounter")
            .field("retention", &R::RETAIN)
            .field("evaluated", &self.evaluated)
            .field("pins", &self.pins)
            .field("changed_since_pass", &self.changed.len())
            .finish()
    }
}

impl<'a, R: Retention> IncrementalCounter<'a, R> {
    /// Allocate a counter bound to `tdd`, with an initially unpinned table of `n_pins` variables.
    ///
    /// # Panics
    ///
    /// Panics if an allocation is refused.
    pub fn new(eng: &Engine, tdd: &'a Tdd, n_pins: usize, convention: SeedConvention) -> Self {
        let _op = eng.limits().begin_operation();
        let cols = (0..tdd.vtree.num_nodes()).map(|i| {
            let width = match R::RETAIN {
                ColumnRetention::All => tdd.effective_width(VtreeIdx(i as u32)),
                ColumnRetention::Frontier => 0,
            };
            CountVec::with_width(eng, width)
        }).collect();
        Self { tdd, cols, pins: vec![None; n_pins], changed: Vec::new(), convention, evaluated: false, _marker: PhantomData }
    }

    /// Change a variable's pin, deferring its affected counts until the next read.
    ///
    /// # Panics
    ///
    /// Panics if `var` is outside the pin table supplied to [`new`](Self::new).
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        assert!(var.idx() < self.pins.len(), "IncrementalCounter::set_pin: {:?} is not below the counter's {} pins", var, self.pins.len());
        if self.pins[var.idx()] == val { return; }
        self.pins[var.idx()] = val;
        if !self.changed.contains(&var) { self.changed.push(var); }
    }

    /// The model count under the current pins, refreshing cached columns without polling limits.
    ///
    /// # Panics
    ///
    /// Panics if the diagram has weighted marginal levels or an allocation is refused.
    pub fn output_count(&mut self, eng: &Engine) -> BigUint {
        self.try_count(eng, None).expect("an unpolled count observes no stop axis")
    }

    /// Refresh and read the root, polling when a gate is supplied.
    pub(crate) fn try_count(&mut self, eng: &Engine, poll: Option<&mut PollGate>) -> Result<BigUint, ApplyError> {
        let _op = eng.limits().begin_operation();
        if self.tdd.is_zero() { return Ok(BigUint::ZERO); }
        self.refresh(eng, poll)?;
        let out = self.tdd.output;
        Ok(match self.cols[out.vtree.idx()].get(out.local.idx()) {
            CountRead::Fast(value) => BigUint::from(value),
            CountRead::Big(value) => value.clone(),
        })
    }

    /// Refresh all columns or the dirty ancestor cone, leaving a failed pass invalidated.
    fn refresh(&mut self, eng: &Engine, mut poll: Option<&mut PollGate>) -> Result<(), ApplyError> {
        if self.evaluated && self.changed.is_empty() { return Ok(()); }
        let incremental = self.evaluated && R::RETAIN == ColumnRetention::All;
        self.evaluated = false;
        let tdd = self.tdd;
        let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
        if incremental {
            let mut in_cone = vec![false; tdd.vtree.num_nodes()];
            let mut cone = Vec::new();
            for &var in &self.changed {
                let Some(mut t) = tdd.vtree.leaf_of(var) else { continue };
                while !in_cone[t.idx()] {
                    in_cone[t.idx()] = true;
                    cone.push(t);
                    match tdd.vtree.node(t).parent() {
                        Some(parent) => t = parent,
                        None => break,
                    }
                }
            }
            for &t in tdd.vtree.bottom_up_subset(cone).levels() {
                fold_level(&fold, eng, tdd, &mut self.cols, t);
                if let Some(gate) = poll.as_deref_mut() {
                    eng.limits().poll(gate, tdd.effective_width(t) as u64)?;
                }
            }
            if let Some(gate) = poll { eng.limits().flush_poll(gate)?; }
        } else {
            if R::RETAIN == ColumnRetention::Frontier {
                for col in &mut self.cols { *col = CountVec::with_width(eng, 0); }
            }
            fold_bottom_up(&fold, eng, tdd, &mut self.cols, R::RETAIN, poll, |cols, ti| {
                let width = tdd.effective_width(VtreeIdx(ti as u32));
                if cols[ti].len() != width { cols[ti] = CountVec::with_width(eng, width); }
            })?;
        }
        self.changed.clear();
        self.evaluated = true;
        Ok(())
    }
}

impl IncrementalCounter<'_, KeepAllColumns> {
    /// Compute and return every fast count slot, preserving overflow sentinels.
    pub(crate) fn into_fast_counts(mut self, eng: &Engine) -> Vec<Vec<u128>> {
        self.refresh(eng, None).expect("an unpolled count observes no stop axis");
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
