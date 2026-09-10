//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, SeedConvention};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::engine::PollGate;
use crate::error::ApplyError;
use crate::diagram::PairsIter;
use crate::value_fold::{
    ColumnRetention, Count, CountRead, CountVec, IntFold, COUNT_OVERFLOW as OVERFLOW,
};
use crate::engine::RecoveryPanic;
use crate::diagram::*;
use crate::vtree::{BottomUpSubset, VarId, VtreeIdx};
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
        for (i, &c) in counts.iter().enumerate() {
            if c == OVERFLOW {
                let bv = level
                    .marginal_counts_big()
                    .and_then(|m| m.get(i).cloned())
                    .expect("marginal OVERFLOW slot without a big entry — level invariant violated");
                col.set_i(eng, i, Count::Big(bv));
            } else {
                col.set_i(eng, i, Count::from_u128(c));
            }
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
    match side.col.fast_val(idx) {
        OVERFLOW => CountRead::Big(sentinel_big(side.col, idx)),
        v => CountRead::Fast(v),
    }
}

/// Borrow the big value of a slot known to hold the overflow sentinel.
/// Precondition: `cols.fast_val(node) == OVERFLOW` (then `big_val` is `Some`
/// by the `CountVec` invariant).
#[inline]
fn sentinel_big(col: &CountVec<RecoveryPanic>, node: usize) -> &BigUint {
    col.big_val(node)
        .expect("CountVec: sentinel fast slot without a big value — invariant violated")
}

/// Column-lifetime policy as a type: [`KeepAllColumns`] or [`KeepFrontier`].
///
/// The policy decides which reads a counter can serve at all — the per-node
/// array and the dirty-cone update exist only under [`KeepAllColumns`] — so it is
/// a type parameter of [`IncrementalCounter`] rather than a field, and the reads it
/// does not support are absent instead of asserting.
pub trait Retention: sealed::Sealed {
    /// The runtime policy the shared walk takes.
    const RETAIN: ColumnRetention;
}

/// Keep every level's column for the counter's lifetime. Costs a column per
/// level; buys the per-node array and the dirty-cone update.
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

/// Whether a counter holds counts yet: [`Unevaluated`] or [`Evaluated`].
///
/// Every column starts at zero, so a read before the first pass returns a
/// count indistinguishable from UNSAT. [`IncrementalCounter::compute`] is the only
/// way to reach [`Evaluated`], and the reads live only there.
pub trait CounterState: sealed::Sealed {}

/// Allocated, pinned, never passed over. No count to read.
#[derive(Debug, Clone, Copy)]
pub struct Unevaluated;

/// Carries the counts of one completed pass under the pins in force at it.
#[derive(Debug, Clone, Copy)]
pub struct Evaluated;

impl CounterState for Unevaluated {}
impl CounterState for Evaluated {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::KeepAllColumns {}
    impl Sealed for super::KeepFrontier {}
    impl Sealed for super::Unevaluated {}
    impl Sealed for super::Evaluated {}
}

/// Incremental pinned model counter for a Gray-code cofactor sum.
///
/// One hybrid `CountVec` column per vtree level (u128-primary, `BigUint` side
/// table on overflow, which keeps most of the arithmetic off the heap; the
/// discipline is shared with the apply and marginalize contexts, see
/// `value_fold`).
/// Under [`KeepAllColumns`] it holds the full per-node count array; after one
/// [`compute`](Self::compute), flipping a few variables' pins and calling
/// [`recompute_dirty`](Self::recompute_dirty) on just the affected vtree levels
/// (the "dirty cone" from those leaves to the root) updates the root count in
/// `O(cone)` instead of the `O(|D|)` of a fresh full pass — every unaffected
/// node's cached count is reused verbatim. The result equals `pinned_counts`.
///
/// The two type parameters are the counter's capabilities rather than
/// documentation: `R` ([`Retention`]) decides whether the per-node array and
/// the dirty-cone update exist, and `S` ([`CounterState`]) whether there is
/// anything to read.
///
/// The counter OWNS its columns and pins (sized from `tdd` at construction) and
/// does not borrow the diagram — every method takes `tdd` as an argument. One
/// counter therefore serves many evaluations of the same diagram: re-pin, then
/// either a dirty-cone [`recompute_dirty`](Self::recompute_dirty) under
/// [`KeepAllColumns`] or a fresh [`compute`](Self::compute). Callers must pass the
/// same `tdd` the counter was sized from; a structurally different diagram is a
/// logic error, since the arrays would be mis-sized.
pub struct IncrementalCounter<R: Retention, S: CounterState> {
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    /// The leaf-seed convention this counter pins with.
    convention: SeedConvention,
    _marker: PhantomData<(R, S)>,
}

impl<R: Retention> IncrementalCounter<R, Unevaluated> {
    /// Allocate the count array with pin slots `0..n_pins`. No pass run yet.
    ///
    /// `convention` is the leaf seed a pinned variable gets
    /// ([`SeedConvention`]); with zero pins the two coincide.
    ///
    /// Under [`KeepAllColumns`] every level's column is sized here and kept; under
    /// [`KeepFrontier`] the columns are allocated on write and freed as parents
    /// complete, so pre-sizing them would commit exactly the whole-diagram
    /// array that policy exists to avoid.
    pub fn new(eng: &Engine, tdd: &Tdd, n_pins: usize, convention: SeedConvention) -> Self {
        let cols = (0..tdd.vtree.num_nodes())
            .map(|i| match R::RETAIN {
                ColumnRetention::All => {
                    CountVec::with_width(eng, tdd.effective_width(VtreeIdx(i as u32)))
                }
                ColumnRetention::Frontier => CountVec::with_width(eng, 0),
            })
            .collect();
        Self { cols, pins: vec![None; n_pins], convention, _marker: PhantomData }
    }
}

impl<R: Retention, S: CounterState> IncrementalCounter<R, S> {
    /// Set one variable's pin (does not recompute). `var.idx()` must be `< n_pins`.
    #[inline]
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        self.pins[var.idx()] = val;
    }

    /// Full bottom-up pass under the current pins (every leaf + every internal
    /// level), yielding the counter its counts can be read from.
    ///
    /// This is also how a counter that already holds counts takes a fresh pass
    /// after re-pinning — the only route under [`KeepFrontier`], which has no
    /// incremental path.
    #[must_use = "the pass produces a new counter; the receiver is consumed"]
    pub fn compute(self, eng: &Engine, tdd: &Tdd) -> IncrementalCounter<R, Evaluated> {
        self.try_compute(eng, tdd, None)
            .expect("an unpolled pass observes no stop axis")
    }

    /// [`compute`](Self::compute) under a stop axis: the pass is cut between
    /// levels, where every level below the cut holds a complete column and
    /// nothing has been read yet.
    ///
    /// # Errors
    ///
    /// Propagates the armed stop, polled at every internal level boundary. The
    /// cut counter is dropped with its partial columns — there is no reading a
    /// pass that did not finish.
    pub(crate) fn try_compute(
        mut self,
        eng: &Engine,
        tdd: &Tdd,
        poll: Option<&mut PollGate>,
    ) -> Result<IncrementalCounter<R, Evaluated>, ApplyError> {
        let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
        let cols = &mut self.cols;
        if R::RETAIN == ColumnRetention::Frontier {
            // Free-before-rebuild: drop the previous pass's surviving column
            // (the root's, plus any level this pass will not revisit) before
            // allocating anything new, so two passes' peaks never overlap.
            for c in cols.iter_mut() {
                *c = CountVec::with_width(eng, 0);
            }
        }
        fold_bottom_up(&fold, eng, tdd, cols, R::RETAIN, poll, |cols, ti| {
            // Under `KeepAllColumns` the constructor pre-sized every column and
            // nothing shrinks them, so this is one length compare per level;
            // under `KeepFrontier` it is the allocate-on-write step for a
            // column that starts — or was freed — empty. A freshly allocated
            // column is all-zero, which is what a fresh counter's column holds,
            // so slots no pass writes (tombstones, which the fold skips) read
            // the same under both policies.
            let w = tdd.effective_width(VtreeIdx(ti as u32));
            if cols[ti].len() != w {
                cols[ti] = CountVec::with_width(eng, w);
            }
        })?;
        Ok(IncrementalCounter {
            cols: self.cols,
            pins: self.pins,
            convention: self.convention,
            _marker: PhantomData,
        })
    }
}

impl<R: Retention> IncrementalCounter<R, Evaluated> {
    /// The current root (output) model count.
    #[inline]
    pub fn output_count(&self, tdd: &Tdd) -> BigUint {
        let (t, i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
        match self.cols[t].get(i) {
            CountRead::Fast(v) => BigUint::from(v),
            CountRead::Big(b) => b.clone(),
        }
    }
}

impl IncrementalCounter<KeepAllColumns, Evaluated> {
    /// Recompute exactly `levels`. Leaf levels are re-seeded from the current
    /// pins; internal levels are re-summed from their (already-updated)
    /// children, which the subset's bottom-up order guarantees are current.
    ///
    /// Only [`KeepAllColumns`] offers this: the dirty-cone update re-reads cached
    /// columns outside `levels`, which [`KeepFrontier`] frees as parents
    /// complete.
    pub fn recompute_dirty(&mut self, eng: &Engine, tdd: &Tdd, levels: &BottomUpSubset) {
        let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
        for &t in levels.levels() {
            fold_level(&fold, eng, tdd, &mut self.cols, t);
        }
    }

    /// Consume the counter, returning the per-node u128 count columns
    /// (`fast[t][i]`) and discarding the `BigUint` side table. A slot that
    /// counted past `u128` saturates to `OVERFLOW` (`u128::MAX`) and its exact
    /// magnitude is dropped. ZERO is exact: the u128 array is authoritative for
    /// zero — only a *non-zero* overflow ever spills to the Big side table — so
    /// `fast[t][i] == 0` iff node `(t,i)` has no models. For callers that need
    /// only monotone ordering, a small-threshold compare, and exact-zero
    /// detection, never an overflowed node's exact value.
    ///
    /// Only [`KeepAllColumns`] offers this: [`KeepFrontier`] keeps the root column
    /// alone, so there is no per-node array to hand out.
    pub(crate) fn into_fast_counts(self) -> Vec<Vec<u128>> {
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
