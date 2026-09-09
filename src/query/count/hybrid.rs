//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, SeedConvention};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::engine::PollGate;
use crate::error::ApplyError;
use crate::diagram::PairsIter;
use crate::counts::{
    Count, CountRead, CountVec, RecoveryPanic, STREAM_OVERFLOW as OVERFLOW,
};
use crate::counts::ColumnRetention;
use crate::diagram::*;
use crate::vtree::{BottomUpSubset, VarId, VtreeIdx};
use std::marker::PhantomData;

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(super) struct HybridCounts<'a> {
    pub(super) pins: &'a [Option<bool>],
    pub(super) convention: SeedConvention,
}

impl LevelFold for HybridCounts<'_> {
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

    /// A frozen level's counts are pin-independent — summed out before any pin
    /// existed — so they are read across verbatim.
    fn frozen_column(
        &self,
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        col: &mut CountVec<RecoveryPanic>,
    ) {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a frozen level carries counts");
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

    /// Two passes, and the second one only where the first overflowed.
    ///
    /// The sentinel ⟺ big-slot invariant, the exact-max promotion, and the
    /// stale-overflow clear on recompute (a node may stop overflowing when pins
    /// change) are all owned by [`CountVec::set`] / [`Count::from_u128`].
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, CountVec<RecoveryPanic>>,
        right: Side<'_, CountVec<RecoveryPanic>>,
    ) -> Count {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs.clone() {
            let lc = match left.view.child(pair.left) {
                ChildRef::Value(ValueRef::Inline(c)) => c as u128,
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                    left.col.fast_val(idx as usize)
                }
            };
            // A zero operand contributes 0·rc = 0: skip without even resolving
            // rc. On a pinned cofactor evaluation these dominate — pinning the
            // relaxed variables leaves half to nine tenths of the pairs with a
            // zero operand — so this avoids the bulk of the multiplies.
            if lc == 0 {
                continue;
            }
            let rc = match right.view.child(pair.right) {
                ChildRef::Value(ValueRef::Inline(c)) => c as u128,
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                    right.col.fast_val(idx as usize)
                }
            };
            if rc == 0 {
                continue;
            }
            // OVERFLOW × 1 would not trip checked_mul, so test the sentinel.
            if lc == OVERFLOW || rc == OVERFLOW {
                overflowed = true;
                break;
            }
            match lc.checked_mul(rc).and_then(|p| total.checked_add(p)) {
                Some(v) => total = v,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if !overflowed {
            return Count::from_u128(total);
        }
        let mut bt = BigUint::ZERO;
        for pair in pairs {
            // Resolve each operand to (u128 view, Some(&big) iff it overflowed).
            // Skip zero operands before any allocation — zero is always a clean
            // u128 (only OVERFLOW forces a big read). Then dispatch by width:
            //   both small  → u128 mul (no BigUint operand allocs at all);
            //   mixed       → scalar mul `&big * u128` (no small-operand alloc,
            //                 faster than promoting to BigUint + general mul);
            //   both big    → `&big * &big`, operands borrowed not cloned.
            let (lu, lbig) = match left.view.child(pair.left) {
                ChildRef::Value(ValueRef::Inline(0)) => continue,
                ChildRef::Value(ValueRef::Inline(c)) => (c as u128, None),
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                    let idx = idx as usize;
                    match left.col.fast_val(idx) {
                        0 => continue,
                        OVERFLOW => (OVERFLOW, Some(sentinel_big(left.col, idx))),
                        v => (v, None),
                    }
                }
            };
            let (ru, rbig) = match right.view.child(pair.right) {
                ChildRef::Value(ValueRef::Inline(0)) => continue,
                ChildRef::Value(ValueRef::Inline(c)) => (c as u128, None),
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                    let idx = idx as usize;
                    match right.col.fast_val(idx) {
                        0 => continue,
                        OVERFLOW => (OVERFLOW, Some(sentinel_big(right.col, idx))),
                        v => (v, None),
                    }
                }
            };
            match (lbig, rbig) {
                (None, None) => match lu.checked_mul(ru) {
                    Some(p) => bt += p,
                    None => bt += BigUint::from(lu) * BigUint::from(ru),
                },
                (Some(lb), None) => bt += lb * ru,
                (None, Some(rb)) => bt += rb * lu,
                (Some(lb), Some(rb)) => bt += lb * rb,
            }
        }
        Count::Big(bt)
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

/// Column-lifetime policy as a type: [`AllColumns`] or [`FrontierOnly`].
///
/// The policy decides which reads a counter can serve at all — the per-node
/// array and the dirty-cone update exist only under [`AllColumns`] — so it is
/// a type parameter of [`PinnedCounter`] rather than a field, and the reads it
/// does not support are absent instead of asserting.
pub trait Retention: sealed::Sealed {
    /// The runtime policy the shared walk takes.
    const RETAIN: ColumnRetention;
}

/// Keep every level's column for the counter's lifetime. Costs a column per
/// level; buys the per-node array and the dirty-cone update.
#[derive(Debug, Clone, Copy)]
pub struct AllColumns;

/// Keep only the walk frontier: each child column is freed as its parent's
/// completes. Peak is the frontier rather than the whole diagram, and the root
/// count is the only read.
#[derive(Debug, Clone, Copy)]
pub struct FrontierOnly;

impl Retention for AllColumns {
    const RETAIN: ColumnRetention = ColumnRetention::All;
}

impl Retention for FrontierOnly {
    const RETAIN: ColumnRetention = ColumnRetention::Frontier;
}

/// Whether a counter holds counts yet: [`Fresh`] or [`Computed`].
///
/// Every column starts at zero, so a read before the first pass returns a
/// count indistinguishable from UNSAT. [`PinnedCounter::compute`] is the only
/// way to reach [`Computed`], and the reads live only there.
pub trait CounterState: sealed::Sealed {}

/// Allocated, pinned, never passed over. No count to read.
#[derive(Debug, Clone, Copy)]
pub struct Fresh;

/// Carries the counts of one completed pass under the pins in force at it.
#[derive(Debug, Clone, Copy)]
pub struct Computed;

impl CounterState for Fresh {}
impl CounterState for Computed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::AllColumns {}
    impl Sealed for super::FrontierOnly {}
    impl Sealed for super::Fresh {}
    impl Sealed for super::Computed {}
}

/// Incremental pinned model counter for a Gray-code cofactor sum.
///
/// One hybrid `CountVec` column per vtree level (u128-primary, `BigUint` side
/// table on overflow — keeps 99%+ of arithmetic off the heap; the discipline
/// shared with the apply/marginalize contexts, see `tididi/src/counts.rs`).
/// Under [`AllColumns`] it holds the full per-node count array; after one
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
/// counter therefore serves many evaluations of the SAME diagram: re-pin, then
/// either a dirty-cone [`recompute_dirty`](Self::recompute_dirty) under
/// [`AllColumns`] or a fresh [`compute`](Self::compute). Callers MUST pass the
/// same `tdd` the counter was sized from; a structurally different diagram is a
/// logic error, since the arrays would be mis-sized.
pub struct PinnedCounter<R: Retention, S: CounterState> {
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    /// The leaf-seed convention this counter pins with.
    convention: SeedConvention,
    _marker: PhantomData<(R, S)>,
}

impl<R: Retention> PinnedCounter<R, Fresh> {
    /// Allocate the count array with pin slots `0..n_pins`. No pass run yet.
    ///
    /// `convention` is the leaf seed a pinned variable gets
    /// ([`SeedConvention`]); with zero pins the two coincide.
    ///
    /// Under [`AllColumns`] every level's column is sized here and kept; under
    /// [`FrontierOnly`] the columns are allocated on write and freed as parents
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

impl<R: Retention, S: CounterState> PinnedCounter<R, S> {
    /// Set one variable's pin (does not recompute). `var.idx()` must be `< n_pins`.
    #[inline]
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        self.pins[var.idx()] = val;
    }

    /// Full bottom-up pass under the current pins (every leaf + every internal
    /// level), yielding the counter its counts can be read from.
    ///
    /// This is also how a counter that already holds counts takes a fresh pass
    /// after re-pinning — the only route under [`FrontierOnly`], which has no
    /// incremental path.
    #[must_use = "the pass produces a new counter; the receiver is consumed"]
    pub fn compute(self, eng: &Engine, tdd: &Tdd) -> PinnedCounter<R, Computed> {
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
    ) -> Result<PinnedCounter<R, Computed>, ApplyError> {
        let fold = HybridCounts { pins: &self.pins, convention: self.convention };
        let cols = &mut self.cols;
        if R::RETAIN == ColumnRetention::Frontier {
            // Free-before-rebuild: drop the previous pass's surviving column
            // (the root's, plus any level this pass will not revisit) BEFORE
            // allocating anything new, so two passes' peaks never overlap.
            for c in cols.iter_mut() {
                *c = CountVec::with_width(eng, 0);
            }
        }
        fold_bottom_up(&fold, eng, tdd, cols, R::RETAIN, poll, |cols, ti| {
            // Under `AllColumns` the constructor pre-sized every column and
            // nothing shrinks them, so this is one length compare per level;
            // under `FrontierOnly` it is the allocate-on-write step for a
            // column that starts — or was freed — empty. A freshly allocated
            // column is all-zero, which is what a fresh counter's column holds,
            // so slots no pass writes (tombstones, which the fold skips) read
            // the same under both policies.
            let w = tdd.effective_width(VtreeIdx(ti as u32));
            if cols[ti].len() != w {
                cols[ti] = CountVec::with_width(eng, w);
            }
        })?;
        Ok(PinnedCounter {
            cols: self.cols,
            pins: self.pins,
            convention: self.convention,
            _marker: PhantomData,
        })
    }
}

impl<R: Retention> PinnedCounter<R, Computed> {
    /// The current root (output) model count.
    #[inline]
    pub fn root_count(&self, tdd: &Tdd) -> BigUint {
        let (t, i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
        match self.cols[t].get(i) {
            CountRead::Fast(v) => BigUint::from(v),
            CountRead::Big(b) => b.clone(),
        }
    }
}

impl PinnedCounter<AllColumns, Computed> {
    /// Recompute exactly `levels`. Leaf levels are re-seeded from the current
    /// pins; internal levels are re-summed from their (already-updated)
    /// children, which the subset's bottom-up order guarantees are current.
    ///
    /// Only [`AllColumns`] offers this: the dirty-cone update re-reads cached
    /// columns outside `levels`, which [`FrontierOnly`] frees as parents
    /// complete.
    pub fn recompute_dirty(&mut self, eng: &Engine, tdd: &Tdd, levels: &BottomUpSubset) {
        let fold = HybridCounts { pins: &self.pins, convention: self.convention };
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
    /// detection (sat-prune MC-priority), never an overflowed node's exact value.
    ///
    /// Only [`AllColumns`] offers this: [`FrontierOnly`] keeps the root column
    /// alone, so there is no per-node array to hand out.
    pub(crate) fn into_fast_counts(self) -> Vec<Vec<u128>> {
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
