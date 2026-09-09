//! What kind of apply is running: which levels it visits, which shortcuts it
//! may take, and how it hands its result back.
//!
//! A conjunction is either FULL — every level of the vtree is rebuilt from the
//! two operands — or RESTRICTED to an ancestor-closed set `R`, where the
//! accumulator's own levels ride through untouched and only `R` is rebuilt.
//! The two differ in a handful of specific places, and every one of them is a
//! method here: the driver names the difference once, at the top, and then
//! never asks again.

use std::sync::Arc;

use crate::diagram::{Tdd, TddLevel, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::WeightStore;

use super::restrict::Restrict;
use super::route::LevelWalk;

/// The levels an apply visits. Under a restriction only `R ∪ children(R)` is
/// ever indexed, so whole-array sweeps become `O(|R|)` walks.
pub(super) enum TouchedLevels<'a> {
    All(std::ops::Range<usize>),
    Some(std::slice::Iter<'a, VtreeIdx>),
}

impl Iterator for TouchedLevels<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            TouchedLevels::All(r) => r.next(),
            TouchedLevels::Some(it) => it.next().map(|t| t.idx()),
        }
    }
}

/// How a finished level array becomes the output diagram.
///
/// Separate from [`ApplyPlan`] because the clause conjunction shares this half
/// and nothing else: it builds no product grid, so it has no level walk, no
/// identity vectors and no sparse machinery — but it does end the same way,
/// carrying the accumulator's outstanding contraction debt forward and moving
/// its frozen weights across.
pub(crate) trait OutputPlan {
    /// Turn the finished level array into the output diagram.
    ///
    /// `acc` is the accumulator the levels came from; a plan that rebuilt only
    /// part of it merges the two here.
    fn finish(
        &self,
        acc: &mut Tdd,
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
        weights: Option<WeightStore>,
    ) -> Tdd;
}

/// Carry the accumulator's outstanding contraction debt forward, seeded with
/// the levels this apply rebuilt, and attach the weights.
///
/// Seeding only the rebuilt levels is exact, not merely sound: a rebuilt set
/// that is ancestor-closed has a DESCENDANT-closed complement, so an off-set
/// level's own pairs, its parent's pairs and its whole subtree are bit-identical
/// to the accumulator's. A contraction sweep seeded there would re-run the
/// accumulator's own last sweep on the same bytes and fire nothing. Whatever
/// the accumulator still owed is carried over rather than dropped, which is
/// what keeps this exact for a caller that does not minimize between applies
/// (`with_levels_dirty`'s second obligation).
fn finish_rebuilt(
    acc: &mut Tdd,
    vtree: Arc<Vtree>,
    levels: Vec<TddLevel>,
    output: TddNodeId,
    rebuilt: &[VtreeIdx],
    weights: Option<WeightStore>,
) -> Tdd {
    let mut dirty_contract = std::mem::take(&mut acc.dirty.contract);
    let mut dirty_leaf_contract = std::mem::take(&mut acc.dirty.leaf_contract);
    dirty_contract.reserve(rebuilt.len());
    dirty_leaf_contract.reserve(rebuilt.len());
    for &t in rebuilt {
        dirty_contract.push(t.0);
        dirty_leaf_contract.push(t.0);
    }
    let mut out =
        Tdd::with_levels_dirty(vtree, levels, output, dirty_contract, dirty_leaf_contract);
    out.weights = weights;
    out
}

/// One clause conjoined into an accumulator: only the clause's spine — the
/// Steiner tree of its variables' leaves — was rebuilt, and every other level
/// rode through as the identity in the same array.
pub(crate) struct ClausePlan<'a>(pub(crate) &'a [VtreeIdx]);

impl OutputPlan for ClausePlan<'_> {
    fn finish(
        &self,
        acc: &mut Tdd,
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
        weights: Option<WeightStore>,
    ) -> Tdd {
        finish_rebuilt(acc, vtree, levels, output, self.0, weights)
    }
}

/// The shape of one conjunction: full, or restricted to an ancestor-closed set.
pub(super) trait ApplyPlan: OutputPlan {
    /// The level indices this apply reads or writes.
    fn touched(&self, num_nodes: usize) -> TouchedLevels<'_>;

    /// The bottom-up build order.
    fn walk<'a>(
        &'a self,
        vtree: &'a Vtree,
    ) -> impl Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + 'a;

    /// Whether the identity fast paths may fire, and with them the operand
    /// child-level drops that precede them.
    ///
    /// False under a restriction: `R` is by construction the set of levels
    /// where no fast path fires, `c1` is the accumulator whose off-`R` levels
    /// ride through into the output verbatim, and the output-child marginality
    /// the guards read lives in `c1`'s levels rather than the fresh array.
    fn takes_fast_paths(&self) -> bool;

    /// Whether an output level can still be sitting in the accumulator rather
    /// than in the fresh level array — true exactly under a restriction, where
    /// the two are merged only at the tail.
    fn output_lives_in_accumulator(&self) -> bool;

    /// Whether the entry-marginality snapshot is meaningful.
    ///
    /// It exists to recover an operand child that an identity fast path stole
    /// mid-sweep, so an apply that takes no fast path needs none.
    fn tracks_entry_marginal(&self) -> bool {
        self.takes_fast_paths()
    }

    /// The leaves to seed, or `None` for every leaf of the vtree.
    fn leaf_children(&self) -> Option<&[VtreeIdx]>;

    /// Whether any level's product grid is big enough to make the sparse
    /// machinery worth setting up.
    fn might_use_sparse(
        &self,
        vtree: &Vtree,
        c1_widths: &[usize],
        c2_widths: &[usize],
        min_grid: usize,
    ) -> bool;


    /// Seed the two identity vectors the product construction reads.
    ///
    /// `c2_identity[t]` is true when `c2` computes constant-true over subtree
    /// `t`, so `c1`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    /// construction can `mem::swap` them into the output instead of running
    /// the per-node inner loop. `c1_identity` is the symmetric case, where
    /// `c2`'s nodes are cloned across — `c2` is immutable, so it cannot be
    /// swapped from. It is what makes conjoining a node's two children cheap:
    /// the left child's diagram is identity over the right subtree's levels,
    /// and vice versa.
    ///
    /// The vectors are lazily accreted, so one can read false for a child that
    /// is structurally identity, sending it to the dense-grid fallback instead
    /// of the pass-through. Completing the predicate was measured and did not
    /// pay — the misses are rare and land on tiny grids.
    ///
    /// # Errors
    ///
    /// Propagates a refused buffer reservation.
    fn seed_identity(
        &self,
        eng: &crate::engine::Engine,
        run: &mut super::setup::ApplyRun,
        c1: &Tdd,
        c2: &Tdd,
        vtree: &Vtree,
        num_nodes: usize,
    ) -> Result<(), crate::error::ApplyError>;

    /// Seed the levels the generic loop would have carried through by an
    /// identity fast path, which a restricted apply skips.
    fn seed_carried_levels(&self, run: &mut super::setup::ApplyRun, vtree: &Vtree);
}

/// Every level rebuilt from the two operands.
pub(super) struct FullPlan;

impl ApplyPlan for FullPlan {
    #[inline]
    fn touched(&self, num_nodes: usize) -> TouchedLevels<'_> {
        TouchedLevels::All(0..num_nodes)
    }

    #[inline]
    fn walk<'a>(
        &'a self,
        vtree: &'a Vtree,
    ) -> impl Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + 'a {
        LevelWalk::Depth(vtree.internal_bottomup())
    }

    #[inline]
    fn takes_fast_paths(&self) -> bool {
        true
    }

    #[inline]
    fn output_lives_in_accumulator(&self) -> bool {
        false
    }

    #[inline]
    fn leaf_children(&self) -> Option<&[VtreeIdx]> {
        None
    }

    fn might_use_sparse(
        &self,
        vtree: &Vtree,
        c1_widths: &[usize],
        c2_widths: &[usize],
        min_grid: usize,
    ) -> bool {
        vtree.internal_bottomup().any(|(t, _, _)| {
            c1_widths[t.idx()].saturating_mul(c2_widths[t.idx()]) > min_grid
        })
    }


    /// A leaf is identity iff only the One label is referenced by parent pairs;
    /// an internal node iff it is width-1 with both children identity. Leaf
    /// identity is precomputed by scanning parent pairs for non-One refs, and
    /// the internal fixpoint accretes as the sweep goes up.
    fn seed_identity(
        &self,
        eng: &crate::engine::Engine,
        run: &mut super::setup::ApplyRun,
        c1: &Tdd,
        c2: &Tdd,
        vtree: &Vtree,
        num_nodes: usize,
    ) -> Result<(), crate::error::ApplyError> {
        super::identity::init_leaf_identity(eng, &mut run.c2_identity, c2, vtree, num_nodes)?;
        super::identity::init_leaf_identity(eng, &mut run.c1_identity, c1, vtree, num_nodes)
    }

    #[inline]
    fn seed_carried_levels(&self, _run: &mut super::setup::ApplyRun, _vtree: &Vtree) {}

}

impl OutputPlan for FullPlan {
    fn finish(
        &self,
        _acc: &mut Tdd,
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
        weights: Option<WeightStore>,
    ) -> Tdd {
        let mut out = Tdd::with_levels(vtree, levels, output);
        out.weights = weights;
        out
    }
}

/// Only the ancestor-closed set `R` rebuilt; every other level rides through
/// in the accumulator.
pub(super) struct RestrictedPlan<'a>(pub(super) &'a Restrict<'a>);

impl ApplyPlan for RestrictedPlan<'_> {
    #[inline]
    fn touched(&self, _num_nodes: usize) -> TouchedLevels<'_> {
        TouchedLevels::Some(self.0.touched.iter())
    }

    #[inline]
    fn walk<'a>(
        &'a self,
        vtree: &'a Vtree,
    ) -> impl Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + 'a {
        // `R` in `topo_pos` order — the full walk with the levels that would
        // take an identity fast path removed.
        // The type parameter is the full walk's iterator, which this variant
        // never constructs; naming it keeps both impls' return types the same
        // shape.
        LevelWalk::<'_, std::iter::Empty<_>>::Restricted(self.0.rebuild.iter(), vtree)
    }

    #[inline]
    fn takes_fast_paths(&self) -> bool {
        false
    }

    #[inline]
    fn output_lives_in_accumulator(&self) -> bool {
        true
    }

    #[inline]
    fn leaf_children(&self) -> Option<&[VtreeIdx]> {
        Some(self.0.leaf_children)
    }

    /// The caller supplies what the full pre-scan would have computed, derived
    /// in `O(|R|)` from the accumulator's cached widest-internal width plus the
    /// spine levels (every off-spine level is `k1 × 1`). Matched rather than
    /// forced either way, so the sparse routes fire at exactly the levels a
    /// full apply would fire them at.
    #[inline]
    fn might_use_sparse(&self, _: &Vtree, _: &[usize], _: &[usize], _: usize) -> bool {
        self.0.might_use_sparse
    }


    /// Restricted mode derives both identity vectors from the spine
    /// certificate instead of scanning every leaf's parent pairs twice:
    ///
    /// * `c2_identity[t] = !on_spine[t]` — the batch is width-1
    ///   constant-true off its spine by construction, which is exactly the
    ///   fixpoint the generic path's FP1 accretes at every off-spine level.
    /// * `c1_identity` all-false — it is read ONLY by FP2's guard (the other
    ///   readers are the declined sparse routes and `plan_marg_level`'s
    ///   `left_pt_c2`, whose `c2_ref` conjunct is false because the batch
    ///   has no marginal levels and no `marg_inlined_*` flags). Restricted
    ///   mode never takes a fast path, so an all-false vector just means
    ///   every level in `R` is rebuilt — and a rebuild against a width-1
    ///   constant-true operand reproduces the carried level.
    ///
    /// Only `touched` entries are ever read, so only they are written; the
    /// pooled buffers keep whatever stale values they had elsewhere.
    fn seed_identity(
        &self,
        eng: &crate::engine::Engine,
        run: &mut super::setup::ApplyRun,
        _c1: &Tdd,
        _c2: &Tdd,
        _vtree: &Vtree,
        num_nodes: usize,
    ) -> Result<(), crate::error::ApplyError> {
        let lim = eng.limits();
        let r = self.0;
        lim.try_resize(&mut run.c2_identity, num_nodes, false)?;
        lim.try_resize(&mut run.c1_identity, num_nodes, false)?;
        for &t in r.touched {
            run.c2_identity[t.idx()] = !r.on_spine[t.idx()];
            run.c1_identity[t.idx()] = false;
        }
        Ok(())
    }

    fn seed_carried_levels(&self, run: &mut super::setup::ApplyRun, vtree: &Vtree) {
        super::drive::seed_restricted_carried_levels(run, self.0, vtree);
    }

}

impl OutputPlan for RestrictedPlan<'_> {
    /// Merge `R` back into the accumulator's array.
    ///
    /// Every level off `R` rode through untouched — it is still the
    /// accumulator's own level, byte for byte, in the accumulator's own
    /// allocation. Moving the `|R|` rebuilt levels across is the whole output
    /// step; the fresh array goes back to the pool holding only empty levels
    /// (the leaf-marginal seeding sweep, its one other writer, is skipped
    /// under a restriction).
    ///
    /// SWAP rather than assign: the accumulator's superseded level at `t` goes
    /// back into the fresh array, so its `nodes`/`pairs` arenas are reused by
    /// the next merge's rebuild instead of being freed here and reallocated
    /// there.
    fn finish(
        &self,
        acc: &mut Tdd,
        vtree: Arc<Vtree>,
        mut levels: Vec<TddLevel>,
        output: TddNodeId,
        weights: Option<WeightStore>,
    ) -> Tdd {
        let r = self.0;
        for &t in r.rebuild {
            let ti = t.idx();
            std::mem::swap(&mut acc.levels[ti], &mut levels[ti]);
        }
        std::mem::swap(&mut acc.levels, &mut levels);
        finish_rebuilt(acc, vtree, levels, output, r.rebuild, weights)
    }
}
