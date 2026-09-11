//! What kind of apply is running: which levels it visits, which shortcuts it
//! may take, and how it hands its result back.
//!
//! A conjunction is either `Full` — every level of the vtree is rebuilt from the
//! two operands — or `Restricted` to an ancestor-closed set `R`, where the
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

/// Carry the accumulator's outstanding contraction debt forward, seeded with
/// the levels this apply rebuilt, and attach the weights.
///
/// Seeding only the rebuilt levels is exact, not merely sound: a rebuilt set
/// that is ancestor-closed has a descendant-closed complement, so an off-set
/// level's own pairs, its parent's pairs and its whole subtree are bit-identical
/// to the accumulator's. A contraction sweep seeded there would re-run the
/// accumulator's own last sweep on the same bytes and fire nothing. Whatever
/// the accumulator still owed is carried over rather than dropped, which is
/// what keeps this exact for a caller that does not minimize between applies
/// (`with_levels_dirty`'s second obligation).
///
/// Shared with the clause conjunction, which rebuilds only its clause's spine
/// — the Steiner tree of its variables' leaves — and lets every other level
/// ride through as the identity in the same array. It builds no product grid,
/// so it has no level walk, no identity vectors and no sparse machinery, but it
/// ends the same way.
pub(crate) fn finish_rebuilt(
    acc: &mut Tdd,
    vtree: Arc<Vtree>,
    levels: Vec<TddLevel>,
    output: TddNodeId,
    rebuilt: &[VtreeIdx],
    weights: Option<WeightStore>,
) -> Tdd {
    let carried = acc.take_worklists();
    let mut out = Tdd::with_levels_dirty(vtree, levels, output, carried, rebuilt);
    out.weights = weights;
    out
}

/// The shape of one conjunction.
#[derive(Clone, Copy)]
pub(super) enum ApplyPlan<'a> {
    /// Every level rebuilt from the two operands.
    Full,
    /// Only the ancestor-closed set `R` rebuilt; every other level rides
    /// through in the accumulator.
    Restricted(&'a Restrict<'a>),
}

impl<'a> ApplyPlan<'a> {
    /// The level indices this apply reads or writes.
    #[inline]
    pub(super) fn touched(self, num_nodes: usize) -> TouchedLevels<'a> {
        match self {
            ApplyPlan::Full => TouchedLevels::All(0..num_nodes),
            ApplyPlan::Restricted(r) => TouchedLevels::Some(r.touched.iter()),
        }
    }

    /// The bottom-up build order: every internal level, or `R` in `topo_pos`
    /// order — the full walk with the levels that would take an identity fast
    /// path removed.
    #[inline]
    pub(super) fn walk(
        self,
        vtree: &'a Vtree,
    ) -> impl Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + 'a {
        match self {
            ApplyPlan::Full => LevelWalk::Depth(vtree.internal_bottomup()),
            ApplyPlan::Restricted(r) => LevelWalk::Restricted(r.rebuild.iter(), vtree),
        }
    }

    /// Whether the identity fast paths may fire, and with them the operand
    /// child-level drops that precede them.
    ///
    /// False under a restriction: `R` is by construction the set of levels
    /// where no fast path fires, `f` is the accumulator whose off-`R` levels
    /// ride through into the output verbatim, and the output-child marginality
    /// the guards read lives in `f`'s levels rather than the fresh array.
    ///
    /// It is also what makes the entry-marginality snapshot worth taking: the
    /// snapshot exists to recover an operand child that an identity fast path
    /// stole mid-sweep, so an apply that takes no fast path needs none.
    #[inline]
    pub(super) fn takes_fast_paths(self) -> bool {
        matches!(self, ApplyPlan::Full)
    }

    /// Whether an output level can still be sitting in the accumulator rather
    /// than in the fresh level array — true exactly under a restriction, where
    /// the two are merged only at the tail.
    #[inline]
    pub(super) fn output_lives_in_accumulator(self) -> bool {
        matches!(self, ApplyPlan::Restricted(_))
    }

    /// The leaves to seed, or `None` for every leaf of the vtree.
    #[inline]
    pub(super) fn leaf_children(self) -> Option<&'a [VtreeIdx]> {
        match self {
            ApplyPlan::Full => None,
            ApplyPlan::Restricted(r) => Some(r.leaf_children),
        }
    }

    /// Whether any level's product grid is big enough to make the sparse
    /// machinery worth setting up.
    ///
    /// A restriction takes the value the caller supplies, which is what the
    /// full pre-scan would have computed, derived in `O(|R|)` from the
    /// accumulator's cached widest-internal width plus the spine levels (every
    /// off-spine level is `left_width × 1`). Matched rather than forced either
    /// way, so the sparse routes fire at exactly the levels a full apply would
    /// fire them at.
    pub(super) fn might_use_sparse(
        self,
        vtree: &Vtree,
        left_widths: &[usize],
        right_widths: &[usize],
        min_grid: usize,
    ) -> bool {
        match self {
            ApplyPlan::Full => vtree.internal_bottomup().any(|(t, _, _)| {
                left_widths[t.idx()].saturating_mul(right_widths[t.idx()]) > min_grid
            }),
            ApplyPlan::Restricted(r) => r.might_use_sparse,
        }
    }

    /// Seed the two identity vectors the product construction reads.
    ///
    /// `right_identity[t]` is true when `g` computes constant-true over subtree
    /// `t`, so `f`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    /// construction can `mem::swap` them into the output instead of running
    /// the per-node inner loop. `left_identity` is the symmetric case, where
    /// `g`'s nodes are cloned across — `g` is immutable, so it cannot be
    /// swapped from. It is what makes conjoining a node's two children cheap:
    /// the left child's diagram is identity over the right subtree's levels,
    /// and vice versa.
    ///
    /// The vectors are lazily accreted, so one can read false for a child that
    /// is structurally identity, sending it to the dense-grid fallback instead
    /// of the pass-through. The predicate is deliberately incomplete: the
    /// misses are rare and land on tiny grids.
    ///
    /// A full apply derives them structurally: a leaf is identity iff only the
    /// One label is referenced by parent pairs; an internal node iff it is
    /// width-1 with both children identity. Leaf identity is precomputed by
    /// scanning parent pairs for non-One refs, and the internal fixpoint
    /// accretes as the sweep goes up.
    ///
    /// A restriction derives both from the spine certificate instead of
    /// scanning every leaf's parent pairs twice:
    ///
    /// * `right_identity[t] = !on_spine[t]` — the batch is width-1
    ///   constant-true off its spine by construction, which is exactly the
    ///   fixpoint the generic path's FP1 accretes at every off-spine level.
    /// * `left_identity` all-false — it is read only by FP2's guard (the other
    ///   readers are the declined sparse routes and `plan_marginal_level`'s
    ///   `left_pt_c2`, whose `right_ref` conjunct is false because the batch
    ///   has no marginal levels and no `marginal_inlined_*` flags). Restricted
    ///   mode never takes a fast path, so an all-false vector just means
    ///   every level in `R` is rebuilt — and a rebuild against a width-1
    ///   constant-true operand reproduces the carried level.
    ///
    /// Only `touched` entries are ever read under a restriction, so only they
    /// are written; the pooled buffers keep whatever stale values they had
    /// elsewhere.
    ///
    /// # Errors
    ///
    /// Propagates a refused buffer reservation.
    pub(super) fn seed_identity(
        self,
        eng: &crate::engine::Engine,
        run: &mut super::setup::ApplyRun,
        f: &Tdd,
        g: &Tdd,
        vtree: &Vtree,
        num_nodes: usize,
    ) -> Result<(), crate::limits::ApplyError> {
        match self {
            ApplyPlan::Full => {
                super::identity::init_leaf_identity(
                    eng, &mut run.right_identity, g, vtree, num_nodes,
                )?;
                super::identity::init_leaf_identity(
                    eng, &mut run.left_identity, f, vtree, num_nodes,
                )
            }
            ApplyPlan::Restricted(r) => {
                let lim = eng.limits();
                lim.try_resize(&mut run.right_identity, num_nodes, false)?;
                lim.try_resize(&mut run.left_identity, num_nodes, false)?;
                for &t in r.touched {
                    run.right_identity[t.idx()] = !r.on_spine[t.idx()];
                    run.left_identity[t.idx()] = false;
                }
                Ok(())
            }
        }
    }

    /// Seed the levels the generic loop would have carried through by an
    /// identity fast path, which a restricted apply skips.
    #[inline]
    pub(super) fn seed_carried_levels(self, run: &mut super::setup::ApplyRun, vtree: &Vtree) {
        if let ApplyPlan::Restricted(r) = self {
            super::drive::seed_restricted_carried_levels(run, r, vtree);
        }
    }

    /// Turn the finished level array into the output diagram.
    ///
    /// `acc` is the accumulator the levels came from; a restriction merges `R`
    /// back into its array. Every level off `R` rode through untouched — it is
    /// still the accumulator's own level, byte for byte, in the accumulator's
    /// own allocation. Moving the `|R|` rebuilt levels across is the whole
    /// output step; the fresh array goes back to the pool holding only empty
    /// levels (the leaf-marginal seeding sweep, its one other writer, is
    /// skipped under a restriction).
    ///
    /// Swap rather than assign: the accumulator's superseded level at `t` goes
    /// back into the fresh array, so its `nodes`/`pairs` arenas are reused by
    /// the next merge's rebuild instead of being freed here and reallocated
    /// there.
    pub(super) fn finish(
        self,
        acc: &mut Tdd,
        vtree: Arc<Vtree>,
        mut levels: Vec<TddLevel>,
        output: TddNodeId,
        weights: Option<WeightStore>,
    ) -> Tdd {
        match self {
            ApplyPlan::Full => {
                let mut out = Tdd::from_levels_unchecked(vtree, levels, output);
                out.weights = weights;
                out
            }
            ApplyPlan::Restricted(r) => {
                for &t in r.rebuild {
                    let ti = t.idx();
                    std::mem::swap(&mut acc.levels[ti], &mut levels[ti]);
                }
                std::mem::swap(&mut acc.levels, &mut levels);
                finish_rebuilt(acc, vtree, levels, output, r.rebuild, weights)
            }
        }
    }
}
