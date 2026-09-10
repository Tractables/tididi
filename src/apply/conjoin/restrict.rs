//! MergeScope-bounded ("restricted") conjunction — the O(spine) batch merge used by
//! the indicator-compile def loop.
//!
//! # Why
//!
//! The def loop folds a small batch diagram (a handful of definition clauses)
//! into a huge accumulator. The generic apply
//! (`apply_and_fallible_inner`) walks every internal vtree level on every such
//! merge — snapshotting widths, seeding leaf identity, laying out grids, and
//! visiting each level only to take an identity fast path. On a vtree with
//! hundreds of thousands of levels that fixed cost dominates the merge even
//! though the batch touches a few thousand levels. This module runs the same
//! apply over a restricted level set instead.
//!
//! # The restricted set `R`
//!
//! `R` is not just the batch's spine. It is exactly the set of internal levels
//! the generic apply does not dispatch to `take_level_fast_path`:
//!
//! * **`S`** — the batch's spine (the ancestor-closed union of the root-paths
//!   of its clauses' variable leaves), as reported by `mark_clause_levels`.
//!   Off `S` the batch is constant-true with width 1, so the
//!   generic apply takes FP1 there and carries the accumulator's level through
//!   by reference. Every `S` internal node has an on-spine child (it is an
//!   ancestor of a clause leaf), so FP1's `right_identity[left] &&
//!   right_identity[right]` guard never holds on `S` — the generic apply rebuilds
//!   all of it.
//!
//! * **`AncClosure(P)`** — `P` is the set of *structural* levels with a
//!   *marginal* child (the parents of the maximal marginal subtrees the
//!   marginalize-behind-the-frontier schedule has already summed out). FP1's
//!   last guard (`!(!f.is_marginal(t) && (out_left_marginal || out_right_marginal))`)
//!   and FP2's mirror of it both decline there, so `P` is rebuilt — and a
//!   rebuilt level sets NEITHER identity flag, so every ancestor of a `P` level
//!   fails the same guard and is rebuilt too, all the way to the root.
//!
//! Both pieces are ancestor-closed, so `R` is ancestor-closed and its
//! complement is descendant-closed — which is what makes the contraction seed
//! at the end exact rather than merely sound (same argument as
//! `conjoin_clause::conjoin_clause_into`; see the note at its `dirty_contract`
//! seed).
//!
//! # What the restricted apply skips
//!
//! Relative to the generic path, restricted mode:
//! * iterates `R` instead of `vtree.internal_bottomup()`;
//! * computes widths / grid layout / live counts only over `R ∪ children(R)`;
//! * skips `init_leaf_identity` on both operands (see `right_identity` below);
//! * skips the leaf-marginalization seeding sweep (the batch has no marginal
//!   levels and the accumulator's marginal leaves are already in place — the
//!   output array is merged back into the accumulator's, so they survive);
//! * skips `take_level_fast_path` (no level in `R` takes one — see above) and
//!   the `drop_dead_operand_level` calls (they would free accumulator levels
//!   that ride through untouched);
//! * restricts the end-of-apply `tag_all_marginal_side_slots` sweep to `R` (only a
//!   structural level with a marginal child does any work there, and off `R`
//!   neither the level nor its children changed);
//! * merges the rebuilt levels back into the accumulator's own level array
//!   instead of returning a fresh full-length one.
//!
//! `left_identity` is passed all-false and `right_identity` as `!on_spine`. Both are
//! exactly what the generic path holds at every index restricted mode reads,
//! with one deliberate exception: if the batch happens to be constant-true at
//! an `S` level (a tautological fold), or the accumulator constant-true at an
//! `R` level, the generic path takes FP1/FP2 there and restricted mode rebuilds
//! instead. Both rebuilds reproduce the carried level (a `k×1` / `1×k` product
//! against a width-1 identity operand emits the same nodes in the same order).
//! A test keeps that claim honest: it merges the same batches both ways and
//! asserts bit-identity.
//!
//! # Declines
//!
//! The restricted path is an optimization, never a semantic change: anything it
//! is not proven exact for falls back to the generic merge, which is still THE
//! apply for every other caller.

use crate::engine::Engine;
use crate::engine::pool::Pool;
use crate::apply::scoped_flags::ScopedFlags;

use crate::diagram::{self, Tdd};
use crate::vtree::VtreeIdx;

use super::ApplyError;

/// Every buffer one engine's restricted applies reuse between calls.
///
/// The two flag arrays hold an all-false invariant between calls, restored by
/// `RestrictPlan::recycle`.
#[derive(Default)]
pub(crate) struct RestrictScratch {
    /// `on_spine[t]` — the batch's spine union. All-false between calls.
    spine_flags: Pool<Vec<bool>>,
    /// `in_rebuild[t]` — membership in `R`. All-false between calls.
    rebuild_flags: Pool<Vec<bool>>,
    /// `R`, bottom-up (children before parents).
    rebuild: Pool<Vec<VtreeIdx>>,
    /// `R ∪ children(R)` — every index whose width / grid / live count the
    /// restricted apply reads or writes.
    touched: Pool<Vec<VtreeIdx>>,
    /// The leaves in `touched` (the restricted `apply_leaf_levels` domain).
    leaf_children: Pool<Vec<VtreeIdx>>,
}

impl RestrictScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.spine_flags.drain();
        self.rebuild_flags.drain();
        self.rebuild.drain();
        self.touched.drain();
        self.leaf_children.drain();
    }
}


/// Where a batch can reach into the accumulator, and how wide the accumulator
/// is — everything a restricted merge needs beyond the two diagrams.
///
/// The two widths are the whole-diagram quantities a restricted merge cannot
/// re-read cheaply, since it never sweeps the accumulator's level array.
/// [`RebuiltWidths`], returned alongside a successful merge, is how a caller keeps
/// both current across a run of merges.
pub struct MergeScope<'a> {
    /// The vtree levels the batch constrains. It may over-approximate — the
    /// merge re-filters — but it must not be short: a level the batch touches
    /// and this omits would be carried through stale.
    pub levels: &'a [VtreeIdx],
    /// Parents of the batch's marginal levels, in the same over-approximating
    /// sense as `levels`.
    pub marginal_parents: &'a [VtreeIdx],
    /// The accumulator's [`Tdd::max_width`].
    pub acc_max_width: usize,
    /// The widest `width()` over the accumulator's internal levels,
    /// tombstoned slots included.
    pub acc_widest_internal: usize,
}

/// The restriction the apply core runs under. Borrowed from a [`RestrictPlan`].
pub(super) struct Restrict<'a> {
    /// Internal levels to rebuild, children before parents (`internal_topo`
    /// order, so the per-level results are the generic path's verbatim).
    pub(super) rebuild: &'a [VtreeIdx],
    /// `in_rebuild[t.idx()]`.
    pub(super) in_rebuild: &'a [bool],
    /// `on_spine[t.idx()]` — `g` (the batch) is constant-true exactly off this
    /// set, which is what makes `right_identity == !on_spine` correct.
    pub(super) on_spine: &'a [bool],
    /// `rebuild ∪ children(rebuild)`.
    pub(super) touched: &'a [VtreeIdx],
    /// The leaves of `touched`.
    pub(super) leaf_children: &'a [VtreeIdx],
    /// The value the generic apply's global pre-scan would compute for this
    /// operand pair, derived in `O(|spine|)` by `build_plan` from the caller's
    /// cached widest-internal-level width. Matched exactly (rather than forced
    /// either way) so the sparse / sparse-marginal routes fire at the same levels
    /// they would generically.
    pub(super) might_use_sparse: bool,
}

/// Pooled backing storage for a [`Restrict`].
///
/// Every buffer goes back to its pool when the plan drops, so the error path
/// out of the merge needs no recycling of its own.
pub(super) struct RestrictPlan<'a> {
    rebuild: Vec<VtreeIdx>,
    in_rebuild: ScopedFlags<'a>,
    on_spine: ScopedFlags<'a>,
    touched: Vec<VtreeIdx>,
    leaf_children: Vec<VtreeIdx>,
    might_use_sparse: bool,
    eng: &'a Engine,
}

impl RestrictPlan<'_> {
    fn as_restrict(&self) -> Restrict<'_> {
        Restrict {
            rebuild: &self.rebuild,
            in_rebuild: &self.in_rebuild,
            on_spine: &self.on_spine,
            touched: &self.touched,
            leaf_children: &self.leaf_children,
            might_use_sparse: self.might_use_sparse,
        }
    }

}

impl Drop for RestrictPlan<'_> {
    fn drop(&mut self) {
        let pool = self.eng.restrict_pool();
        pool.rebuild.put(std::mem::take(&mut self.rebuild));
        pool.touched.put(std::mem::take(&mut self.touched));
        pool.leaf_children.put(std::mem::take(&mut self.leaf_children));
    }
}

/// The two accumulator width maxima `conjoin_batch` is handed and
/// gives back, taken over the levels it rebuilt.
///
/// Those are the only levels a restricted merge can have widened — every other
/// level rides through byte for byte — so a caller that keeps both quantities
/// as running maxima folds these two numbers in after each merge instead of
/// re-sweeping the whole level array. That is the point of the type: sweeping
/// a multi-million-node accumulator once per merge would cost more than the
/// merge.
///
/// The two maxima are different quantities and both are needed:
///
/// * `live` counts live nodes only, and is exactly what [`Tdd::max_width`]
///   reports for the merged diagram.
/// * `raw_internal` counts tombstoned slots too, and is taken over internal
///   levels only. It is the quantity that decides the apply's grid layout, so
///   it cannot be recovered from `live`.
///
/// `live_at` and `raw_at` name the level each maximum was read at. A caller
/// keeping a running maximum needs them: without the level, a later merge that
/// rebuilds that same level cannot tell whether the recorded maximum still
/// stands or has just been invalidated.
#[derive(Clone, Copy, Debug)]
pub struct RebuiltWidths {
    /// Widest rebuilt level counting live nodes only.
    pub live: usize,
    /// The level `live` was read at.
    pub live_at: VtreeIdx,
    /// Widest rebuilt internal level counting tombstoned slots too.
    pub raw_internal: usize,
    /// The level `raw_internal` was read at.
    pub raw_at: VtreeIdx,
}

impl RebuiltWidths {
    /// Read both maxima off `rebuild` in the merged diagram. `rebuild` is
    /// internal-only and non-empty for every call that reaches here (an empty
    /// spine declines), so the `at` fields always name a level that was walked.
    fn over(merged: &Tdd, rebuild: &[VtreeIdx]) -> Self {
        let mut m = RebuiltWidths {
            live: 0,
            live_at: merged.output.vtree,
            raw_internal: 0,
            raw_at: merged.output.vtree,
        };
        for &t in rebuild {
            let l = &merged.levels[t.idx()];
            let lw = l.live_width();
            if lw > m.live {
                m.live = lw;
                m.live_at = t;
            }
            let w = l.width();
            if w > m.raw_internal {
                m.raw_internal = w;
                m.raw_at = t;
            }
        }
        m
    }
}

/// Outcome of `conjoin_batch`.
pub enum BatchMergeOutcome {
    /// The restricted merge ran. The diagram is `acc ∧ batch` — bit for bit what
    /// the generic conjunction would have produced — and the [`RebuiltWidths`] is
    /// both width maxima re-read over the levels it rebuilt.
    Merged(Tdd, RebuiltWidths),
    /// The restricted merge declined. Both operands come back untouched, in the
    /// order they were passed, for the caller to hand to
    /// `conjoin_owned`. This is
    /// not an answer and never a failure.
    Declined(Tdd, Tdd),
}

/// Conjoin a small `batch` diagram into a large `acc`, visiting only the vtree
/// levels the batch can have changed.
///
/// [`conjoin_owned`](super::conjoin_owned) walks every
/// internal vtree level on every call. When one operand is small — a handful of
/// clauses folded together — and the other is a large accumulator, that fixed
/// per-level cost dominates a merge whose real work touches a small fraction of
/// the vtree. This entry point runs the same apply over a restricted level set
/// instead, and produces the bit-identical diagram, provided the caller's
/// `spine` argument meets the contract below.
///
/// # The `spine` contract
///
/// `spine` must name every vtree node at which `batch` is not constant true:
/// Every internal node whose level in `batch` is wider than one, and every leaf
/// `batch` reaches other than through the constant-one child. It must be
/// **ancestor closed** — if a node is listed, so is every node on its path to
/// the root. Order does not matter and duplicates are harmless. Listing extra
/// nodes only costs time; listing too few is **unsound**, so a caller that
/// cannot certify the set must use the generic conjunction instead.
///
/// The set is a by-product of building the batch, which is why it is a
/// parameter rather than something derived here. A caller that folds `k`
/// clauses into `batch` obtains it by marking each clause variable's leaf and
/// walking to the root, stopping at the first node already marked: `O(|spine|)`
/// over the whole fold rather than `O(k × height)`, and already paid for. The
/// alternative — recovering it here — is a sweep of the entire vtree per merge,
/// which is the cost this entry point exists to avoid.
///
/// # The accumulator's arguments
///
/// `marginal_parents` names the levels of `acc` that have a marginalized child, the
/// levels a preceding marginalization left behind. Only the caller knows when
/// the accumulator was last marginalized, so this too is passed in. It is a
/// seed set: naming a level that has itself since gone marginal is fine and is
/// re-filtered here, but as with `spine`, it must not be short.
///
/// `acc_max_width` and `acc_widest_internal` are `acc`'s [`Tdd::max_width`] and
/// the widest `width()` over its internal levels (tombstoned slots included).
/// They are the two whole-diagram quantities a restricted merge cannot re-read
/// cheaply. [`RebuiltWidths`], returned alongside a successful merge, is how a
/// caller keeps both current across a run of merges without ever re-sweeping
/// the level array.
///
/// # Declining
///
/// Returns [`BatchMergeOutcome::Declined`], with both operands intact and in the order
/// they were passed, whenever the restricted path is not provably exact for
/// this call: an empty spine, a batch wider than the accumulator, operands that
/// do not share a vtree, a constant-false operand, a marginalized batch root,
/// or an armed cap or weight context, neither of which the restricted level
/// set models. A caller must treat a decline as "run the
/// generic conjunction on these operands"; it is not an answer, and it is never
/// a failure.
///
/// # Errors
///
/// Propagates [`ApplyError`] from the apply core. As with every other owned
/// apply entry point, an `Err` means both operands are spent — they must not be
/// reused, only rebuilt.
pub fn conjoin_batch(
    eng: &Engine,
    acc: Tdd,
    batch: Tdd,
    spine: &MergeScope<'_>,
) -> Result<BatchMergeOutcome, ApplyError> {
    if decline_reason(eng, &acc, &batch, spine.levels, spine.acc_max_width).is_some() {
        return Ok(BatchMergeOutcome::Declined(acc, batch));
    }
    let plan = build_plan(
        eng, &acc, &batch, spine.levels, spine.marginal_parents, spine.acc_widest_internal,
    );

    let mut acc = acc;
    let mut batch = batch;
    let result = {
        let r = plan.as_restrict();
        let out = super::apply_and_fallible_restricted(eng, &mut acc, &mut batch, &r);
        diagram::return_levels(eng, diagram::PoolSlot::First, std::mem::take(&mut acc.levels));
        diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut batch.levels));
        out
    };
    // `RebuiltWidths` has to be read before the plan drops its buffers back into
    // their pools.
    let merged = match result {
        Ok(t) => {
            let m = RebuiltWidths::over(&t, &plan.rebuild);
            (t, m)
        }
        Err(e) => return Err(e),
    };
    Ok(BatchMergeOutcome::Merged(merged.0, merged.1))
}

#[path = "restrict_plan.rs"]
mod restrict_plan;
use restrict_plan::{build_plan, decline_reason};
