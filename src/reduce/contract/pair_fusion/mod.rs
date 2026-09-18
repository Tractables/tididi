//! Fuse pairs that share a structural child by summing their marginal values.

mod plan;
mod rewrite;


use crate::Engine;
use crate::limits::OperationError;
use crate::diagram::{MarginalSide, Tdd, ValueRef};
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildSide, boundary_marginal_levels_into};
use crate::value::slots::SlotValues;
use crate::value::{IntFold, WeightFold};

use super::scratch::ContractScratch;

use plan::{collect_fusion_plans, allocate_fusion_slots};
use rewrite::rebuild_parent_level;

/// Stats returned by the pair-fusion sweeps.
#[derive(Debug, Clone, Default)]
pub(crate) struct PairFusionStats {
    /// Total number of (parent node, `x_idx`) groups fused (each removes
    /// `group_size - 1` parent pair entries and references one `R_new` slot).
    /// Counts applied rewrites only: a weighted leaf group whose value the pinned
    /// column cannot represent is dropped before Phase 3 and not counted (the
    /// contract fixpoint reads a nonzero count as "the diagram changed").
    pub(crate) fusion_groups: usize,
}

/// Fuse pairs only at the named boundary parents; an empty slice does no work.
/// Allocation refusals return [`OperationError::OverBudget`].
pub(crate) fn fuse_pairs_at_parents(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_vtree_idxs: &[VtreeIdx],
) -> Result<PairFusionStats, OperationError> {
    // Borrow the pooled scratch; the contract fixpoint calls `fuse_pairs_inner`
    // directly with the scratch it already holds.
    let mut scratch = eng.reduce_scratch().contract.checkout(eng.limits());
    fuse_pairs_inner(eng, tdd, Some(parent_vtree_idxs), &mut scratch)
}

/// Per-(node, `x_idx`) fusion plan over one value domain. Phase 1 builds
/// these; Phase 2 fills `new_ref`.
struct PlanEntry<V> {
    node_idx: usize,
    x_idx: u32,
    /// The fused value: the domain's sum over the group's occurrence multiset.
    value: V,
    // Filled in Phase 2 with the fully-encoded marginal-side ref to write
    // into the fused parent pair: a tagged inline count (bit-30 set)
    // when `value` fits the inline threshold under emit mode, else a
    // bare slot index (bit-30 clear). In the slot case plans whose
    // `value` matches share a slot (existing or newly allocated).
    // Sound under multiset pair lists.
    new_ref: u32,
}

/// Same-left pair fusion, in place: at every boundary marginal level, each
/// parent node's group of pairs `(L, R1), (L, R2), …` (one explicit-side ref,
/// several marginal-side refs) becomes a single pair `(L, R_new)` with value
/// `c(R1) + c(R2) + …`. Old slots stay for other parents; `minimize` compacts
/// unreferenced ones. Plans with equal values share a slot, so a parent may
/// hold the same pair twice; pair lists are multisets, so each occurrence
/// keeps its own contribution.
///
/// Weighted mode sums through the `WeightStore`, in the exact domain only (the
/// `Log` domain is skipped); at a weight-marginal leaf a group folds only onto
/// a value the pinned column already holds, and is otherwise left alone.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when an allocation is refused. The node whose
/// Phase-3 re-encode failed is left mid-rewrite, so the diagram must be discarded.
pub(super) fn fuse_pairs_inner(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_filter: Option<&[VtreeIdx]>,
    scratch: &mut ContractScratch,
) -> Result<PairFusionStats, OperationError> {
    // Weighted marginalization keeps its values in the `WeightStore` and
    // `marginal_counts` is `None`: the exact domain runs the weighted arm, the
    // `Log` domain is skipped (signed-log addition is order-dependent).
    let weighted = match tdd.weights.as_ref() {
        Some(ws) if ws.is_log() => return Ok(PairFusionStats::default()),
        Some(_) => true,
        None => false,
    };
    let mut stats = PairFusionStats::default();
    boundary_marginal_levels_into(tdd, parent_filter, &mut scratch.boundaries);
    // Indexed so the per-boundary work can borrow `scratch.pair_fusion` (a
    // disjoint field) while this list stays live. Snapshotting the set before
    // the loop is safe: fusion never marginalizes a level, and invariant 5
    // forbids un-marginalizing one, so the snapshot cannot go stale.
    for bi in 0..scratch.boundaries.len() {
        let (v, parent, side) = scratch.boundaries[bi];
        // Weighted reads and mints go through this level's `WeightStore` vec, so
        // a marginal level with no store yet is not fusable.
        if weighted && !tdd.weights.as_ref().is_some_and(|ws| ws.is_set(v.idx())) {
            continue;
        }
        if weighted {
            fuse_boundary::<WeightFold>(eng, tdd, v, parent, side, scratch, &mut stats)?;
        } else {
            fuse_boundary::<IntFold>(eng, tdd, v, parent, side, scratch, &mut stats)?;
        }
    }
    Ok(stats)
}

/// The three phases at one boundary: marginal level `v` under `parent`, on
/// `side` of its pairs.
fn fuse_boundary<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    parent: VtreeIdx,
    side: ChildSide,
    scratch: &mut ContractScratch,
    stats: &mut PairFusionStats,
) -> Result<(), OperationError> {
    // Phase 1: full-scan parent's nodes; collect per-(node, x_idx) groups
    // with > 1 distinct marginal-side index. Compute the fused value for each.
    let mut plans: Vec<PlanEntry<D::Value>> =
        collect_fusion_plans::<D>(eng, tdd, parent, v, side, &mut scratch.pair_fusion)?;
    if plans.is_empty() {
        return Ok(());
    }

    // Phase 2: encode each plan's value as a marginal-side ref, an inline count
    // or a value-keyed slot (`allocate_fusion_slots`). `any_inline` makes
    // Phase 3 raise the parent's marginal-side inline marker, without which
    // readers decode the bit-30-tagged ref as a grid coordinate.
    //
    // A weight-marginal leaf's column is pinned
    // (`test_helpers::check::marginal::check_leaf_columns_pinned`), so nothing
    // is minted there: a plan resolves by lookup in the column and is dropped
    // when the column lacks its value. A dropped plan leaves its group's pairs
    // untouched, since Phase 3 rewrites only the x-indices a surviving plan
    // names: one un-fused redex, never a wrong value. `retain_mut` keeps the
    // ascending-`node_idx` order Phase 3 needs.
    let any_inline = if D::LEAF_PINNED && tdd.vtree.node(v).is_leaf() {
        plans.retain_mut(|plan| match D::leaf_ref(tdd, v, &plan.value) {
            Some(raw) => {
                plan.new_ref = raw;
                true
            }
            None => false,
        });
        if plans.is_empty() {
            return Ok(());
        }
        plans.iter().any(|plan| {
            matches!(ValueRef::from_raw(MarginalSide(plan.new_ref)), ValueRef::Inline(_))
        })
    } else {
        allocate_fusion_slots::<D>(eng, tdd, v, &mut plans)?
    };

    // Counted after Phase 2, which may drop leaf plans: `fusion_groups` counts
    // applied rewrites only, and the contract fixpoint loops while it is nonzero.
    stats.fusion_groups += plans.len();

    // Phase 3: rewrite parent pair lists for affected nodes. `plans` is
    // emitted grouped by ascending `node_idx` in Phase 1 (and the leaf-lookup
    // filter above preserves that order), which is the cursor-walk
    // precondition. See `rebuild_parent_level`.
    rebuild_parent_level(eng, tdd, parent, side, any_inline, &plans)
}

#[cfg(test)]
pub(crate) mod tests;
