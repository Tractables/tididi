//! Fuse pairs that share a structural child by summing their marginal values.

mod plan;
mod rewrite;


use crate::Engine;
use crate::limits::{OperationError, Transient};
use crate::diagram::{EncodedChildRef, ChildDecoder, Tdd, ValueRef};
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildSide, boundary_marginal_levels_into};
use crate::value::slots::SlotValues;
use crate::value::{IntFold, WeightFold};

use super::scratch::ContractScratch;

use plan::{collect_fusion_plans, FusionSlots};
use rewrite::{ParentRewrite, BITMAP_MIN_PLANS};

/// Fuse pairs only at the named boundary parents; an empty slice does no work.
/// Allocation refusals return [`OperationError::OverBudget`], a fired stop
/// rule [`OperationError::Stopped`].
pub(crate) fn fuse_pairs_at_parents(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_vtree_idxs: &[VtreeIdx],
) -> Result<(), OperationError> {
    // Borrow the pooled scratch; the contract fixpoint calls `fuse_pairs_inner`
    // directly with the scratch it already holds.
    let mut scratch = eng.scratch.reduce.contract.checkout(eng);
    fuse_pairs_inner(eng, tdd, Some(parent_vtree_idxs), None, &mut scratch)?;
    Ok(())
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
/// An unweighted sweep that runs to the end records each parent it fused
/// with one marginal child in the diagram's worklists, and a later sweep
/// skips a recorded parent: the record lasts until an edit pushes the level
/// (`Dirty::is_fused`).
///
/// With `listed`, the parent nodes listed, ascending, are the only ones read:
/// the caller knows no other node of the one parent `parent_filter` names
/// holds a group.
///
/// Returns the number of (parent node, `x_idx`) groups fused, each removing
/// `group_size - 1` parent pair entries and referencing one `R_new` slot.
/// It counts applied rewrites only: a weighted leaf group whose value the
/// pinned column cannot represent is dropped before Phase 3 and not counted
/// (the contract fixpoint reads a nonzero count as "the diagram changed").
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when an allocation is refused. The node whose
/// Phase-3 re-encode failed is left mid-rewrite, so the diagram must be discarded.
///
/// `Err(OperationError::Stopped)` when a stop rule fires. The sweep tests for
/// one once per stride of the pairs it reads or rewrites, only ever between
/// two nodes, so the diagram stays valid: some groups are left unfused.
pub(super) fn fuse_pairs_inner(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_filter: Option<&[VtreeIdx]>,
    listed: Option<&[u32]>,
    scratch: &mut ContractScratch,
) -> Result<usize, OperationError> {
    debug_assert!(listed.is_none() || parent_filter.is_some_and(|p| p.len() == 1), "a list is of one parent's nodes");
    // Weighted marginalization keeps its values in the `WeightStore` and
    // `marginal_counts` is `None`: the exact domain runs the weighted arm, a
    // bounded-precision domain is skipped (its addition is order-dependent).
    let weighted = match tdd.weights.as_ref() {
        Some(ws) if ws.is_bounded() => return Ok(0),
        Some(_) => true,
        None => false,
    };
    let mut fused = 0;
    boundary_marginal_levels_into(tdd, parent_filter, &mut scratch.boundaries);
    // Indexed so the per-boundary work can borrow `scratch.pair_fusion` (a
    // disjoint field) while this list stays live. Snapshotting the set before
    // the loop is safe: fusion never marginalizes a level, and invariant 5
    // forbids un-marginalizing one, so the snapshot cannot go stale.
    for bi in 0..scratch.boundaries.len() {
        let (v, parent, side) = scratch.boundaries[bi];
        // A fusion that ran to the end at `parent` left no group there, and
        // an edit that could make one, merging the nodes a pair names or
        // adding pairs, pushes the level and so drops the record.
        if tdd.dirty.is_fused(parent.0) {
            #[cfg(debug_assertions)]
            crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(
                tdd, Some(&[parent]), "a parent recorded fused",
            );
            continue;
        }
        // Weighted reads and mints go through this level's `WeightStore` vec, so
        // a marginal level with no store yet is not fusable.
        if weighted && !tdd.weights.as_ref().is_some_and(|ws| ws.is_set(v.idx())) {
            continue;
        }
        let boundary = Boundary { v, parent, side };
        if weighted {
            fuse_boundary::<WeightFold>(eng, tdd, boundary, listed, scratch, &mut fused)?;
        } else {
            fuse_boundary::<IntFold>(eng, tdd, boundary, listed, scratch, &mut fused)?;
        }
    }
    // A weighted group the pinned column cannot hold stays unfused, so only
    // an unweighted fusion is recorded.
    if !weighted {
        for bi in 0..scratch.boundaries.len() {
            record_fused(eng, tdd, scratch.boundaries[bi].1);
        }
    }
    Ok(fused)
}

/// Record that `parent`'s pairs hold no group pair fusion would fuse, as a
/// fusion run to the end or a conjunction that sums a child out as it
/// builds the parent leaves them (`Dirty::is_fused`). Not under a parent
/// with both children marginal, where a fusion on one side can mint a
/// group on the other.
pub(crate) fn record_fused(eng: &Engine, tdd: &mut Tdd, parent: VtreeIdx) {
    let (left, right) = tdd.vtree.children(parent);
    if !(tdd.levels[left.idx()].is_marginal() && tdd.levels[right.idx()].is_marginal()) {
        tdd.dirty.set_fused(eng, parent.0);
    }
}

/// The plans Phase 1 collects before Phases 2 and 3 apply them. A level can
/// hold tens of millions of plans; a chunk keeps the plan buffer this size,
/// and its nodes are rewritten while the pairs Phase 1 read are still in
/// cache.
const CHUNK_PLANS: usize = 1 << 12;

/// One marginal child `v` under `parent`, on `side` of its pairs.
#[derive(Clone, Copy)]
pub(super) struct Boundary {
    pub(super) v: VtreeIdx,
    pub(super) parent: VtreeIdx,
    pub(super) side: ChildSide,
}

/// The three phases at one `boundary`, reading the parent nodes `listed`
/// where given.
fn fuse_boundary<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    boundary: Boundary,
    listed: Option<&[u32]>,
    scratch: &mut ContractScratch,
    fused: &mut usize,
) -> Result<(), OperationError> {
    let mut rewrite = ParentRewrite::new(eng.limits(), BITMAP_MIN_PLANS);
    let fused = match listed {
        Some(nodes) => {
            debug_assert!(nodes.is_sorted(), "listed fusion nodes ascend");
            let nodes = nodes.iter().map(|&n| n as usize);
            fuse_chunks::<D>(eng, tdd, boundary, nodes, &mut rewrite, scratch, fused)
        }
        None => {
            let all = 0..tdd.levels[boundary.parent.idx()].nodes().len();
            fuse_chunks::<D>(eng, tdd, boundary, all, &mut rewrite, scratch, fused)
        }
    };
    // Every chunk rewritten before a stop is a valid level, so the sweep
    // runs on every exit.
    rewrite.finish(tdd, boundary.parent);
    fused
}

/// [`fuse_boundary`] over the parent nodes `nodes`, ascending, a chunk of
/// [`CHUNK_PLANS`] plans at a time: Phase 1 collects a chunk, Phase 2 encodes
/// its values, Phase 3 rewrites its nodes, and the next chunk follows. Phase 3
/// moves no node's pairs but the chunk's own, and Phase 2 only appends slots,
/// so the next chunk reads the pairs and values a scan of the whole level
/// would; slots are minted in the same order, so the level and store are the
/// ones one chunk would leave.
fn fuse_chunks<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    boundary: Boundary,
    mut nodes: impl Iterator<Item = usize>,
    rewrite: &mut ParentRewrite<'_>,
    scratch: &mut ContractScratch,
    fused: &mut usize,
) -> Result<(), OperationError> {
    let Boundary { v, parent, side } = boundary;
    let lim = eng.limits();
    // A test pins the chunk size on its thread and counts the boundaries
    // applied in more than one chunk.
    #[cfg(test)]
    let chunk = tests::pinned_chunk_plans().unwrap_or(CHUNK_PLANS);
    #[cfg(not(test))]
    let chunk = CHUNK_PLANS;
    let mut plans: Transient<'_, Vec<PlanEntry<D::Value>>> = Transient::new(lim, Vec::new());
    let mut slots = FusionSlots::<D>::new(lim);
    #[cfg(test)]
    let mut applied = 0;
    loop {
        // Phase 1: scan the chunk's nodes; collect per-(node, x_idx) groups
        // with > 1 distinct marginal-side index. Compute the fused value for
        // each.
        plans.clear();
        let done =
            collect_fusion_plans::<D>(eng, tdd, boundary, &mut nodes, chunk, &mut plans, &mut scratch.pair_fusion)?;
        if !plans.is_empty() {
            // Phase 2: encode each plan's value as a marginal-side ref, an
            // inline count or a value-keyed slot (`FusionSlots`). `any_inline`
            // makes Phase 3 raise the parent's marginal-side inline marker,
            // without which readers decode the bit-30-tagged ref as a grid
            // coordinate.
            //
            // A weight-marginal leaf's column is pinned
            // (`test_helpers::check::marginal::check_leaf_columns_pinned`), so
            // nothing is minted there: a plan resolves by lookup in the column
            // and is dropped when the column lacks its value. A dropped plan
            // leaves its group's pairs untouched, since Phase 3 rewrites only
            // the x-indices a surviving plan names: one un-fused redex, never a
            // wrong value. `retain_mut` keeps the ascending-`node_idx` order
            // Phase 3 needs.
            let any_inline = if D::LEAF_PINNED && tdd.vtree.node(v).is_leaf() {
                plans.retain_mut(|plan| match D::leaf_ref(tdd, v, &plan.value) {
                    Some(raw) => {
                        plan.new_ref = raw;
                        true
                    }
                    None => false,
                });
                plans.iter().any(|plan| {
                    matches!(ChildDecoder::marginal().value(EncodedChildRef::from_raw(plan.new_ref)), ValueRef::Inline(_))
                })
            } else {
                slots.allocate(eng, tdd, v, &mut plans)?
            };
            // Counted after Phase 2, which may drop leaf plans: `fused`
            // counts applied rewrites only, and the contract fixpoint loops
            // while it is nonzero.
            *fused += plans.len();
            // Phase 3: rewrite the parent pair lists of the chunk's nodes.
            if !plans.is_empty() {
                rewrite.apply(eng, tdd, parent, side, any_inline, &plans)?;
                #[cfg(test)]
                {
                    applied += 1;
                    if applied == 2 {
                        tests::note_chunked_boundary();
                    }
                }
            }
        }
        if done {
            return Ok(());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
