//! The top-down twin-contraction sweep.
//!
//! Soundness: contracting twins at a
//! child level edits only (a) that child's own pairs — changing the *contexts of
//! its children* — and (b) the parent's pair lists (a dedup), which changes the
//! *context of the sibling* but leaves the parent's own set of nodes bit-identical. So a
//! contraction can create fresh twins only in the sibling and below — never in
//! the parent or any ancestor.
//!
//! Therefore a single top-down sweep over the vtree suffices: process internal
//! nodes parents-before-children; at each parent bring its two children to a
//! joint fixed point (the only place we iterate); then descend. Once a parent is
//! finalized nothing processed later can reopen a twin above it. The max-heap is
//! keyed by `topo_pos` (postorder position, root largest) so the "parents first"
//! invariant holds even after a rotation has scrambled raw node indices, and even
//! when a parent is activated dynamically by an ancestor firing.

use crate::diagram::{ChildSide, Pass, Tdd};
use crate::Engine;
use std::collections::BinaryHeap;

use crate::vtree::VtreeIdx;

use crate::limits::OperationError;

use super::scratch::ContractScratch;
use super::fingerprint::find_twin_groups;
use super::merge::contract_twins;

/// Contract one child level `t1` (with parent `parent`) if it has twins.
/// Returns `Ok(true)` iff a productive contraction fired.
#[inline]
fn contract_child(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1: VtreeIdx,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    if tdd.vtree.node(t1).is_leaf() {
        return Ok(false);
    }
    // Marginal-side twins (slots sharing a parent context) are pair-fusion
    // redexes, and fusion merges them by summing through the seeded slot map,
    // which keeps slot-count uniqueness (invariant 10). Summing counts here
    // would not, so a marginal child is left to fusion.
    if tdd.levels[t1.idx()].is_marginal() {
        return Ok(false);
    }
    let width = tdd.levels[t1.idx()].slot_count();
    if width <= 1 {
        return Ok(false);
    }
    if !tdd.levels[parent.idx()].has_multi_pair() {
        return Ok(false);
    }
    let (parent_left, _parent_right) = tdd.vtree.children(parent);
    let t1_side = if parent_left == t1 { ChildSide::Left } else { ChildSide::Right };

    let found = find_twin_groups(eng, tdd, parent, t1_side, width, scratch)?;
    if !found {
        return Ok(false);
    }

    // The merge below is the only reader of `has_marginal_below`, so fill it here —
    // on the first merge this scratch checkout attempts — instead of once per
    // sweep. See `ContractScratch::has_marginal_below_valid` for why one fill covers
    // the rest of the checkout.
    if !scratch.has_marginal_below_valid {
        super::duplicate_pair_resolve::compute_has_marginal_below_into(tdd, &mut scratch.has_marginal_below);
        scratch.has_marginal_below_valid = true;
    }
    let merged = contract_twins(eng, tdd, t1, parent, t1_side, scratch)?;
    if merged == 0 {
        // Every found group was overlap-filtered: the level is unchanged, and
        // reporting progress would spin the sibling-pair loop.
        return Ok(false);
    }
    Ok(true)
}

/// Enqueue vtree node `p` as a parent to process, if eligible and not already
/// queued. Keyed by `topo_pos` (postorder position, root largest), so the
/// max-heap pops the root-most pending parent first. Leaves and levels without
/// a multi-pair node have no contractable children and are skipped.
#[inline]
pub(super) fn push_parent(
    tdd: &Tdd,
    scratch: &mut ContractScratch,
    heap: &mut BinaryHeap<(u32, u32)>,
    num_nodes: usize,
    p: usize,
) {
    if p < num_nodes
        && !tdd.vtree.node(VtreeIdx(p as u32)).is_leaf()
        && tdd.levels[p].has_multi_pair()
        && !scratch.needs_check[p]
    {
        scratch.needs_check[p] = true;
        heap.push((tdd.vtree.topo_pos(VtreeIdx(p as u32)), p as u32));
    }
}

/// Seed the contraction max-heap with every eligible parent in `dirty_parents`.
fn seed_contract_heap(
    tdd: &Tdd,
    dirty_parents: &[u32],
    scratch: &mut ContractScratch,
    heap: &mut BinaryHeap<(u32, u32)>,
    num_nodes: usize,
) {
    for &dp in dirty_parents {
        push_parent(tdd, scratch, heap, num_nodes, dp as usize);
    }
}

/// Re-queue the parent that was mid-process (`current`) and every parent still
/// in `heap` on an error exit from a sweep, clearing their `needs_check` so
/// the pooled scratch is all-false again for the next sweep.
///
/// The sweep drained the worklist into the heap, and the worklist is
/// maintained incrementally, never rebuilt, so a parent dropped here would
/// keep its stale contexts until something else re-dirties it: a size leak,
/// not a soundness fault. Already-processed parents are clean and stay out.
#[inline]
fn restore_pending_dirty(
    tdd: &mut Tdd,
    scratch: &mut ContractScratch,
    current: Option<u32>,
    heap: &BinaryHeap<(u32, u32)>,
) {
    if let Some(p) = current {
        tdd.dirty.requeue(Pass::Contract, [p]);
        scratch.needs_check[p as usize] = false;
    }
    for &(_topo_pos, p) in heap.iter() {
        tdd.dirty.requeue(Pass::Contract, [p]);
        scratch.needs_check[p as usize] = false;
    }
}

/// Contract every twin in the diagram in one top-down pass over the parents
/// on the twin-contraction worklist, which every site that mutates a level's
/// pair list feeds; the module doc says why one pass suffices. Cost is
/// O(|dirty|) plus the work at each popped parent, not O(num_vtree_nodes).
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if a budget-gated rewrite step fails, or
/// `Err(OperationError::Stopped)` if the caller's wall passed while the walk was
/// running and the reduce poll is armed. Either way the diagram is well-formed and
/// the unprocessed parents are back in `dirty_contract`, so a later minimize
/// resumes them.
pub(crate) fn contract_all_twins(
    eng: &Engine,
    tdd: &mut Tdd,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let num_nodes = tdd.vtree.num_nodes();

    let dirty_parents = tdd.dirty.take(Pass::Contract);
    if dirty_parents.is_empty() {
        return Ok(());
    }

    let mut scratch = eng.reduce_scratch().contract.checkout(lim);
    // On OOM here the heap is not yet built, so restore the intact taken worklist
    // wholesale — dropping it would leak the whole dirty set.
    if let Err(e) = lim.try_resize(&mut scratch.needs_check, num_nodes, false) {
        tdd.dirty.restore(Pass::Contract, dirty_parents);
        return Err(e);
    }
    // The unit of work is the parent: a level whose pairs were mutated is a
    // parent whose children's contexts may have moved.
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::new();
    seed_contract_heap(tdd, &dirty_parents, &mut scratch, &mut heap, num_nodes);

    // The walk's one preemption point, amortized. With no stop axis installed the
    // poll short-circuits before any clock read, so the meter below costs an add
    // and a predicted-not-taken branch per popped parent.
    let mut poll = lim.gate();

    // Each parent is popped at most once: any node that could reopen its twins
    // is a strict ancestor (larger `topo_pos`), hence already popped and
    // finalized before it. The only iteration is the inner sibling-pair
    // fixed point.
    while let Some((_topo_pos, p_raw)) = heap.pop() {
        let p_idx = p_raw as usize;
        // Preemption point, metered in nodes of the parent's level, the unit
        // `contract_child`'s work scales with. On `Err` the popped parent and
        // the rest of the heap go back to the worklist.
        if let Err(e) = poll.poll(tdd.levels[p_idx].slot_count() as u64 + 1) {
            restore_pending_dirty(tdd, &mut scratch, Some(p_raw), &heap);
            return Err(e);
        }
        scratch.needs_check[p_idx] = false;

        // Guard against stale state (a prior contraction may have flipped
        // `has_multi_pair` or the node is a leaf after a structural change).
        if tdd.vtree.node(VtreeIdx(p_idx as u32)).is_leaf() || !tdd.levels[p_idx].has_multi_pair() {
            continue;
        }
        let parent = VtreeIdx(p_raw);
        let (left, right) = tdd.vtree.children(parent);

        let is_marginal_boundary = tdd.levels[left.idx()].is_marginal()
            || tdd.levels[right.idx()].is_marginal();
        let (left_fired, right_fired) = match joint_contract_fixpoint(
            eng,
            tdd, parent, left, right, is_marginal_boundary, &mut scratch,
        ) {
            Ok(v) => v,
            Err(e) => {
                restore_pending_dirty(tdd, &mut scratch, Some(p_raw), &heap);
                return Err(e);
            }
        };

        // A child that fired had its own pairs unioned, moving its children's
        // contexts — so enqueue it as a (deeper) parent. Strictly downward, so
        // the heap only ever grows toward the leaves and the single-pass
        // invariant holds.
        if left_fired {
            push_parent(tdd, &mut scratch, &mut heap, num_nodes, left.idx());
        }
        if right_fired {
            push_parent(tdd, &mut scratch, &mut heap, num_nodes, right.idx());
        }
    }

    Ok(())
}

/// Sibling-pair joint fixed point at one parent. Returns whether the left and
/// right child each fired at least once.
///
/// Contracting one child dedups the parent's pairs, which can equalize the
/// other child's contexts, so both children are scanned per iteration until
/// neither fires. When `is_marginal_boundary`, pair fusion runs at the parent
/// each iteration too: fusion rewrites the parent's pair lists, which can
/// create twins at either child, and contraction can mint fusion redexes.
///
/// Termination: each productive contraction lowers the explicit node count,
/// and each productive fusion keeps it and lowers the total pair count (k ≥ 2
/// pairs at one child become one), so the pair (node count, pair count)
/// strictly decreases lexicographically at every productive step.
fn joint_contract_fixpoint(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    is_marginal_boundary: bool,
    scratch: &mut ContractScratch,
) -> Result<(bool, bool), OperationError> {
    let mut left_fired = false;
    let mut right_fired = false;
    loop {
        // As in `canonicalize_content_twins`: termination is argued, not
        // bounded, so the round boundary is where cancellation cuts in.
        eng.limits().check_stop()?;
        let mut changed = false;
        if contract_child(eng, tdd, parent, left, scratch)? {
            changed = true;
            left_fired = true;
        }
        if contract_child(eng, tdd, parent, right, scratch)? {
            changed = true;
            right_fired = true;
        }
        if is_marginal_boundary {
            // The inner form, so fusion reuses this run's already-taken
            // `scratch` instead of re-borrowing the pool.
            let stats = crate::reduce::contract::pair_fusion::fuse_pairs_inner(
                eng,
                tdd, Some(&[parent]), scratch,
            )?;
            if stats.fusion_groups > 0 {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok((left_fired, right_fired))
}

#[cfg(test)]
#[path = "tests/strategies/mod.rs"]
mod tests;
