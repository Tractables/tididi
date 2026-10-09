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
//! joint fixed point (the only place we iterate, and only with a marginal
//! level: without one, the sibling's contexts move but its twins stay as
//! they were, see [`joint_contract_fixpoint`]); then descend. Once a parent is
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
use super::fingerprint::{find_listed_twin_groups, find_twin_groups, named_by, pair_mass, screens};
use super::merge::contract_twins;

/// Which nodes of a child level a twin search reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reach {
    /// Every node.
    Whole,
    /// The nodes listed in `scratch.reach` for the child's side, which hold
    /// every node that can have a twin (`fingerprint::listed`).
    Listed,
}

/// A listed search reads every entry of the parent level once and then
/// fingerprints all of the listed nodes' entries, without the screens a
/// search of the whole level runs first under a parent of more than a few
/// entries (`fingerprint::screens`): there, a list of more than half of the
/// level is searched whole. Where a parent has at least this many nodes,
/// listing its children is itself weighed against the whole searches
/// (`list_changed_children`).
const LISTED_WHOLE_MIN_WIDTH: usize = 1 << 16;

// A test counts the listed searches, to check the sweep takes them.
#[cfg(test)]
use super::tests::note_listed_search;

#[cfg(not(test))]
#[inline(always)]
fn note_listed_search() {}

/// Contract one child level `t1` (with parent `parent`) if it has twins,
/// searching the nodes `reach` names. Returns `Ok(true)` iff a productive
/// contraction fired, and then adds its survivors to the list
/// `scratch.listing` keeps of `t1` where the sweep lists.
#[inline]
fn contract_child(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1: VtreeIdx,
    reach: Reach,
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
    if tdd.levels[t1.idx()].slot_count() <= 1 {
        return Ok(false);
    }
    if !tdd.levels[parent.idx()].has_multi_pair() {
        return Ok(false);
    }
    let (parent_left, _parent_right) = tdd.vtree.children(parent);
    let t1_side = if parent_left == t1 { ChildSide::Left } else { ChildSide::Right };
    let side = t1_side as usize;
    let width = tdd.levels[t1.idx()].slot_count();
    // The search reads the nodes `reach` names. Charged before a parent's
    // description can answer for it, so a described level meters the work
    // its written form does.
    scratch.searched += match reach {
        Reach::Listed => scratch.reach[side].len(),
        Reach::Whole => width,
    } as u64;
    // A parent held as the description of its pairs shows from its digits
    // alone that the child has no twins, in most cases; then the grouping
    // below would find none, and need not read the parent's pairs.
    if let Some(d) = tdd.levels[parent.idx()].implicit()
        && d.twin_free(t1_side, tdd.levels[t1.idx()].slot_count())
    {
        return Ok(false);
    }

    let found = match reach {
        Reach::Listed if !screens(&tdd.levels[parent.idx()]) || 2 * scratch.reach[side].len() <= width => {
            note_listed_search();
            let listed = std::mem::take(&mut scratch.reach[side]);
            let found = find_listed_twin_groups(eng, tdd, parent, t1_side, &listed, scratch);
            scratch.reach[side] = listed;
            let found = found?;
            if cfg!(debug_assertions) {
                check_listed_search(eng, tdd, t1, parent, t1_side, found, scratch)?;
            }
            found
        }
        _ => find_twin_groups(eng, tdd, t1, parent, t1_side, scratch)?,
    };
    if !found {
        return Ok(false);
    }

    // Where the sweep lists, the list of `t1` it keeps has its room for the
    // survivors before the merge changes anything, so a refused reservation
    // leaves the diagram as it was. Whether the diagram has a marginal level
    // is read at the first merge, once per checkout, as the merge plan reads
    // it.
    let lists = !diagram_marginal(tdd, scratch);
    if lists {
        reserve_survivors(eng, t1, scratch)?;
    }
    let merged = contract_twins(eng, tdd, t1, parent, t1_side, scratch)?;
    if merged == 0 {
        // Every found group was overlap-filtered: the level is unchanged, and
        // reporting progress would spin the sibling-pair loop.
        return Ok(false);
    }
    if lists {
        note_survivors(t1, scratch);
    }
    Ok(true)
}

/// A listed search must find every group a search of the whole level does:
/// the whole level is searched again and the two compared, in debug builds.
/// A difference means some level off the worklist was not at its fixpoint.
fn check_listed_search(
    eng: &Engine,
    tdd: &Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    found: bool,
    scratch: &mut ContractScratch,
) -> Result<(), OperationError> {
    let listed = (found, scratch.group_starts.clone(), scratch.flat_groups.clone());
    let whole = find_twin_groups(eng, tdd, t1, parent, t1_side, scratch)?;
    assert_eq!(
        (whole, &scratch.group_starts, &scratch.flat_groups),
        (listed.0, &listed.1, &listed.2),
        "the twins of level {} under {} listed {} of {} nodes and missed a group",
        t1.0,
        parent.0,
        scratch.reach[t1_side as usize].len(),
        tdd.levels[t1.idx()].slot_count(),
    );
    Ok(())
}

/// Before a contraction at `t1`, where the sweep lists: room in what the
/// sweep has changed at `t1` for one survivor per group found, the most the
/// merge plans.
fn reserve_survivors(eng: &Engine, t1: VtreeIdx, scratch: &mut ContractScratch) -> Result<(), OperationError> {
    let ContractScratch { group_starts, listing, .. } = scratch;
    let list = listing.list_mut(eng.limits(), t1.idx())?;
    eng.limits().reserve(list, group_starts.len())
}

/// After a contraction at `t1`, where the sweep lists: add its survivors,
/// in the level's new numbering, to what the sweep has changed at `t1`,
/// whose earlier entries are renumbered, in the room [`reserve_survivors`]
/// had.
fn note_survivors(t1: VtreeIdx, scratch: &mut ContractScratch) {
    let ContractScratch { remap, merge, listing, .. } = scratch;
    let list = listing.listed_mut(t1.idx());
    for node in list.iter_mut() {
        *node = remap.final_remap[*node as usize].0;
    }
    debug_assert!(list.capacity() - list.len() >= merge.group_plans.len(), "the survivors' room was had before the merge");
    for plan in &merge.group_plans {
        let keep = merge.sel[plan.start as usize];
        list.push(remap.final_remap[keep as usize].0);
    }
    list.sort_unstable();
    list.dedup();
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
    let pending = current.into_iter().chain(heap.iter().map(|&(_, p)| p));
    tdd.dirty.requeue(Pass::Contract, pending.inspect(|&p| scratch.needs_check[p as usize] = false));
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
/// the unprocessed parents are back on the `Pass::Contract` worklist, so a
/// later minimize resumes them.
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

    let mut scratch = eng.scratch.reduce.contract.checkout(eng);
    // On OOM here the heap is not yet built, so restore the intact taken worklist
    // wholesale — dropping it would leak the whole dirty set.
    if let Err(e) = lim.try_resize(&mut scratch.needs_check, num_nodes, false) {
        tdd.dirty.restore(Pass::Contract, dirty_parents);
        return Err(e);
    }
    // Without a marginal level a parent the sweep reaches only through a
    // contraction above it has its children searched at the nodes that
    // contraction changed (`fingerprint::listed`); a parent on the worklist
    // has them searched whole. Whether the diagram has a marginal level is
    // read only once a contraction fired, which is what puts a parent off
    // the worklist on the heap: a sweep that merges nothing never reads it.
    if let Err(e) = start_listing(eng, &mut scratch, num_nodes, &dirty_parents) {
        tdd.dirty.restore(Pass::Contract, dirty_parents);
        return Err(e);
    }
    // The unit of work is the parent: a level whose pairs were mutated is a
    // parent whose children's contexts may have moved. Every dirty parent
    // may go in at once.
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(dirty_parents.len());
    for &dp in &dirty_parents {
        push_parent(tdd, &mut scratch, &mut heap, num_nodes, dp as usize);
    }

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
        let reach = if scratch.listing.whole(p_idx) || diagram_marginal(tdd, &mut scratch) {
            Reach::Whole
        } else {
            match list_changed_children(eng, tdd, parent, &mut scratch) {
                Ok(reach) => reach,
                Err(e) => {
                    restore_pending_dirty(tdd, &mut scratch, Some(p_raw), &heap);
                    return Err(e);
                }
            }
        };
        let (left_fired, right_fired) = match joint_contract_fixpoint(
            eng,
            tdd, parent, left, right, is_marginal_boundary, reach, &mut scratch,
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
        // The children's searches read their nodes, which the poll above,
        // metered by the parent, does not see. Charged once the fired
        // children are on the heap, so a stop here hands every pending
        // parent back.
        if let Err(e) = poll.poll(std::mem::take(&mut scratch.searched)) {
            restore_pending_dirty(tdd, &mut scratch, None, &heap);
            return Err(e);
        }
    }

    Ok(())
}

/// Whether the diagram has a marginal level, read once per scratch checkout.
fn diagram_marginal(tdd: &Tdd, scratch: &mut ContractScratch) -> bool {
    *scratch.diagram_marginal.get_or_insert_with(|| tdd.has_marginal_level())
}

/// Set up the listed searches of one sweep: the parents on the worklist are
/// searched whole, and no level has changed yet. In time independent of the
/// vtree's size, past the first sweep over a diagram of its size
/// ([`Listing`](super::scratch::Listing)).
fn start_listing(
    eng: &Engine,
    scratch: &mut ContractScratch,
    num_nodes: usize,
    dirty_parents: &[u32],
) -> Result<(), OperationError> {
    scratch.listing.start(eng.limits(), num_nodes, dirty_parents)?;
    for reach in &mut scratch.reach {
        reach.clear();
    }
    Ok(())
}

/// Listing a parent's children reads its changed nodes' pairs, and each
/// listed search reads every pair of the parent once more: where the changed
/// nodes of a parent of at least [`LISTED_WHOLE_MIN_WIDTH`] nodes hold more
/// than this share of its pairs, the whole searches' screens are no dearer.
const LISTED_MAX_PAIR_SHARE: usize = 4;

/// For a parent the sweep reached through contractions above it: list, per
/// child side, the nodes the pairs of the parent's changed nodes name, and
/// say so; or, where those nodes hold most of a wide parent's pairs, list
/// nothing and say the children are searched whole.
fn list_changed_children(
    eng: &Engine,
    tdd: &Tdd,
    parent: VtreeIdx,
    scratch: &mut ContractScratch,
) -> Result<Reach, OperationError> {
    let (left, right) = tdd.vtree.children(parent);
    let changed = scratch.listing.take(parent.idx());
    // Nodes and arena together bound the parent's pairs from above.
    let level = &tdd.levels[parent.idx()];
    if level.nodes().len() >= LISTED_WHOLE_MIN_WIDTH
        && LISTED_MAX_PAIR_SHARE * pair_mass(tdd, parent, &changed) > level.nodes().len() + level.arena_len()
    {
        scratch.listing.put(parent.idx(), changed);
        return Ok(Reach::Whole);
    }
    let mut listed = Ok(());
    for (t1, side) in [(left, ChildSide::Left), (right, ChildSide::Right)] {
        let ContractScratch { reach, reach_bits, .. } = &mut *scratch;
        reach[side as usize].clear();
        if listed.is_ok() && !tdd.vtree.node(t1).is_leaf() && !tdd.levels[t1.idx()].is_marginal() {
            listed = named_by(eng.limits(), tdd, t1, parent, side, &changed, reach_bits, &mut reach[side as usize]);
        }
    }
    scratch.listing.put(parent.idx(), changed);
    listed.map(|()| Reach::Listed)
}

/// Sibling-pair joint fixed point at one parent. Returns whether the left and
/// right child each fired at least once.
///
/// Without a marginal level each child is searched once, at the nodes
/// `reach` names. Every twin group found there merges whole and its survivor
/// keeps the members' contexts, so the child has no twins left; and the
/// merge leaves the sibling's twins as they were. A sibling node's context
/// holds a member beside a parent node `p` exactly where the member's context
/// holds that sibling node beside `p`, so it holds every member of the group
/// beside `p` or none. The merge (and the dedup of the parent's pairs after
/// it) puts the survivor in place of all of them, in every such context
/// alike, so two of the sibling's contexts are equal after it exactly where
/// they were before. Debug builds search both children whole afterwards and
/// assert that neither has a twin.
///
/// With a marginal level a group the overlap filter held back is looked at
/// again, and both children are searched whole per iteration until neither
/// fires. The first round is the one above either way, and whether the
/// diagram has a marginal level is read only once it fired, so that a
/// parent whose children have no twin costs no scan of the diagram's
/// levels. When `is_marginal_boundary`, pair fusion runs at the parent each
/// iteration too: fusion rewrites the parent's pair lists, which can create
/// twins at either child, and contraction can mint fusion redexes.
///
/// Termination: each productive contraction lowers the explicit node count,
/// and each productive fusion keeps it and lowers the total pair count (k ≥ 2
/// pairs at one child become one), so the pair (node count, pair count)
/// strictly decreases lexicographically at every productive step.
#[allow(clippy::too_many_arguments)]
fn joint_contract_fixpoint(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    is_marginal_boundary: bool,
    reach: Reach,
    scratch: &mut ContractScratch,
) -> Result<(bool, bool), OperationError> {
    let mut left_fired = false;
    let mut right_fired = false;
    if !is_marginal_boundary {
        // The first round: each child searched once, at the nodes `reach`
        // names. Without a marginal level it is the whole fixpoint; with one
        // (`reach` is then `Whole`), a round that fires nothing is the
        // fixpoint too, so whether the diagram has one is read only once a
        // child fired, once per checkout.
        eng.limits().check_stop()?;
        let left_once = contract_child(eng, tdd, parent, left, reach, scratch);
        let right_once = left_once.and_then(|_| contract_child(eng, tdd, parent, right, reach, scratch));
        for list in &mut scratch.reach {
            list.clear();
        }
        (left_fired, right_fired) = (left_once?, right_once?);
        if !(left_fired || right_fired) || !diagram_marginal(tdd, scratch) {
            #[cfg(debug_assertions)]
            if !diagram_marginal(tdd, scratch) {
                for child in [left, right] {
                    debug_assert_no_twins(eng, tdd, parent, child, scratch)?;
                }
            }
            return Ok((left_fired, right_fired));
        }
        debug_assert_eq!(reach, Reach::Whole, "a diagram with a marginal level is searched whole");
    }
    // A child fired in a diagram with a marginal level, or the parent borders
    // one: rounds of whole searches until neither child fires.
    let (mut scan_left, mut scan_right) = (true, true);
    loop {
        // As in `Reduction::content_twins`: termination is argued, not
        // bounded, so the round boundary is where cancellation cuts in, with
        // the last round's searches charged first.
        eng.limits().poll_host_work(std::mem::take(&mut scratch.searched))?;
        let mut changed = false;
        if std::mem::take(&mut scan_left) && contract_child(eng, tdd, parent, left, Reach::Whole, scratch)? {
            changed = true;
            left_fired = true;
        }
        if std::mem::take(&mut scan_right) && contract_child(eng, tdd, parent, right, Reach::Whole, scratch)? {
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
        (scan_left, scan_right) = (true, true);
    }
    Ok((left_fired, right_fired))
}

/// A child the joint fixpoint of a diagram without a marginal level is done
/// with has no twin under `parent`: searched whole, in debug builds.
#[cfg(debug_assertions)]
fn debug_assert_no_twins(
    eng: &Engine,
    tdd: &Tdd,
    parent: VtreeIdx,
    child: VtreeIdx,
    scratch: &mut ContractScratch,
) -> Result<(), OperationError> {
    let level = &tdd.levels[child.idx()];
    if tdd.vtree.node(child).is_leaf()
        || level.is_marginal()
        || level.slot_count() <= 1
        || !tdd.levels[parent.idx()].has_multi_pair()
    {
        return Ok(());
    }
    let side = if tdd.vtree.children(parent).0 == child { ChildSide::Left } else { ChildSide::Right };
    let found = find_twin_groups(eng, tdd, child, parent, side, scratch)?;
    assert!(!found, "level {} keeps twins under {} after its joint fixpoint", child.0, parent.0);
    Ok(())
}

#[cfg(test)]
#[path = "tests/sweep/mod.rs"]
mod tests;
