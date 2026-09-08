use crate::engine::Engine;
use std::collections::BinaryHeap;

use crate::marg_slots::ChildSide;
use crate::vtree::VtreeIdx;

use crate::engine::PollGate;

use crate::error::ApplyError;
use crate::diagram::*;

use super::scratch::{ContractScratch, take_scratch, return_scratch};
use super::fingerprint::find_twin_groups;
use super::merge::contract_twins;

/// Contract all twin nodes across the entire TDD in a single top-down pass.
///
/// ## Why one pass suffices
///
/// A contraction can create fresh twins only below the contracted level, never
/// at or above its parent, so one parents-before-children sweep reaches the
/// global canonical form. The full argument lives with the implementation —
/// see the "Top-down contraction" soundness note on
/// `contract_all_twins_topdown` below.
///
/// ## Sparse seed via `tdd.dirty.contract`
///
/// Sites that mutate a level's pair list (rotate, leaf-twin rewrite, full
/// minimize after prune) push the parent index into `tdd.dirty.contract`, and
/// operations that REBUILD a diagram hand the list to `Tdd::with_levels_dirty`
/// (the clause apply names its spine; `Tdd::with_levels` names every internal
/// level, the conservative default). We consume that list to seed the heap with
/// the dirty *parents* — O(|dirty|) instead of O(num_vtree_nodes) per call. In
/// the rotation-search hot path, |dirty| is typically 2 (the rotated v_idx and
/// w_idx), vs num_vtree_nodes ≈ 13 600 on Berger feature models.
pub(crate) fn contract_all_twins(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    let r = contract_all_twins_topdown(eng, tdd, None);
    r
}

/// Locality-asserting variant: under rotation locality, the only level
/// that can have fresh twins after `restructure_after_*_rotation` is the
/// newly-introduced inner-node level (`w_idx` in the rotation info). The
/// outer level at `v_idx` inherits canonicity from the pre-rotation `v_idx`
/// level by parent-context bijection (same node count and same parent
/// contexts at the unchanged grandparent), and every other level is
/// bit-identical pre/post.
///
/// In debug builds this routine asserts no productive twin merge fires at
/// any level except `expected_only`. In release it behaves exactly like
/// `contract_all_twins`.
#[cfg(debug_assertions)]
pub(crate) fn contract_all_twins_with_locality(
    eng: &Engine,
    tdd: &mut Tdd,
    expected_only: VtreeIdx,
) -> Result<(), ApplyError> {
    contract_all_twins_topdown(eng, tdd, Some(expected_only))
}

// ── Top-down contraction ──────────────────────────────────────────────────
//
// Soundness: contracting twins at a
// child level edits only (a) that child's own pairs — changing the *contexts of
// its children* — and (b) the parent's pair lists (a dedup), which changes the
// *context of the sibling* but leaves the parent's NODE SET bit-identical. So a
// contraction can create fresh twins only in the sibling and below — never in
// the parent or any ancestor.
//
// Therefore a single top-down sweep over the vtree suffices: process internal
// nodes parents-before-children; at each parent bring its two children to a
// joint fixed point (the only place we iterate); then descend. Once a parent is
// finalized nothing processed later can reopen a twin above it. The max-heap is
// keyed by `topo_pos` (postorder position, root largest) so the "parents first"
// invariant holds even after a rotation has scrambled raw node indices, and even
// when a parent is activated dynamically by an ancestor firing.

/// Contract one child level `t1` (with parent `parent`) if it has twins.
/// Returns `Ok(true)` iff a productive contraction fired. Factored out of the
/// top-down pass so the sibling-pair fixed-point loop can call it on each child.
#[inline]
fn try_contract_child(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1: VtreeIdx,
    scratch: &mut ContractScratch,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] expected_only: Option<VtreeIdx>,
) -> Result<bool, ApplyError> {
    if tdd.vtree.node(t1).is_leaf() {
        return Ok(false);
    }
    // Marginal-side twin contraction is DELETED — p-fusion subsumes it.
    // Marginal-side "twins" (slots sharing the same parent context) are
    // definitionally co-located p-fusion redexes; p-fusion already merges them
    // by summing through the seeded SlotInterner (preserving slot-count
    // uniqueness, including
    // u128→BigUint overflow promotion). Summing counts in-place here (the old
    // in-place path) violated it: two distinct slots can end
    // up holding the same count value WITHOUT re-interning, so explicit-side
    // twins whose pair lists differ only by those equal-valued slot indices would
    // never contract. With this guard, fusion is the ONLY mechanism for
    // marginal-side redexes; the explicit sibling side still contracts normally.
    if tdd.levels[t1.idx()].is_marginal() {
        return Ok(false);
    }
    let width = tdd.levels[t1.idx()].width();
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

    // Rotation-locality tightening: after `restructure_after_*_rotation`
    // the only level that can have fresh twins is the newly-introduced
    // inner-node level (`expected_only`); a productive merge anywhere else means
    // the rotation-locality claim is wrong or this call fed a stale context.
    //
    // This tightening holds only for Boolean (determinism-canonical) diagrams.
    // Once any level is marginal, `restructure` full-expands the rotation as a
    // multiset (no Boolean dedup — the marg_ctx path in `tdd/restructure/relevel.rs`), so
    // fresh twins can legitimately surface at the outer level too. The
    // marginal-rotation fuzz tests (`tdd/restructure/relevel.rs`) exercise exactly this, so
    // gate the single-level locality assert on a marginal-free diagram.
    #[cfg(debug_assertions)]
    if let Some(expected) = expected_only {
        let has_marginal = tdd.levels.iter().any(|l| l.is_marginal());
        if !has_marginal {
            assert_eq!(
                t1, expected,
                "rotation locality: productive twin merge at level {} (expected only at {})",
                t1.0, expected.0,
            );
        }
    }

    // The merge below is the only reader of `has_marg_below`, so fill it here —
    // on the first merge this scratch checkout attempts — instead of once per
    // sweep. See `ContractScratch::has_marg_below_valid` for why one fill covers
    // the rest of the checkout.
    if !scratch.has_marg_below_valid {
        super::dup_resolve::compute_has_marg_below_into(tdd, &mut scratch.has_marg_below);
        scratch.has_marg_below_valid = true;
    }
    let merged = contract_twins(eng, tdd, t1, parent, t1_side, scratch)?;
    if merged == 0 {
        // Every found group was overlap-filtered (multiplicity-carrying twins
        // at a plain level — unmergeable without forking): the level is
        // unchanged; report no-progress or the sibling-pair loop spins.
        return Ok(false);
    }
    // Marginal twins are handled by exactly two mechanisms: generic twin
    // contraction (identical raw-multiset twins, including equal-count slots
    // via the birth-time value dedup on the marginalize path) and p-fusion at this
    // parent level, wired into `contract_all_twins_topdown`'s per-parent
    // fixpoint loop below for the same-explicit-different-count redexes that
    // survive or are minted by contraction.
    //
    // Both point strictly DOWNWARD, and any replacement must too: a rewrite
    // that redirects a grandparent's refs upward violates the top-down worklist
    // invariant.

    Ok(true)
}

/// Enqueue vtree node `p` as a parent to process, if eligible and not already
/// queued. Keyed by `topo_pos` (postorder position: the root has the largest
/// value), so a max-heap yields the shallowest pending parent — closest to the
/// root — first, giving parents-before-children processing even on a rotated
/// vtree. `topo_pos` is maintained incrementally by the vtree's rotate fixups,
/// so this is O(1) per push with no per-call rank rebuild. Leaf nodes and
/// pair-less (marginal / single-pair) levels have no contractable children and
/// are skipped.
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

/// Seed the contraction max-heap from `dirty_parents`.
///
/// Each dirty parent is enqueued in `heap` (keyed by `topo_pos`, so root-most
/// pops first). Called once per `contract_all_twins_topdown` invocation; split
/// out so the heap-setup logic can be read separately from the main loop.
#[inline(always)]
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

/// Restore the still-pending contraction worklist on an error exit from a
/// top-down sweep.
///
/// A sweep `mem::take`s `tdd.dirty.contract` into the topo-heap, so a mid-sweep
/// `Err` — race-lane `Deadline` preemption or `OverBudget` from `contract_twins`
/// — would otherwise drop every parent that had not yet been popped. Those
/// levels keep stale contexts and, being absent from `dirty_contract` (and from
/// any re-seeding — the worklist is maintained incrementally, not rebuilt
/// between sweeps), are never re-contracted until something
/// else re-dirties them: a permanent canonicity/size leak (sound, since twins
/// are count-exact, but a leak). This re-queues the parent that was mid-process
/// when the error fired (`current`) plus every parent still in `heap`, and
/// clears their `needs_check` so the pooled scratch re-enters the all-false
/// invariant the next sweep relies on. Already-processed parents are
/// intentionally NOT re-queued: they are canonically clean (re-seeding them
/// would only cost no-op rescans), matching the benign steady state of a level
/// born `contracted=false` that never had a twin. Restoring even on a
/// poison-path `Err` (whose TDD the caller drops) is harmless.
#[inline]
fn restore_pending_dirty(
    tdd: &mut Tdd,
    scratch: &mut ContractScratch,
    current: Option<u32>,
    heap: &BinaryHeap<(u32, u32)>,
) {
    if let Some(p) = current {
        tdd.dirty.contract.push(p);
        scratch.needs_check[p as usize] = false;
    }
    for &(_topo_pos, p) in heap.iter() {
        tdd.dirty.contract.push(p);
        scratch.needs_check[p as usize] = false;
    }
}

/// Top-down twin contraction. See the module note above for soundness.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated rewrite step fails, or
/// `Err(ApplyError::Deadline)` if the caller's wall passed while the walk was
/// running and the reduce poll is armed. Either way the diagram is well-formed and
/// the unprocessed parents are back in `dirty_contract`, so a later minimize
/// resumes them.
pub(crate) fn contract_all_twins_topdown(
    eng: &Engine,
    tdd: &mut Tdd,
    expected_only: Option<VtreeIdx>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let num_nodes = tdd.vtree.num_nodes();

    let dirty_parents = std::mem::take(&mut tdd.dirty.contract);
    if dirty_parents.is_empty() {
        return Ok(());
    }

    let mut scratch = take_scratch(eng);
    // On OOM here the heap is not yet built, so restore the intact taken worklist
    // wholesale — dropping it would leak the whole dirty set.
    if let Err(e) = lim.try_resize(&mut scratch.needs_check, num_nodes, false) {
        tdd.dirty.contract = dirty_parents;
        return_scratch(eng, scratch);
        return Err(e);
    }
    // Seed the heap with the dirty *parents* themselves: a level whose pairs
    // were mutated is exactly a parent whose children's contexts may have moved.
    // The unit of work is the parent, since contraction reads the parent's
    // pairs. The heap is keyed by `topo_pos` and is a max-heap, so the root-most
    // pending parent pops first.
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::new();
    seed_contract_heap(tdd, &dirty_parents, &mut scratch, &mut heap, num_nodes);

    // The walk's ONE preemption point, amortized. With no stop axis installed the
    // poll short-circuits before any clock read, so the meter below costs an add
    // and a predicted-not-taken branch per popped parent.
    let mut poll = PollGate::new(lim.reduce_poll_stride());

    // Process parents shallow-first. Each parent is popped at most once: any
    // node that could reopen its twins is a strict ancestor (larger topo_pos),
    // hence already popped and finalized before it (induction from the root).
    // The only iteration is the inner sibling-pair fixed point.
    //
    while let Some((_topo_pos, p_raw)) = heap.pop() {
        let p_idx = p_raw as usize;
        // The mid-loop preemption point: this walk is the most expensive phase of
        // a minimize and runs BETWEEN two applies of one bottom-up step, so
        // without it a caller's wall is observed only where the step ends —
        // which on a near-root leaf compile is minutes away. Metered in nodes of
        // the parent's level (the unit `try_contract_child`'s work scales with),
        // and it aborts through the deadline arm every other cut in the compile
        // already takes. `Err` restores the popped parent and the rest of the
        // heap to `dirty_contract` exactly as the OOM arms below do, so a cut
        // walk leaves a well-formed diagram with its pending work intact.
        if let Err(e) = lim.poll(&mut poll, tdd.levels[p_idx].width() as u64 + 1) {
            restore_pending_dirty(tdd, &mut scratch, Some(p_raw), &heap);
            return_scratch(eng, scratch);
            return Err(e);
        }
        scratch.needs_check[p_idx] = false;

        // Guard against stale state (a prior contraction may have flipped
        // has_multi_pair or the node is a leaf after a structural change).
        if tdd.vtree.node(VtreeIdx(p_idx as u32)).is_leaf() || !tdd.levels[p_idx].has_multi_pair() {
            continue;
        }
        let parent = VtreeIdx(p_raw);
        let (left, right) = tdd.vtree.children(parent);

        let is_marg_boundary = tdd.levels[left.idx()].is_marginal()
            || tdd.levels[right.idx()].is_marginal();
        let (left_fired, right_fired) = match joint_contract_fixpoint(
            eng,
            tdd, parent, left, right, is_marg_boundary, &mut scratch, expected_only,
        ) {
            Ok(v) => v,
            Err(e) => {
                restore_pending_dirty(tdd, &mut scratch, Some(p_raw), &heap);
                return_scratch(eng, scratch);
                return Err(e);
            }
        };

        // A child that fired had its own pairs unioned, moving its children's
        // contexts — so enqueue it as a (deeper) parent. Strictly downward, so
        // the heap only ever grows toward the leaves and the single-pass
        // invariant holds.
        if left_fired {
            push_parent(tdd, &mut scratch, &mut heap, num_nodes, left.idx());
            // Feed the content-twin worklist: the left child's pair list changed, so the
            // current parent (p_raw) may have new content-twins if it is a
            // boundary parent.  Also push the fired child itself: if it is a
            // marginal level, its own boundary-parent (p) needs rescanning.
            tdd.dirty.c2_rescan.push(p_raw);
            tdd.dirty.c2_rescan.push(left.0);
        }
        if right_fired {
            push_parent(tdd, &mut scratch, &mut heap, num_nodes, right.idx());
            tdd.dirty.c2_rescan.push(p_raw);
            tdd.dirty.c2_rescan.push(right.0);
        }
    }

    return_scratch(eng, scratch);

    Ok(())
}

/// Sibling-pair joint fixed point at one parent. Returns whether the left and
/// right child each fired at least once.
/// Sibling-pair joint fixed point: contracting one child dedups the
/// parent's pairs, which can equalize the other child's contexts, so we
/// alternate until neither fires. We scan each child per iteration,
/// restarting whenever one fires, until both are clean.
///
/// At marginal-boundary parents (at least one child is marginal),
/// also run p-fusion per iteration. Fusion changes the parent's pair lists,
/// which can create new twins at either child; twin contraction can mint new
/// p-fusion redexes. The joint fixpoint (twin contract + fusion) at this
/// parent terminates because each productive step strictly decreases the
/// lexicographic measure (explicit node count, total pair count, distinct
/// referenced slots). Zero-cost gate: p-fusion is only called when the
/// parent is a marginal boundary (one or both children are marginal).
///
/// The measure argument covers the WEIGHTED arm unchanged, on the SECOND
/// component: a productive fusion group has k ≥ 2 pairs at one x and
/// replaces all k with exactly one, so total pair count drops by k−1 ≥ 1,
/// and fusion never adds an explicit node (first component fixed). Minting
/// a fresh value can RAISE the third component (a `WeightStore` slot on
/// intern-table exhaustion; the interned `MargRef::Inline` form adds no
/// level slot at all) — irrelevant lexicographically, since the second
/// component already fell. Fusion is also idempotent within one call: after
/// the rewrite each fused x carries exactly ONE pair, so an immediately
/// repeated sweep reports `fusion_groups == 0` and cannot re-set `changed`.
fn joint_contract_fixpoint(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    is_marg_boundary: bool,
    scratch: &mut ContractScratch,
    expected_only: Option<VtreeIdx>,
) -> Result<(bool, bool), ApplyError> {
    let mut left_fired = false;
    let mut right_fired = false;
    loop {
        let mut changed = false;
        match try_contract_child(eng, tdd, parent, left, scratch, expected_only) {
            Ok(true) => { changed = true; left_fired = true; }
            Ok(false) => {}
            Err(e) => return Err(e),
        }
        match try_contract_child(eng, tdd, parent, right, scratch, expected_only) {
            Ok(true) => { changed = true; right_fired = true; }
            Ok(false) => {}
            Err(e) => return Err(e),
        }
        // Step 2: run p-fusion at this parent if it is a
        // marginal boundary. Fusion rewrites the parent's pair lists
        // (same-explicit-different-count redexes → one summed slot),
        // which can create new twins at either child — so loop again if
        // it fired. No-op cost on non-marginal-boundary parents.
        if is_marg_boundary {
            // Call the inner directly (not the pooled `apply_p_fusion_at_parents`
            // wrapper) so the fusion grouping scatter reuses this contract run's
            // already-taken `scratch` instead of re-borrowing the pool.
            let fus_res = crate::reduce::contract::p_fusion::apply_p_fusion_inner(
                eng,
                tdd, Some(&[parent]), scratch,
            );
            match fus_res {
                Ok(stats) if stats.fusion_groups > 0 => {
                    changed = true;
                }
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        if !changed {
            break;
        }
    }
    Ok((left_fired, right_fired))
}

#[cfg(test)]
#[path = "strategies_deadline_tests.rs"]
mod strategies_deadline_tests;
