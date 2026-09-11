//! Twin-group contraction: merging nodes that share a parent context.

use crate::diagram::Changed;
use crate::engine::Engine;
use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::limits::ApplyError;
use crate::diagram::{NodeIdx, Tdd};

use super::scratch::{ContractScratch, MergeBuffers};

mod data;
mod plan;
mod rewrite;

use data::{compact_and_fork_down, merge_twin_data};
use plan::{plan_groups, reserve_transactional, MergePolicy};
pub(in crate::reduce::contract) use plan::{GroupAction, GroupPlan};
#[cfg(test)]
use data::{compact_explicit_level, merge_two_internal_twins};
use rewrite::{build_final_remap, rewrite_parent};

/// Merge all twin groups at level t1, then compact.
///
/// Twins are nodes with identical parent contexts (same (parent, sibling)
/// reference set). Since they always co-occur, their Boolean functions can
/// be disjoined (ORed) into a single node without changing the diagram's overall
/// function. Contraction merges nodes with identical *context* (looking up),
/// as opposed to deduplication which would merge nodes with identical *data*
/// (looking down).
///
/// ## Steps
///
/// 1. **Merge twin data**: for each group, union all members' input pairs
///    into the first ("kept") node.
///
/// 2. **Build composed remap**: merge_target (twin → kept) composed with
///    compact indices (gap removal). Apply to parent pairs + sort/dedup.
///
/// 3. **Compact**: remove merged-away nodes from the level in-place.
///
/// 4. **Reclaim**: the merges left every source pair range unreferenced (the
///    union was appended at the arena tail) and the parent rewrite shrank pair
///    lists in place. Both levels' arenas are swept if that garbage now
///    dominates them (`TddLevel::compact_pairs_if_stale`) — t1's the moment its
///    node compaction lands (before fork-down re-grows the arenas), the
///    parent's once its rewrite has finished.
pub(super) fn contract_twins(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) -> Result<usize, ApplyError> {
    let lim = eng.limits();
    // Pair lists at parent (remap+dedup below) and t1 (twin merge in
    // merge_twin_data) are about to be mutated.
    tdd.invalidate(parent, Changed::PAIRS);
    tdd.invalidate(t1, Changed::PAIRS);

    // Lazy unpack: we read `find_twin_groups` via the packed-safe iterator
    // path (see `for_each_target_sibling`), but the mutation below uses
    // slice-based push/filter/pop on `pairs`/`pairs_mut`. Unpack parent
    // and t1 only here — productive merge path, rare relative to the
    // find_twin_groups scan.
    //
    // t1 is never a marginal level here: the sole caller `contract_child`
    // returns early on a marginal t1 (marginal-side redexes go to pair fusion, not
    // twin contraction), so this path only ever rewrites explicit-side refs —
    // no marginal slot/inline handling is needed below.
    let width = tdd.levels[t1.idx()].width();

    // Step 1: Merge twin data — combine each group into its first ("kept") node.
    //
    // `merge_target[i]` maps each node to the kept node it merges into.
    // Canonical nodes (merge_target[i] == i) survive; others are absorbed.
    //
    // The three level-width buffers (`merge_target`, `duplicate_redirect`,
    // `final_remap`) are grown fallibly and before any mutation — a grow that
    // trips the budget must surface here, ahead of the grand reserve, not after
    // Pass B has already merged twins.
    // `final_remap` is only filled in Step 2, but it is sized here for that
    // reason.
    lim.try_resize(&mut scratch.merge_target, width, 0u32)?;
    lim.try_resize(&mut scratch.final_remap, width, NodeIdx(0))?;
    for i in 0..width { scratch.merge_target[i] = i as u32; }
    let policy = MergePolicy::decide(tdd, t1, parent, scratch);
    scratch.duplicate_redirect.clear();
    lim.try_resize(&mut scratch.duplicate_redirect, width, false)?;
    // Working buffers, checked out of the scratch (cleared on take) instead of
    // freshly allocated per call — see `scratch::MergeBuffers`. Parked back at
    // both productive exits.
    let mut bufs = scratch.take_merge_buffers();

    plan_groups(tdd, t1, &policy, scratch, &mut bufs);
    reserve_transactional(eng, tdd, t1, parent, &bufs)?;
    let merged_members = commit_group_actions(tdd, t1, &policy, scratch, &mut bufs);
    if merged_members == 0 {
        // Nothing merged: level untouched, no compaction or parent rewrite
        // needed. Returning 0 lets contract_child report no-progress.
        scratch.put_merge_buffers(bufs);
        return Ok(0);
    }

    build_final_remap(scratch, width);
    rewrite_parent(tdd, parent, t1_side, scratch);
    compact_and_fork_down(eng, tdd, t1, &bufs.resolve_keeps, scratch)?;

    // Step 4: reclaim the parent's shrunk pair lists. (t1's garbage — the far
    // larger mass — is swept inside `compact_and_fork_down`, as early as it is
    // legal to, so fork-down grows the arenas on an already-compacted t1.)
    //
    // This is the safe point and no earlier one is: the parent rewrite has
    // finished, so no pair-arena offset is held across the call — the caller
    // obligation documented on `compact_pairs_if_stale` (types/level.rs).
    tdd.levels[parent.idx()].compact_pairs_if_stale();
    scratch.put_merge_buffers(bufs);
    Ok(merged_members)
}

/// Commit each planned group: point its absorbed members at the survivor and,
/// for a concat plan, union their pair lists into it. Returns the number of
/// members merged away — 0 means the level is unchanged.
fn commit_group_actions(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    policy: &MergePolicy,
    scratch: &mut ContractScratch,
    bufs: &mut MergeBuffers,
) -> usize {
    // Members actually merged away. 0 ⇒ every group was overlap-filtered:
    // the level is unchanged and the caller must not treat this as progress
    // (the groups will be re-found by the next scan; reporting progress here
    // spins the sibling-pair fixed-point loop forever).
    let mut merged_members = 0usize;
    let MergeBuffers { sel, group_plans, resolve_keeps, .. } = bufs;
    for p in &*group_plans {
        let members = &sel[p.start as usize..p.end as usize];
        let keep = members[0];
        match p.action {
            GroupAction::Concat => {
                for &idx in &members[1..] {
                    scratch.merge_target[idx as usize] = keep;
                }
                merged_members += members.len() - 1;
                if policy.t1_scalable {
                    // Concat-all path: the survivor may now hold duplicate
                    // pairs, which fork-down resolves after compaction.
                    resolve_keeps.push(keep);
                }
                merge_twin_data(tdd, t1, members, /*allow_dups=*/ policy.t1_scalable);
            }
            GroupAction::DupRedirect => {
                for &idx in &members[1..] {
                    scratch.merge_target[idx as usize] = keep;
                    scratch.duplicate_redirect[idx as usize] = true;
                }
                merged_members += members.len() - 1;
            }
        }
    }
    merged_members
}

#[cfg(test)]
#[path = "../merge_tests.rs"]
mod tests;
