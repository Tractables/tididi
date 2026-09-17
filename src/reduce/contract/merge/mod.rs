//! Twin-group contraction: merging nodes that share a parent context.

use crate::diagram::Changed;
use crate::Engine;
use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;
use crate::diagram::{NodeIdx, Tdd};

use super::scratch::{ContractScratch, MergeBuffers, MergeRemap};

mod data;
mod plan;
mod rewrite;

use data::{compact_and_fork_down, merge_twin_data};
use plan::{plan_groups, reserve_transactional, MergePolicy};
pub(in crate::reduce::contract) use plan::{GroupAction, GroupPlan};
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
/// 2. **Rewrite the parent**: drop the pairs that reference merged-away twins
///    and remap the rest through the composed (twin → kept → compact) map.
///
/// 3. **Compact**: remove merged-away nodes from the level in place, then
///    sweep each level's arena once its garbage dominates
///    (`TddLevel::compact_pairs_if_stale`).
///
/// Returns the number of members merged away; 0 means the level is unchanged.
pub(super) fn contract_twins(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) -> Result<usize, OperationError> {
    let lim = eng.limits();
    // t1 is never a marginal level here: `contract_child` returns early on one
    // (marginal-side redexes go to pair fusion), so only explicit-side refs are
    // rewritten below.
    let width = tdd.levels[t1.idx()].slot_count();

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
    let ContractScratch { remap, merge: bufs, duplicate, group_starts, flat_groups, has_marginal_below, .. } = scratch;
    lim.try_resize(&mut remap.merge_target, width, 0u32)?;
    lim.try_resize(&mut remap.final_remap, width, NodeIdx(0))?;
    for i in 0..width { remap.merge_target[i] = i as u32; }
    let policy = MergePolicy::decide(tdd, t1, parent, has_marginal_below);
    remap.duplicate_redirect.clear();
    lim.try_resize(&mut remap.duplicate_redirect, width, false)?;
    bufs.clear();

    plan_groups(tdd, t1, &policy, group_starts, flat_groups, bufs);
    reserve_transactional(eng, tdd, t1, parent, bufs)?;
    let merged_members = commit_group_actions(tdd, t1, &policy, remap, bufs);
    if merged_members == 0 {
        // Nothing merged: level untouched, no compaction or parent rewrite
        // needed. Returning 0 lets `contract_child` report no-progress.
        return Ok(0);
    }
    // The parent's pair lists are remapped and deduplicated below; t1's nodes
    // were merged, so its pair lists and the identity of its nodes changed.
    tdd.invalidate(parent, Changed::PAIRS);
    tdd.invalidate(t1, Changed::PAIRS | Changed::NODES);

    build_final_remap(remap, width);
    rewrite_parent(tdd, parent, t1_side, remap);
    compact_and_fork_down(eng, tdd, t1, &bufs.resolve_keeps, remap, duplicate)?;

    // Reclaim the parent's shrunk pair lists; legal only now that the rewrite
    // is done and no pair-arena offset is held across the call (the caller
    // obligation on `compact_pairs_if_stale`). t1's arena was swept inside
    // `compact_and_fork_down`.
    tdd.levels[parent.idx()].compact_pairs_if_stale();
    Ok(merged_members)
}

/// Commit each planned group: point its absorbed members at the survivor and,
/// for a concat plan, union their pair lists into it. Returns the number of
/// members merged away — 0 means the level is unchanged.
fn commit_group_actions(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    policy: &MergePolicy,
    remap: &mut MergeRemap,
    bufs: &mut MergeBuffers,
) -> usize {
    // 0 ⇒ every group was overlap-filtered and the level is unchanged; reporting
    // progress then would spin the caller's fixpoint loop.
    let mut merged_members = 0usize;
    let MergeBuffers { sel, group_plans, resolve_keeps, .. } = bufs;
    for p in &*group_plans {
        let members = &sel[p.start as usize..p.end as usize];
        let keep = members[0];
        match p.action {
            GroupAction::Concat => {
                for &idx in &members[1..] {
                    remap.merge_target[idx as usize] = keep;
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
                    remap.merge_target[idx as usize] = keep;
                    remap.duplicate_redirect[idx as usize] = true;
                }
                merged_members += members.len() - 1;
            }
        }
    }
    merged_members
}

#[cfg(test)]
mod tests;
