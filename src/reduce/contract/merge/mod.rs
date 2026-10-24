//! Twin-group contraction: merging nodes that share a parent context.

use crate::Engine;
use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;
use crate::diagram::{NodeIdx, Tdd};

use super::scratch::{ContractScratch, MergeBuffers, MergeRemap};

mod data;
mod plan;
mod rewrite;

use data::{fork_down, rebuild_level, reserve_rebuilt_level};
use plan::{plan_groups, MergePolicy};
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
/// 1. **Plan**: decide each group's action and point its absorbed members at
///    the first ("kept") node; reserve the rebuilt level's arena.
///
/// 2. **Rewrite the parent**: drop the pairs that reference merged-away twins
///    and remap the rest through the composed (twin → kept → compact) map,
///    then sweep the parent's arena once its garbage dominates
///    (`TddLevel::compact_pairs_if_stale`).
///
/// 3. **Rebuild**: one pass over the level in node order unions each group's
///    pairs into its kept node, drops the merged-away nodes and leaves the
///    level's arena with no garbage.
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

    // Step 1: plan each group's merge into its first ("kept") node.
    //
    // `merge_target[i]` maps each node to the kept node it merges into.
    // Canonical nodes (merge_target[i] == i) survive; others are absorbed.
    //
    // The three level-width buffers (`merge_target`, `duplicate_redirect`,
    // `final_remap`) are grown fallibly and before any mutation, so a grow
    // that trips the budget surfaces before the diagram changes.
    // `final_remap` is only filled in Step 2, but it is sized here for that
    // reason.
    let ContractScratch {
        remap, merge: bufs, duplicate, group_starts, flat_groups, has_marginal_below, has_marginal_below_valid,
        diagram_marginal, ..
    } = scratch;
    lim.try_resize(&mut remap.merge_target, width, 0u32)?;
    lim.try_resize(&mut remap.final_remap, width, NodeIdx(0))?;
    for i in 0..width { remap.merge_target[i] = i as u32; }
    let diagram_marginal = *diagram_marginal.get_or_insert_with(|| tdd.has_marginal_level());
    let mut policy = MergePolicy::decide(tdd, t1, parent, diagram_marginal);
    remap.duplicate_redirect.clear();
    lim.try_resize(&mut remap.duplicate_redirect, width, false)?;
    bufs.clear();

    // A plain level of a diagram with a marginal level reads the marginal
    // map, a pass over every level of the diagram, filled once per scratch
    // checkout: where a child side of `t1` has marginalization below it,
    // the plan concatenates every member without the overlap filter. A
    // diagram without a marginal level never fills it; its twin supports
    // are disjoint, and the plan concatenates every member anyway.
    if policy.plain_level && diagram_marginal {
        if !*has_marginal_below_valid {
            super::duplicate_pair::compute_has_marginal_below_into(lim, tdd, has_marginal_below)?;
            *has_marginal_below_valid = true;
        }
        policy.t1_scalable = MergePolicy::scalable(tdd, t1, has_marginal_below);
    }
    // The rebuild reads the members' stored pairs, so an implicit t1 is
    // built stored first. It is built before the plan, which reads every
    // member's pairs by index, twice where the overlap filter runs: a stored
    // node's pairs are read off its words, an implicit level's generated
    // from its description one node at a time.
    if !group_starts.is_empty() {
        tdd.levels[t1.idx()].store_if_implicit(lim)?;
        tdd.levels.mark_changed(t1);
    }
    plan_groups(lim, tdd, t1, &policy, group_starts, flat_groups, bufs)?;
    // At most one fork-down survivor per planned group.
    lim.reserve_exact(&mut bufs.resolve_keeps, bufs.group_plans.len())?;
    let merged_members = mark_group_members(&policy, remap, bufs);
    if merged_members == 0 {
        // Nothing merged: level untouched, no compaction or parent rewrite
        // needed. Returning 0 lets `contract_child` report no-progress.
        return Ok(0);
    }
    // Everything the merge allocates is had before it changes anything: the
    // rebuilt level's arenas, and the room an implicit parent is rewritten
    // into where a pair changes.
    let arenas = reserve_rebuilt_level(lim, &tdd.levels[t1.idx()], remap, bufs)?;
    let parent_room = match tdd.levels[parent.idx()].pairs.implicit() {
        Some(_) => Some(tdd.levels[parent.idx()].store_room(lim)?),
        None => None,
    };
    // The parent's pair lists are remapped and deduplicated below; t1's nodes
    // were merged, so its pair lists and the identity of its nodes changed.
    tdd.invalidate(parent);
    tdd.invalidate_with_parent(t1);
    tdd.levels.mark_changed(parent);

    build_final_remap(remap, width);
    rewrite_parent(lim, tdd, parent, t1_side, remap, parent_room);
    // Reclaim the parent's shrunk pair lists; legal only now that the rewrite
    // is done and no pair-arena offset is held across the call (the caller
    // obligation on `compact_pairs_if_stale`).
    tdd.levels[parent.idx()].compact_pairs_if_stale();
    // A repeated pair is legal where a level of the diagram is marginal, and
    // where fork-down resolves it right after the rebuild; elsewhere debug
    // builds check the unions for one.
    let check_disjoint = cfg!(debug_assertions)
        && !policy.t1_scalable
        && !diagram_marginal
        && !tdd.levels[t1.idx()].any_value_ref_side();
    rebuild_level(&mut tdd.levels[t1.idx()], arenas, remap, bufs, check_disjoint);
    fork_down(eng, tdd, t1, &bufs.resolve_keeps, remap, duplicate)?;
    Ok(merged_members)
}

/// Point each planned group's absorbed members at its survivor, and list
/// the concat survivors fork-down resolves. Returns the number of members
/// merged away — 0 means the level is unchanged. Touches only the scratch:
/// the rebuild unions the pairs.
fn mark_group_members(policy: &MergePolicy, remap: &mut MergeRemap, bufs: &mut MergeBuffers) -> usize {
    // 0 ⇒ every group was overlap-filtered and the level is unchanged; reporting
    // progress then would spin the caller's fixpoint loop.
    let mut merged_members = 0usize;
    let MergeBuffers { sel, group_plans, resolve_keeps, .. } = bufs;
    for p in &*group_plans {
        let members = &sel[p.start as usize..p.end as usize];
        let keep = members[0];
        for &idx in &members[1..] {
            remap.merge_target[idx as usize] = keep;
        }
        merged_members += members.len() - 1;
        match p.action {
            // Concat-all path: the survivor may hold duplicate pairs, which
            // fork-down resolves after the rebuild.
            GroupAction::Concat if policy.t1_scalable => resolve_keeps.push(keep),
            GroupAction::Concat => {}
            GroupAction::DupRedirect => {
                for &idx in &members[1..] {
                    remap.duplicate_redirect[idx as usize] = true;
                }
            }
        }
    }
    merged_members
}

#[cfg(test)]
mod tests;
