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

use data::{fork_down, rebuild_level, reserve_rebuilt};
use plan::{plan_groups, MergePolicy};
pub(in crate::reduce::contract) use plan::{GroupAction, GroupPlan};
use rewrite::rewrite_parent;
pub(in crate::reduce::contract) use rewrite::{survivor, DISCARDED, DROPPED};

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
/// 1. **Plan**: decide each group's action, mark its absorbed members and
///    size the rebuilt level's arena from its pair count and the groups.
///
/// 2. **Rebuild**: one pass over the level in node order unions each group's
///    pairs into its kept node, drops the merged-away nodes, leaves the
///    level's arena with no garbage, and numbers the survivors, pointing
///    each member at its survivor's new index.
///
/// 3. **Rewrite the parent**: drop the pairs that reference merged-away twins
///    and remap the rest through that map, then sweep the parent's arena
///    once its garbage dominates (`TddLevel::compact_pairs_if_stale`).
///
/// The nodes in `scratch.unnamed`, which no pair of the parent names, are
/// dropped with their pairs in the same rebuild.
///
/// Returns the number of members merged away and nodes dropped; 0 means
/// the level is unchanged.
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
    // `remap.absorbed` marks the members merged away; every other node
    // survives. `final_remap` takes each node to its survivor's new index.
    // Both are grown fallibly and before any mutation, so a grow that trips
    // the budget surfaces before the diagram changes.
    let ContractScratch {
        remap, merge: bufs, duplicate, group_starts, flat_groups, has_marginal_below, has_marginal_below_valid,
        diagram_marginal, unnamed, ..
    } = scratch;
    lim.try_resize(&mut remap.final_remap, width, NodeIdx(0))?;
    let words = width.div_ceil(64);
    lim.try_resize(&mut remap.absorbed, words, 0u64)?;
    remap.absorbed[..words].fill(0);
    let diagram_marginal = *diagram_marginal.get_or_insert_with(|| tdd.has_marginal_level());
    let mut policy = MergePolicy::decide(tdd, t1, parent, diagram_marginal);
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
    if !group_starts.is_empty() || !unnamed.is_empty() {
        tdd.levels[t1.idx()].store_if_implicit(lim)?;
        tdd.levels.mark_changed(t1);
    }
    plan_groups(lim, tdd, t1, &policy, group_starts, flat_groups, bufs)?;
    // At most one fork-down survivor per planned group.
    lim.reserve_exact(&mut bufs.resolve_keeps, bufs.group_plans.len())?;
    let merged_members = mark_group_members(&policy, remap, bufs);
    for &u in unnamed.iter() {
        debug_assert!(remap.absorbed[(u / 64) as usize] >> (u % 64) & 1 == 0, "an unnamed node is in no group");
        remap.absorbed[(u / 64) as usize] |= 1 << (u % 64);
        remap.final_remap[u as usize] = NodeIdx(DISCARDED);
    }
    if merged_members + unnamed.len() == 0 {
        // Nothing merged: level untouched, no compaction or parent rewrite
        // needed. Returning 0 lets `contract_child` report no-progress.
        return Ok(0);
    }
    // Everything the merge allocates is had before it changes anything: the
    // rebuilt level's arenas, and the room an implicit parent is rewritten
    // into where a pair changes.
    let arenas = reserve_rebuilt(lim, &tdd.levels[t1.idx()], remap, bufs, unnamed)?;
    let parent_room = match tdd.levels[parent.idx()].pairs.implicit() {
        Some(_) => Some(tdd.levels[parent.idx()].store_room(lim)?),
        None => None,
    };
    // The parent's pair lists are remapped and deduplicated below; t1's nodes
    // were merged, so its pair lists and the identity of its nodes changed.
    // The parent's fusion record stands unless a group redirects
    // ([`redirects`]).
    // A level that only lost nodes no pair names keeps its fusion record.
    if merged_members == 0 {
        tdd.invalidate_shrunk(t1);
    } else {
        tdd.invalidate(t1);
    }
    if redirects(&bufs.group_plans) {
        tdd.invalidate(parent);
    } else {
        tdd.invalidate_shrunk(parent);
    }
    tdd.levels.mark_changed(parent);

    // A repeated pair is legal where a level of the diagram is marginal, and
    // where fork-down resolves it right after the rebuild; elsewhere debug
    // builds check the unions for one.
    let check_disjoint = cfg!(debug_assertions)
        && !policy.t1_scalable
        && !diagram_marginal
        && !tdd.levels[t1.idx()].any_value_ref_side();
    rebuild_level(&mut tdd.levels[t1.idx()], arenas, remap, bufs, check_disjoint);
    rewrite_parent(lim, tdd, parent, t1_side, remap, parent_room);
    // Reclaim the parent's shrunk pair lists; legal only now that the rewrite
    // is done and no pair-arena offset is held across the call (the caller
    // obligation on `compact_pairs_if_stale`).
    tdd.levels[parent.idx()].compact_pairs_if_stale();
    if !unnamed.is_empty() {
        let (left, right) = tdd.vtree.children(t1);
        for c in [left, right] {
            if !tdd.vtree.node(c).is_leaf() {
                tdd.dirty.add_loose(c.0);
            }
        }
    }
    fork_down(eng, tdd, t1, &bufs.resolve_keeps, remap, duplicate)?;
    Ok(merged_members + unnamed.len())
}

/// Whether a planned group redirects content-equal members onto its
/// survivor. The parent rewrite keeps the members' pairs, which then repeat
/// the survivor's, so a parent node can hold a pair fusion group it did not
/// hold before. Every other merge drops each pair that names a member, and
/// the pair naming the survivor with the same sibling in its place is
/// already there, twins having equal contexts: a parent node gains no
/// repeated reference, and the parent's fusion record stands.
fn redirects(plans: &[GroupPlan]) -> bool {
    plans.iter().any(|p| p.action == GroupAction::DupRedirect)
}

/// After [`contract_twins`] merged something, whether a group redirected
/// content-equal members onto its survivor ([`redirects`]).
pub(in crate::reduce::contract) fn merged_redirects(scratch: &ContractScratch) -> bool {
    redirects(&scratch.merge.group_plans)
}

/// After [`contract_twins`] merged something, whether every member of every
/// group in `scratch` merged into its survivor by concatenation. Where one
/// did not, the level can still have twins: the overlap filter held it
/// back, to merge in a later round, or it was redirected onto its survivor
/// as a content-equal duplicate, which multiplies the survivor's contexts.
pub(in crate::reduce::contract) fn merged_whole(scratch: &ContractScratch) -> bool {
    let plans = &scratch.merge.group_plans;
    plans.len() == scratch.group_starts.len()
        && scratch.merge.sel.len() == scratch.flat_groups.len()
        && plans.iter().all(|p| p.action == GroupAction::Concat)
}

/// Mark each planned group's absorbed members, and list the concat
/// survivors fork-down resolves. A member's `final_remap` entry holds its
/// survivor's old index, with [`DROPPED`] set unless the member is a
/// content-equal redirect, until [`rebuild_level`] numbers the
/// survivors. Returns the number of members merged away — 0 means the level
/// is unchanged. Touches only the scratch: the rebuild unions the pairs.
fn mark_group_members(policy: &MergePolicy, remap: &mut MergeRemap, bufs: &mut MergeBuffers) -> usize {
    // 0 ⇒ every group was overlap-filtered and the level is unchanged; reporting
    // progress then would spin the caller's fixpoint loop.
    let mut merged_members = 0usize;
    let MergeBuffers { sel, group_plans, resolve_keeps, .. } = bufs;
    for p in &*group_plans {
        let members = &sel[p.start as usize..p.end as usize];
        let keep = members[0];
        // A content-equal redirect keeps the members' parent pairs, remapped
        // onto the survivor; every other member's are dropped.
        let dropped = if p.action == GroupAction::DupRedirect { 0 } else { DROPPED };
        for &idx in &members[1..] {
            remap.absorbed[(idx / 64) as usize] |= 1 << (idx % 64);
            remap.final_remap[idx as usize] = NodeIdx(keep | dropped);
        }
        merged_members += members.len() - 1;
        // Concat-all path: the survivor may hold duplicate pairs, which
        // fork-down resolves after the rebuild.
        if p.action == GroupAction::Concat && policy.t1_scalable {
            resolve_keeps.push(keep);
        }
    }
    merged_members
}

#[cfg(test)]
mod tests;
