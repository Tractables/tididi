//! Deciding what each twin group does, and reserving its arena growth up front.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::OperationError;
use crate::diagram::Tdd;

use super::super::scratch::MergeBuffers;

/// What the commit pass does with one twin group, decided by `plan_groups`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::reduce::contract) enum GroupAction {
    /// Concatenate the selected members' pair lists into the survivor. The only
    /// action that grows t1's arena, hence the only one the grand reserve charges.
    Concat,
    /// Redirect content-equal members onto the survivor without touching its
    /// pair list — the parent rewrite keeps their pairs and pair fusion sums the
    /// multiplicity. Appends nothing.
    DupRedirect,
}

/// One decided twin group: its action plus the `start..end` range of the flat
/// selected-member buffer holding the members it acts on, survivor first.
pub(in crate::reduce::contract) struct GroupPlan {
    pub(in crate::reduce::contract) action: GroupAction,
    pub(in crate::reduce::contract) start: u32,
    pub(in crate::reduce::contract) end: u32,
}

/// The per-level rules this contraction runs under, decided once before any
/// group acts.
pub(super) struct MergePolicy {
    pub(super) plain_level: bool,
    pub(super) parent_marginal: bool,
    pub(super) t1_scalable: bool,
}

impl MergePolicy {
    /// Read the level and parent flags that decide which merges are legal here.
    pub(super) fn decide(
        tdd: &Tdd,
        t1: VtreeIdx,
        parent: VtreeIdx,
        has_marginal_below: &[bool],
    ) -> Self {
    // At a plain (no inlined side) level, twin members whose supports overlap
    // (share a pair) are not concat-merged: the union would hold duplicate
    // pairs, which carry a multiplicity only at marginal-flagged levels and
    // break determinism (invariant 1) elsewhere. The greedy filter in
    // `plan_groups` accepts pairwise-disjoint members, one hash-set pass over
    // the group's pairs.
        let plain_level = !tdd.levels[t1.idx()].any_inlined_side();
    // Content-equal twins at a plain level merge under a marginal-flagged
    // parent: the survivor's pair list already is the shared function, and the
    // member's parent pairs are remapped onto the survivor, where the resulting
    // duplicates are legal multiset entries that pair fusion sums. Under a
    // plain parent the multiplicity has nowhere to live, so they stay apart.
        let parent_marginal = tdd.levels[parent.idx()].any_inlined_side();
    // When a child side of t1 has marginalization below it, overlapping twins
    // concat-merge unconditionally and `compact_and_fork_down` folds the
    // resulting duplicate pairs where it can; see the module doc of
    // `duplicate_pair_resolve`. False unless `plain_level`.
        let t1_scalable = if plain_level {
        let (t1_l, t1_r) = tdd.vtree.children(t1);
        has_marginal_below.get(t1_l.idx()).copied().unwrap_or(false)
            || has_marginal_below.get(t1_r.idx()).copied().unwrap_or(false)
    } else {
        false
    };
        Self { plain_level, parent_marginal, t1_scalable }
    }
}

/// Pass A: decide every group's action before any of them commits, so the
/// reserve can ask for exactly the pair mass that will be appended (the
/// overlap filter drops members; a duplicate-redirect group appends nothing).
/// Deciding first is equivalent to deciding on commit: a node belongs to at
/// most one twin group, and committing a group only re-points its own members
/// and appends at the arena tail. `sel` holds each acting group's members
/// contiguously, survivor first.
pub(super) fn plan_groups(
    tdd: &Tdd,
    t1: VtreeIdx,
    policy: &MergePolicy,
    group_starts: &[u32],
    flat_groups: &[u32],
    bufs: &mut MergeBuffers,
) {
    let MergeBuffers {
        filtered, duplicate_members, keep_pairs_sorted, member_pairs, seen_pairs, sel, group_plans, ..
    } = bufs;
    let plain_level = policy.plain_level;
    let t1_scalable = policy.t1_scalable;
    let parent_marginal = policy.parent_marginal;
    {
        // The per-group buffers are cleared before each group below.
        let level = &tdd.levels[t1.idx()];
        for g in 0..group_starts.len() {
            let start = group_starts[g] as usize;
            let end = if g + 1 < group_starts.len() { group_starts[g + 1] as usize } else { flat_groups.len() };
            let group = &flat_groups[start..end];
            let keep = group[0];
            if !plain_level || t1_scalable {
                // Concat every member, overlapping or not: at a marginal-flagged
                // level duplicate pairs are legal multiset entries, and on the
                // scalable plain path fork-down resolves them right after
                // compaction (`duplicate_pair_resolve`).
                let sel_start = sel.len();
                sel.extend_from_slice(group);
                group_plans.push(GroupPlan {
                    action: GroupAction::Concat,
                    start: sel_start as u32,
                    end: sel.len() as u32,
                });
                continue;
            }
            filtered.clear();
            duplicate_members.clear();
            seen_pairs.clear();
            keep_pairs_sorted.clear();
            filtered.push(keep);
            for p in level.pairs_of_idx(keep as usize) {
                seen_pairs.insert((p.left.0, p.right.0));
                keep_pairs_sorted.push((p.left.0, p.right.0));
            }
            keep_pairs_sorted.sort_unstable();
            for &idx in &group[1..] {
                let mut overlap = false;
                member_pairs.clear();
                for p in level.pairs_of_idx(idx as usize) {
                    let lr = (p.left.0, p.right.0);
                    overlap |= seen_pairs.contains(&lr);
                    member_pairs.push(lr);
                }
                if !overlap {
                    for &(l, r) in member_pairs.iter() {
                        seen_pairs.insert((l, r));
                    }
                    filtered.push(idx);
                } else if parent_marginal {
                    member_pairs.sort_unstable();
                    if member_pairs == keep_pairs_sorted {
                        duplicate_members.push(idx);
                    }
                }
            }
            // Dups are redirected only when no two members have disjoint
            // supports; a mixed group concatenates first, and the duplicate member
            // is then no longer content-equal to the grown survivor, so it
            // stays a separate node.
            let take_dups = !duplicate_members.is_empty() && filtered.len() < 2;
            let sel_start = sel.len();
            let action = if take_dups {
                // Redirect the content-equal members onto the survivor without
                // touching its pair list; the parent rewrite keeps the members'
                // pairs and pair fusion sums the multiplicity. Appends nothing to
                // the arena, so this group is charged nothing below.
                sel.push(keep);
                sel.extend_from_slice(duplicate_members);
                GroupAction::DupRedirect
            } else if filtered.len() >= 2 {
                // Disjoint-support concat merge. Content-equal members (if any)
                // are skipped this round: the survivor's function grows, so a
                // duplicate redirect against the grown survivor would no longer carry
                // the right value. They are re-examined on the next fixpoint
                // iteration (and stay unmerged if no longer content-equal —
                // sound, non-canonical residue).
                sel.extend_from_slice(filtered); // filtered[0] == keep
                GroupAction::Concat
            } else {
                // Nothing in this group can act this round.
                continue;
            };
            group_plans.push(GroupPlan { action, start: sel_start as u32, end: sel.len() as u32 });
        }
    }
}

/// Reserve the commit pass's whole arena growth up front, so a refused
/// reservation bails before any mutation and the commit pass can use plain,
/// infallible push/extend.
///
/// Sized from the decided actions: only a `Concat` appends, and only the
/// members it selected; `needed_ext` is one `MultiPairRange` per concat, the
/// most `finalize_merged_node` pushes per group.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when the reservation is refused; nothing has
/// been mutated at that point.
pub(super) fn reserve_transactional(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    bufs: &MergeBuffers,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (sel, group_plans) = (&bufs.sel, &bufs.group_plans);
    let mut needed_pairs = 0usize;
    let mut needed_ext = 0usize;
    {
        let level = &tdd.levels[t1.idx()];
        for p in group_plans.iter() {
            if p.action != GroupAction::Concat {
                continue;
            }
            for &idx in &sel[p.start as usize..p.end as usize] {
                needed_pairs += level.pair_count_at(idx as usize);
            }
            needed_ext += 1;
        }
    }
    if needed_ext > 0 {
        // Immutable sizing borrow above ends here; take the mutable arena borrow.
        let level = &mut tdd.levels[t1.idx()];
        lim.reserve_exact(&mut level.pairs, needed_pairs)?;
        lim.reserve_exact(&mut level.multi_pairs, needed_ext)?;
    }
    Ok(())
}
