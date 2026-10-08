//! Deciding what each twin group does, and reserving its arena growth up front.

use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};
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
    /// No level of the diagram is marginal. Determinism (invariant 1) then
    /// makes twin supports pairwise disjoint, the overlap filter could drop
    /// no member, and the plan concatenates every member without it.
    pub(super) disjoint_supports: bool,
}

impl MergePolicy {
    /// Read the level and parent flags that decide which merges are legal
    /// here. `t1_scalable` starts false; the caller sets it from
    /// [`scalable`](Self::scalable) at a plain level of a diagram with a
    /// marginal level.
    pub(super) fn decide(tdd: &Tdd, t1: VtreeIdx, parent: VtreeIdx, diagram_marginal: bool) -> Self {
        // At a plain (no inlined side) level, twin members whose supports
        // overlap (share a pair) are not concat-merged: the union would hold
        // duplicate pairs, which carry a multiplicity only at marginal-flagged
        // levels and break determinism (invariant 1) elsewhere. The greedy
        // filter in `plan_groups` accepts pairwise-disjoint members, one
        // hash-set pass over the group's pairs.
        let plain_level = !tdd.levels[t1.idx()].any_value_ref_side();
        // Content-equal twins at a plain level merge under a marginal-flagged
        // parent: the survivor's pair list already is the shared function, and
        // the member's parent pairs are remapped onto the survivor, where the
        // resulting duplicates are legal multiset entries that pair fusion
        // sums. Under a plain parent the multiplicity has nowhere to live, so
        // they stay apart.
        let parent_marginal = tdd.levels[parent.idx()].any_value_ref_side();
        Self { plain_level, parent_marginal, t1_scalable: false, disjoint_supports: !diagram_marginal }
    }

    /// Whether a child side of the plain level `t1` has marginalization
    /// below it. Overlapping twins there concat-merge unconditionally and
    /// `compact_and_fork_down` folds the resulting duplicate pairs where it
    /// can; see the module doc of `duplicate_pair`.
    pub(super) fn scalable(tdd: &Tdd, t1: VtreeIdx, has_marginal_below: &[bool]) -> bool {
        let (t1_l, t1_r) = tdd.vtree.children(t1);
        has_marginal_below.get(t1_l.idx()).copied().unwrap_or(false)
            || has_marginal_below.get(t1_r.idx()).copied().unwrap_or(false)
    }
}

/// Pass A: decide every group's action before any of them commits, so the
/// reserve can ask for exactly the pair mass that will be appended (the
/// overlap filter drops members; a duplicate-redirect group appends nothing).
/// Deciding first is equivalent to deciding on commit: a node belongs to at
/// most one twin group, and committing a group only re-points its own members
/// and appends at the arena tail. `sel` holds each acting group's members
/// contiguously, survivor first.
///
/// The planner buffers are reserved up front, sized by the widest group and
/// its pair mass, so a refused reservation returns before any group is
/// planned and the loop itself pushes without checking.
pub(super) fn plan_groups(
    lim: &Limits,
    tdd: &Tdd,
    t1: VtreeIdx,
    policy: &MergePolicy,
    group_starts: &[u32],
    flat_groups: &[u32],
    bufs: &mut MergeBuffers,
) -> Result<(), OperationError> {
    let MergeBuffers {
        filtered, duplicate_members, keep_pairs_sorted, member_pairs, seen_pairs, sel, group_plans, ..
    } = bufs;
    let plain_level = policy.plain_level;
    // Without a marginal level no pair can repeat, and every group takes
    // the concat-all plan below; the filter's set of pairs was most of a
    // Boolean diagram's contraction. `concat_twin_pairs` checks the
    // disjointness in debug builds.
    let t1_scalable = policy.t1_scalable || policy.disjoint_supports;
    let parent_marginal = policy.parent_marginal;
    let level = &tdd.levels[t1.idx()];
    let group_bounds = |g: usize| {
        let start = group_starts[g] as usize;
        let end = if g + 1 < group_starts.len() { group_starts[g + 1] as usize } else { flat_groups.len() };
        (start, end)
    };
    lim.reserve_exact(sel, flat_groups.len())?;
    lim.reserve_exact(group_plans, group_starts.len())?;
    if plain_level && !t1_scalable {
        // The overlap filter below holds one group's members and pairs at a
        // time: `filtered` and `duplicate_members` up to the widest group,
        // `keep_pairs_sorted` and `member_pairs` up to the longest pair list,
        // `seen_pairs` up to one group's pair mass.
        let (mut max_members, mut max_pairs, mut max_mass) = (0usize, 0usize, 0usize);
        for g in 0..group_starts.len() {
            let (start, end) = group_bounds(g);
            max_members = max_members.max(end - start);
            let mut mass = 0usize;
            for &idx in &flat_groups[start..end] {
                let n = level.pair_count_at(idx as usize);
                max_pairs = max_pairs.max(n);
                mass += n;
            }
            max_mass = max_mass.max(mass);
        }
        lim.reserve_exact(filtered, max_members)?;
        lim.reserve_exact(duplicate_members, max_members)?;
        lim.reserve_exact(keep_pairs_sorted, max_pairs)?;
        lim.reserve_exact(member_pairs, max_pairs)?;
        lim.reserve_set(seen_pairs, max_mass)?;
        seen_pairs.clear();
    }
    // The per-group buffers are cleared before each group below.
    for g in 0..group_starts.len() {
        let (start, end) = group_bounds(g);
        let group = &flat_groups[start..end];
        let keep = group[0];
        if !plain_level || t1_scalable {
            // Concat every member, overlapping or not: at a marginal-flagged
            // level duplicate pairs are legal multiset entries, and on the
            // scalable plain path fork-down resolves them right after
            // compaction (`duplicate_pair`).
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
        debug_assert!(seen_pairs.is_empty(), "the previous group emptied the overlap set");
        keep_pairs_sorted.clear();
        filtered.push(keep);
        for p in level.pairs_iter_of_idx(keep as usize) {
            seen_pairs.insert((p.left.0, p.right.0));
            keep_pairs_sorted.push((p.left.0, p.right.0));
        }
        keep_pairs_sorted.sort_unstable();
        for &idx in &group[1..] {
            let mut overlap = false;
            member_pairs.clear();
            for p in level.pairs_iter_of_idx(idx as usize) {
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
        // Empty the overlap set for the next group. It holds the pairs of the
        // members kept in `filtered` and has the capacity of the largest
        // group's pairs, and `clear` costs that capacity: after one large
        // group, every small one would pay for it. A set much larger than this
        // group's pairs has them removed one by one instead.
        let mass: usize = filtered.iter().map(|&i| level.pair_count_at(i as usize)).sum();
        if seen_pairs.capacity() > 4 * mass.max(16) {
            for &idx in filtered.iter() {
                for p in level.pairs_iter_of_idx(idx as usize) {
                    seen_pairs.remove(&(p.left.0, p.right.0));
                }
            }
        } else {
            seen_pairs.clear();
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
    Ok(())
}

/// Reserve the commit pass's whole arena growth up front, so a refused
/// reservation bails before any mutation and the commit pass can use plain,
/// infallible push/extend.
///
/// Sized from the decided actions: only a `Concat` appends, and only the
/// members it selected; `needed_ext` is one `PairRange` per concat, the
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
        lim.reserve_exact(level.pairs.stored_mut(), needed_pairs)?;
        level.ranges.reserve_exact(lim, needed_ext)?;
    }
    Ok(())
}
