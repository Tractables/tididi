//! Deciding what each twin group does, and reserving its arena growth up front.

use crate::engine::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::ApplyError;
use crate::diagram::Tdd;

use super::super::scratch::{ContractScratch, MergeBuffers};

/// What the commit pass does with one twin group, decided by the sizing pass
/// (see `contract_twins` Pass A).
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
///
/// `pub(super)` because the buffer of these is pooled in `scratch::MergeBuffers`
/// (a plain element type — nothing here owns heap).
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
        scratch: &ContractScratch,
    ) -> Self {
    // Twin-group members whose supports overlap (share any pair) must not be
    // concat-merged at a plain (no inlined side) level: the merged support
    // would hold duplicate pairs, which are legal count-carrying multiset
    // entries only at marginal-flagged levels and violate determinism
    // (invariant 1) everywhere else. Overlapping context-equal twins arise
    // when a boundary-parent content merge rewrites a grandparent's refs and
    // two grandparent pairs collapse onto the same child — the nodes are
    // sound (each contributes its own count), but the shared pair's
    // multiplicity has no representation at a plain level, so the nodes must
    // stay separate (folding them would require forking the multiplicity
    // down to the nearest marginal-flagged descendant). The filter greedily
    // accepts pairwise-disjoint
    // members; cost is one hash-set pass over the group's pairs, only on
    // levels where determinism no longer guarantees disjointness.
        let plain_level = !tdd.levels[t1.idx()].any_inlined_side();
    // Content-equal twins at a plain level are free to merge when the parent level is
    // marginal-flagged: the survivor's pair list is already the shared function
    // (no concat — concat would mint duplicate pairs at the plain level), and
    // the member's parent pairs are kept and remapped onto the survivor. The
    // resulting duplicate (survivor, marginal) parent pairs are legal
    // count-carrying multiset entries there, and the sibling-pair joint
    // fixpoint's pair fusion folds them into one summed count — multiplicity is
    // summed, never set-dedup'd (which would halve the model count). Under a
    // plain parent the multiplicity has no representation, so those twins
    // stay unmerged.
        let parent_marginal = tdd.levels[parent.idx()].any_inlined_side();
    // Concat-all eligibility: when a child side of t1 has marginalization
    // below it, overlapping twins concat-merge unconditionally. The duplicate
    // pairs that mints in the survivor are legal count-carrying multiset
    // entries there (`duplicate_pair_resolve`'s module doc), and `compact_and_fork_down`
    // folds them into a scaled count wherever that costs O(1) — i.e. where a
    // child of t1 is itself marginal, a strictly narrower condition than this
    // one. Where it does not, the survivor simply keeps the duplicates. When
    // no side has marginalization below at all, the multiplicity has nowhere
    // to live even in principle, so fall back to the greedy disjoint filter
    // (with the up-fold `duplicate_redirect` under a marginal parent) or skip.
    //
    // False unless `plain_level` by construction, so it doubles as the
    // "concat-all, then fork down afterwards" predicate below.
        let t1_scalable = if plain_level {
        let (t1_l, t1_r) = tdd.vtree.children(t1);
        scratch.has_marginal_below.get(t1_l.idx()).copied().unwrap_or(false)
            || scratch.has_marginal_below.get(t1_r.idx()).copied().unwrap_or(false)
    } else {
        false
    };
        Self { plain_level, parent_marginal, t1_scalable }
    }
}

/// Pass A: decide every group's action before committing any of them, so the
/// grand reserve can ask for exactly the pair mass that will be appended
/// rather than the whole group's — the overlap filter drops members, and a
/// duplicate-redirect group concatenates nothing at all.
///
/// Deciding first is equivalent to deciding as each group commits, because the
/// decisions are order-independent: a node
/// belongs to at most one twin group (`find_twin_groups`' counting sort gives
/// each node a single representative), and committing a group only re-points
/// its own survivor and appends at the arena tail — it never rewrites a slot
/// another group's filter reads.
///
/// `sel` holds each acting group's members contiguously, survivor first; the
/// filter's own selection is a subset of the group, so this is bounded by
/// `flat_groups` (≤ width u32s) — noise against the pair mass it is sizing.
pub(super) fn plan_groups(
    tdd: &Tdd,
    t1: VtreeIdx,
    policy: &MergePolicy,
    scratch: &ContractScratch,
    bufs: &mut MergeBuffers,
) {
    let MergeBuffers {
        filtered, duplicate_members, keep_pairs_sorted, member_pairs, seen_pairs, sel, group_plans, ..
    } = bufs;
    let plain_level = policy.plain_level;
    let t1_scalable = policy.t1_scalable;
    let parent_marginal = policy.parent_marginal;
    {
        // `filtered` / `duplicate_members` / `keep_pairs_sorted` / `member_pairs` /
        // `seen_pairs` are the pooled `MergeBuffers` checked out above; every
        // one is `clear()`ed per group below, so the retained capacity carries
        // across calls without carrying state.
        let level = &tdd.levels[t1.idx()];
        for g in 0..scratch.group_starts.len() {
            let start = scratch.group_starts[g] as usize;
            let end = if g + 1 < scratch.group_starts.len() { scratch.group_starts[g + 1] as usize } else { scratch.flat_groups.len() };
            let group = &scratch.flat_groups[start..end];
            let keep = group[0];
            if !plain_level || t1_scalable {
                // Concat all members, overlapping or not. At a marginal-flagged
                // level duplicate pairs are legal count-carrying multiset
                // entries; on the scalable plain path (`t1_scalable` is only
                // ever set under `plain_level`) they carry the merged twins'
                // shared multiplicity and are resolved by fork-down scaling
                // right after compaction (duplicate_pair_resolve) — never set-dedup'd,
                // which would undercount.
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

/// Reserve the commit pass's whole arena growth up front, in one go.
///
/// Reserving per group inside `concat_twin_pairs`, interleaved with survivor
/// growth, would leave a window in which group g fails after groups 0..g-1 have
/// already grown their survivors — parent not yet rewritten — silently
/// overcounting. Reserving here means a refusal bails before any mutation, with
/// the count unchanged, and lets the commit pass use plain, infallible
/// push/extend. Total allocation is the same either way.
///
/// Sized from the decided actions, so it is the exact concatenation total: only
/// a `Concat` appends, and only the members it selected. `needed_ext` is one
/// `MultiPairRange` per concat (worst case: `finalize_merged_node` / `encode_multi`
/// push at most one multi_pairs entry per merged group); a `DupRedirect` group never
/// reaches either. `needed_ext == 0` ⇒ nothing will be appended, so there is
/// nothing to reserve on `t1`.
///
/// The parent's `multi_pairs` is reserved here too, and for the same reason. The
/// parent rewrite that follows shrinks pair lists in place, and a node that
/// shrinks to a single pair which cannot inline needs one `MultiPairRange` — the one
/// allocation in an otherwise infallible walk, and the one whose refusal would
/// otherwise leave a half-rewritten diagram behind. An upper bound is cheap:
/// at most one entry per multi-pair node of the parent, since only a multi-pair
/// node can shrink — an inline node just remaps its single pair. It is a plain
/// `reserve`, so the arena grows the way the pushes it replaces grew it.
/// Reserving here makes the rewrite's push infallible, so the whole pass keeps
/// its one property — on `Err`, nothing was touched.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` when the reservation is refused; nothing has
/// been mutated at that point.
pub(super) fn reserve_transactional(
    eng: &Engine,
    tdd: &mut Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    bufs: &MergeBuffers,
) -> Result<(), ApplyError> {
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
    let parent_ext = tdd.levels[parent.idx()]
        .nodes
        .iter()
        .filter(|n| n.is_multi())
        .count();
    if parent_ext > 0 {
        lim.reserve(&mut tdd.levels[parent.idx()].multi_pairs, parent_ext)?;
    }
    Ok(())
}
