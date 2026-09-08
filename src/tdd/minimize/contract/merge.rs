use crate::tdd::marg_slots::ChildSide;
use crate::vtree::VtreeIdx;

use crate::tdd::limits::ApplyError;
use crate::tdd::types::*;

use super::scratch::{ContractScratch, MergeBuffers};
use crate::tdd::limits::{try_push, try_resize};

/// What the commit pass does with one twin group, decided by the sizing pass
/// (see `contract_twins` Pass A).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GroupAction {
    /// Concatenate the selected members' pair lists into the survivor. The ONLY
    /// action that grows t1's arena, hence the only one the grand reserve charges.
    Concat,
    /// Redirect content-equal members onto the survivor without touching its
    /// pair list — the parent rewrite keeps their pairs and p-fusion sums the
    /// multiplicity. Appends nothing.
    DupRedirect,
}

/// One decided twin group: its action plus the `start..end` range of the flat
/// selected-member buffer holding the members it acts on, SURVIVOR FIRST.
///
/// `pub(super)` because the buffer of these is pooled in `scratch::MergeBuffers`
/// (a plain POD element type — nothing here owns heap).
pub(super) struct GroupPlan {
    pub(super) action: GroupAction,
    pub(super) start: u32,
    pub(super) end: u32,
}

/// Merge all twin groups at level t1, then compact.
///
/// Twins are nodes with identical parent contexts (same (parent, sibling)
/// reference set). Since they always co-occur, their Boolean functions can
/// be disjoined (ORed) into a single node without changing the TDD's overall
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
    tdd: &mut Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) -> Result<usize, ApplyError> {
    // Pair lists at parent (remap+dedup below) and t1 (twin merge in
    // merge_twin_data) are about to be mutated, so any prior leaf-contract
    // verdict is invalidated. The next contract_leaf_twins pass will re-check
    // both — push to dirty_leaf_contract so the worklist finds them in
    // O(|dirty|).
    tdd.scratch.dirty_leaf_contract.push(parent.idx() as u32);
    tdd.scratch.dirty_leaf_contract.push(t1.idx() as u32);

    // Lazy unpack: we read `find_twin_groups` via the packed-safe iterator
    // path (see `for_each_target_sibling`), but the mutation below uses
    // slice-based push/filter/pop on `pairs`/`pairs_mut`. Unpack parent
    // and t1 only here — productive merge path, rare relative to the
    // find_twin_groups scan.
    //
    // t1 is NEVER a marginal level here: the sole caller `try_contract_child`
    // returns early on a marginal t1 (marginal-side redexes go to p-fusion, not
    // twin contraction), so this path only ever rewrites explicit-side refs —
    // no marg slot/inline handling is needed below.
    let width = tdd.levels[t1.idx()].width();

    // Step 1: Merge twin data — combine each group into its first ("kept") node.
    //
    // `merge_target[i]` maps each node to the kept node it merges into.
    // Canonical nodes (merge_target[i] == i) survive; others are absorbed.
    //
    // The three level-width buffers (`merge_target`, `dup_redirect`,
    // `final_remap`) are grown fallibly and BEFORE any mutation — a grow that
    // trips the budget must surface here, ahead of the grand reserve, not after
    // Pass B has already merged twins (the cross-group poison window the hoisted
    // grand reserve closes).
    // `final_remap` is only filled in Step 2, but it is sized here for that
    // reason.
    try_resize(&mut scratch.merge_target, width, 0u32)?;
    try_resize(&mut scratch.final_remap, width, LocalNodeIdx(0))?;
    for i in 0..width { scratch.merge_target[i] = i as u32; }
    // Twin-group members whose supports OVERLAP (share any pair) must NOT be
    // concat-merged at a plain (marg_flags == 0) level: the merged support
    // would hold duplicate pairs, which are legal count-carrying multiset
    // entries only at marg-flagged levels and violate determinism
    // (Invariant 2) everywhere else. Overlapping context-equal twins arise
    // when a boundary-parent content merge rewrites a grandparent's refs and
    // two grandparent pairs collapse onto the same child — the nodes are
    // sound (each contributes its own count), but the shared pair's
    // multiplicity has no representation at a plain level, so the nodes must
    // stay separate (folding them would require forking the multiplicity
    // down to the nearest marg-flagged descendant). The filter greedily
    // accepts pairwise-disjoint
    // members; cost is one hash-set pass over the group's pairs, only on
    // levels where determinism no longer guarantees disjointness.
    let plain_level = tdd.levels[t1.idx()].marg_flags == 0;
    // Debug-only: whether a repeated `(L, R)` in a concatenated support is a
    // legal multiset entry rather than an Invariant-2 violation. Diagram-scoped,
    // not level-scoped: once ANY level is marginal, every count consumer folds
    // `Σ_pairs c(l)·c(r)` and the content-twin merge (`content_twin.rs`) rewrites
    // refs at PLAIN levels too, so a plain-level node can arrive here already
    // holding the same pair twice — concatenating it with a disjoint twin then
    // carries that duplicate through. Only a purely Boolean diagram still
    // guarantees set-ness, which is where the check stays armed. `cfg!` is a
    // compile-time constant, so the O(levels) scan is dead code in release.
    let dups_legal = cfg!(debug_assertions) && tdd.has_marginal_level();
    // Content-equal twins at a plain level CAN merge when the parent level is
    // marg-flagged: the survivor's pair list is already the shared function
    // (no concat — concat would mint duplicate pairs at the plain level), and
    // the member's parent pairs are KEPT and remapped onto the survivor. The
    // resulting duplicate (survivor, marg) parent pairs are legal
    // count-carrying multiset entries there, and the sibling-pair joint
    // fixpoint's p-fusion folds them into one summed count — multiplicity is
    // SUMMED, never set-dedup'd (which would halve the model count). Under a
    // plain parent the multiplicity has no representation, so those twins
    // stay unmerged.
    let parent_marg = tdd.levels[parent.idx()].marg_flags != 0;
    // Concat-all eligibility: when a child side of t1 has marginalization
    // below it, overlapping twins concat-merge unconditionally. The duplicate
    // pairs that mints in the survivor are legal count-carrying multiset
    // entries there (`dup_resolve`'s module doc), and `compact_and_fork_down`
    // folds them into a scaled count wherever that costs O(1) — i.e. where a
    // CHILD of t1 is itself marginal, a strictly narrower condition than this
    // one. Where it does not, the survivor simply keeps the duplicates. When
    // no side has marginalization below at all, the multiplicity has nowhere
    // to live even in principle, so fall back to the greedy disjoint filter
    // (with the up-fold `dup_redirect` under a marg parent) or skip.
    //
    // FALSE unless `plain_level` by construction, so it doubles as the
    // "concat-all AND fork down afterwards" predicate below.
    let t1_scalable = if plain_level {
        let (t1_l, t1_r) = tdd.vtree.children(t1);
        scratch.has_marg_below.get(t1_l.idx()).copied().unwrap_or(false)
            || scratch.has_marg_below.get(t1_r.idx()).copied().unwrap_or(false)
    } else {
        false
    };
    scratch.dup_redirect.clear();
    try_resize(&mut scratch.dup_redirect, width, false)?;
    // Working buffers, checked out of the scratch (cleared on take) instead of
    // freshly allocated per call — see `scratch::MergeBuffers`. Destructured
    // into owned locals so the body below is unchanged; parked back at both
    // productive exits.
    //
    // `resolve_keeps` is u32-wide throughout, matching `scratch.flat_groups`
    // (the twin groups its entries are built from) and `scratch.merge_target`.
    let MergeBuffers {
        mut resolve_keeps,
        mut filtered,
        mut dup_members,
        mut keep_pairs_sorted,
        mut member_pairs,
        mut seen_pairs,
        mut sel,
        mut group_plans,
    } = scratch.take_merge_buffers();
    // Members actually merged away. 0 ⇒ every group was overlap-filtered:
    // the level is unchanged and the caller must NOT treat this as progress
    // (the groups will be re-found by the next scan; reporting progress here
    // spins the sibling-pair fixed-point loop forever).
    let mut merged_members = 0usize;

    // ── Pass A: decide every group's action BEFORE committing any of them ────
    //
    // The overlap filter used to run interleaved with the merges, which forced
    // the grand reserve below to size by the whole group's pair mass — including
    // the members the filter drops and the dup-redirect groups that concatenate
    // nothing at all. Deciding first lets the reserve ask for exactly the mass
    // that will be appended.
    //
    // Deciding every group before committing any is byte-identical to the
    // interleaved form because the decisions are order-independent: a node
    // belongs to at most ONE twin group (`find_twin_groups`' counting sort gives
    // each node a single representative), and committing a group only re-points
    // its OWN survivor and appends at the arena tail — it never rewrites a slot
    // another group's filter reads.
    //
    // `sel` holds each acting group's members contiguously, SURVIVOR FIRST; the
    // filter's own selection is a subset of the group, so this is bounded by
    // `flat_groups` (≤ width u32s) — noise against the pair mass it is sizing.
    // It and `group_plans` are pooled with the rest of the merge buffers (both
    // checked out empty above).
    {
        // `filtered` / `dup_members` / `keep_pairs_sorted` / `member_pairs` /
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
                // Concat ALL members, overlapping or not. At a marg-flagged
                // level duplicate pairs are legal count-carrying multiset
                // entries; on the scalable plain path (`t1_scalable` is only
                // ever set under `plain_level`) they carry the merged twins'
                // shared multiplicity and are resolved by fork-down scaling
                // right after compaction (dup_resolve) — never set-dedup'd,
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
            dup_members.clear();
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
                    for &(l, r) in &member_pairs {
                        seen_pairs.insert((l, r));
                    }
                    filtered.push(idx);
                } else if parent_marg {
                    member_pairs.sort_unstable();
                    if member_pairs == keep_pairs_sorted {
                        dup_members.push(idx);
                    }
                }
            }
            // Dups are redirected only when no two members have disjoint
            // supports; a mixed group concatenates first, and the dup member
            // is then no longer content-equal to the grown survivor, so it
            // stays a separate node.
            let take_dups = !dup_members.is_empty() && filtered.len() < 2;
            let sel_start = sel.len();
            let action = if take_dups {
                // Redirect the content-equal members onto the survivor without
                // touching its pair list; the parent rewrite keeps the members'
                // pairs and p-fusion sums the multiplicity. Appends nothing to
                // the arena, so this group is charged nothing below.
                sel.push(keep);
                sel.extend_from_slice(&dup_members);
                GroupAction::DupRedirect
            } else if filtered.len() >= 2 {
                // Disjoint-support concat merge. Content-equal members (if any)
                // are skipped this round: the survivor's function grows, so a
                // dup-redirect against the grown survivor would no longer carry
                // the right value. They are re-examined on the next fixpoint
                // iteration (and stay unmerged if no longer content-equal —
                // sound, non-canonical residue).
                sel.extend_from_slice(&filtered); // filtered[0] == keep
                GroupAction::Concat
            } else {
                // Nothing in this group can act this round.
                continue;
            };
            group_plans.push(GroupPlan { action, start: sel_start as u32, end: sel.len() as u32 });
        }
    }

    // ── Hoisted grand reserve — make the merge loop transactional.
    //
    // Per-group `try_reserve`s inside `concat_twin_pairs`, interleaved with
    // survivor growth, left a cross-group poison window: group g failing after
    // groups 0..g-1 already grew their survivors — parent not yet rewritten —
    // silently overcounts. Reserve the WHOLE loop's arena growth ONCE, up front;
    // on failure bail before any mutation (count unchanged, un-poisoned). The
    // merge loop below then commits with plain, infallible push/extend. Total
    // allocation is identical — the reserves just move earlier.
    //
    // Sized from Pass A's DECIDED actions, so it is the exact concatenation
    // total: only a `Concat` appends, and only the members it selected.
    // `needed_ext` is one `ExtMulti` per concat (worst case:
    // `finalize_merged_node` / `encode_multi` push at most one ext entry per
    // merged group); a `DupRedirect` group never reaches either.
    // `needed_ext == 0` ⇒ nothing will be appended (no group concats, possibly
    // none acts at all), so there is nothing to reserve.
    let mut needed_pairs = 0usize;
    let mut needed_ext = 0usize;
    {
        let level = &tdd.levels[t1.idx()];
        for p in &group_plans {
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
        // Injection point (test-only): on the FIXED path this hoisted reserve is
        // where an OverBudget surfaces — before ANY mutation (see the
        // OverBudget-safety tests in `minimize::tests`).
        #[cfg(test)]
        if super::scratch::fail_point() {
            return Err(ApplyError::OverBudget);
        }
        crate::tdd::limits::budget_reserve_exact(&mut level.pairs, needed_pairs)?;
        #[cfg(test)]
        if super::scratch::fail_point() {
            return Err(ApplyError::OverBudget);
        }
        crate::tdd::limits::budget_reserve_exact(&mut level.ext, needed_ext)?;
    }

    // ── Pass B: commit the decided actions. Every allocation they need is
    // already reserved, so nothing below can fail.
    for p in &group_plans {
        let members = &sel[p.start as usize..p.end as usize];
        let keep = members[0];
        match p.action {
            GroupAction::Concat => {
                for &idx in &members[1..] {
                    scratch.merge_target[idx as usize] = keep;
                }
                merged_members += members.len() - 1;
                if t1_scalable {
                    // Concat-all path: the survivor may now hold duplicate
                    // pairs, which fork-down resolves after compaction.
                    resolve_keeps.push(keep);
                }
                // `dups_legal` widens `allow_dups` beyond the fork-down path:
                // once ANY level is marginal a plain-level node can arrive here
                // already holding a pair twice, so the debug set-ness scan in
                // `concat_twin_pairs` must stand down for the whole diagram.
                merge_twin_data(tdd, t1, members, /*allow_dups=*/ t1_scalable || dups_legal);
            }
            GroupAction::DupRedirect => {
                for &idx in &members[1..] {
                    scratch.merge_target[idx as usize] = keep;
                    scratch.dup_redirect[idx as usize] = true;
                }
                merged_members += members.len() - 1;
            }
        }
    }
    if merged_members == 0 {
        // Nothing merged: level untouched, no compaction or parent rewrite
        // needed. Returning 0 lets try_contract_child report no-progress.
        scratch.put_merge_buffers(MergeBuffers {
            resolve_keeps, filtered, dup_members, keep_pairs_sorted, member_pairs, seen_pairs,
            sel, group_plans,
        });
        return Ok(0);
    }
    // Step 2: Build composed remap: old_index → final_compact_index.
    //
    // Pass a: assign dense indices 0,1,2,... to surviving (canonical) nodes.
    // Pass b: merged-away nodes inherit their kept node's dense index. This
    //         must happen in a second pass so pass a has settled every canonical
    //         node's new index before any non-canonical node reads it.
    //
    // Example with 5 nodes, twins {0,2} merged into 0, {3,4} into 3:
    //   merge_target = [0, 1, 0, 3, 3]
    //   After (a): final_remap[0]=0, final_remap[1]=1, final_remap[3]=2
    //   After (b): final_remap[2]=0, final_remap[4]=2
    debug_assert!(scratch.final_remap.len() >= width, "final_remap sized before Pass B");
    let mut next = 0u32;
    for i in 0..width {
        if scratch.merge_target[i] == i as u32 {
            scratch.final_remap[i] = LocalNodeIdx(next);
            next += 1;
        }
    }
    for i in 0..width {
        let target = scratch.merge_target[i] as usize;
        if target != i {
            scratch.final_remap[i] = scratch.final_remap[target];
        }
    }

    // Filter out pairs referencing non-canonical twins and apply gap-closing remap.
    //
    // Twins have identical parent contexts, so every pair referencing a
    // non-canonical twin has a duplicate referencing the canonical twin.
    // Deleting non-canonical pairs is O(n) and avoids the O(n log n) sort
    // that sort_and_dedup_level_pairs would require. This is order-independent:
    // the filter keys on *which* twin a pair references, not on position, so it
    // removes exactly the post-remap duplicates whatever order the pairs are in
    // (pair lists are unordered sets; the result need not be
    // sorted because find_twin_groups canonicalizes signature slices itself).
    //
    // t1 is never marginal here (see the guard note at the top of this fn), so
    // the parent's refs on the t1 side are plain node indices: `merge_target` /
    // `final_remap` index them directly — no marg-slot mask, no `slot_raw` retag,
    // and no marg-side inline refs to pass through verbatim.
    // Mid-parent-rewrite poison backstop: the rewrite below mutates in place —
    // if its single remaining fallible allocation OverBudgets mid-loop the
    // diagram is structurally broken with no clean rollback. Capture the error
    // in a local and break; the `tdd.scratch.poisoned` write happens after the
    // `parent_level` borrow ends (below the loop).
    let mut poison_w2: Option<ApplyError> = None;
    // Arena garbage from the whole rewrite, accumulated and noted ONCE below:
    // the only reader (`compact_pairs_if_stale`) runs after the loop, so the
    // per-iteration saturating add bought nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    for node_idx in 0..parent_level.nodes.len() {
        if parent_level.nodes[node_idx].is_inline() {
            // Inline node: 1 pair, field is a (left) or b (right).
            // By the twin invariant, this pair always references a canonical node,
            // so the pair is never filtered out — only remapped.
            let node = &mut parent_level.nodes[node_idx];
            let tv_old = if t1_side == ChildSide::Left { node.a } else { node.b };
            let tv_new = scratch.final_remap[tv_old as usize].0;
            if t1_side == ChildSide::Left {
                node.a = tv_new;
            } else {
                node.b = tv_new;
            }
            // new_len == 1, no has_multi_pair update
        } else if parent_level.nodes[node_idx].is_multi() {
            // Pair-fusion dirty tracking: a fusion redex is two pairs at one node
            // sharing an explicit-side ref but with distinct marg-side refs. The
            // rewrite below can mint one when a twin absorb / dup-redirect remaps
            // two of this node's refs onto the same survivor (or when a marginal
            // twin merge keeps duplicate `(X, s)` pairs whose counts p-fusion
            // sums).
            let new_len = {
                let pairs = parent_level.pairs_mut(node_idx);
                let mut write = 0;
                for read in 0..pairs.len() {
                    let field_raw =
                        if t1_side == ChildSide::Left { pairs[read].left.0 }
                        else { pairs[read].right.0 };
                    let field_val = LocalNodeIdx(field_raw);
                    // Keep pairs referencing canonical (surviving) twins, and
                    // pairs referencing dup-redirected content-equal twins —
                    // the latter remap onto the survivor, minting a duplicate
                    // (survivor, marg) pair whose count p-fusion sums.
                    if scratch.merge_target[field_val.idx()] == field_val.0
                        || scratch.dup_redirect[field_val.idx()]
                    {
                        let tv_new = scratch.final_remap[field_val.idx()];
                        let mut pair = pairs[read];
                        let f = if t1_side == ChildSide::Left { &mut pair.left } else { &mut pair.right };
                        *f = tv_new;
                        pairs[write] = pair;
                        write += 1;
                    }
                    // else: dropped (non-canonical) pair — omitted from the
                    // compacted list.
                }
                write
            };
            let old_len = parent_level.multi_len_at(node_idx);
            if new_len < old_len {
                // The slots past `new_len` are now unreferenced arena; the inline
                // re-encode abandons the survivor's slot as well. (Both `break`
                // paths below skip the accumulation — that diagram is dropped.)
                let mut abandoned = old_len - new_len;
                if new_len == 1 {
                    let surviving_start = parent_level.multi_start_at(node_idx);
                    let surviving = parent_level.pairs[surviving_start];
                    if surviving.can_inline() {
                        parent_level.nodes[node_idx] = TddNodeData::inline(surviving);
                        abandoned += 1;
                    } else {
                        // Can't inline (would alias leaf/multi_extended encoding):
                        // keep it as an extended multi of len 1. ALIAS the pair
                        // already sitting at this node's own `multi_start` slot —
                        // do NOT push a fresh copy. The in-place compaction above
                        // left the single survivor at `surviving_start`, and the
                        // parent pair arena is only `shrink_to_fit`'d later (never
                        // compacted mid-rewrite), so the slot stays valid. Aliasing
                        // removes the fallible *pairs* push that used to open the
                        // mid-rewrite poison window (the grand reserve is on the T1 level,
                        // not this parent level, so it cannot cover a parent-arena
                        // push).
                        let ext_idx = parent_level.ext.len();
                        // The one irreducible fallible allocation of the whole
                        // parent rewrite. If it OverBudgets we are MID-REWRITE
                        // (some parent pairs already remapped, this node not yet
                        // re-encoded) → the diagram is structurally broken with no
                        // clean rollback. Flag the TDD poisoned (after the borrow
                        // ends) so every count consumer refuses it, then bail.
                        //
                        // Injection point (test-only): this backstop is exercised
                        // by arming `fail_point` to fire here.
                        #[cfg(test)]
                        if super::scratch::fail_point() {
                            poison_w2 = Some(ApplyError::OverBudget);
                            break;
                        }
                        match try_push(
                            &mut parent_level.ext,
                            ExtMulti { start: surviving_start as u64, len: 1 },
                        ) {
                            Ok(()) => {
                                parent_level.nodes[node_idx] =
                                    TddNodeData::multi_extended(ext_idx as u32);
                            }
                            Err(e) => {
                                poison_w2 = Some(e);
                                break;
                            }
                        }
                    }
                } else {
                    parent_level.set_pair_len(node_idx, new_len as u32);
                }
                dead_acc += abandoned;
            }
        }
    }
    parent_level.note_dead_pairs(dead_acc);

    // Poison backstop: the `parent_level` borrow has ended, so we can flag the TDD.
    // A mid-rewrite OverBudget left the parent structurally inconsistent; mark
    // it poisoned (query::model_count asserts `!poisoned`) and propagate the
    // error so the caller drops the diagram and recovers via Shannon split.
    if let Some(e) = poison_w2 {
        tdd.scratch.poisoned = true;
        return Err(e);
    }

    compact_and_fork_down(
        tdd,
        t1,
        &resolve_keeps,
        scratch,
    )?;

    // Step 4: reclaim the parent's shrunk pair lists. (t1's garbage — the far
    // larger mass — is swept inside `compact_and_fork_down`, as early as it is
    // legal to, so fork-down grows the arenas on an already-compacted t1.)
    //
    // This is the safe point and no earlier one is: the parent rewrite has
    // finished, so no pair-arena offset is held across the call — the caller
    // obligation documented on `compact_pairs_if_stale` (types/level.rs).
    tdd.levels[parent.idx()].compact_pairs_if_stale();
    scratch.put_merge_buffers(MergeBuffers {
        resolve_keeps, filtered, dup_members, keep_pairs_sorted, member_pairs, seen_pairs,
        sel, group_plans,
    });
    Ok(merged_members)
}

/// Compact the t1 level after twin contraction and run fork-down duplicate
/// resolution for any survivors that were merged via the concat-all path
/// (Step 3b of `contract_twins`).
///
/// Compaction removes absorbed twins in-place; the freed arena is then swept
/// (see the sweep note inline) before fork-down resolution replaces each run of
/// k equal pairs in a survivor with one pair whose marginal side is scaled by k
/// — multiplicity is preserved in counts, never set-dedup'd. It only fires where
/// that scale is O(1) (t1 has a marginal child); elsewhere the run stays as k
/// legal multiset terms, which sum to the same count (see `dup_resolve`'s cost
/// policy). Fork-down runs after compaction so survivor indices are final.
#[inline(always)]
fn compact_and_fork_down(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    resolve_keeps: &[u32],
    scratch: &mut ContractScratch,
) -> Result<(), ApplyError> {
    // Step 3: Compact the level in-place (keep only alive nodes). t1 is never
    // marginal here (see the guard note in `contract_twins`), so only the
    // explicit-level compaction is reachable.
    compact_explicit_level(&mut tdd.levels[t1.idx()], &scratch.merge_target);

    // Reclaim t1's merge garbage HERE, at the moment its dead fraction is
    // maximal and known: every merged union has been appended at the arena tail
    // and every absorbed member's node has just been dropped, so up to half the
    // arena is unreferenced — and fork-down below is what GROWS the arenas
    // again (scaled clones at t1's children, a re-encoded survivor here).
    // Sweeping before that growth is what keeps the two from being resident
    // together at the peak; the sweep only slides live ranges down, preserving
    // every node's pair slice and its order byte-for-byte.
    //
    // Legal here: the caller obligation on `compact_pairs_if_stale`
    // (types/level.rs) is to hold no pair-arena offset across the call, and
    // nothing live at this point is one — `merge_target`/`final_remap`/
    // `resolve_keeps`/`tdd.output.local` are all NODE indices, and fork-down
    // resolves each node to its slice through `pairs_of_idx` at use time.
    // What fork-down leaves behind (shrunk survivor tails) is charged to
    // `dead_pairs` and waits for the next contraction's sweep, exactly as the
    // counter is designed for.
    tdd.levels[t1.idx()].compact_pairs_if_stale();

    // Update output if it points to t1
    if tdd.output.vtree == t1 {
        tdd.output.local = scratch.final_remap[tdd.output.local.idx()];
    }

    // Fork-down resolution: survivors merged on the concat-all path may hold
    // duplicate pairs (overlapping twin supports). Fold each run of k equal
    // pairs into one pair whose marginal side carries the factor k, where that
    // is an O(1) count scale; otherwise leave the run — multiplicity is
    // preserved either way, never set-dedup'd. Runs after compaction so
    // survivor indices are final.
    // One scratch for the whole loop (cleared per node inside the callee): the
    // resolver runs once per survivor, so its three working buffers were three
    // fresh allocations per NODE — the finest granularity on this path.
    for &old_keep in resolve_keeps {
        let new_idx = scratch.final_remap[old_keep as usize].idx();
        super::dup_resolve::resolve_duplicate_pairs_in_node(tdd, t1, new_idx, &mut scratch.dup)?;
    }
    Ok(())
}

/// Merge a group of twin nodes' data into the first node (the "kept" node).
///
/// Only called on internal nodes (leaf levels are marginal and never contracted).
/// Unions input pair sets of twin nodes. Paths by group/pair count:
///   - 2 twins with 1 pair each: inline, no allocation
///   - otherwise: arena-internal concatenation (`concat_twin_pairs`) —
///     concatenation IS the union since pair lists are unordered sets
fn merge_twin_data(
    tdd: &mut Tdd,
    t1: VtreeIdx,
    group: &[u32],
    allow_dups: bool,
) {
    let keep = group[0] as usize;
    let level = &mut tdd.levels[t1.idx()];

    // Leaf levels are marginal — contract_all_twins never calls this for leaves.
    debug_assert!(
        level.nodes[keep].is_internal(),
        "merge_twin_data called on leaf node — leaf levels should be skipped"
    );

    // Internal twins: union input pair sets.
    if group.len() == 2 {
        merge_two_internal_twins(level, keep, group[1] as usize, allow_dups);
    } else {
        merge_many_internal_twins(level, keep, group, allow_dups);
    }
}

/// Merge two internal twin nodes — the most common case.
///
/// Concatenates both nodes' pair lists at the arena tail via
/// `extend_from_within` (no temp buffers), then updates the kept node's
/// pair_start/pair_len. Pair lists are unordered sets and
/// `find_twin_groups` canonicalizes each signature slice before comparing, so
/// no consumer needs the union sorted — plain concatenation IS the union.
/// Duplicate `(L, R)` entries across (and within) the inputs are legitimate
/// multiset entries at marginal-child levels — count-keyed slot sharing
/// (`apply_p_fusion`) lets each occurrence carry one historical plan's
/// `c(L)·c(R)` contribution — and concatenation
/// preserves them by construction. At fully non-marginal levels determinism
/// (Invariant 2) guarantees the supports are disjoint (checked debug-only in
/// `concat_twin_pairs`).
///
/// Do NOT reintroduce an ordered merge through temp buffers: on pathological
/// nodes the two transient copies land at exactly the moment memory is
/// tightest.
fn merge_two_internal_twins(
    level: &mut TddLevel,
    keep: usize,
    other: usize,
    allow_dups: bool,
) {
    // 1+1 fast path: merge two single-pair nodes without allocation.
    // After the inline encoding, single-pair nodes are inline (pair in the node itself).
    let keep_len = level.pair_count_at(keep);
    let other_len = level.pair_count_at(other);
    if keep_len == 1 && other_len == 1 {
        let pa = level.pairs_of_idx(keep)[0];
        let pb = level.pairs_of_idx(other)[0];
        // pa == pb is permitted and both copies must survive — see this
        // function's doc for why duplicates are legitimate multiset entries.
        // The grand reserve charged these two pairs (keep_len + other_len), so
        // the pushes cannot reallocate — plain push.
        let new_start = level.pairs.len();
        debug_assert!(
            level.pairs.capacity() - level.pairs.len() >= 2,
            "merge_two_internal_twins: hoisted grand reserve under-sized pairs capacity"
        );
        // A 1-pair node is normally inline (owning no arena slot), but the
        // extended encoding also carries len-1 nodes; if `keep` was one, the
        // re-encode below abandons its slot. `other`'s is accounted when
        // compaction drops its node.
        let abandoned = level.arena_pairs_at(keep);
        level.pairs.push(pa);
        level.pairs.push(pb);
        let data = level.encode_multi(new_start, 2);
        level.nodes[keep] = data;
        level.note_dead_pairs(abandoned);
        return;
    }

    concat_twin_pairs(
        level,
        keep,
        &[keep as u32, other as u32],
        keep_len + other_len,
        allow_dups,
    );
}

/// Merge 3+ internal twin nodes. Rare in practice — most twin groups have
/// exactly 2 members. Same concatenation-is-union argument as
/// `merge_two_internal_twins` (the previous sort here existed only to support
/// a windows-based duplicate assert, now done debug-only in
/// `concat_twin_pairs`).
fn merge_many_internal_twins(
    level: &mut TddLevel,
    keep: usize,
    group: &[u32],
    allow_dups: bool,
) {
    let total: usize = group.iter().map(|&idx| level.pair_count_at(idx as usize)).sum();
    concat_twin_pairs(level, keep, group, total, allow_dups);
}

/// Concatenate the pair lists of `group`'s nodes at the arena tail and point
/// `keep` at the result. `total` must be the exact summed pair count.
///
/// Sources are ranges of the arena itself (or inline node data), so
/// `extend_from_within` copies arena→arena with no temp buffer. Infallible: the
/// arena growth of the WHOLE merge loop (a group can total ~1B pairs ≈ 8 GiB of
/// `InputPair`, asserted 8 bytes in types.rs) is charged ONCE up front by the
/// hoisted grand reserve in `contract_twins`, which bails before any
/// mutation on OverBudget. By the time we get here the capacity is guaranteed,
/// so the extends/pushes below cannot reallocate — hence plain `push`/`extend`.
///
/// Every source range is left behind as dead arena (the union is a tail copy).
/// Those slots are counted into `TddLevel::dead_pairs` — here for the survivor,
/// in `compact_explicit_level` for the absorbed members — and reclaimed by the
/// sweep at the end of `contract_twins`.
fn concat_twin_pairs(
    level: &mut TddLevel,
    keep: usize,
    group: &[u32],
    total: usize,
    allow_dups: bool,
) {
    let new_start = level.pairs.len();
    debug_assert!(
        level.pairs.capacity() - level.pairs.len() >= total,
        "concat_twin_pairs: hoisted grand reserve under-sized pairs capacity"
    );
    for &idx in group {
        let d = level.nodes[idx as usize];
        if d.is_inline() {
            level.pairs.push(d.inline_pair());
        } else {
            // `pair_range_at` handles both normal and extended multi encodings;
            // the range is owned, so the immutable borrow ends before the extend.
            let r = level.pair_range_at(idx as usize);
            level.pairs.extend_from_within(r);
        }
    }
    debug_assert_eq!(level.pairs.len() - new_start, total);
    // At fully non-marginal levels, Invariant 2 (determinism) guarantees twin
    // supports are pairwise disjoint, so the concatenation has no duplicates.
    // Debug-only full check — stronger than an adjacency-only test, since
    // concatenation can place equal pairs anywhere. Skipped when the level carries marg
    // markers — there duplicate `(L, R)` entries are legitimate multiset
    // entries (see `merge_two_internal_twins`).
    // `allow_dups`: the caller is on the concat-then-fork-down path (plain
    // scalable level) and resolves the duplicates immediately after
    // compaction (dup_resolve) — transient duplicates are expected there.
    #[cfg(debug_assertions)]
    if level.marg_flags == 0 && !allow_dups {
        let mut chk: Vec<InputPair> = level.pairs[new_start..].to_vec();
        chk.sort_unstable();
        debug_assert!(
            chk.windows(2).all(|w| w[0] != w[1]),
            "twin contraction (concat merge): duplicate pair across twin supports — Invariant 2 violation"
        );
    }
    #[cfg(not(debug_assertions))]
    let _ = allow_dups;
    // The survivor is about to point at the tail copy, abandoning its own source
    // range; the absorbed members' ranges are accounted when compaction drops
    // their nodes (`compact_explicit_level`).
    let abandoned = level.arena_pairs_at(keep);
    finalize_merged_node(level, keep, new_start, total);
    level.note_dead_pairs(abandoned);
}

/// Finalize node at `level.nodes[keep]` from a merged pair sequence already
/// written to `level.pairs[new_start..new_start + new_len]`. Used by the two-twin
/// and many-twin merge paths after they've appended the deduplicated pairs.
///
/// Encoding picks:
/// - `new_len == 1` + pair fits inline: pop the speculative pair from the arena
///   and store the pair directly in the node (no heap traffic).
/// - `new_len == 1` + pair cannot inline (e.g. high bit set): keep the pair in
///   the arena, record a 1-element `ExtMulti` side-entry, and tag the node as
///   `multi_extended`. (Required because the packed single-pair encoding aliases
///   leaf or multi-extended encodings when the high bits are set.)
/// - `new_len >= 2`: delegate to `encode_multi`, which picks packed vs extended
///   based on whether `new_start`/`new_len` fit in the packed bit-budget.
#[inline]
fn finalize_merged_node(
    level: &mut TddLevel,
    keep: usize,
    new_start: usize,
    new_len: usize,
) {
    if new_len == 1 {
        let pair = level.pairs[new_start];
        if pair.can_inline() {
            level.pairs.pop();
            level.nodes[keep] = TddNodeData::inline(pair);
        } else {
            // The grand reserve charged one `ExtMulti` per group on
            // `level.ext`, so this push cannot reallocate — plain push.
            let ext_idx = level.ext.len();
            debug_assert!(
                level.ext.capacity() > level.ext.len(),
                "finalize_merged_node: hoisted grand reserve under-sized ext capacity"
            );
            level.ext.push(ExtMulti { start: new_start as u64, len: 1 });
            level.nodes[keep] = TddNodeData::multi_extended(ext_idx as u32);
        }
    } else {
        level.nodes[keep] = level.encode_multi(new_start, new_len);
    }
}

/// Compact a non-marginal (explicit) level in-place after twin contraction.
///
/// Walks `level.nodes`, keeping only entries whose `merge_target[read] == read`
/// (i.e. canonical survivors; absorbed twins are skipped). Survivors are shifted
/// left in-place via `swap` and the vec is truncated. Mirrors the prune-side
/// compaction in `prune.rs:154` for the non-marginal case.
///
/// Also the accounting point for absorbed twins' pair ranges: dropping the node
/// is what makes its range unreferenced (whether the merge copied the content to
/// the survivor's tail range, or a dup-redirect left the pair list untouched).
#[inline(always)]
fn compact_explicit_level(level: &mut TddLevel, merge_target: &[u32]) {
    let n = level.nodes.len();
    let mut write = 0usize;
    // Accumulated and noted once after the walk: the counter's only reader
    // (`compact_pairs_if_stale`) runs later, so per-drop adds bought nothing.
    let mut dead_acc = 0usize;
    for read in 0..n {
        if merge_target[read] == read as u32 {
            if write < read {
                level.nodes.swap(write, read);
            }
            write += 1;
        } else {
            // `swap` only ever writes to positions ≤ the current `read`, so
            // `nodes[read]` is still this slot's original node here.
            dead_acc += level.arena_pairs_at(read);
        }
    }
    level.nodes.truncate(write);
    level.note_dead_pairs(dead_acc);
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod pairs_arena_sweep_tests;
