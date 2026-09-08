//! Rewriting the parent level's refs onto the surviving twins.

use crate::engine::Limits;
use crate::marg_slots::ChildSide;
use crate::vtree::VtreeIdx;

use crate::error::ApplyError;
use crate::diagram::{ExtMulti, LocalNodeIdx, Tdd, TddNodeData};

use super::super::scratch::ContractScratch;

/// Build the composed remap `old_index → final_compact_index`.
///
/// Pass a assigns dense indices 0,1,2,... to surviving (canonical) nodes; pass b
/// lets merged-away nodes inherit their kept node's dense index. The second pass
/// is separate so pass a has settled every canonical node's new index before any
/// non-canonical node reads it.
///
/// Example with 5 nodes, twins {0,2} merged into 0, {3,4} into 3:
///   merge_target = [0, 1, 0, 3, 3]
///   After (a): final_remap[0]=0, final_remap[1]=1, final_remap[3]=2
///   After (b): final_remap[2]=0, final_remap[4]=2
pub(super) fn build_final_remap(scratch: &mut ContractScratch, width: usize) {
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
}

/// Filter out parent pairs referencing non-canonical twins and apply the
/// gap-closing remap.
///
/// Twins have identical parent contexts, so every pair referencing a
/// non-canonical twin has a duplicate referencing the canonical twin.
/// Deleting non-canonical pairs is O(n) and avoids the O(n log n) sort
/// that sort_and_dedup_level_pairs would require. This is order-independent:
/// the filter keys on *which* twin a pair references, not on position, so it
/// removes exactly the post-remap duplicates whatever order the pairs are in
/// (pair lists are unordered sets; the result need not be sorted because
/// find_twin_groups canonicalizes signature slices itself).
///
/// t1 is never marginal here (see the guard note in `contract_twins`), so the
/// parent's refs on the t1 side are plain node indices: `merge_target` /
/// `final_remap` index them directly — no marg-slot mask, no `slot_raw` retag,
/// and no marg-side inline refs to pass through verbatim.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` if the one irreducible `ext` push is refused.
/// The rewrite mutates in place, so that leaves the diagram structurally broken
/// with no clean rollback: the TDD is flagged poisoned before the error returns.
pub(super) fn rewrite_parent(
    lim: &Limits,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) -> Result<(), ApplyError> {
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
                        if super::super::scratch::fail_point() {
                            poison_w2 = Some(ApplyError::OverBudget);
                            break;
                        }
                        match lim.try_push(
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
    Ok(())
}
