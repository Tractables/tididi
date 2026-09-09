//! Rewriting the parent level's refs onto the surviving twins.

use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::diagram::{ExtMulti, NodeIdx, Tdd, TddLevel, TddNodeData};

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
            scratch.final_remap[i] = NodeIdx(next);
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
/// Infallible: every allocation it could need was reserved before the pass
/// mutated anything ([`super::plan::reserve_transactional`]).
pub(super) fn rewrite_parent(
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) {
    // Arena garbage from the whole rewrite, accumulated and noted ONCE below:
    // the only reader (`compact_pairs_if_stale`) runs after the loop, so the
    // per-iteration saturating add bought nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    for node_idx in 0..parent_level.nodes.len() {
        if parent_level.nodes[node_idx].is_inline() {
            remap_inline_node(parent_level, node_idx, t1_side, scratch);
        } else if parent_level.nodes[node_idx].is_multi() {
            let old_len = parent_level.multi_len_at(node_idx);
            let new_len = keep_canonical_pairs(parent_level, node_idx, t1_side, scratch);
            if new_len < old_len {
                dead_acc += shrink_node(parent_level, node_idx, old_len, new_len);
            }
        }
    }
    parent_level.note_dead_pairs(dead_acc);
}

/// Point an inline node's single pair at the survivor of its T1-side twin
/// class. By the twin invariant that pair always references a canonical node,
/// so it is never dropped — only remapped, and the node stays inline.
fn remap_inline_node(
    level: &mut TddLevel,
    node_idx: usize,
    t1_side: ChildSide,
    scratch: &ContractScratch,
) {
    let node = &mut level.nodes[node_idx];
    let tv_old = if t1_side == ChildSide::Left { node.a } else { node.b };
    let tv_new = scratch.final_remap[tv_old as usize].0;
    if t1_side == ChildSide::Left {
        node.a = tv_new;
    } else {
        node.b = tv_new;
    }
}

/// Compact a multi-pair node in place, keeping the pairs whose T1-side ref
/// survives the merge and remapping each onto its survivor. Returns how many
/// pairs are left; the node's recorded length is not touched.
///
/// A dup-redirected content-equal twin is kept, not dropped: remapping it onto
/// the survivor mints a duplicate `(survivor, marg)` pair whose count pair
/// fusion sums. That is also how a fusion redex is minted here — two pairs at
/// one node sharing an explicit-side ref with distinct marg-side refs.
fn keep_canonical_pairs(
    level: &mut TddLevel,
    node_idx: usize,
    t1_side: ChildSide,
    scratch: &ContractScratch,
) -> usize {
    let pairs = level.pairs_mut(node_idx);
    let mut write = 0;
    for read in 0..pairs.len() {
        let field_raw = if t1_side == ChildSide::Left { pairs[read].left.0 } else { pairs[read].right.0 };
        let field_val = NodeIdx(field_raw);
        if scratch.merge_target[field_val.idx()] == field_val.0
            || scratch.dup_redirect[field_val.idx()]
        {
            let mut pair = pairs[read];
            let f = if t1_side == ChildSide::Left { &mut pair.left } else { &mut pair.right };
            *f = scratch.final_remap[field_val.idx()];
            pairs[write] = pair;
            write += 1;
        }
        // else: dropped (non-canonical) pair — omitted from the compacted list.
    }
    write
}

/// Re-encode a node whose pair list just shrank to `new_len`, and return how
/// many arena slots that abandoned.
///
/// A node down to one pair goes back to the inline encoding where the pair
/// allows it, which abandons its slot too. Where it does not (the pair would
/// alias the leaf / extended-multi encoding), the node becomes an extended
/// multi of length 1 that ALIASES the slot the survivor already sits in — the
/// compaction above left it at the node's own `multi_start`, and the parent
/// pair arena is never compacted mid-rewrite. Aliasing is what removes the
/// fallible pairs push here; the grand reserve is on the T1 level, not this
/// parent level, so it could not have covered one.
///
/// The `ext` entry it may need was reserved before the pass mutated anything,
/// so the push here cannot fail.
fn shrink_node(
    level: &mut TddLevel,
    node_idx: usize,
    old_len: usize,
    new_len: usize,
) -> usize {
    let abandoned = old_len - new_len;
    if new_len != 1 {
        level.set_pair_len(node_idx, new_len as u32);
        return abandoned;
    }
    let surviving_start = level.multi_start_at(node_idx);
    let surviving = level.pairs[surviving_start];
    if surviving.can_inline() {
        level.nodes[node_idx] = TddNodeData::inline(surviving);
        return abandoned + 1;
    }
    let ext_idx = level.ext.len();
    level.ext.push(ExtMulti { start: surviving_start as u64, len: 1 });
    level.nodes[node_idx] = TddNodeData::multi_extended(ext_idx as u32);
    abandoned
}
