//! Rewriting the parent level's refs onto the surviving twins.

use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::diagram::{NodeIdx, NodeKind, Tdd, TddLevel};

use super::super::scratch::MergeRemap;

/// Build the composed remap `old_index → final_compact_index`.
///
/// Pass a assigns dense indices 0,1,2,... to surviving (canonical) nodes; pass b
/// lets merged-away nodes inherit their kept node's dense index. The second pass
/// is separate so pass a has settled every canonical node's new index before any
/// non-canonical node reads it.
///
/// Example with 5 nodes, twins {0,2} merged into 0, {3,4} into 3:
///   merge_target = [0, 1, 0, 3, 3]
///   after (a): `final_remap[0]=0, final_remap[1]=1, final_remap[3]=2`
///   after (b): `final_remap[2]=0, final_remap[4]=2`
pub(super) fn build_final_remap(remap: &mut MergeRemap, width: usize) {
    debug_assert!(remap.final_remap.len() >= width, "final_remap sized before Pass B");
    let mut next = 0u32;
    for i in 0..width {
        if remap.merge_target[i] == i as u32 {
            remap.final_remap[i] = NodeIdx(next);
            next += 1;
        }
    }
    for i in 0..width {
        let target = remap.merge_target[i] as usize;
        if target != i {
            remap.final_remap[i] = remap.final_remap[target];
        }
    }
}

/// Filter out parent pairs referencing non-canonical twins and apply the
/// gap-closing remap.
///
/// Twins have identical parent contexts, so every pair referencing a
/// non-canonical twin has a duplicate referencing the canonical one; dropping
/// the non-canonical pairs is an O(n) filter keyed on which twin a pair
/// references, whatever order the pairs are in. t1 is never marginal here (see
/// `contract_twins`), so the t1-side refs are plain node indices that
/// `merge_target` / `final_remap` index directly.
///
/// Infallible: every allocation it could need was reserved before the pass
/// mutated anything (`reserve_transactional`).
pub(super) fn rewrite_parent(
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    remap: &MergeRemap,
) {
    // Arena garbage from the whole rewrite, accumulated and noted in one charge
    // below: the only reader (`compact_pairs_if_stale`) runs after the loop, so a
    // per-iteration saturating add buys nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    for node_idx in 0..parent_level.nodes.len() {
        if matches!(parent_level.nodes[node_idx].kind(), NodeKind::Inline(_)) {
            remap_inline_node(parent_level, node_idx, t1_side, &remap.final_remap);
        } else if parent_level.nodes[node_idx].kind().pairs_in_arena() {
            let old_len = parent_level.multi_len_at(node_idx);
            let new_len = keep_canonical_pairs(parent_level, node_idx, t1_side, remap);
            if new_len < old_len {
                let start = parent_level.multi_start_at(node_idx);
                dead_acc += parent_level.reencode_shrunk(node_idx, start, old_len, new_len);
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
    final_remap: &[NodeIdx],
) {
    let node = &mut level.nodes[node_idx];
    let tv_old = if t1_side == ChildSide::Left { node.a } else { node.b };
    let tv_new = final_remap[tv_old as usize].0;
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
/// A duplicate redirected content-equal twin is kept, not dropped: remapping it onto
/// the survivor mints a duplicate `(survivor, marginal)` pair whose count pair
/// fusion sums. That is also how a fusion redex is minted here — two pairs at
/// one node sharing an explicit-side ref with distinct marginal-side refs.
fn keep_canonical_pairs(
    level: &mut TddLevel,
    node_idx: usize,
    t1_side: ChildSide,
    remap: &MergeRemap,
) -> usize {
    let pairs = level.pairs_mut(node_idx);
    let mut write = 0;
    for read in 0..pairs.len() {
        let field_raw = if t1_side == ChildSide::Left { pairs[read].left.0 } else { pairs[read].right.0 };
        let field_val = NodeIdx(field_raw);
        if remap.merge_target[field_val.idx()] == field_val.0
            || remap.duplicate_redirect[field_val.idx()]
        {
            let mut pair = pairs[read];
            let f = if t1_side == ChildSide::Left { &mut pair.left } else { &mut pair.right };
            *f = remap.final_remap[field_val.idx()].into();
            pairs[write] = pair;
            write += 1;
        }
        // else: dropped (non-canonical) pair — omitted from the compacted list.
    }
    write
}
