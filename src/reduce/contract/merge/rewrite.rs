//! Rewriting the parent level's refs onto the surviving twins.

use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildPair, NodeIdx, StoreRoom, Tdd, TddLevel};
use crate::limits::Limits;

use super::super::scratch::MergeRemap;

/// Set in a merged-away node's `final_remap` entry, above its survivor's
/// new index, where the parent rewrite drops the pairs naming it: every
/// member but a content-equal one redirected onto its survivor. The parent
/// rewrite reads one entry per pair; readers that want the survivor mask it
/// off ([`survivor`]). A stored side never has bit 31 set, so no node index
/// reaches it.
pub(in crate::reduce::contract) const DROPPED: u32 = 1 << 31;

/// The `final_remap` entry of a node the merge drops with its pairs, no
/// pair naming it: it has no survivor.
pub(in crate::reduce::contract) const DISCARDED: u32 = u32::MAX;

/// The new index of the survivor of node `old` after the merge.
#[inline]
pub(in crate::reduce::contract) fn survivor(final_remap: &[NodeIdx], old: usize) -> NodeIdx {
    NodeIdx(final_remap[old].0 & !DROPPED)
}

/// Filter out parent pairs referencing non-canonical twins and apply the
/// gap-closing remap.
///
/// Twins have identical parent contexts, so every pair referencing a
/// non-canonical twin has a duplicate referencing the canonical one; dropping
/// the non-canonical pairs is an O(n) filter keyed on which twin a pair
/// references, whatever order the pairs are in. t1 is never marginal here (see
/// `contract_twins`), so the t1-side refs are plain node indices that
/// `final_remap` indexes directly.
///
/// Infallible: every allocation it could need was reserved before the pass
/// mutated anything (`room`, an implicit parent's, through `lim`).
pub(super) fn rewrite_parent(
    lim: &Limits,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    remap: &MergeRemap,
    room: Option<StoreRoom>,
) {
    // Arena garbage from the whole rewrite, accumulated and noted in one charge
    // below: the only reader (`compact_pairs_if_stale`) runs after the loop, so a
    // per-iteration saturating add buys nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    // An implicit parent's pairs are read off the description through the
    // same filter, and the level is built stored where one changes.
    if parent_level.pairs.implicit().is_some() {
        debug_assert!(room.is_some(), "an implicit parent's room is reserved");
        parent_level
            .rewrite_described_in(lim, room, false, |_, _, _, pair| {
                let (pair, kept) = canonical_pair(pair, t1_side, remap);
                kept.then_some(pair)
            })
            .expect("an implicit parent is rewritten in its room");
        // Built stored at its description's places, in node order: its
        // arena is left as the stored route below leaves it.
        parent_level.compact_in_node_order();
        return;
    }
    // Where the arena holds the ranges in node order, each node's kept pairs
    // move down to a write cursor as they are rewritten, which leaves the
    // arena with no dead pairs and the parent no sweep to run; otherwise
    // each node shrinks in place and its tail is left dead.
    let slide = parent_level.ranges_in_node_order();
    let mut cursor = 0usize;
    for node_idx in 0..parent_level.nodes().len() {
        let kind = parent_level.node(node_idx).kind();
        let Some(range) = parent_level.arena_range(kind) else {
            remap_inline_node(parent_level, node_idx, t1_side, &remap.final_remap);
            continue;
        };
        if slide {
            let new_len = move_canonical_pairs(parent_level, range, cursor, t1_side, remap);
            parent_level.reencode_moved(node_idx, cursor, new_len);
            if new_len >= 2 {
                cursor += new_len;
            }
        } else {
            let new_len = keep_canonical_pairs(parent_level, node_idx, t1_side, remap);
            if new_len < range.len() {
                dead_acc += parent_level.reencode_shrunk(node_idx, range.start, range.len(), new_len);
            }
        }
    }
    if slide {
        parent_level.pairs.stored_mut().truncate(cursor);
        parent_level.dead_pairs = 0;
    } else {
        parent_level.note_dead_pairs(dead_acc);
    }
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
    let node = &mut level.nodes.stored_mut()[node_idx];
    let tv_old = if t1_side == ChildSide::Left { node.a } else { node.b };
    let tv_new = survivor(final_remap, tv_old as usize).0;
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
        let (pair, kept) = canonical_pair(pairs[read], t1_side, remap);
        pairs[write] = pair;
        write += usize::from(kept);
    }
    write
}

/// [`keep_canonical_pairs`] for a node whose pairs `range` are moved down to
/// `to`, at most `range.start`: its kept pairs are written from `to` on.
/// Returns how many pairs are left.
fn move_canonical_pairs(
    level: &mut TddLevel,
    range: std::ops::Range<usize>,
    to: usize,
    t1_side: ChildSide,
    remap: &MergeRemap,
) -> usize {
    debug_assert!(to <= range.start, "the write cursor stays behind the reads");
    let pairs = level.pairs.stored_mut();
    let mut write = to;
    for read in range {
        let (pair, kept) = canonical_pair(pairs[read], t1_side, remap);
        pairs[write] = pair;
        write += usize::from(kept);
    }
    write - to
}

/// A parent pair after the merge, remapped onto its T1-side ref's survivor,
/// and whether it is kept: where that ref survives the merge or was
/// redirected as a content-equal duplicate. A pair not kept duplicates one
/// naming the survivor. The callers write every pair at their write cursor
/// and advance it by the flag, with no branch on it: which pairs are kept
/// follows no pattern a branch predictor learns.
#[inline]
fn canonical_pair(mut pair: ChildPair, t1_side: ChildSide, remap: &MergeRemap) -> (ChildPair, bool) {
    let f = if t1_side == ChildSide::Left { &mut pair.left } else { &mut pair.right };
    let to = remap.final_remap[f.0 as usize].0;
    debug_assert_ne!(to, DISCARDED, "a pair names a node the merge drops as unnamed");
    *f = NodeIdx(to & !DROPPED).into();
    (pair, to & DROPPED == 0)
}
