//! Rewriting the parent level's refs onto the surviving twins.

use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildPair, NodeIdx, StoreRoom, Tdd, TddLevel};
use crate::limits::Limits;

use super::super::scratch::MergeRemap;

/// Set in a merged-away node's `final_remap` entry, above its survivor's
/// index before the merge, where the parent rewrite drops the pairs naming
/// it: every member but a content-equal one redirected onto its survivor.
/// The parent rewrite reads one entry per pair and needs only the bit;
/// readers that want the survivor's new index read it off the survivor's
/// entry ([`survivor`]). A stored side never has bit 31 set, so no node
/// index reaches it.
pub(in crate::reduce::contract) const DROPPED: u32 = 1 << 31;

/// The `final_remap` entry of a node the merge drops with its pairs, no
/// pair naming it: it has no survivor.
pub(in crate::reduce::contract) const DISCARDED: u32 = u32::MAX;

/// The new index of the survivor of node `old` after the merge: its own
/// entry, or, for a member whose pairs were dropped, its survivor's.
#[inline]
pub(in crate::reduce::contract) fn survivor(final_remap: &[NodeIdx], old: usize) -> NodeIdx {
    let to = final_remap[old].0;
    debug_assert_ne!(to, DISCARDED, "a node the merge drops as unnamed has no survivor");
    if to & DROPPED == 0 { NodeIdx(to) } else { final_remap[(to & !DROPPED) as usize] }
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
///
/// `ordered` says the parent is known to hold its ranges in node order,
/// which is then not read again. Returns whether the parent is left so,
/// and its live pairs, counted as they are written: the merge leaves them
/// for the close.
pub(super) fn rewrite_parent(
    lim: &Limits,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    remap: &MergeRemap,
    room: Option<StoreRoom>,
    ordered: bool,
) -> (bool, usize) {
    // Arena garbage from the whole rewrite, accumulated and noted in one charge
    // below: the only reader (`compact_pairs_if_stale`) runs after the loop, so a
    // per-iteration saturating add buys nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    // An implicit parent's pairs are read off the description through the
    // same filter, and the level is built stored where one changes.
    if let Some(described) = parent_level.implicit().map(|d| d.pairs()) {
        debug_assert!(room.is_some(), "an implicit parent's room is reserved");
        let mut dropped = 0usize;
        parent_level
            .rewrite_described_in(lim, room, false, |_, _, _, pair| {
                let (pair, kept) = canonical_pair_on(pair, t1_side, &remap.final_remap);
                dropped += usize::from(!kept);
                kept.then_some(pair)
            })
            .expect("an implicit parent is rewritten in its room");
        // Built stored at its description's places, in node order: its
        // arena is left as the stored route below leaves it.
        return (parent_level.compact_in_node_order(), described - dropped);
    }
    // Where the arena holds the ranges in node order, each node's kept pairs
    // move down to a write cursor as they are rewritten, which leaves the
    // arena with no dead pairs and the parent no sweep to run; otherwise
    // each node shrinks in place and its tail is left dead. The side is a
    // constant of each slide's loop.
    debug_assert!(!ordered || parent_level.ranges_in_node_order(), "a parent known in node order is so");
    if ordered || parent_level.ranges_in_node_order() {
        let final_remap = &remap.final_remap;
        let live = match t1_side {
            ChildSide::Left => parent_level.slide_pairs(|pair| canonical_pair::<true>(pair, final_remap)),
            ChildSide::Right => parent_level.slide_pairs(|pair| canonical_pair::<false>(pair, final_remap)),
        };
        return (true, live);
    }
    let mut live = 0usize;
    for node_idx in 0..parent_level.nodes().len() {
        let kind = parent_level.node(node_idx).kind();
        let Some(range) = parent_level.arena_range(kind) else {
            remap_inline_node(parent_level, node_idx, t1_side, &remap.final_remap);
            live += 1;
            continue;
        };
        let new_len = keep_canonical_pairs(parent_level, node_idx, t1_side, remap);
        live += new_len;
        if new_len < range.len() {
            dead_acc += parent_level.reencode_shrunk(node_idx, range.start, range.len(), new_len);
        }
    }
    parent_level.note_dead_pairs(dead_acc);
    (false, live)
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
        let (pair, kept) = canonical_pair_on(pairs[read], t1_side, &remap.final_remap);
        pairs[write] = pair;
        write += usize::from(kept);
    }
    write
}

/// A parent pair after the merge, remapped onto its T1-side ref's survivor,
/// and whether it is kept: where that ref survives the merge or was
/// redirected as a content-equal duplicate. A pair not kept duplicates one
/// naming the survivor. The callers write every pair at their write cursor
/// and advance it by the flag, with no branch on it: which pairs are kept
/// follows no pattern a branch predictor learns. `LEFT` says whether T1 is
/// the left child.
#[inline(always)]
fn canonical_pair<const LEFT: bool>(mut pair: ChildPair, final_remap: &[NodeIdx]) -> (ChildPair, bool) {
    let f = if LEFT { &mut pair.left } else { &mut pair.right };
    let to = final_remap[f.0 as usize].0;
    debug_assert_ne!(to, DISCARDED, "a pair names a node the merge drops as unnamed");
    *f = NodeIdx(to & !DROPPED).into();
    (pair, to & DROPPED == 0)
}

/// [`canonical_pair`] with T1's side read at run time.
#[inline]
fn canonical_pair_on(pair: ChildPair, t1_side: ChildSide, final_remap: &[NodeIdx]) -> (ChildPair, bool) {
    match t1_side {
        ChildSide::Left => canonical_pair::<true>(pair, final_remap),
        ChildSide::Right => canonical_pair::<false>(pair, final_remap),
    }
}
