//! Twin contraction at leaf-adjacent levels: rewrite
//! `(Pos_x, S) + (Neg_x, S) → (One_x, S)`.
//!
//! This is the same operation as `contract_twins` (merge nodes that share
//! parent context) but specialized for leaves. Leaf labels (Pos/Neg/One) are
//! *implicit* indices in parent pair lists rather than stored nodes, so the
//! generic `contract_twins` data path — which iterates a level's `nodes` Vec
//! — cannot reach them. `contract_leaf_twins` walks parent pair lists
//! directly, recognizing the `(Pos_x, S) + (Neg_x, S)` co-occurrence as the
//! leaf-side twin pattern, and rewrites it to the canonical `(One_x, S)`.
//!
//! ## All-or-nothing per leaf vtree node
//!
//! The leaf-mode determinism invariant requires
//! that the set of labels referenced at any leaf vtree node is a subset of
//! `{Pos, Neg}` (literal mode) or `{One}` (One mode), never both. Partial
//! contraction — collapsing some Pos/Neg pairs to One while leaving others as
//! literals at the same leaf — would mix modes and violate the invariant. So
//! each leaf is treated atomically: either every parent pair list at that
//! leaf's parent level admits the rewrite (every `(Pos, S)` has a matching
//! `(Neg, S)` partner inside the same pair list, and vice versa), or we leave
//! the leaf untouched.

use crate::diagram::Changed;
use crate::engine::Engine;
use crate::diagram::ChildSide;
use crate::diagram::{MultiPairRange, InputPair, LeafLabel, NodeIdx, Tdd};
use crate::vtree::{VtreeIdx, VtreeNode};

const POS: NodeIdx = NodeIdx(LeafLabel::Pos as u32);
const NEG: NodeIdx = NodeIdx(LeafLabel::Neg as u32);
const ONE: NodeIdx = NodeIdx(LeafLabel::One as u32);

/// Rewrite `(Pos_x, S) + (Neg_x, S)` pairs to `(One_x, S)` wherever feasible —
/// the leaf-specialized form of twin contraction.
///
/// For each internal vtree node `v` whose left or right child is a leaf, check
/// whether every parent pair list on that side admits the rewrite. If yes, do
/// it. Otherwise leave `v`'s level unchanged on that side.
///
/// Returns true if any rewrite happened (caller may want to re-run twin
/// contraction to catch newly-equivalent parents).
pub(crate) fn contract_leaf_twins(eng: &Engine, tdd: &mut Tdd) -> bool {
    let vtree = tdd.vtree.clone();
    let n = vtree.num_nodes();
    // Consume the dirty list. Sites that mutate pair lists push here (rotate,
    // contract_twins, prune-driven full reset); a rebuilt diagram is seeded by
    // its constructor — every internal level from `from_levels_unchecked`, just the
    // rewritten spine from `with_levels_dirty`. Per-call cost is O(|dirty|)
    // instead of O(num_vtree_nodes).
    let dirty = tdd.take_leaf_worklist();
    if dirty.is_empty() {
        return false;
    }
    let mut changed = false;
    for &vi_raw in &dirty {
        let vi = vi_raw as usize;
        if vi >= n { continue; }
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        // No already-contracted cache — always re-classify. A duplicate dirty
        // entry (rotate/contract_twins can push the same vi more than once) is
        // reprocessed, but re-classifying an already-contracted level is a
        // no-op (every contractible pair was already removed), so this is sound.
        if vtree.node(left).is_leaf() {
            changed |= try_contract_leaf_twins(eng, tdd, VtreeIdx(vi_raw), ChildSide::Left);
        }
        if vtree.node(right).is_leaf() {
            changed |= try_contract_leaf_twins(eng, tdd, VtreeIdx(vi_raw), ChildSide::Right);
        }
    }
    changed
}

/// Attempt to contract literal pairs on one side of `parent_vi`'s level.
/// `side = ChildSide::Left` means the leaf is the left child (we contract `pair.left`).
fn try_contract_leaf_twins(eng: &Engine, tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) -> bool {
    let level = &tdd.levels[parent_vi.idx()];
    if level.width() == 0 { return false; }

    // Singleton-pair witness pre-pass (the leaf-contraction corollary). A
    // singleton pair list whose relevant-side
    // label is a literal (Pos or Neg) is an instant non-contractibility
    // witness for the whole level — the missing opposite-polarity partner
    // cannot exist within a length-1 pair list, so leaf contraction at this
    // parent is impossible, and `try_contract_leaf_twins` must abort the level.
    //
    // O(1) per internal node vs `classify`'s O(pair-list-size) collect +
    // sort + compare. Cheaper than reaching the classify loop when any such
    // witness exists, which is the common case in compiled CNFs.
    // Direct slice access (single-instruction `pairs[0]` deref).
    for i in 0..level.nodes.len() {
        if !level.nodes[i].is_internal() { continue; }
        let pairs = level.pairs_of_idx(i);
        if pairs.len() == 1 {
            let label = if side == ChildSide::Left { pairs[0].left } else { pairs[0].right };
            if label == POS || label == NEG {
                return false;
            }
        }
    }

    let mut any_literal = false;
    for i in 0..level.nodes.len() {
        if !level.nodes[i].is_internal() { continue; }
        let pairs = level.pairs_of_idx(i);
        match classify(pairs, side) {
            Class::AllContractible { has_literal } => {
                any_literal |= has_literal;
            }
            Class::NotContractible => return false,
        }
    }
    if !any_literal { return false; }

    rewrite_level(eng, tdd, parent_vi, side);
    true
}

enum Class {
    /// Pair list is OK to rewrite on this side. `has_literal` = at least one
    /// Pos/Neg pair exists (so this is mode-literal in part); if false, the
    /// pair list is purely One on this side and the rewrite is a no-op here.
    AllContractible { has_literal: bool },
    /// At least one Pos has no matching Neg (or vice versa) within this pair
    /// list. Contracting would change the function; bail.
    NotContractible,
}

fn classify(pairs: &[InputPair], side: ChildSide) -> Class {
    let mut pos: Vec<NodeIdx> = Vec::new();
    let mut neg: Vec<NodeIdx> = Vec::new();
    let mut has_one = false;
    for p in pairs {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        let partner = if side == ChildSide::Left { p.right } else { p.left };
        if label == POS { pos.push(partner); }
        else if label == NEG { neg.push(partner); }
        else if label == ONE { has_one = true; }
        else { return Class::NotContractible; }
    }
    let has_literal = !pos.is_empty() || !neg.is_empty();
    if has_literal && has_one {
        // Mode-mixed input. On a STRUCTURAL leaf this shouldn't happen if
        // check_determinism passes. On a WEIGHT-marginal leaf it is expected and
        // benign: the refs there are value selectors into the pinned column, not
        // Boolean children, and `marginalize::canonicalize_leaf_refs_at_parent`
        // deliberately folds equal-valued slots together (Neg → Pos when w⁺ = w⁻,
        // Pos → One when w⁻ = 0), which mixes the label sets. Bailing is the right
        // answer either way — for the weighted case the same collapse is reached by
        // `duplicate_pair_resolve`'s pinned-column fold on the duplicate run canon produces.
        return Class::NotContractible;
    }
    pos.sort();
    neg.sort();
    if pos == neg {
        Class::AllContractible { has_literal }
    } else {
        Class::NotContractible
    }
}

/// Apply the `(Pos_x, S) + (Neg_x, S) → (One_x, S)` rewrite to every pair list
/// at `parent_vi`'s level, IN PLACE.
///
/// Precondition (established by `try_contract_leaf_twins`, the only caller):
/// Every internal node at the level is `Class::AllContractible` on `side` — its
/// labels on that side are all `One`, or all literals whose `(Pos, S)` and
/// `(Neg, S)` multisets are equal. Mode-mixed and unmatched lists vetoed the
/// level before we got here.
///
/// ## In-place cursor, not a rebuild
///
/// The rewrite is monotone shrinking per node: a `(Pos, S)` pair becomes one
/// `(One, S)` pair, its `(Neg, S)` partner is dropped, a `One` pair is copied
/// verbatim — so a node's new pair count is `old / 2` (matched literals) or
/// `old` (pure `One`), never more. Each node is therefore rewritten with a write
/// cursor trailing a read cursor **inside the node's own arena range**: both
/// start at the range's first slot and the write index advances at most once per
/// read, so the write can never overtake the read. Ranges at a level are
/// pairwise disjoint — every arena writer appends a fresh tail range and only
/// ever re-points a node at its own slots, the property
/// `compact_pairs_if_stale` verifies before sliding — so one node's cursor can
/// never reach another node's pairs either.
///
/// The predecessor materialized every rewritten list into a
/// `Vec<Vec<InputPair>>` (a second full copy of the level plus one allocation
/// per node) and then `clear()`ed and re-pushed the whole level. Node indices
/// are unchanged by construction here, where the rebuild had to re-derive them
/// from push order.
fn rewrite_level(eng: &Engine, tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) {
    let level = &mut tdd.levels[parent_vi.idx()];
    for i in 0..level.nodes.len() {
        // Tombstone slots (index-stable conjoin) and leaf words own no
        // pair range and are left exactly as they are. That is also what keeps
        // `n_tombstones` correct for free: the rebuild had to record every
        // tombstone before `clear()` and re-push it, or an unreferenced dead
        // slot would have come back as a live empty-multi node.
        if !level.nodes[i].is_internal() {
            continue;
        }
        if level.nodes[i].is_inline() {
            // Single-pair node: its one pair is labelled `One` on `side`. A lone
            // literal has no opposite-polarity partner inside a length-1 list, so
            // it is an instant non-contractibility witness — the singleton
            // pre-pass in `try_contract_leaf_twins` aborts the level on it, well
            // before this function runs. `One` pairs are copied verbatim by the
            // rewrite, so there is nothing to do.
            debug_assert!(
                {
                    let p = level.nodes[i].inline_pair();
                    let label = if side == ChildSide::Left { p.left } else { p.right };
                    label != POS && label != NEG
                },
                "leaf rewrite: a contractible level cannot hold a single-pair literal node"
            );
            continue;
        }
        let range = level.pair_range_at(i);
        let (start, old_len) = (range.start, range.len());
        let mut w = start;
        for r in start..start + old_len {
            let p = level.pairs[r];
            let label = if side == ChildSide::Left { p.left } else { p.right };
            if label == NEG {
                // Dropped: its matching Pos contributes the (One, partner) pair
                // for this context. No re-sort and no dedup: pair lists are
                // unordered and twin contraction is order-independent, and
                // `classify` already rejected the mode-mixed lists that could
                // have produced a *new* duplicate. Duplicates already present
                // in the input (legal in a marginalized diagram) are carried
                // through one-for-one, which is what the multiset count
                // recurrence needs.
                continue;
            }
            let np = if label == POS {
                if side == ChildSide::Left {
                    InputPair { left: ONE, right: p.right }
                } else {
                    InputPair { left: p.left, right: ONE }
                }
            } else {
                p
            };
            debug_assert!(w <= r, "leaf rewrite: write cursor overtook the read cursor");
            level.pairs[w] = np;
            w += 1;
        }
        let new_len = w - start;
        debug_assert!(
            new_len == old_len || new_len * 2 == old_len,
            "leaf rewrite: `classify` admits a pure-One list (unchanged) or a matched \
             literal list (halved), got {new_len} of {old_len}"
        );
        if new_len == old_len {
            // Pure-`One` node (or the empty-multi placeholder): every store above
            // was an identity copy, and the node word already says `new_len`.
            continue;
        }
        // Shrink the node onto the prefix the cursor wrote: re-encode via the
        // shared epilogue (`TddLevel::reencode_shrunk_multi`, also used by
        // `pair_fusion::rebuild_parent_level`) — inline when the sole survivor
        // allows it, a length-1 extended range over the cursor's slot
        // otherwise, or a plain `set_pair_len` shrink. The tail slots it
        // abandons are unreferenced arena, accounted to `dead_pairs` for the
        // level's own sweep below — that counter is a sweep TRIGGER only, so
        // reaching it by accumulation instead of the rebuild's reset-to-zero
        // can shift *when* a sweep runs, never what it produces.
        //
        // `rewrite_level` is called only from the infallible
        // `contract_leaf_twins` (a debug-only rotation-locality invariant check calls it
        // with no error path), so the one fallible arm inside the helper (a
        // fresh `multi_pairs` push when the node isn't already extended) maps its
        // `Err` to the same abort `Vec::push` itself would have raised on
        // allocation failure — this changes nothing observable, it just
        // routes the failure through the same fallible primitive the rest of
        // the crate uses instead of an unchecked `push`.
        let dead = level.reencode_shrunk_multi(eng, i, start, old_len, new_len).unwrap_or_else(|_| {
            std::alloc::handle_alloc_error(core::alloc::Layout::new::<MultiPairRange>())
        });
        level.note_dead_pairs(dead);
    }

    // `inlined_sides` is deliberately left alone: the rewrite copies every
    // marginal-side ref through verbatim, so a marker saying that side holds inline
    // counts still describes the level. The other state a rebuild's `clear()`
    // used to reset needs no action either — `marginal_counts` is already `None`
    // (a marginal level has empty `nodes`, so the caller never finds a literal
    // and never calls us), the marginal width fields are only ever written on
    // marginal levels, and no tombstone moved.
    // The rebuild compacted the arena as a side effect of refilling it; the
    // cursor leaves the dropped slots in place instead. Hand that to the level's
    // one compaction policy — a no-op until the garbage passes its threshold,
    // then a single memmove plus `shrink_arrays` (the same call `merge.rs` makes
    // after its rewrites), and the only step here that actually returns pages:
    // `clear()` retained the arena's capacity, so the rebuild freed nothing.
    // No pair-arena offset is held across this call.
    level.compact_pairs_if_stale();
    tdd.invalidate(parent_vi, Changed::PAIRS);
}
