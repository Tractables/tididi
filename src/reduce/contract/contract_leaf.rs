//! Twin contraction at leaf-adjacent levels: rewrite
//! `(Pos_x, S) + (Neg_x, S) → (One_x, S)`.
//!
//! The same operation as `contract_twins` (merge nodes that share parent
//! context), specialized for leaves: leaf labels (Pos/Neg/One) are implicit
//! indices in parent pair lists rather than stored nodes, so the generic path
//! over a level's `nodes` cannot reach them. `contract_leaf_twins` walks the
//! parent pair lists directly and rewrites the co-occurrence to `(One_x, S)`.
//!
//! The rewrite is all-or-nothing per leaf: the labels referenced at a leaf
//! must be a subset of `{Pos, Neg}` or of `{One}`, never both
//! (`test_helpers::check::check_determinism`), so a leaf's parent level is
//! rewritten only when every pair list there admits it (every `(Pos, S)` has
//! its `(Neg, S)` partner in the same list, and vice versa).

use crate::diagram::Pass;
use crate::Engine;
use crate::diagram::ChildSide;
use crate::diagram::{EncodedChildRef, ChildPair, NodeKind, Tdd, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Rewrite `(Pos_x, S) + (Neg_x, S)` pairs to `(One_x, S)` wherever feasible —
/// the leaf-specialized form of twin contraction.
///
/// For each internal vtree node `v` whose left or right child is a leaf, check
/// whether every parent pair list on that side admits the rewrite. If yes, do
/// it. Otherwise leave `v`'s level unchanged on that side.
///
/// Returns true if any rewrite happened (caller may want to re-run twin
/// contraction to catch newly-equivalent parents).
///
/// # Errors
///
/// The rewrite allocates nothing, so this does not refuse; the signature
/// matches the other contraction passes.
pub(crate) fn contract_leaf_twins(eng: &Engine, tdd: &mut Tdd) -> Result<bool, OperationError> {
    let vtree = tdd.vtree.clone();
    let n = vtree.num_nodes();
    // Every site that mutates a pair list pushes its level here, so the
    // per-call cost is O(|dirty|) instead of O(num_vtree_nodes).
    let dirty = tdd.dirty.take(Pass::LeafContract);
    if dirty.is_empty() {
        return Ok(false);
    }
    let mut changed = false;
    for (k, &vi_raw) in dirty.iter().enumerate() {
        if vi_raw as usize >= n { continue; }
        match contract_leaf_sides(eng, tdd, &vtree, VtreeIdx(vi_raw)) {
            Ok(fired) => changed |= fired,
            Err(e) => {
                tdd.dirty.requeue(Pass::LeafContract, dirty[k..].iter().copied());
                return Err(e);
            }
        }
    }
    Ok(changed)
}

/// Contract each leaf child of `vi`, left side then right. A duplicate dirty
/// entry is reprocessed; re-classifying an already-contracted level finds no
/// literal pair and does nothing.
fn contract_leaf_sides(eng: &Engine, tdd: &mut Tdd, vtree: &Vtree, vi: VtreeIdx) -> Result<bool, OperationError> {
    let (left, right) = match *vtree.node(vi) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => return Ok(false),
    };
    let mut changed = false;
    if vtree.node(left).is_leaf() {
        changed |= try_contract_leaf_twins(eng, tdd, vi, ChildSide::Left)?;
    }
    if vtree.node(right).is_leaf() {
        changed |= try_contract_leaf_twins(eng, tdd, vi, ChildSide::Right)?;
    }
    Ok(changed)
}

/// Attempt to contract literal pairs on one side of `parent_vi`'s level.
/// `side = ChildSide::Left` means the leaf is the left child (we contract `pair.left`).
fn try_contract_leaf_twins(eng: &Engine, tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) -> Result<bool, OperationError> {
    let level = &tdd.levels[parent_vi.idx()];
    if level.slot_count() == 0 { return Ok(false); }

    // Singleton-pair witness pre-pass: a length-1 pair list whose label on
    // `side` is a literal cannot hold the opposite-polarity partner, so the
    // level is not contractible. O(1) per node, against `classify`'s
    // collect-and-sort per pair list.
    for i in 0..level.nodes.len() {
        if !level.nodes[i].is_internal() { continue; }
        let pairs = level.pairs_of_idx(i);
        if pairs.len() == 1 {
            let label = if side == ChildSide::Left { pairs[0].left } else { pairs[0].right };
            if label == POS_LEAF_IDX.into() || label == NEG_LEAF_IDX.into() {
                return Ok(false);
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
            Class::NotContractible => return Ok(false),
        }
    }
    if !any_literal { return Ok(false); }

    rewrite_level(tdd, parent_vi, side);
    Ok(true)
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

fn classify(pairs: &[ChildPair], side: ChildSide) -> Class {
    let mut pos: Vec<EncodedChildRef> = Vec::new();
    let mut neg: Vec<EncodedChildRef> = Vec::new();
    let mut has_one = false;
    for p in pairs {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        let partner = if side == ChildSide::Left { p.right } else { p.left };
        if label == POS_LEAF_IDX.into() { pos.push(partner); }
        else if label == NEG_LEAF_IDX.into() { neg.push(partner); }
        else if label == ONE_LEAF_IDX.into() { has_one = true; }
        else { return Class::NotContractible; }
    }
    let has_literal = !pos.is_empty() || !neg.is_empty();
    if has_literal && has_one {
        // Mode-mixed list. On a structural leaf `check_determinism` forbids
        // it; on a weight-marginal leaf it is expected, since the refs there
        // select values in the pinned column and
        // `marginal::leaf::canonicalize_leaf_refs_at_parent` folds equal-valued
        // labels together. Bail either way.
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
/// at `parent_vi`'s level, in place.
///
/// Precondition: every internal node at the level is `Class::AllContractible`
/// on `side` (its labels there are all `One`, or literals whose `(Pos, S)` and
/// `(Neg, S)` multisets are equal).
///
/// # Soundness
///
/// Each node is rewritten with a write cursor trailing a read cursor inside
/// the node's own arena range. A `(Pos, S)` pair becomes `(One, S)`, its
/// `(Neg, S)` partner is dropped and a `One` pair is copied, so the write
/// index advances at most once per read and never overtakes it; ranges at a
/// level are pairwise disjoint (`compact_pairs_if_stale` verifies this before
/// sliding), so a cursor never reaches another node's pairs. Node indices are
/// unchanged, and nothing is allocated.
fn rewrite_level(tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) {
    tdd.rewrite_level(parent_vi, |level| {
        for i in 0..level.nodes.len() {
            // Tombstone slots and leaf words own no pair range and are left as
            // they are, which also keeps `n_tombstones` correct.
            if !level.nodes[i].is_internal() {
                continue;
            }
            if let NodeKind::Inline(p) = level.nodes[i].kind() {
                // A single-pair node is labelled `One` on `side` (the singleton
                // pre-pass in `try_contract_leaf_twins` aborted the level on a
                // lone literal), and `One` pairs are copied verbatim.
                debug_assert!(
                    {
                        let label = if side == ChildSide::Left { p.left } else { p.right };
                        label != POS_LEAF_IDX.into() && label != NEG_LEAF_IDX.into()
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
                if label == NEG_LEAF_IDX.into() {
                    // Dropped: its matching Pos contributes the (One, partner)
                    // pair. No re-sort and no dedup: pair lists are unordered,
                    // `classify` rejected the mode-mixed lists that could mint a
                    // new duplicate, and duplicates already present (legal in a
                    // marginalized diagram) must be carried through one-for-one.
                    continue;
                }
                let np = if label == POS_LEAF_IDX.into() {
                    if side == ChildSide::Left {
                        ChildPair::new(ONE_LEAF_IDX, p.right)
                    } else {
                        ChildPair::new(p.left, ONE_LEAF_IDX)
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
            // Shrink the node onto the prefix the cursor wrote. The abandoned tail
            // slots are unreferenced arena, tallied into `dead_pairs` for the
            // sweep below; that counter only decides when a sweep runs.
            let dead = level.reencode_shrunk(i, start, old_len, new_len);
            level.note_dead_pairs(dead);
        }

        // `inlined_sides` still describes the level: every marginal-side ref was
        // copied through verbatim. The dropped slots stay in the arena until the
        // level's compaction threshold; no pair-arena offset is held across it.
        level.compact_pairs_if_stale();
    });
}
