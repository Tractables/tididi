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
use crate::limits::{Limits, OperationError, Transient};
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
/// [`OperationError::OverBudget`] when the partner lists a level is classified
/// with cannot be reserved; the levels already rewritten stay rewritten and
/// the rest stay queued for the next call.
pub(crate) fn contract_leaf_twins(eng: &Engine, tdd: &mut Tdd) -> Result<bool, OperationError> {
    let vtree = tdd.vtree.clone();
    let n = vtree.num_nodes();
    // Every site that mutates a pair list pushes its level here, so the
    // per-call cost is O(|dirty|) instead of O(num_vtree_nodes).
    let mut dirty = tdd.dirty.take(Pass::LeafContract);
    if dirty.is_empty() {
        return Ok(false);
    }
    // The list may name a level more than once ([`Dirty::dedup_above`]).
    dirty.sort_unstable();
    dirty.dedup();
    // Hints only: a level is refused off its hinted node where that node
    // refuses, whatever level the hint was taken at, and is classified in
    // full otherwise. Without room for them there are none.
    let mut refused = eng.scratch.reduce.leaf_refused.checkout_preserving(eng);
    if refused.len() < 2 * n && eng.limits().try_resize(&mut refused, 2 * n, NO_HINT).is_err() {
        refused.clear();
    }
    let mut changed = false;
    for (k, &vi_raw) in dirty.iter().enumerate() {
        if vi_raw as usize >= n { continue; }
        match contract_leaf_sides(eng, tdd, &vtree, VtreeIdx(vi_raw), &mut refused) {
            Ok(fired) => changed |= fired,
            Err(e) => {
                tdd.dirty.requeue(Pass::LeafContract, dirty[k..].iter().copied());
                return Err(e);
            }
        }
    }
    Ok(changed)
}

/// An entry of the refusal hints that names no node.
const NO_HINT: u32 = u32::MAX;

/// Contract each leaf child of `vi`, left side then right. A duplicate dirty
/// entry is reprocessed; re-classifying an already-contracted level finds no
/// literal pair and does nothing. `refused` holds the hints
/// [`try_contract_leaf_twins`] reads and writes.
fn contract_leaf_sides(eng: &Engine, tdd: &mut Tdd, vtree: &Vtree, vi: VtreeIdx, refused: &mut [u32]) -> Result<bool, OperationError> {
    let (left, right) = match *vtree.node(vi) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => return Ok(false),
    };
    let mut changed = false;
    if vtree.node(left).is_leaf() {
        changed |= try_contract_leaf_twins(eng, tdd, vi, ChildSide::Left, refused)?;
    }
    if vtree.node(right).is_leaf() {
        changed |= try_contract_leaf_twins(eng, tdd, vi, ChildSide::Right, refused)?;
    }
    Ok(changed)
}

/// Attempt to contract literal pairs on one side of `parent_vi`'s level.
/// `side = ChildSide::Left` means the leaf is the left child (we contract `pair.left`).
///
/// One node that refuses the rewrite refuses it for the level, so a stored
/// level is first tested at the node that refused the last classification
/// on the same vtree node and side, its entry in `refused` (any diagram's
/// level there, so a hint only): a level a conjunction rebuilt or a merge
/// changed mostly refuses where it refused before. A level refused is
/// noted there with its first refusing node.
fn try_contract_leaf_twins(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_vi: VtreeIdx,
    side: ChildSide,
    refused: &mut [u32],
) -> Result<bool, OperationError> {
    let level = &tdd.levels[parent_vi.idx()];
    if level.slot_count() == 0 { return Ok(false); }
    // A stored level's nodes are read as slices of its arena.
    let Some(stored) = level.stored() else {
        return try_contract_described(eng, tdd, parent_vi, side);
    };
    let hint = refused.get_mut(2 * parent_vi.idx() + side as usize);
    if let Some(&at) = hint.as_deref()
        && let Some(node) = stored.nodes().get(at as usize)
        && matches!(classify_narrow(stored.of(node), side), Some(Class::NotContractible))
    {
        return Ok(false);
    }

    // One pass in node order, stopping at the first node that refuses the
    // rewrite: most levels that refuse it refuse it near the start. A node
    // of up to [`SMALL_PAIRS`] pairs is classified on the stack
    // ([`classify_narrow`]), a wider one in two lists reused across the
    // level's nodes so the sort buffers are allocated once per level, not
    // per node, and charged only while the level is classified.
    let lim = eng.limits();
    let mut pos: Transient<'_, Vec<EncodedChildRef>> = Transient::new(lim, Vec::new());
    let mut neg: Transient<'_, Vec<EncodedChildRef>> = Transient::new(lim, Vec::new());
    let mut any_literal = false;
    for (i, node) in stored.nodes().iter().enumerate() {
        let pairs = stored.of(node);
        let class = match classify_narrow(pairs, side) {
            Some(class) => class,
            None => classify(lim, pairs.iter().copied(), side, &mut pos, &mut neg)?,
        };
        match class {
            Class::AllContractible { has_literal } => any_literal |= has_literal,
            Class::NotContractible => {
                if let Some(hint) = hint {
                    *hint = i as u32;
                }
                return Ok(false);
            }
        }
    }
    if !any_literal { return Ok(false); }

    rewrite_level(lim, tdd, parent_vi, side)?;
    Ok(true)
}

/// [`try_contract_leaf_twins`] on a level held as the description of its
/// pairs. A node's pairs are its first pair shifted by the offsets every
/// node shares, so whether a node admits the rewrite, and whether it holds a
/// literal, depends only on its first pair's label on `side`: one node of
/// each label is classified, off the description, about the side's distinct
/// labels times a node's pairs, not the level's pairs. A level that admits
/// the rewrite is built stored for it ([`rewrite_level`]).
fn try_contract_described(eng: &Engine, tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) -> Result<bool, OperationError> {
    let d = tdd.levels[parent_vi.idx()].implicit().expect("an implicit level");
    let mut labels: rustc_hash::FxHashMap<i64, usize> = rustc_hash::FxHashMap::default();
    d.each_side_first(side, |label, node| {
        labels.entry(label).or_insert(node);
    });
    let lim = eng.limits();
    let mut pos: Transient<'_, Vec<EncodedChildRef>> = Transient::new(lim, Vec::new());
    let mut neg: Transient<'_, Vec<EncodedChildRef>> = Transient::new(lim, Vec::new());
    let mut any_literal = false;
    for &node in labels.values() {
        match classify(lim, d.places(node), side, &mut pos, &mut neg)? {
            Class::AllContractible { has_literal } => any_literal |= has_literal,
            Class::NotContractible => return Ok(false),
        }
    }
    if !any_literal { return Ok(false); }
    rewrite_level(lim, tdd, parent_vi, side)?;
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

/// [`classify`] of a stored node's pair list of at most [`SMALL_PAIRS`]
/// pairs, collecting nothing in memory, or `None` for a wider one. A node of
/// one pair admits the rewrite only where its label on `side` is `One`: a
/// lone literal has no opposite-polarity partner.
#[inline(always)]
fn classify_narrow(pairs: &[ChildPair], side: ChildSide) -> Option<Class> {
    Some(match *pairs {
        [pair] => {
            let label = if side == ChildSide::Left { pair.left } else { pair.right };
            if label == ONE_LEAF_IDX.into() {
                Class::AllContractible { has_literal: false }
            } else {
                Class::NotContractible
            }
        }
        [p, q] => classify_two(p, q, side),
        ref pairs if pairs.len() <= SMALL_PAIRS => classify_small(pairs, side),
        _ => return None,
    })
}

/// [`classify`] of a pair list of two pairs, read off their labels with
/// nothing collected: both `One`, or `Pos` and `Neg` with one partner.
#[inline]
fn classify_two(p: ChildPair, q: ChildPair, side: ChildSide) -> Class {
    let split = |p: ChildPair| if side == ChildSide::Left { (p.left, p.right) } else { (p.right, p.left) };
    let ((a, x), (b, y)) = (split(p), split(q));
    let (one, pos, neg): (EncodedChildRef, EncodedChildRef, EncodedChildRef) =
        (ONE_LEAF_IDX.into(), POS_LEAF_IDX.into(), NEG_LEAF_IDX.into());
    if a == one && b == one {
        Class::AllContractible { has_literal: false }
    } else if x == y && (a == pos && b == neg || a == neg && b == pos) {
        Class::AllContractible { has_literal: true }
    } else {
        Class::NotContractible
    }
}

/// The widest node [`classify_small`] classifies on the stack.
const SMALL_PAIRS: usize = 8;

/// [`classify`] of a pair list of at most [`SMALL_PAIRS`] pairs, its
/// partners collected and sorted on the stack.
#[inline]
fn classify_small(pairs: &[ChildPair], side: ChildSide) -> Class {
    let (one, pos_label, neg_label): (EncodedChildRef, EncodedChildRef, EncodedChildRef) =
        (ONE_LEAF_IDX.into(), POS_LEAF_IDX.into(), NEG_LEAF_IDX.into());
    let (mut pos, mut neg) = ([one; SMALL_PAIRS], [one; SMALL_PAIRS]);
    let (mut np, mut nn, mut has_one) = (0usize, 0usize, false);
    for p in pairs {
        let (label, partner) = if side == ChildSide::Left { (p.left, p.right) } else { (p.right, p.left) };
        if label == pos_label {
            pos[np] = partner;
            np += 1;
        } else if label == neg_label {
            neg[nn] = partner;
            nn += 1;
        } else if label == one {
            has_one = true;
        } else {
            return Class::NotContractible;
        }
    }
    let has_literal = np + nn > 0;
    // A mode-mixed list refuses, as in `classify`, and so do partner lists
    // of different lengths.
    if (has_literal && has_one) || np != nn {
        return Class::NotContractible;
    }
    pos[..np].sort_unstable();
    neg[..nn].sort_unstable();
    if pos[..np] == neg[..nn] { Class::AllContractible { has_literal } } else { Class::NotContractible }
}

/// `pos` and `neg` are the caller's buffers for the partners of the `Pos` and
/// `Neg` pairs; they are cleared here and grown against `lim`.
fn classify(
    lim: &Limits,
    pairs: impl Iterator<Item = ChildPair>,
    side: ChildSide,
    pos: &mut Vec<EncodedChildRef>,
    neg: &mut Vec<EncodedChildRef>,
) -> Result<Class, OperationError> {
    pos.clear();
    neg.clear();
    let mut has_one = false;
    for p in pairs {
        let label = if side == ChildSide::Left { p.left } else { p.right };
        let partner = if side == ChildSide::Left { p.right } else { p.left };
        if label == POS_LEAF_IDX.into() { lim.try_push(pos, partner)?; }
        else if label == NEG_LEAF_IDX.into() { lim.try_push(neg, partner)?; }
        else if label == ONE_LEAF_IDX.into() { has_one = true; }
        else { return Ok(Class::NotContractible); }
    }
    let has_literal = !pos.is_empty() || !neg.is_empty();
    if has_literal && has_one {
        // Mode-mixed list. On a structural leaf `check_determinism` forbids
        // it; on a weight-marginal leaf it is expected, since the refs there
        // select values in the pinned column and
        // `marginal::leaf::canonicalize_leaf_refs_at_parent` folds equal-valued
        // labels together. Bail either way.
        return Ok(Class::NotContractible);
    }
    pos.sort_unstable();
    neg.sort_unstable();
    Ok(if pos == neg { Class::AllContractible { has_literal } } else { Class::NotContractible })
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
/// unchanged, and nothing is allocated but the room an implicit level is
/// stored in, through `lim`, before the rewrite starts.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when that room is refused; the level is
/// then as it was.
fn rewrite_level(lim: &Limits, tdd: &mut Tdd, parent_vi: VtreeIdx, side: ChildSide) -> Result<(), OperationError> {
    // A contracted level leaves the description: an implicit one that
    // admits the rewrite is built stored where its pairs lie, and the close
    // at the end of the operation describes what is affine again.
    tdd.levels[parent_vi.idx()].store_if_implicit(lim)?;
    tdd.rewrite_level(parent_vi, |level| {
        for i in 0..level.nodes().len() {
            if let NodeKind::Inline(p) = level.node(i).kind() {
                // A single-pair node is labelled `One` on `side` (the first
                // pass of `try_contract_leaf_twins` refused the level on a
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
            let arena = level.pairs.stored_mut();
            for r in start..start + old_len {
                let p = arena[r];
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
                arena[w] = np;
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

        // `value_ref_sides` still describes the level: every marginal-side ref was
        // copied through verbatim. The dropped slots stay in the arena until the
        // level's compaction threshold; no pair-arena offset is held across it.
        level.compact_pairs_if_stale();
    });
    Ok(())
}
