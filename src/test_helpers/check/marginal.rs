//! Structural invariant checks for a marginalized diagram: slot
//! canonicalization, orphan, twin and fusion-redex detection, pair-fusion
//! saturation, and model-count preservation.
//!
//! The invariants are numbered in `docs/architecture.md` and stated there once.
//! This module decides five of them: 5 (marginality permanent and
//! downward-closed), 7 (inline discipline), 8 (pair-fusion saturation), 9 (twin
//! canonicality) and 10 (value-slot uniqueness).
//!
//! Invariants 8, 9 and 10 hold at the fixpoint of twin contraction,
//! canonicalization by value, and pair fusion, at every boundary-marginal level
//! — a marginal child of a structural parent. Each is a post-pass property, so
//! a check applies where its pass has just run and not at an arbitrary point:
//! a twin merge can recreate a fusion redex, and a store born at an apply's
//! emit reaches value-slot uniqueness only at the slot prune that follows
//! tagging. A parent whose other child is also marginal is exempt from 8, since
//! there is no structural side there.
//!
//! Every check decodes marginal-side references through the post-tagger
//! encoding, so none applies before the tagger has run.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::diagram::{BigSide, InputPair, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildSide, boundary_marginal_levels};
use crate::value::Count;
use crate::value::slots::{RefSlotScratch, count_key_at, referenced_marginal_slots};

/// Invariant 7: among the slots actually referenced from a structural
/// parent's marginal-side pair refs, no slot holds an inline-eligible count
/// (at most `MARGINAL_INLINE_MAX`, with no big-overflow entry).
///
/// Scope:
/// - **Unreferenced/stale slots are exempt** — count vectors are never shrunk
///   count vectors are never shrunk, so a slot whose
///   refs were all retagged inline legitimately retains its small count.
/// - **Deep-marginal and root levels are exempt** — a marginal level whose
///   parent is itself marginal (or that has no parent) carries no parent pair
///   refs to inline into; its counts (e.g. the final model count) must live in
///   explicit slots regardless of magnitude.
///
/// Slot count uniqueness lives in invariant 10 (`check_slot_count_uniqueness`): with
/// `prune_value_slots` collecting orphaned slots, all slots at a marginal level
/// must carry pairwise-distinct counts at the canonical-form fixpoint.
pub fn check_inline_discipline(tdd: &Tdd) -> Result<(), String> {
    let mut slots = RefSlotScratch::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        let vlevel = &tdd.levels[v.idx()];
        let Some(counts) = vlevel.marginal_counts() else {
            continue;
        };
        let big = vlevel.marginal_counts_big();
        for &s in referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut slots) {
            let i = s as usize;
            if i >= counts.len() {
                continue; // Out-of-range refs are the ref-bounds check's job.
            }
            let has_big = big.and_then(|b| b.get(i)).is_some();
            if counts[i] <= crate::diagram::MARGINAL_INLINE_MAX as u128 && !has_big {
                return Err(format!(
                    "invariant 7 violation at marginal level {} (parent {}) slot {}: referenced count {} \u{2264} the inline maximum ({}); must be inline at parent refs",
                    v.idx(),
                    parent.idx(),
                    i,
                    counts[i],
                    crate::diagram::MARGINAL_INLINE_MAX,
                ));
            }
        }
    }
    Ok(())
}

/// Collect node `n`'s pairs.
fn node_pairs_into(level: &TddLevel, n: usize, out: &mut Vec<InputPair>) {
    out.clear();
    out.extend_from_slice(level.pairs_of_idx(n));
}

/// Invariant 8, pair-fusion saturation: within each boundary-marginal parent
/// node, every non-marginal-side child ref appears in at most one pair — a
/// group that shares one is exactly a pair-fusion redex. `filter`, when given,
/// restricts the walk to those parent vtree nodes (mirroring
/// `fuse_pairs_at_parents`). Returns `Err` describing the first redex found.
///
/// Parents with two marginal children are out of scope. When *both* children of `parent`
/// are marginal, `boundary_marginal_levels` yields the parent twice (once per
/// child) and the "x" ref F would key on is itself a marginal ref, not a
/// non-marginal child — the premise of the invariant (pairwise-mutex explicit
/// children) does not apply. Such a parent is also not a fusion fixpoint after
/// a single sweep: the second boundary's fusion can hand two groups the same
/// marginal ref by design (count-keyed slot sharing / equal inline counts, see
/// `fuse_pairs_inner` Phase 2), recreating a same-x group at the first
/// boundary. Those duplicate pairs are sound under multiset pair lists and the
/// next sweep closes them. Skipping keeps the check faithful to what F
/// actually claims.
pub fn check_pair_fusion_saturation(tdd: &Tdd, filter: Option<&[VtreeIdx]>) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if let Some(f) = filter
            && !f.contains(&parent) {
                continue;
            }
        let (pleft, pright) = tdd.vtree.children(parent);
        let sibling = match side {
            ChildSide::Left => pright,
            ChildSide::Right => pleft,
        };
        if tdd.levels[sibling.idx()].is_marginal() {
            continue; // both-marginal parent — see doc above
        }
        let plevel = &tdd.levels[parent.idx()];
        for n in 0..plevel.nodes.len() {
            if plevel.nodes[n].is_leaf() {
                continue;
            }
            node_pairs_into(plevel, n, &mut pairs_buf);
            if pairs_buf.len() < 2 {
                continue;
            }
            seen.clear();
            for p in &pairs_buf {
                let x = match side {
                    ChildSide::Right => p.left.0,
                    ChildSide::Left => p.right.0,
                };
                if !seen.insert(x) {
                    return Err(format!(
                        "invariant 8 (pair-fusion saturation) violation at parent level {} (marginal child {}, side {:?}): \
                         node {} holds \u{2265}2 pairs sharing non-marginal child ref {:#x}",
                        parent.idx(),
                        v.idx(),
                        side,
                        n,
                        x,
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Invariant 9, twin canonicality: no two non-leaf nodes at any canonicalized
/// level carry equal pair multisets — equal-pair-list nodes are twins and
/// must have merged. Returns `Err` describing the first twin pair found.
///
/// The level set is `contract::content_twin::content_twin_scan_levels`, the same one the
/// merge itself walks (every explicit level of a marginalized diagram, empty
/// otherwise), so the checker is exactly the merge's postcondition and cannot
/// drift from it.
pub fn check_twin_canonicality(tdd: &Tdd) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    for parent in crate::reduce::contract::content_twin::content_twin_scan_levels(tdd) {
        let plevel = &tdd.levels[parent.idx()];
        let mut key_to_node: FxHashMap<Vec<(u32, u32)>, usize> = FxHashMap::default();
        for n in 0..plevel.nodes.len() {
            if plevel.nodes[n].is_leaf() {
                continue;
            }
            node_pairs_into(plevel, n, &mut pairs_buf);
            let mut key: Vec<(u32, u32)> =
                pairs_buf.iter().map(|p| (p.left.0, p.right.0)).collect();
            key.sort_unstable();
            if let Some(&m) = key_to_node.get(&key) {
                return Err(format!(
                    "invariant 9 (twin canonicality) violation at parent level {}: nodes {} and {} \
                     have identical pair multisets ({} pairs) — unmerged twins",
                    parent.idx(),
                    m,
                    n,
                    key.len(),
                ));
            }
            key_to_node.insert(key, n);
        }
    }
    Ok(())
}

/// How many slots a marginal level actually stores, in whichever domain it was
/// marginal into: integer counts live inside the level, weighted values in the
/// diagram's external store.
fn stored_slot_count(tdd: &Tdd, left_idx: usize) -> usize {
    if tdd.levels[left_idx].is_weight_marginal() {
        return tdd
            .weights()
            .and_then(|ws| ws.level(left_idx))
            .map_or(0, |c| c.len());
    }
    tdd.levels[left_idx].marginal_counts().map_or(0, |c| c.len())
}

/// Invariant 4 garbage-freedom: after the slot prune, every slot index in
/// `0..store_len` at a boundary marginal level (a marginal child of a
/// structural parent) is referenced by at least one parent marginal-side ref.
/// An unreferenced slot is an orphan `prune_value_slots` should have
/// collected. The output level's store (`tdd.output.vtree`) is exempt: it
/// holds the final count or a component sub-diagram's count, and the prune
/// never touches it.
pub fn check_no_orphan_slots(tdd: &Tdd) -> Result<(), String> {
    let out_v = tdd.output.vtree;
    let mut slots = RefSlotScratch::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        let store_len = stored_slot_count(tdd, v.idx());
        if store_len == 0 {
            continue; // nothing to check
        }
        let referenced = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut slots);
        let ref_count = referenced.len();
        if tdd.levels[v.idx()].is_weight_marginal() {
            // Weighted stores are full width and are never pruned, so an
            // unreferenced slot is not garbage there. What the bare-slot
            // encoding does require is that every reference names a slot the
            // column actually holds.
            for &s in referenced {
                if (s as usize) >= store_len {
                    return Err(format!(
                        "invariant 4 (garbage-freedom) violation at boundary weight-marginal level {} \
                         (non-marginal parent {}, side {:?}): reference to slot {} is past \
                         the end of a {}-slot column",
                        v.idx(),
                        parent.idx(),
                        side,
                        s,
                        store_len,
                    ));
                }
            }
            continue;
        }
        if ref_count == store_len {
            // Fast path: every slot is referenced (dense store).
            continue;
        }
        // Build a bitset of referenced slots.
        let mut ref_set = vec![false; store_len];
        for &s in referenced {
            if (s as usize) < store_len {
                ref_set[s as usize] = true;
            }
        }
        for (slot, referenced) in ref_set.iter().enumerate().take(store_len) {
            if !referenced {
                return Err(format!(
                    "invariant 4 (garbage-freedom) violation at boundary marginal level {} \
                     (non-marginal parent {}, side {:?}): slot {} is not \
                     referenced by any parent ref (store_len={}, referenced={})",
                    v.idx(),
                    parent.idx(),
                    side,
                    slot,
                    store_len,
                    ref_count,
                ));
            }
        }
    }

    Ok(())
}

/// Invariant 10 in the weighted domain. Weighted values are not deduped — a weighted
/// store is full width and a parent's reference is the node's own index — so
/// the integer uniqueness claim does not apply and two nodes may legitimately
/// carry the same value. What must hold instead is the identity that bare-slot
/// encoding rests on: the column has exactly one slot per node of the level.
/// A level whose store was freed as subsumed reports width 0 and holds no
/// column, which satisfies it trivially.
fn check_weight_column_is_full_width(tdd: &Tdd, left_idx: usize) -> Result<(), String> {
    let width = tdd.levels[left_idx].width();
    let stored = stored_slot_count(tdd, left_idx);
    if stored != width {
        return Err(format!(
            "invariant 10 (weighted) violation at weight-marginal level {left_idx}: the store holds \
             {stored} slots for a level of width {width} — a weighted column is one \
             slot per node",
        ));
    }
    Ok(())
}

/// Invariant 10: at every marginal level, all slot count keys are pairwise distinct.
/// The count is the anonymous identity of a marginal node, so two slots with
/// equal counts are the same node stored twice. invariant 10 is enforced at birth by
/// `dedup_fresh_store` for stores the marginalize pass builds, and at post-tagger
/// slot-prune (`prune_value_slots`) for apply-emit-born stores. This check is
/// a postcondition verifier, not a trigger for a rewrite pass.
pub fn check_slot_count_uniqueness(tdd: &Tdd) -> Result<(), String> {
    for (left_idx, level) in tdd.levels.iter().enumerate() {
        if level.is_weight_marginal() {
            check_weight_column_is_full_width(tdd, left_idx)?;
            continue;
        }
        let Some(counts) = level.marginal_counts() else {
            continue;
        };
        check_store_counts(counts, level.marginal_counts_big())
            .map_err(|e| format!("invariant 10 (slot count uniqueness) violation at marginal level {left_idx}: {e}"))?;
    }
    Ok(())
}

/// Invariant 10 for one integer store: the counts are pairwise distinct,
/// every overflow sentinel has its entry in the big table, and the big table
/// holds nothing else. The store-birth tests call it on a store before any
/// level holds it.
pub fn check_store_counts(counts: &[u128], big: Option<&BigSide>) -> Result<(), String> {
    let mut key_to_slot: FxHashMap<Count, usize> = FxHashMap::default();
    let mut sentinels = 0usize;
    for i in 0..counts.len() {
        let has_big = big.and_then(|b| b.get(i)).is_some();
        if counts[i] == u128::MAX && !has_big {
            return Err(format!("slot {i} holds the overflow sentinel with no marginal_counts_big entry"));
        }
        sentinels += usize::from(counts[i] == u128::MAX);
        let key = count_key_at(counts, big, i);
        if let Some(&prev) = key_to_slot.get(&key) {
            return Err(format!("slots {prev} and {i} carry equal counts"));
        }
        key_to_slot.insert(key, i);
    }
    // Other direction of the same invariant: the sparse overflow table is
    // keyed by slot, so an entry whose fast cell is no longer the sentinel
    // is dead weight and a stale value a later rekey would carry forward.
    // The loop above proved every sentinel has an entry; equal counts then
    // prove there are no extras.
    let entries = big.map_or(0, |b| b.len());
    if entries != sentinels {
        return Err(format!(
            "marginal_counts_big holds {entries} entries for {sentinels} overflow slots — \
             a stale entry at a slot that no longer overflows",
        ));
    }
    Ok(())
}

/// A marginal level whose parent is marginal holds no value store. The parent's
/// aggregate is all a reader above can reach, so the marginalize step frees
/// each child's store as the parent becomes marginal
/// (`marginal::free_subsumed_marginal_children`) and no later pass refills it.
/// A weight-marginal leaf is exempt: its column is the pinned leaf cache
/// (invariant 11), live whatever its parent is.
pub fn check_subsumed_stores_empty(tdd: &Tdd) -> Result<(), String> {
    let bad = subsumed_marginal_data_violations(tdd);
    if bad.is_empty() {
        return Ok(());
    }
    let bad: Vec<usize> = bad.iter().map(|v| v.idx()).collect();
    Err(format!(
        "marginal levels {bad:?} under a marginal parent still hold value slots",
    ))
}

/// Full canonical form: invariants 7, 8, 9 and 10 together, and no value
/// store under a marginal parent.
///
/// Valid at the fixpoint of twin contraction, canonicalization and pair
/// fusion, followed by the slot prune — in practice on a freshly minimized
/// diagram immediately after a full `fuse_pairs` sweep and `prune_value_slots`.
pub fn check_marginal_canonical_form(tdd: &Tdd) -> Result<(), String> {
    check_inline_discipline(tdd)?;
    check_pair_fusion_saturation(tdd, None)?;
    check_twin_canonicality(tdd)?;
    check_slot_count_uniqueness(tdd)?;
    check_subsumed_stores_empty(tdd)
}

/// Debug-only enforcement of invariant 8 at the one moment it is guaranteed:
/// immediately after a `fuse_pairs` / `fuse_pairs_at_parents` sweep, with the
/// same parent filter the sweep used. Compiled out of release builds.
///
/// # Panics
///
/// Panics if a same-structural-child pair the sweep should have fused survives.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_pair_fusion_saturated(tdd: &Tdd, filter: Option<&[VtreeIdx]>, label: &str) {
    if tdd.weights().is_some() {
        return;
    }
    if let Err(e) = check_pair_fusion_saturation(tdd, filter) {
        panic!("pair-fusion saturation violated immediately after pair_fusion [{label}]: {e}");
    }
}

/// Invariant 11: every weight-marginal vtree leaf advertises exactly
/// `LEAF_WIDTH` slots, and — when its column is installed — that column equals
/// the `leaf_val` triple in `LeafLabel` order.
///
/// This is the one invariant that makes bare leaf-label refs and `ValueRef::Slot`
/// refs interchangeable at a leaf, which is what lets `marginalize_leaf_weighted`
/// flip a leaf marginal without rewriting a single parent ref. Every pass that
/// could break it (slot-prune compaction, duplicate resolution twin-fold minting, weighted
/// pair fusion allocation, subsumption reclaim) declines to touch leaves; the check
/// is run on a hot, frequently-run path (slot-prune entry) so a regression in
/// any of them surfaces immediately instead of as a silently low weighted count.
///
/// Four things are checked, in the order a breakage shows up:
///   1. the level advertises `LEAF_WIDTH` slots (catches a `weight_width`
///      bump — how the weighted slot mint records a
///      minted slot);
///   2. No parent ref into the leaf names a slot ≥ `LEAF_WIDTH` (catches a minted
///      ref that outlived the width, and is the check that fails closest to the
///      real damage: a `Slot(3)` ref is decoded by every label-first reader —
///      `query::count`, `query::sat`, `validate`, `duplicate_pair_resolve` — as
///      `LeafLabel::from_idx(3)`, the never-satisfied `ZERO` sentinel, so the
///      models under it vanish with no error anywhere, and `prune_unreachable`
///      indexes the neighbouring level's remap window with it);
///   3. the installed column equals the `leaf_val` triple in label order
///      (catches compaction / erasure / reordering);
///   4. Every bare leaf-side ref is the canonical slot of its value class
///      (`leaf_canon_map`) — catches a site
///      that creates a leaf-side ref and skips the canon pass. That is a size
///      regression rather than a wrong count (a non-canonical ref still resolves
///      to the right value), so it has no other symptom: without this check the
///      twin cascade would just quietly stop firing on the affected leaves.
///      Exact domain only, and only once the column is installed — the canon
///      partition is undefined otherwise.
///
/// `Ok(())` whenever the diagram carries no weight store.
#[cfg(debug_assertions)]
pub fn check_leaf_columns_pinned(tdd: &Tdd) -> Result<(), String> {
    use crate::diagram::semiring::weight_key;
    use crate::diagram::{leaf_canon_map, LeafLabel, LEAF_WIDTH};
    use crate::value::slots::referenced_marginal_slots;
    use crate::vtree::VtreeNode;
    let Some(ws) = tdd.weights.as_ref() else {
        return Ok(());
    };
    let mut scratch = RefSlotScratch::default();
    for i in 0..tdd.levels.len() {
        let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(VtreeIdx(i as u32)) else { continue };
        if !tdd.levels[i].is_weight_marginal() {
            continue;
        }
        if tdd.levels[i].width() != LEAF_WIDTH {
            return Err(format!(
                "weight-marginal leaf level {i} advertises {} slots, not `LEAF_WIDTH`",
                tdd.levels[i].width()
            ));
        }
        if let Some(parent) = tdd.vtree.node(VtreeIdx(i as u32)).parent() {
            let side = match tdd.vtree.node(parent) {
                VtreeNode::Internal { left, .. } if left.idx() == i => ChildSide::Left,
                _ => ChildSide::Right,
            };
            let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut scratch);
            if let Some(&last) = refs.last()
                && (last as usize) >= LEAF_WIDTH
            {
                return Err(format!(
                    "weight-marginal leaf level {i} is referenced at slot {last} — outside \
                     the label range, so every label-first reader decodes it as the zero \
                     sentinel and drops that branch's mass"
                ));
            }
            // #4 — canonicality. Every surviving ref must already name the
            // smallest slot of its value class; anything else means some site
            // minted a leaf-side ref without running
            // `canonicalize_leaf_refs_at_parent`.
            if !ws.is_log()
                && let Some(col) = ws.level(i)
            {
                let canon = leaf_canon_map(col);
                for &s in refs {
                    // Out-of-range refs are check #2's report, not ours.
                    if (s as usize) < LEAF_WIDTH && canon[s as usize] != s {
                        return Err(format!(
                            "weight-marginal leaf level {i} is referenced at a non-canonical \
                             slot {s} (canonical slot for that value is {}) — a leaf-side ref \
                             was created without the equal-value canon pass, so the twin \
                             cascade cannot fire there",
                            canon[s as usize]
                        ));
                    }
                }
            }
        }
        let Some(col) = ws.level(i) else { continue };
        if col.len() != LEAF_WIDTH {
            return Err(format!(
                "weight-marginal leaf level {i} column holds {} values — it was \
                 compacted, erased or appended to",
                col.len()
            ));
        }
        for (k, slot_val) in col.iter().enumerate() {
            if weight_key(slot_val) != weight_key(&ws.leaf_val(var, LeafLabel::from_idx(k))) {
                return Err(format!(
                    "weight-marginal leaf level {i} column slot {k} is not the \
                     label-ordered `leaf_val` cache"
                ));
            }
        }
    }
    Ok(())
}

pub use super::marginal_counts::{assert_model_count_preserved, model_count_snapshot};
pub(crate) use super::marginal_counts::subsumed_marginal_data_violations;

#[cfg(test)]
mod tests;
