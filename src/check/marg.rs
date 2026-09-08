//! Structural invariant checks for marginal (post-marginalization) TDDs:
//! slot canonicalization, orphan/twin/fusion-redex detection, P-saturation,
//! and model-count-preservation assertions.
//!
//! # The marginal invariants
//!
//! This module doc is the ONE definition of the invariant names the marginal
//! machinery uses; every other site cites `validate::marg` rather than
//! restating them.
//!
//! **I1 — marginality is permanent.** Once a vtree node is marginalized it
//! stays marginalized: no compile or reduction step turns a marginal level back
//! into a structural one. (Its mass may later roll *up* into a marginalized
//! parent, but the node itself never un-marginalizes.)
//!
//! **I2 — inline discipline.** Among slots actually REFERENCED from a
//! non-marginal parent's marg-side pair refs, none holds an inline-eligible
//! count: the writers encode such counts directly into the parent ref. This is
//! the same statement as C4 below.
//!
//! At the fixpoint of contract-twins → canonicalize-by-count → p-fusion, every
//! boundary-marginal level `v` (a marginal child of a non-marginal parent `P`)
//! satisfies a canonical form. A parent node's pairs case-split over pairwise
//! mutex non-marginal children, so each node is a *function* from distinct
//! x-children to marginal counts:
//!
//! **C1 — P-saturation.** Within a parent node, no two pairs share the same
//! non-marginal-side child ref. Established by `apply_p_fusion`, which fuses
//! every same-x group; a later twin merge can recreate one, so C1 holds
//! immediately after a fusion sweep and not at arbitrary points. A parent whose
//! OTHER child is also marginal is exempt: there is no non-marginal side there,
//! and one sweep is not a fixpoint across both boundaries (see
//! [`check_p_saturation`]).
//!
//! **C2 — twin canonicality.** No two nodes at the parent level have equal pair
//! multisets; such nodes are twins and must have been merged.
//!
//! **C3 — slot count uniqueness.** At every marginal level all slot counts are
//! pairwise distinct — the count is the anonymous identity of a marginal node.
//! A store built by marginalization is C3 from birth (`dedup_fresh_store`
//! merges duplicates at mint time); a store born from an apply's emit reaches
//! C3 at the slot-prune after tagging (`prune_marg_slots`, which also collects
//! orphan slots). Dedup at the emit site is forbidden on that path.
//!
//! **C4 — inline discipline**, the same statement as I2. No REFERENCED slot
//! holds an inline-eligible count; small counts live inline in the parent refs.
//! Stale slots, and deep-marginal or root levels that have no parent refs, are
//! exempt.
//!
//! All checks decode marg-side refs with `MargRef::from_raw`, the post-tagger
//! encoding; they do not apply before the tagger has run.

use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::diagram::{InputPair, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::marg_slots::{
    boundary_marginal_levels, count_key_at, referenced_marg_slots, ChildSide, CountKey, RefSlotScratch,
};

/// TDD-wide marginal invariant check (**I2**, inline discipline — see the
/// module doc): among slots actually REFERENCED from a non-marginal parent's
/// marg-side pair refs, no slot holds an inline-eligible count
/// (`≤ MARG_INLINE_MAX` with no big-overflow entry).
///
/// Scope (mirrors C3):
/// - **Unreferenced/stale slots are exempt** — count vectors are never shrunk
///   (the no-shrink invariant is described in `types.rs`), so a slot whose
///   refs were all retagged inline legitimately retains its small count.
/// - **Deep-marginal and root levels are exempt** — a marginal level whose
///   parent is itself marginal (or that has no parent) carries no parent pair
///   refs to inline into; its counts (e.g. the final model count) must live in
///   explicit slots regardless of magnitude.
///
/// Slot count uniqueness lives in C3 (`check_slot_count_uniqueness`): with
/// `prune_marg_slots` collecting orphaned slots, ALL slots at a marginal level
/// must carry pairwise-distinct counts at the canonical-form fixpoint.
pub fn check_tdd_marg_invariants(tdd: &Tdd) -> Result<(), String> {
    let mut slots = RefSlotScratch::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        let vlevel = &tdd.levels[v.idx()];
        let Some(counts) = vlevel.marginal_counts.as_ref() else { continue };
        let big = vlevel.marginal_counts_big.as_ref();
        for &s in referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut slots) {
            let i = s as usize;
            if i >= counts.len() {
                continue; // Out-of-range refs are the ref-bounds check's job.
            }
            let has_big = big.and_then(|b| b.get(i)).is_some();
            if counts[i] <= crate::diagram::marg_inline_max() as u128 && !has_big {
                return Err(format!(
                    "I2 violation at marg level {} (parent {}) slot {}: referenced count {} \u{2264} marg_inline_max ({}); must be inline at parent refs",
                    v.idx(), parent.idx(), i, counts[i], crate::diagram::marg_inline_max(),
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

/// C1: within each boundary-marginal parent node, every non-marginal-side
/// child ref appears in at most one pair. `filter`, when given, restricts the
/// walk to those parent vtree nodes (mirroring `apply_p_fusion_at_parents`).
///
/// BOTH-MARGINAL PARENTS ARE OUT OF SCOPE. When *both* children of `parent`
/// are marginal, `boundary_marginal_levels` yields the parent twice (once per
/// child) and the "x" ref C1 would key on is itself a marg ref, not a
/// non-marginal child — the premise of the invariant (pairwise-mutex explicit
/// children) does not apply. Such a parent is also not a fusion fixpoint after
/// a single sweep: the second boundary's fusion can hand two groups the SAME
/// marg ref by design (count-keyed slot sharing / equal inline counts, see
/// `apply_p_fusion_inner` Phase 2), recreating a same-x group at the first
/// boundary. Those duplicate pairs are sound under multiset pair lists and the
/// next sweep closes them (the explicit side's inline marker is raised by then,
/// so `collect_fusion_plans` takes its opaque-key path). Skipping keeps the
/// check faithful to what C1 actually claims.
pub(crate) fn check_p_saturation(tdd: &Tdd, filter: Option<&[VtreeIdx]>) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if let Some(f) = filter {
            if !f.contains(&parent) {
                continue;
            }
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
                        "C1 (P-saturation) violation at parent level {} (marg child {}, side {:?}): \
                         node {} holds \u{2265}2 pairs sharing non-marginal child ref {:#x}",
                        parent.idx(), v.idx(), side, n, x,
                    ));
                }
            }
        }
    }
    Ok(())
}



/// C2: no two non-leaf nodes at any C2-canonicalized level carry equal pair
/// multisets — equal-pair-list nodes are twins and must have merged.
///
/// The level set is `contract::content_twin::c2_scan_levels`, the same one the
/// merge itself walks (every explicit level of a marginalized diagram, empty
/// otherwise), so the checker is exactly the merge's postcondition and cannot
/// drift from it.
pub(crate) fn check_twin_canonicality(tdd: &Tdd) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    for parent in crate::reduce::contract::content_twin::c2_scan_levels(tdd) {
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
                    "C2 (twin canonicality) violation at parent level {}: nodes {} and {} \
                     have identical pair multisets ({} pairs) — unmerged twins",
                    parent.idx(), m, n, key.len(),
                ));
            }
            key_to_node.insert(key, n);
        }
    }
    Ok(())
}

/// C4 garbage-freedom: post-fixpoint-including-prune, boundary stores contain
/// exactly the referenced slots; no orphaned or dead-store entries remain.
///
/// Two garbage classes (mirroring `prune_marg_slots`):
///
/// 1. **Boundary orphans** — every slot index in `0..store_len` at a boundary
///    marginal level (marginal child of a non-marginal parent) must be
///    referenced by at least one parent marg-side ref.  An unreferenced slot is
///    a boundary orphan that `prune_marg_slots` should have collected.
///
/// 2. **Dead deep stores** — a marginal level whose vtree-parent is also
///    marginal holds a dead store (consumed at cascade-marginalize time).
///    After prune the store must be empty (`len == 0`).
///
/// **Exemptions** (mirrors `slot_prune.rs`):
/// - The output level's store (`tdd.output.vtree`) is never touched — it holds
///   the final count or a component sub-TDD's count.
/// - The root marginal level (no parent) is also exempt (its store is the
///   final model-count store, not consumed by any parent).
pub fn check_no_orphan_slots(tdd: &Tdd) -> Result<(), String> {
    let out_v = tdd.output.vtree;
    let mut slots = RefSlotScratch::default();

    // --- (1) Dead deep stores must be empty ---
    for i in 0..tdd.levels.len() {
        if !tdd.levels[i].is_marginal() {
            continue;
        }
        let v = VtreeIdx(i as u32);
        if v == out_v {
            continue;
        }
        let Some(parent) = tdd.vtree.node(v).parent() else { continue };
        if !tdd.levels[parent.idx()].is_marginal() {
            continue; // boundary level: checked below
        }
        // Deep store: parent is also marginal.
        let counts = tdd.levels[i].marginal_counts.as_ref().unwrap();
        if !counts.is_empty() {
            return Err(format!(
                "C4 (garbage-freedom) violation at marg level {} (deep store, \
                 marginal parent {}): expected empty store after prune, \
                 found {} non-empty slots",
                i,
                parent.idx(),
                counts.len(),
            ));
        }
    }

    // --- (2) Boundary stores must have no orphan slots ---
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        let store_len = tdd.levels[v.idx()]
            .marginal_counts
            .as_ref()
            .map_or(0, |c| c.len());
        if store_len == 0 {
            continue; // nothing to check
        }
        let referenced =
            referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut slots);
        let ref_count = referenced.len();
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
        for slot in 0..store_len {
            if !ref_set[slot] {
                return Err(format!(
                    "C4 (garbage-freedom) violation at boundary marg level {} \
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

/// C3: at every marginal level, ALL slot count keys are pairwise distinct.
/// The count is the anonymous identity of a marginal node, so two slots with
/// equal counts are the same node stored twice. C3 is enforced at birth by
/// `dedup_fresh_store` for compile_marginalize-path stores, and at post-tagger
/// slot-prune (`prune_marg_slots`) for apply-emit-born stores. This check is
/// a postcondition verifier, not a trigger for a rewrite pass.
pub fn check_slot_count_uniqueness(tdd: &Tdd) -> Result<(), String> {
    let mut key_to_slot: FxHashMap<CountKey, usize> = FxHashMap::default();
    for (li, level) in tdd.levels.iter().enumerate() {
        let Some(counts) = level.marginal_counts.as_ref() else { continue };
        let big = level.marginal_counts_big.as_ref();
        key_to_slot.clear();
        let mut sentinels = 0usize;
        for i in 0..counts.len() {
            let has_big = big.and_then(|b| b.get(i)).is_some();
            if counts[i] == u128::MAX && !has_big {
                return Err(format!(
                    "C3 walk at marg level {li}: slot {i} holds the OVERFLOW \
                     sentinel with no marginal_counts_big entry",
                ));
            }
            sentinels += usize::from(counts[i] == u128::MAX);
            let key = count_key_at(counts, big, i);
            if let Some(&prev) = key_to_slot.get(&key) {
                return Err(format!(
                    "C3 (slot count uniqueness) violation at marg level {li}: \
                     slots {prev} and {i} carry equal counts",
                ));
            }
            key_to_slot.insert(key, i);
        }
        // Other direction of the same invariant: the sparse overflow table is
        // keyed by slot, so an entry whose fast cell is no longer the sentinel
        // is dead weight AND a stale value a later rekey would carry forward.
        // The loop above proved every sentinel has an entry; equal counts then
        // prove there are no extras.
        let entries = big.map_or(0, |b| b.len());
        if entries != sentinels {
            return Err(format!(
                "C3 walk at marg level {li}: marginal_counts_big holds {entries} \
                 entries for {sentinels} OVERFLOW slots — stale entry at a slot \
                 that no longer overflows",
            ));
        }
    }
    Ok(())
}

/// Full canonical-form check (see the module doc): C4/I2
/// (`check_tdd_marg_invariants`) + C1 + C2 + C3. Valid at the contract → canon → p-fusion → slot-prune fixpoint — in
/// practice: on a freshly minimized TDD immediately after a full
/// `apply_p_fusion` sweep followed by `prune_marg_slots`.
pub fn check_marg_canonical_form(tdd: &Tdd) -> Result<(), String> {
    check_tdd_marg_invariants(tdd)?;
    check_p_saturation(tdd, None)?;
    check_twin_canonicality(tdd)?;
    check_slot_count_uniqueness(tdd)
}

/// No p-fusion redexes: at every boundary-marginal parent level, no node holds
/// two pairs sharing the same EXPLICIT-side child ref. This is the change-C name
/// for C1 (P-saturation): a p-fusion redex is exactly a same-explicit-different-
/// count pair group, so a TDD at the joint fixpoint of twin-contract + p-fusion
/// satisfies this. Delegates to `check_p_saturation` (the full-TDD C1 scan).
///
/// Returns `Err` on the first redex found (same format as C1).
pub fn check_no_fusion_redexes(tdd: &Tdd) -> Result<(), String> {
    // weighted mode: marginal levels carry no integer counts; skip the
    // count-reading checks.
    if tdd.weights().is_some() {
        return Ok(());
    }
    check_p_saturation(tdd, None)
}

/// No unmerged twins: no two distinct non-leaf nodes at a boundary-marginal
/// parent level carry identical pair multisets. This is the change-C name for
/// C2 (twin canonicality): any surviving pair of twins is a contraction redex.
/// Delegates to `check_twin_canonicality` (the full-TDD C2 scan).
///
/// Returns `Err` on the first twin pair found (same format as C2).
pub fn check_no_twins(tdd: &Tdd) -> Result<(), String> {
    // weighted mode: marginal levels carry no integer counts; skip the
    // count-reading checks.
    if tdd.weights().is_some() {
        return Ok(());
    }
    check_twin_canonicality(tdd)
}



/// Debug-only enforcement of C1 at the moments it is guaranteed: immediately
/// after an `apply_p_fusion` / `apply_p_fusion_at_parents` sweep (pass the same
/// parent filter the sweep used). Panics with the violation. Compiled out of
/// release builds; always on in debug (cargo test).
///
/// # Panics
///
/// Panics if P-saturation is violated (a same-left-child pair that the sweep
/// should have fused survives).
pub fn debug_assert_p_saturated(tdd: &Tdd, filter: Option<&[VtreeIdx]>, label: &str) {
    // weighted mode: marginal levels carry no integer counts; skip the
    // count-reading checks.
    if tdd.weights().is_some() {
        return;
    }
    if let Err(e) = check_p_saturation(tdd, filter) {
        panic!("(P)-saturation violated immediately after p_fusion [{label}]: {e}");
    }
}

// ── Count-preservation localizer ─────────────────────────────────────────
//
// A *count-neutral* marginal rewrite — p_fusion, contract's marginal pass,
// reexpand — must leave the TDD's model count unchanged: it re-encodes / merges
// marginal nodes but represents the same set of models. `mc_snapshot` /
// `mc_assert_preserved` bracket one such rewrite and panic, naming the op, when
// the count moved. Each snapshot is a full `model_count`, so the caller decides
// where (and whether) to place the pair.

// ── C3: slot count uniqueness ─────────────────────────────────────────────────

/// Check that a marginal store satisfies **C3** (each count value appears in at
/// most one slot). Returns `Ok(())` when all slot values are distinct, or
/// `Err(description)` naming the first duplicate pair found.
///
/// This is the constructor invariant for stores built by `dedup_fresh_store`
/// or through a seeded `SlotInterner` map, and also the postcondition for
/// apply-emit-born stores after `prune_marg_slots`. It is weaker than a full
/// `check_tdd_marg_invariants` sweep; use it in unit tests immediately after store
/// birth (or after slot-prune) to confirm C3 holds. Production code relies on
/// C3 being guaranteed by construction or slot-prune and does NOT call this on
/// every store.
#[cfg(test)]
pub(crate) fn check_store_counts_c3(
    counts: &[u128],
    big: Option<&crate::diagram::BigSide>,
) -> Result<(), String> {
    let mut seen: FxHashMap<CountKey, usize> = FxHashMap::default();
    for i in 0..counts.len() {
        let key = count_key_at(counts, big, i);
        if let Some(&first) = seen.get(&key) {
            return Err(format!(
                "C3 violation: slot {} and slot {} share the same count value ({:?})",
                first, i, key
            ));
        }
        seen.insert(key, i);
    }
    Ok(())
}

/// Marginal levels under a marginal parent that still hold per-node data.
///
/// A marginal level whose parent is also marginal is subsumed by the parent's
/// aggregate and unreachable from the root, so marginalization frees its
/// integer counts, big-overflow entries, and weighted slot carrier as the parent
/// marginalizes. Returns the levels where any of that data survived: each is
/// dead memory. A weight-marginal vtree leaf is exempt — its 3-slot column is
/// the compile-wide `WeightStore::leaf_val` cache that other diagrams decode
/// their leaf-label refs against, not per-diagram data.
pub fn subsumed_marginal_data_violations(tdd: &Tdd) -> Vec<VtreeIdx> {
    let vtree = &tdd.vtree;
    let mut bad = Vec::new();
    for i in 0..vtree.num_nodes() {
        if !tdd.levels[i].is_marginal() {
            continue;
        }
        let Some(parent) = vtree.node(VtreeIdx(i as u32)).parent() else {
            continue;
        };
        if !tdd.levels[parent.idx()].is_marginal() {
            continue;
        }
        let lvl = &tdd.levels[i];
        let has_int = lvl.marginal_counts.as_ref().is_some_and(|c| !c.is_empty());
        let has_big = lvl.marginal_counts_big.as_ref().is_some_and(|b| !b.is_empty());
        let has_wt = lvl.is_weight_marginal()
            && lvl.retired_marg_width != 0
            && !vtree.node(VtreeIdx(i as u32)).is_leaf();
        if has_int || has_big || has_wt {
            bad.push(VtreeIdx(i as u32));
        }
    }
    bad
}

/// Snapshot the TDD's model count for [`mc_assert_preserved`]. `None` in
/// weighted mode, where marginal levels carry no integer counts. Full
/// `model_count` cost — pair it around one count-neutral marginal rewrite at
/// a time.
pub fn mc_snapshot(tdd: &Tdd) -> Option<BigUint> {
    if tdd.weights().is_some() {
        return None;
    }
    Some(crate::query::model_count(tdd))
}

/// Assert the model count is unchanged vs a prior [`mc_snapshot`]. Panics with
/// the op label on mismatch. No-op when the
/// snapshot was `None` (check disabled).
///
/// # Panics
///
/// Panics if the current model count differs from `before` (a count-neutral op
/// changed the count). No-op when `before` is `None`.
pub fn mc_assert_preserved(tdd: &Tdd, before: Option<BigUint>, op: &str) {
    // weighted mode: marginal levels carry no integer counts; skip the
    // count-reading checks.
    if tdd.weights().is_some() {
        return;
    }
    let Some(before) = before else { return };
    let after = crate::query::model_count(tdd);
    if after != before {
        // Surface the multiplicative factor (×2 for the m139 doubler) when it
        // divides cleanly, to make the signature unmistakable in the panic.
        let factor = if before != BigUint::ZERO && &after % &before == BigUint::ZERO {
            format!(" (after = {}× before)", &after / &before)
        } else {
            String::new()
        };
        panic!(
            "count-neutral op `{op}` CHANGED the model count{factor}\n  \
             before = {before}\n  after  = {after}"
        );
    }
}


#[cfg(test)]
#[path = "marg_canonical_form_tests.rs"]
mod canonical_form_tests;

// ── Change-C: joint-fixpoint property tests ──────────────────────────────────
//
// The compile-driven property tests (property tests over `compile_cnf_mc` +
// the end-to-end cascade zero-footprint test), including their `make_cnf` /
// `brute_force_mc_raw` helpers, moved to `tests/tdd_validate_marg_compile.rs`
// (crate-split: `tididi` cannot depend on CNF parsing, which lives in the
// CNF front end, or on compilation, which lives in the downstream driver
// crate).
//
// The directed hand-built fixture for fusion-redex → twin is in contract.rs's
// test module so it can access the private `contract_all_twins_topdown` directly.





// ── Unit tests for C3 construction invariant ─────────────────────────────────
//
// Each test constructs a duplicate-prone store and asserts that after
// `dedup_fresh_store` (or SlotInterner) the result satisfies C3 immediately —
// no post-hoc canon pass required. Tests are authored for compilation; run
// with `cargo test` (no --include-ignored needed).
#[cfg(test)]
#[path = "marg_c3_tests.rs"]
mod c3_tests;
