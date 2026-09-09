//! Count-preservation brackets and the store-level slot checks.
//!
//! Sibling of `marginal.rs`, which holds the structural invariants; these are the
//! ones stated in terms of a marginal store's VALUES.

use num_bigint::BigUint;
#[cfg(test)]
use rustc_hash::FxHashMap;

use crate::diagram::Tdd;
#[cfg(test)]
use crate::reduce::slots::count_key_at;
#[cfg(test)]
use crate::value_fold::Count;
use crate::vtree::VtreeIdx;

// ── Count-preservation localizer ─────────────────────────────────────────
//
// A *count-neutral* marginal rewrite — pair_fusion, contract's marginal pass,
// reexpand — must leave the diagram's model count unchanged: it re-encodes / merges
// marginal nodes but represents the same set of models. `model_count_snapshot` /
// `assert_model_count_preserved` bracket one such rewrite and panic, naming the op, when
// the count moved. Each snapshot is a full `model_count`, so the caller decides
// where (and whether) to place the pair.

// ── Invariant 10: slot count uniqueness ─────────────────────────────────────────────────

/// Check that a marginal store satisfies **invariant 10** (each count value appears in at
/// most one slot). Returns `Ok(())` when all slot values are distinct, or
/// `Err(description)` naming the first duplicate pair found.
///
/// This is the constructor invariant for stores built by `dedup_fresh_store`
/// or through a seeded `SlotInterner` map, and also the postcondition for
/// apply-emit-born stores after `prune_value_slots`. It is weaker than a full
/// `check_inline_discipline` sweep; use it in unit tests immediately after store
/// birth (or after slot-prune) to confirm invariant 10 holds. Production code relies on
/// Invariant 10 being guaranteed by construction or slot-prune and does not call this on
/// every store.
#[cfg(test)]
pub(crate) fn check_store_counts_c3(
    counts: &[u128],
    big: Option<&crate::diagram::BigSide>,
) -> Result<(), String> {
    let mut seen: FxHashMap<Count, usize> = FxHashMap::default();
    for i in 0..counts.len() {
        let key = count_key_at(counts, big, i);
        if let Some(&first) = seen.get(&key) {
            return Err(format!(
                "invariant 10 violation: slot {} and slot {} share the same count value ({:?})",
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
        let has_int = lvl.marginal_counts().is_some_and(|c| !c.is_empty());
        let has_big = lvl.marginal_counts_big().is_some_and(|b| !b.is_empty());
        let has_wt = lvl.is_weight_marginal()
            && lvl.weight_width() != 0
            && !vtree.node(VtreeIdx(i as u32)).is_leaf();
        if has_int || has_big || has_wt {
            bad.push(VtreeIdx(i as u32));
        }
    }
    bad
}

/// Snapshot the diagram's model count for [`assert_model_count_preserved`]. `None` in
/// weighted mode, where marginal levels carry no integer counts. Full
/// `model_count` cost — pair it around one count-neutral marginal rewrite at
/// a time.
pub fn model_count_snapshot(tdd: &Tdd) -> Option<BigUint> {
    if tdd.weights().is_some() {
        return None;
    }
    Some(crate::query::model_count(tdd))
}

/// Assert the model count is unchanged vs a prior [`model_count_snapshot`]. Panics with
/// the op label on mismatch. No-op when the
/// snapshot was `None` (check disabled).
///
/// # Panics
///
/// Panics if the current model count differs from `before` (a count-neutral op
/// changed the count). No-op when `before` is `None`.
pub fn assert_model_count_preserved(tdd: &Tdd, before: Option<BigUint>, op: &str) {
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

// ── Change-C: joint-fixpoint property tests ──────────────────────────────────
//
// The compile-driven property tests (property tests over `compile_cnf_mc` +
// the end-to-end cascade zero-footprint test), including their `make_cnf` /
// `brute_force_mc_raw` helpers, moved to `tests/tdd_validate_marginal_compile.rs`
// which can parse CNF and compile; this crate does neither.
//
// The directed hand-built fixture for fusion-redex → twin is in contract.rs's
// test module so it can access the private `contract_all_twins_topdown` directly.

// ── Unit tests for invariant 10 construction invariant ─────────────────────────────────
//
// Each test constructs a duplicate-prone store and asserts that after
// `dedup_fresh_store` (or SlotInterner) the result satisfies invariant 10 immediately —
// no post-hoc canon pass required. Tests are authored for compilation; run
// with `cargo test` (no --include-ignored needed).
#[cfg(test)]
#[path = "marginal_uniqueness_tests.rs"]
mod uniqueness_tests;
