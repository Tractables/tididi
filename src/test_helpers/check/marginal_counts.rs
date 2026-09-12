//! Count-preservation brackets and the store-level slot checks.
//!
//! Sibling of `marginal.rs`, which holds the structural invariants; these are the
//! ones stated in terms of the values a marginal store holds.

use num_bigint::BigUint;

use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

// ── Count-preservation localizer ─────────────────────────────────────────
//
// A *count-neutral* marginal rewrite — pair fusion, contract's marginal pass,
// the marginal-context expansion in `restructure::relevel` — must leave the
// diagram's model count unchanged: it re-encodes / merges
// marginal nodes but represents the same set of models. `model_count_snapshot` /
// `assert_model_count_preserved` bracket one such rewrite and panic, naming the op, when
// the count moved. Each snapshot is a full `model_count`, so the caller decides
// where (and whether) to place the pair.

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
            "count-neutral op `{op}` changed the model count{factor}\n  \
             before = {before}\n  after  = {after}"
        );
    }
}

#[cfg(test)]
mod tests;
