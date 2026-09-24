//! The store-level slot checks: sibling of `marginal.rs`, which holds the
//! structural invariants; these are stated in terms of the values a marginal
//! store holds.

use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

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
            && lvl.slot_count() != 0
            && !vtree.node(VtreeIdx(i as u32)).is_leaf();
        if has_int || has_big || has_wt {
            bad.push(VtreeIdx(i as u32));
        }
    }
    bad
}

#[cfg(test)]
#[path = "tests/marginal_counts/mod.rs"]
mod tests;
