use crate::test_helpers::{toy, BIG};
use super::*;
use crate::engine::Engine;

/// Invariant 8 negative: two pairs sharing left x=0 with distinct marginal slots is a
/// fusable group — saturation must reject it (and so must the full check).
#[test]
fn detects_unfused_same_structural_child_group() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let err = check_pair_fusion_saturation(&tdd, None).unwrap_err();
    assert!(err.contains("invariant 8"), "wrong violation: {err}");
    assert!(check_marginal_canonical_form(&tdd).is_err());
}

/// Invariant 8 positive + full form: after `fuse_pairs` + `prune_value_slots` the
/// same diagram is canonical — the fused slot survives alone, orphans
/// collected.
#[test]
fn canonical_form_holds_after_p_fusion_and_slot_prune() {
    let eng = Engine::new();
    let mut tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let stats = crate::reduce::contract::pair_fusion::fuse_pairs(&eng, &mut tdd).unwrap();
    assert_eq!(stats.fusion_groups, 1);
    let v = tdd.levels.iter().position(|l| l.is_marginal()).unwrap();
    let slots_after_fusion = tdd.levels[v].marginal_counts().unwrap().len();
    crate::reduce::slot_prune::prune_value_slots(&eng, &mut tdd);
    assert_eq!(
        tdd.levels[v].marginal_counts().unwrap().len(),
        slots_after_fusion - 2,
        "both pre-fusion slots are orphans"
    );
    check_marginal_canonical_form(&tdd).unwrap();
}

/// Invariant 8 filter: a violation at a parent OUTSIDE the filter is not reported
/// (mirrors `fuse_pairs_at_parents` semantics).
#[test]
fn filter_skips_other_parents() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    assert!(check_pair_fusion_saturation(&tdd, Some(&[])).is_ok());
}

/// Invariant 9 negative: two nodes with identical pair lists are unmerged twins.
#[test]
fn detects_unmerged_twins() {
    let tdd = toy(vec![BIG + 1], &[&[(0, 0)], &[(0, 0)]]);
    let err = check_twin_canonicality(&tdd).unwrap_err();
    assert!(err.contains("invariant 9"), "wrong violation: {err}");
}

/// Invariant 10 negative: two slots carrying equal counts should have been
/// canonicalized onto one slot (and the orphan collected).
#[test]
fn c3_detects_duplicate_counts() {
    let tdd = toy(vec![BIG, BIG], &[&[(0, 0), (1, 1)]]);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("invariant 10"), "wrong violation: {err}");
}

/// Invariant 10 covers all slots: a stale duplicate is a violation too — and
/// `prune_value_slots` is the fix (collects the orphan, after which invariant 10 holds).
#[test]
fn c3_rejects_stale_duplicate_until_slot_prune() {
    let eng = &crate::engine::Engine::new();
    let mut tdd = toy(vec![BIG, BIG], &[&[(0, 0)]]);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("invariant 10"), "wrong violation: {err}");
    crate::reduce::slot_prune::prune_value_slots(eng, &mut tdd);
    check_slot_count_uniqueness(&tdd).unwrap();
}

/// invariants 4 and 7 negative: an inline-eligible count parked in a REFERENCED slot.
#[test]
fn c4_rejects_inline_eligible_slot() {
    let tdd = toy(vec![5], &[&[(0, 0)]]);
    let err = check_marginal_canonical_form(&tdd).unwrap_err();
    assert!(err.contains("invariant 7"), "wrong violation: {err}");
}

/// invariants 4 and 7 exemption: an inline-eligible count on an UNREFERENCED slot is
/// fine — count vectors are never shrunk, so a slot whose refs were all
/// retagged inline legitimately retains its small count.
#[test]
fn c4_ignores_stale_inline_eligible_slot() {
    let tdd = toy(vec![BIG, 5], &[&[(0, 0)]]);
    check_inline_discipline(&tdd).unwrap();
}

// ── Invariant 4 garbage-freedom (check_no_orphan_slots) ───────────────────────────

/// Negative: a boundary store with an extra unreferenced slot (slot 1 is
/// orphaned — only slot 0 is referenced). `check_no_orphan_slots` must
/// return an `Err` naming the orphan.
#[test]
fn c4_orphan_slot_detects_unreferenced_boundary_slot() {
    // Slot 0 is referenced (parent has right-ref = 0).
    // Slot 1 is orphaned — no parent ref points to it.
    let tdd = toy(vec![BIG + 10, BIG + 20], &[&[(0, 0)]]);
    let err = check_no_orphan_slots(&tdd).unwrap_err();
    assert!(
        err.contains("invariant 4"),
        "wrong violation kind (expected invariant 4): {err}"
    );
    assert!(
        err.contains('1') || err.contains("slot 1"),
        "error must mention orphan slot 1: {err}"
    );
}

/// Positive: after `prune_value_slots`, the orphan is removed and
/// `check_no_orphan_slots` passes. invariant 10 must also hold.
#[test]
fn c4_orphan_slot_cleared_after_prune() {
    let eng = &crate::engine::Engine::new();
    let mut tdd = toy(vec![BIG + 10, BIG + 20], &[&[(0, 0)]]);
    // Pre-condition: invariant 4 violated.
    assert!(
        check_no_orphan_slots(&tdd).is_err(),
        "pre-prune: expected invariant 4 violation"
    );
    crate::reduce::slot_prune::prune_value_slots(eng, &mut tdd);
    // Post-condition: invariant 4 passes.
    check_no_orphan_slots(&tdd).unwrap();
    // invariant 10 must also hold after prune.
    check_slot_count_uniqueness(&tdd).unwrap();
}

// ── No value store under a marginal parent (check_subsumed_stores_empty) ──

/// A marginal level under a marginal parent still holding counts and an
/// overflow entry fails the check and the full canonical form; the freeing
/// step the marginalize path runs on the parent empties both and the check
/// passes, with the parent's own store untouched.
#[test]
fn subsumed_store_detected_and_freed() {
    use crate::diagram::{NodeIdx, Tdd, TddLevel, TddNodeId};
    use crate::marginal::free_subsumed_marginal_children;
    use crate::vtree::Vtree;
    use num_bigint::BigUint;
    use std::sync::Arc;

    let vtree = Arc::new(Vtree::balanced(4));
    let root = vtree.root();
    let (_v_left, v_right) = vtree.children(root);
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let big = [(1u32, BigUint::from(1_000_000_u64))].into_iter().collect();
    levels[v_right.idx()].become_marginal(vec![42, u128::MAX, 99], Some(big));
    levels[root.idx()].become_marginal(vec![100, 200], None);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, output);

    let err = check_subsumed_stores_empty(&tdd).unwrap_err();
    assert!(err.contains(&v_right.idx().to_string()), "must name the level: {err}");
    assert!(check_marginal_canonical_form(&tdd).is_err());

    free_subsumed_marginal_children(&mut tdd.levels, &vtree, root, None);
    check_subsumed_stores_empty(&tdd).unwrap();
    check_marginal_canonical_form(&tdd).unwrap();
    let deep = &tdd.levels[v_right.idx()];
    assert!(deep.is_marginal(), "the level stays marginal");
    assert_eq!(deep.width(), 0);
    assert!(deep.marginal_counts_big().is_none());
    assert_eq!(tdd.levels[root.idx()].marginal_counts().unwrap(), &[100, 200]);
}

// ── The same checks in the weighted domain ───────────────────────────────
//
// A weighted diagram stores its marginal values in the external `WeightStore`
// instead of the level, and never dedups them, so invariant 10 and invariant 4 claim something
// different there — see `check_weight_column_is_full_width` and the weighted
// arm of `check_no_orphan_slots`. Invariant 9 is about pair multisets and claims the
// same thing in both domains.

use crate::diagram::ValueRef;
use crate::diagram::{RationalWeights, WeightVal};
use crate::test_helpers::{rat, toy_weighted};
use crate::vtree::{Vtree, VtreeNode};
use crate::diagram::{Arithmetic, WeightStore};

fn weighted_store() -> WeightStore {
    WeightStore::new(
        RationalWeights::from_weights(&[(rat(2, 5), rat(3, 11)), (rat(1, 3), rat(-4, 9))]),
        Arithmetic::ExactRational,
    )
}

/// The marginal level `toy_weighted` builds: `balanced(3)`'s root's right child.
fn weighted_marginal_level() -> crate::vtree::VtreeIdx {
    let vtree = Vtree::balanced(3);
    match vtree.node(vtree.root()) {
        VtreeNode::Internal { right, .. } => *right,
        _ => unreachable!("balanced(3) root is internal"),
    }
}

/// Invariant 10 positive, weighted: two slots may carry the same value. Nothing dedups a
/// weighted column, and a parent references a node by its own index, so equal
/// values are not two spellings of one node.
#[test]
fn c3_weighted_allows_equal_values() {
    let tdd = toy_weighted(
        weighted_store(),
        vec![rat(3, 7), rat(3, 7)],
        &[&[(0, 0), (1, 1)]],
    );
    check_slot_count_uniqueness(&tdd).unwrap();
}

/// Invariant 10 negative, weighted: a column short of the level's width breaks the
/// slot-index-is-node-index identity the bare-slot encoding rests on.
#[test]
fn c3_weighted_detects_short_column() {
    let mut tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)]]);
    let marginal = weighted_marginal_level();
    // Claim a third node without giving it a slot.
    tdd.levels[marginal.idx()].become_marginal_weighted(3);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("invariant 10 (weighted)"), "wrong violation: {err}");
}

/// Invariant 4 positive, weighted: an unreferenced slot is not garbage in a weighted
/// column — nothing prunes one, and the column stays full width.
#[test]
fn c4_weighted_allows_unreferenced_slots() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)]]);
    check_no_orphan_slots(&tdd).unwrap();
}

/// Invariant 4 negative, weighted: a reference past the end of the column is a dangling
/// slot, which the bare-slot encoding cannot tolerate.
#[test]
fn c4_weighted_detects_dangling_reference() {
    let dangling = ValueRef::slot_raw(5);
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, dangling)]]);
    let err = check_no_orphan_slots(&tdd).unwrap_err();
    assert!(err.contains("invariant 4"), "wrong violation: {err}");
    assert!(err.contains("past the end"), "wrong invariant 4 arm: {err}");
}

/// Invariant 9, weighted: two root nodes with identical pair lists are unmerged twins
/// there too — the check reads pair multisets, which say nothing about the
/// value domain.
#[test]
fn weighted_detects_unmerged_twins() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)], &[(0, 0)]]);
    let err = check_twin_canonicality(&tdd).unwrap_err();
    assert!(err.contains("invariant 9"), "wrong violation: {err}");
}

/// The weighted values themselves are untouched by the checks above.
#[test]
fn weighted_column_survives_the_checks() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0), (1, 1)]]);
    let ws = tdd.weights().expect("weighted diagram");
    let col = ws.level(weighted_marginal_level().idx()).expect("column installed");
    use crate::diagram::semiring::weight_key;
    let got: Vec<_> = col.iter().map(weight_key).collect();
    let want: Vec<_> = [rat(3, 7), rat(1, 2)]
        .into_iter()
        .map(|r| weight_key(&WeightVal::exact(r)))
        .collect();
    assert!(got == want, "the checks must not disturb the weighted column");
}
