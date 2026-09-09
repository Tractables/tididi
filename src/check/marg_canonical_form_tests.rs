use crate::test_helpers::{toy, BIG};
use super::*;
use crate::engine::Engine;

/// F negative: two pairs sharing left x=0 with distinct marg slots is a
/// fusable group — saturation must reject it (and so must the full check).
#[test]
fn c1_detects_unfused_same_x_group() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let err = check_p_saturation(&tdd, None).unwrap_err();
    assert!(err.contains("F"), "wrong violation: {err}");
    assert!(check_marg_canonical_form(&tdd).is_err());
}

/// F positive + full form: after `fuse_pairs` + `prune_value_slots` the
/// same TDD is canonical — the fused slot survives alone, orphans
/// collected.
#[test]
fn canonical_form_holds_after_p_fusion_and_slot_prune() {
    let eng = Engine::new();
    let mut tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let stats = crate::reduce::contract::pair_fusion::fuse_pairs(&eng, &mut tdd).unwrap();
    assert_eq!(stats.fusion_groups, 1);
    let pruned = crate::reduce::slot_prune::prune_value_slots(&eng, &mut tdd);
    assert_eq!(pruned.slots_freed, 2, "both pre-fusion slots are orphans");
    check_marg_canonical_form(&tdd).unwrap();
}

/// F filter: a violation at a parent OUTSIDE the filter is not reported
/// (mirrors `fuse_pairs_at_parents` semantics).
#[test]
fn c1_filter_skips_other_parents() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    assert!(check_p_saturation(&tdd, Some(&[])).is_ok());
}

/// G negative: two nodes with identical pair lists are unmerged twins.
#[test]
fn c2_detects_unmerged_twins() {
    let tdd = toy(vec![BIG + 1], &[&[(0, 0)], &[(0, 0)]]);
    let err = check_twin_canonicality(&tdd).unwrap_err();
    assert!(err.contains("G"), "wrong violation: {err}");
}

/// C3 negative: two slots carrying equal counts should have been
/// canonicalized onto one slot (and the orphan collected).
#[test]
fn c3_detects_duplicate_counts() {
    let tdd = toy(vec![BIG, BIG], &[&[(0, 0), (1, 1)]]);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("C3"), "wrong violation: {err}");
}

/// C3 covers ALL slots: a stale duplicate is a violation too — and
/// `prune_value_slots` is the fix (collects the orphan, after which C3 holds).
#[test]
fn c3_rejects_stale_duplicate_until_slot_prune() {
    let eng = &crate::engine::Engine::new();
    let mut tdd = toy(vec![BIG, BIG], &[&[(0, 0)]]);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("C3"), "wrong violation: {err}");
    crate::reduce::slot_prune::prune_value_slots(eng, &mut tdd);
    check_slot_count_uniqueness(&tdd).unwrap();
}

/// C4/I2 negative: an inline-eligible count parked in a REFERENCED slot.
#[test]
fn c4_rejects_inline_eligible_slot() {
    let tdd = toy(vec![5], &[&[(0, 0)]]);
    let err = check_marg_canonical_form(&tdd).unwrap_err();
    assert!(err.contains("I2"), "wrong violation: {err}");
}

/// C4/I2 exemption: an inline-eligible count on an UNREFERENCED slot is
/// fine — count vectors are never shrunk, so a slot whose refs were all
/// retagged inline legitimately retains its small count.
#[test]
fn c4_ignores_stale_inline_eligible_slot() {
    let tdd = toy(vec![BIG, 5], &[&[(0, 0)]]);
    check_tdd_marg_invariants(&tdd).unwrap();
}

// ── C4 garbage-freedom (check_no_orphan_slots) ───────────────────────────

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
        err.contains("C4"),
        "wrong violation kind (expected C4): {err}"
    );
    assert!(
        err.contains('1') || err.contains("slot 1"),
        "error must mention orphan slot 1: {err}"
    );
}

/// Positive: after `prune_value_slots`, the orphan is removed and
/// `check_no_orphan_slots` passes. C3 must also hold.
#[test]
fn c4_orphan_slot_cleared_after_prune() {
    let eng = &crate::engine::Engine::new();
    let mut tdd = toy(vec![BIG + 10, BIG + 20], &[&[(0, 0)]]);
    // Pre-condition: C4 violated.
    assert!(
        check_no_orphan_slots(&tdd).is_err(),
        "pre-prune: expected C4 violation"
    );
    crate::reduce::slot_prune::prune_value_slots(eng, &mut tdd);
    // Post-condition: C4 passes.
    check_no_orphan_slots(&tdd).unwrap();
    // C3 must also hold after prune.
    check_slot_count_uniqueness(&tdd).unwrap();
}

// ── The same checks in the weighted domain ───────────────────────────────
//
// A weighted diagram stores its marginal values in the external `WeightStore`
// instead of the level, and never dedups them, so C3 and C4 claim something
// different there — see `check_weight_column_is_full_width` and the weighted
// arm of `check_no_orphan_slots`. G is about pair multisets and claims the
// same thing in both domains.

use crate::diagram::ValueRef;
use crate::diagram::{RationalWeights, WeightVal};
use crate::test_helpers::toy_weighted;
use crate::vtree::{Vtree, VtreeNode};
use crate::diagram::{Arithmetic, WeightStore};
use num_rational::BigRational;

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn weighted_store() -> WeightStore {
    WeightStore::new(
        RationalWeights::from_weights(&[(rat(2, 5), rat(3, 11)), (rat(1, 3), rat(-4, 9))]),
        Arithmetic::ExactRational,
    )
}

/// The marginal level `toy_weighted` builds: `balanced(3)`'s root's right child.
fn weighted_marg_level() -> crate::vtree::VtreeIdx {
    let vtree = Vtree::balanced(3);
    match vtree.node(vtree.root()) {
        VtreeNode::Internal { right, .. } => *right,
        _ => unreachable!("balanced(3) root is internal"),
    }
}

/// C3 positive, weighted: two slots may carry the SAME value. Nothing dedups a
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

/// C3 negative, weighted: a column short of the level's width breaks the
/// slot-index-is-node-index identity the bare-slot encoding rests on.
#[test]
fn c3_weighted_detects_short_column() {
    let mut tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)]]);
    let marg = weighted_marg_level();
    // Claim a third node without giving it a slot.
    tdd.levels[marg.idx()].make_marginal_weighted_with_slots(3);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("C3 (weighted)"), "wrong violation: {err}");
}

/// C4 positive, weighted: an unreferenced slot is NOT garbage in a weighted
/// column — nothing prunes one, and the column stays full width.
#[test]
fn c4_weighted_allows_unreferenced_slots() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)]]);
    check_no_orphan_slots(&tdd).unwrap();
}

/// C4 negative, weighted: a reference past the end of the column is a dangling
/// slot, which the bare-slot encoding cannot tolerate.
#[test]
fn c4_weighted_detects_dangling_reference() {
    let dangling = ValueRef::slot_raw(5);
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, dangling)]]);
    let err = check_no_orphan_slots(&tdd).unwrap_err();
    assert!(err.contains("C4"), "wrong violation: {err}");
    assert!(err.contains("past the end"), "wrong C4 arm: {err}");
}

/// G, weighted: two root nodes with identical pair lists are unmerged twins
/// there too — the check reads pair multisets, which say nothing about the
/// value domain.
#[test]
fn c2_weighted_detects_unmerged_twins() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0)], &[(0, 0)]]);
    let err = check_no_twins(&tdd).unwrap_err();
    assert!(err.contains("G"), "wrong violation: {err}");
}

/// The weighted values themselves are untouched by the checks above.
#[test]
fn weighted_column_survives_the_checks() {
    let tdd = toy_weighted(weighted_store(), vec![rat(3, 7), rat(1, 2)], &[&[(0, 0), (1, 1)]]);
    let ws = tdd.weights().expect("weighted diagram");
    let col = ws.level(weighted_marg_level().idx()).expect("column installed");
    use crate::diagram::semiring::weight_key;
    let got: Vec<_> = col.iter().map(weight_key).collect();
    let want: Vec<_> = [rat(3, 7), rat(1, 2)]
        .into_iter()
        .map(|r| weight_key(&WeightVal::exact(r)))
        .collect();
    assert!(got == want, "the checks must not disturb the weighted column");
}
