use super::test_fixtures::{toy, BIG};
use super::*;

/// C1 negative: two pairs sharing left x=0 with distinct marg slots is a
/// fusable group — saturation must reject it (and so must the full check).
#[test]
fn c1_detects_unfused_same_x_group() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let err = check_p_saturation(&tdd, None).unwrap_err();
    assert!(err.contains("C1"), "wrong violation: {err}");
    assert!(check_marg_canonical_form(&tdd).is_err());
}

/// C1 positive + full form: after `apply_p_fusion` + `prune_marg_slots` the
/// same TDD is canonical — the fused slot survives alone, orphans
/// collected.
#[test]
fn canonical_form_holds_after_p_fusion_and_slot_prune() {
    let _g = crate::tdd::transform::pairwise::conjoin::apply_limits().budget(None).apply();
    let mut tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    let stats = crate::tdd::minimize::contract::p_fusion::apply_p_fusion(&mut tdd).unwrap();
    assert_eq!(stats.fusion_groups, 1);
    let pruned = crate::tdd::minimize::slot_prune::prune_marg_slots(&mut tdd);
    assert_eq!(pruned.slots_freed, 2, "both pre-fusion slots are orphans");
    check_marg_canonical_form(&tdd).unwrap();
}

/// C1 filter: a violation at a parent OUTSIDE the filter is not reported
/// (mirrors `apply_p_fusion_at_parents` semantics).
#[test]
fn c1_filter_skips_other_parents() {
    let tdd = toy(vec![BIG + 1, BIG + 3], &[&[(0, 0), (0, 1)]]);
    assert!(check_p_saturation(&tdd, Some(&[])).is_ok());
}

/// C2 negative: two nodes with identical pair lists are unmerged twins.
#[test]
fn c2_detects_unmerged_twins() {
    let tdd = toy(vec![BIG + 1], &[&[(0, 0)], &[(0, 0)]]);
    let err = check_twin_canonicality(&tdd).unwrap_err();
    assert!(err.contains("C2"), "wrong violation: {err}");
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
/// `prune_marg_slots` is the fix (collects the orphan, after which C3 holds).
#[test]
fn c3_rejects_stale_duplicate_until_slot_prune() {
    let mut tdd = toy(vec![BIG, BIG], &[&[(0, 0)]]);
    let err = check_slot_count_uniqueness(&tdd).unwrap_err();
    assert!(err.contains("C3"), "wrong violation: {err}");
    crate::tdd::minimize::slot_prune::prune_marg_slots(&mut tdd);
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

/// Positive: after `prune_marg_slots`, the orphan is removed and
/// `check_no_orphan_slots` passes. C3 must also hold.
#[test]
fn c4_orphan_slot_cleared_after_prune() {
    let mut tdd = toy(vec![BIG + 10, BIG + 20], &[&[(0, 0)]]);
    // Pre-condition: C4 violated.
    assert!(
        check_no_orphan_slots(&tdd).is_err(),
        "pre-prune: expected C4 violation"
    );
    crate::tdd::minimize::slot_prune::prune_marg_slots(&mut tdd);
    // Post-condition: C4 passes.
    check_no_orphan_slots(&tdd).unwrap();
    // C3 must also hold after prune.
    check_slot_count_uniqueness(&tdd).unwrap();
}
