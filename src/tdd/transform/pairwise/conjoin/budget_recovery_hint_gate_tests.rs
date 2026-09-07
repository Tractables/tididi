use super::*;

/// Regression guard for the mc2026_017 recovery gridlock: hint-driven split
/// selection must stay off on a normal run (no apply-error-path write site
/// exists anymore), so recovery's `pick_branching_vars` sees no hint unless
/// explicitly seeded, and uses the global occurrence ranking — the behavior
/// that solves 017 in 2 splits where hint-driven selection burned the whole
/// wall over-budget. The replay harness's explicit seeding still round-trips.
#[test]
fn recovery_hint_empty_unless_seeded() {
    assert!(
        take_recovery_hint().is_none(),
        "recovery hint must be empty on a fresh run"
    );

    seed_recovery_hint(vec![VarId(1)]);
    let seeded = take_recovery_hint().expect("seeded hint must be readable");
    assert_eq!(seeded.vars, vec![VarId(1)]);
    assert!(take_recovery_hint().is_none(), "take must consume");
}
