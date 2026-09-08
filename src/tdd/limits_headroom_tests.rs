use super::*;

#[test]
fn vas_fallback_arms_without_soft_budget() {
    // Default production: no soft budget armed. The fallback must still hand
    // the emit-growth mode decision a positive headroom (RLIMIT_AS − VAS, or
    // the unlimited constant) — the old `apply_budget_headroom_bytes()`
    // returned `None` here, which forced the exact pre-count walk.
    // `budget_in_flight` is irrelevant when no budget is armed.
    set_apply_budget(None);
    assert_eq!(apply_budget_headroom_bytes(), None);
    let h = apply_headroom_bytes_or_vas();
    assert!(h > 0, "VAS fallback headroom must be positive, got {h}");
}

#[test]
fn vas_margin_subtracted_from_rlimit_headroom() {
    // Branch (2): the soft margin is held back below RLIMIT_AS, on top of
    // the mapped-bytes subtraction. 30 GiB ceiling, 10 GiB mapped ⇒
    // 30 − 1.5 − 10 = 18.5 GiB headroom.
    const GIB: u64 = 1024 * 1024 * 1024;
    assert_eq!(
        vas_headroom_with_margin(30 * GIB, 10 * GIB),
        30 * GIB - SOFT_HEADROOM_MARGIN_BYTES - 10 * GIB,
    );
    assert_eq!(SOFT_HEADROOM_MARGIN_BYTES, 1536 * 1024 * 1024);
}

#[test]
fn vas_margin_saturates_when_ceiling_below_margin_or_mapped() {
    // A ceiling smaller than the margin, or usage already past the margined
    // ceiling, reports zero headroom (⇒ bounded growth) — never wraps.
    assert_eq!(vas_headroom_with_margin(SOFT_HEADROOM_MARGIN_BYTES / 2, 0), 0);
    assert_eq!(vas_headroom_with_margin(4 * 1024 * 1024 * 1024, u64::MAX), 0);
}

#[test]
fn soft_budget_semantics_unchanged() {
    // With a soft budget armed (segmented compile), the value MUST equal the
    // old soft-budget headroom exactly — no VAS floor, no change.
    reset_apply_in_flight();
    set_apply_budget(Some(4096));
    assert_eq!(apply_budget_headroom_bytes(), Some(4096));
    assert_eq!(apply_headroom_bytes_or_vas(), 4096);
    // Restore the default so a reused test thread starts clean.
    set_apply_budget(None);
}
