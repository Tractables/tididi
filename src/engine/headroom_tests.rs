use super::*;

#[test]
fn vas_fallback_arms_without_soft_budget() {
    // Default production: no soft budget armed. The fallback must still hand
    // the emit-growth mode decision a positive headroom (RLIMIT_AS − VAS, or
    // the unlimited constant): the soft-budget headroom is `None` here, and a
    // caller that saw only that would fall back to an exact pre-count walk.
    // The in-flight byte meter is irrelevant when no budget is armed.
    let eng = Engine::new();
    let lim = eng.limits();
    assert_eq!(lim.budget_headroom(), None);
    let h = lim.headroom();
    assert!(h > 0, "VAS fallback headroom must be positive, got {h}");
}

#[test]
fn vas_margin_subtracted_from_rlimit_headroom() {
    // The no-soft-budget arm: the soft margin is held back below RLIMIT_AS, on top of
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
    // With a soft budget armed (segmented compile), the value must equal the
    // old soft-budget headroom exactly — no address space consulted.
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(4096));
    assert_eq!(lim.budget_headroom(), Some(4096));
    assert_eq!(lim.headroom(), 4096);
}
