use super::*;
use crate::Engine;

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

#[test]
fn a_reading_answers_fits_until_charges_pass_an_eighth_of_its_room() {
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
    const ROOM: u64 = 8 << 20;
    let reads = Arc::new(AtomicU64::new(0));
    let mapped = Arc::new(AtomicU64::new(0));
    let (counted, host) = (Arc::clone(&reads), Arc::clone(&mapped));
    let hooks = MemoryHooks::new(
        |_| {},
        move || { counted.fetch_add(1, Ordering::Relaxed); host.load(Ordering::Relaxed) },
        || Some(SOFT_HEADROOM_MARGIN_BYTES + ROOM),
        || {},
    );
    let eng = Engine::new();
    let lim = eng.limits();
    let _hooks = lim.scope(LimitConfig::none().with_memory_hooks(hooks));
    let read = || reads.load(Ordering::Relaxed);
    let _op = lim.begin_operation();
    assert_eq!(lim.headroom_short_of(ROOM), None);
    assert_eq!(read(), 1, "the first question reads the host");
    // The host maps more without a charge: the reading answers on.
    mapped.store(ROOM / 2, Ordering::Relaxed);
    assert_eq!(lim.headroom_short_of(ROOM), None);
    assert_eq!(read(), 1);
    // Bytes charged come off the room; one released and charged again counts twice.
    lim.charge_bytes(ROOM / 32).unwrap();
    lim.release_bytes(ROOM / 32);
    lim.charge_bytes(ROOM / 32).unwrap();
    assert_eq!(lim.headroom_short_of(ROOM - ROOM / 16), None);
    assert_eq!(read(), 1);
    // More than the reading's room left: a read, which the answer is.
    assert_eq!(lim.headroom_short_of(ROOM - ROOM / 16 + 1), Some(ROOM / 2));
    assert_eq!(read(), 2);
    // Past an eighth of the room charged since, the host is read again.
    lim.charge_bytes(ROOM / 16 + 1).unwrap();
    assert_eq!(lim.headroom_short_of(1), None);
    assert_eq!(read(), 3);
    assert_eq!(lim.headroom_short_of(1), None);
    assert_eq!(read(), 3);
    drop(_op);
    // A new operation reads afresh.
    let _op = lim.begin_operation();
    assert_eq!(lim.headroom_short_of(1), None);
    assert_eq!(read(), 4);
}

#[test]
fn an_armed_budget_or_no_ceiling_reads_no_mapped_bytes() {
    use super::super::memory::VAS_UNLIMITED_HEADROOM;
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
    let reads = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&reads);
    let hooks = MemoryHooks::new(|_| {}, move || { counted.fetch_add(1, Ordering::Relaxed); 0 }, || None, || {});
    let eng = Engine::new();
    let lim = eng.limits();
    let _hooks = lim.scope(LimitConfig::none().with_memory_hooks(hooks));
    assert_eq!(lim.headroom_short_of(VAS_UNLIMITED_HEADROOM), None);
    assert_eq!(lim.headroom_short_of(VAS_UNLIMITED_HEADROOM + 1), Some(VAS_UNLIMITED_HEADROOM));
    lim.set_budget(Some(4096));
    assert_eq!(lim.headroom_short_of(4096), None);
    assert_eq!(lim.headroom_short_of(4097), Some(4096));
    assert_eq!(reads.load(Ordering::Relaxed), 0);
}
