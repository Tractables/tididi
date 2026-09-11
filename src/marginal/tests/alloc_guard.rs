//! The width-sized
//! `CountVec<RecoveryPanic>` scratch buffers used by `marginalize_batch`/
//! `ensure_counts` must raise a *recoverable panic* — not an infallible
//! alloc-error abort — when a buffer would exceed the address-space budget.
//! The panic unwinds into the caller's memory-budget recovery path and
//! triggers a Shannon split; an abort would double-fault past it.
use crate::test_helpers::CountVecExt;
use crate::engine::Engine;
use crate::value::{Count, CountVec};
use crate::limits::{RecoveryPanic, ReservePolicy};
use num_bigint::BigUint;

/// A width whose `u128`/`(u32, BigUint)` byte size overflows `isize::MAX`,
/// so `try_reserve_exact` returns `Err(CapacityOverflow)` *without* touching
/// the allocator — deterministic, no real allocation, no abort.
const OVER_BUDGET_WIDTH: usize = usize::MAX / 8;

#[test]
#[should_panic(expected = "triggering recovery split")]
fn with_width_panics_over_budget() {
    let eng = Engine::new();
    let _ = CountVec::<RecoveryPanic>::with_width(&eng, OVER_BUDGET_WIDTH);
}

/// The overflow side table grows one entry at a time (it is keyed by slot, so
/// it is never sized to the column's width), but it still routes every growth
/// through the same guarded `reserve`/`reserve_exact` as the fast column. This
/// pins that policy on the side table's element type — a different size than
/// the fast `Vec<u128>` above, so guarded independently.
#[test]
#[should_panic(expected = "triggering recovery split")]
fn big_side_table_reserve_panics_over_budget() {
    let eng = Engine::new();
    let mut v: Vec<(u32, BigUint)> = Vec::new();
    let _ = <RecoveryPanic as ReservePolicy>::reserve_exact(&eng, &mut v, OVER_BUDGET_WIDTH);
}

#[test]
fn clone_guarded_copies_fast_values_exactly() {
    let eng = Engine::new();
    let mut cv = CountVec::<RecoveryPanic>::with_width(&eng, 4);
    cv.set_i(&eng, 0, Count::Fast(0));
    cv.set_i(&eng, 1, Count::Fast(1));
    cv.set_i(&eng, 2, Count::Big(BigUint::from(u128::MAX))); // sentinel lives in fast[2]
    cv.set_i(&eng, 3, Count::Fast(42));
    let out = cv.clone_guarded(&eng);
    for i in 0..4 {
        assert_eq!(out.fast_val(i), cv.fast_val(i));
    }
}

#[test]
fn clone_guarded_copies_big_overflow_exactly() {
    let eng = Engine::new();
    let mut cv = CountVec::<RecoveryPanic>::with_width(&eng, 4);
    cv.set_i(&eng, 1, Count::Big(BigUint::from(7u32)));
    cv.set_i(&eng, 3, Count::Big(BigUint::from(u128::MAX)));
    let out = cv.clone_guarded(&eng);
    for i in 0..4 {
        assert_eq!(out.big_val(i).cloned(), cv.big_val(i).cloned());
    }
}
