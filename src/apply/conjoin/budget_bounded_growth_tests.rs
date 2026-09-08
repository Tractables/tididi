use super::*;
use crate::limits::{apply_limits, reset_apply_meters, APPLY_LIMITS};
use crate::diagram::{InputPair, LocalNodeIdx};

const MIB: u64 = 1024 * 1024;
const ELEM: u64 = std::mem::size_of::<InputPair>() as u64; // 8

#[test]
fn increment_plentiful_headroom_is_doubling() {
    // half-headroom ≥ cap ⇒ increment == cap (plain doubling, amortized O(n)).
    let cap = 1_000_000;
    let inc = bounded_grow_increment(cap, 1 << 40, ELEM);
    assert_eq!(inc, cap);
}

#[test]
fn increment_shrinking_headroom_is_half_headroom() {
    // half-headroom below cap but above the min-chunk floor ⇒ increment ==
    // headroom/2 elements: transient = cap + inc always fits, increments
    // halve geometrically.
    let cap = 100_000_000;
    let headroom = 64 * MIB; // half = 32 MiB = 4 M pairs
    let inc = bounded_grow_increment(cap, headroom, ELEM);
    assert_eq!(inc as u64, headroom / 2 / ELEM);
    assert!(inc < cap);
}

#[test]
fn increment_near_exhaustion_floors_at_min_chunk() {
    // half-headroom under the floor ⇒ increment == min_chunk (never
    // degenerates to per-push realloc).
    let cap = 100_000_000;
    let inc = bounded_grow_increment(cap, MIB, ELEM);
    assert_eq!(inc as u64, PAIRS_GROW_MIN_CHUNK_BYTES / ELEM);
}

#[test]
fn increment_small_cap_stays_doubling_under_floor() {
    // cap below min_chunk ⇒ increment capped at cap (doubling) even when
    // the floor is larger — small vecs never over-reserve.
    let cap = 1_000;
    let inc = bounded_grow_increment(cap, MIB, ELEM);
    assert_eq!(inc, cap);
}

#[test]
fn grow_pairs_bounded_grows_less_than_doubling_and_charges_budget() {
    // Integration through the real Vec + soft-budget accounting: a full
    // 2 M-pair vec under a 12 MiB soft budget must grow by the 1 M-pair
    // min-chunk floor (half-headroom 6 MiB < 8 MiB floor), i.e. to 3 M
    // capacity — strictly less than doubling's 4 M — and charge exactly
    // that chunk to the in-flight meter.
    reset_apply_meters();
    let _g = apply_limits().budget(Some(12 * MIB)).apply();
    let pair = InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) };
    let cap0 = 2_000_000usize;
    let mut v: Vec<InputPair> = Vec::with_capacity(cap0);
    v.resize(v.capacity(), pair); // len == capacity ⇒ next push must grow
    let cap0 = v.capacity(); // allocator may round up; use the real cap
    grow_pairs_bounded(&mut v).expect("chunk fits the budget");
    let min_chunk = (PAIRS_GROW_MIN_CHUNK_BYTES / ELEM) as usize;
    assert!(
        v.capacity() >= cap0 + min_chunk && v.capacity() < 2 * cap0,
        "bounded growth must add ~min_chunk, not double: cap0={cap0} cap1={}",
        v.capacity()
    );
    let charged = APPLY_LIMITS.with(|l| l.budget_in_flight.get());
    assert!(charged >= min_chunk as u64 * ELEM, "chunk must be budget-accounted");
}

#[test]
fn bounded_growth_flag_resets_at_apply_entry() {
    set_pairs_bounded_growth(true);
    assert!(APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()));
    reset_meters();
    assert!(
        !APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()),
        "apply entry must clear the per-level emit-growth mode"
    );
}
