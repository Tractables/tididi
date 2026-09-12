use super::*;
use crate::engine::Engine;
use crate::limits::growth::DENSE_GROWTH_DECISION_THRESHOLD;
use crate::diagram::{InputPair, NodeIdx};

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
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(12 * MIB));
    let pair = InputPair { left: NodeIdx(0), right: NodeIdx(0) };
    let cap0 = 2_000_000usize;
    let mut v: Vec<InputPair> = Vec::with_capacity(cap0);
    v.resize(v.capacity(), pair); // len == capacity ⇒ next push must grow
    let cap0 = v.capacity(); // allocator may round up; use the real cap
    grow_pairs_bounded(&eng, &mut v).expect("chunk fits the budget");
    let min_chunk = (PAIRS_GROW_MIN_CHUNK_BYTES / ELEM) as usize;
    assert!(
        v.capacity() >= cap0 + min_chunk && v.capacity() < 2 * cap0,
        "bounded growth must add ~min_chunk, not double: cap0={cap0} cap1={}",
        v.capacity()
    );
    let charged = lim.meters().in_flight_bytes;
    assert!(charged >= min_chunk as u64 * ELEM, "chunk must be budget-accounted");
}

#[test]
fn bounded_growth_mode_resets_at_operation_entry() {
    let eng = Engine::new();
    let lim = eng.limits();
    // A level whose emitted-pair bound dwarfs the headroom arms the mode.
    lim.set_budget(Some(MIB));
    lim.begin_level(Some(DENSE_GROWTH_DECISION_THRESHOLD * 2));
    assert!(lim.bounded_growth());
    let _op = lim.begin_operation();
    assert!(
        !lim.bounded_growth(),
        "operation entry must clear the per-level emit-growth mode"
    );
}
