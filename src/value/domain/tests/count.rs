//! Differential validation of `compute_cell_count`'s u128→BigUint overflow
//! fallback — a branch ordinary compiles never reach, because their counts stay
//! far inside u128. Specifically
//! guards the edge the `all_u64` widening-multiply fast path introduced:
//! Both children certified `all_u64`, yet the Σ of `u64×u64` products
//! overflows u128 and must hand off to an exact BigUint accumulation rather
//! than wrap or mis-promote.
use crate::value::{Count, CountRef, COUNT_OVERFLOW};
use crate::value::domain::count::StreamChildCounts;
use crate::value::domain::count::compute_cell_count;
use crate::test_helpers::pair;
use num_bigint::BigUint;

/// The child column is a BORROWED view in production too (`IntFold::child_view`
/// reads the child level's storage in place), so each test owns its slot array
/// and lends it here.
fn child(counts: &[u128]) -> StreamChildCounts<'_> {
    // `certified` is the same certificate scan `child_view` runs over a
    // level's raw marginal arrays, so the test selects the same path production
    // would for these slot values.
    StreamChildCounts { col: CountRef::new(counts, None).certified(), view: crate::diagram::ChildDecoder::structural() }
}

#[test]
fn all_u64_fast_path_sum_overflow_falls_back_to_exact_biguint() {
    // Each product = (2^64-1)^2 ≈ u128::MAX, so the running u128 total
    // overflows on the 2nd add; the fast path must break to the BigUint
    // accumulator. Three pairs → exactly 3·(2^64-1)^2.
    let (ls, rs) = (vec![u64::MAX as u128], vec![u64::MAX as u128]);
    let (left, right) = (child(&ls), child(&rs));
    assert!(left.col.all_u64() && right.col.all_u64(), "fast path must be selected");
    let pairs = [pair(0, 0), pair(0, 0), pair(0, 0)];
    let m = BigUint::from(u64::MAX);
    let expected = (&m * &m) * BigUint::from(3u32);
    match compute_cell_count(&pairs, &left, &right) {
        Count::Big(big) => assert_eq!(big, expected, "overflow fallback miscounted"),
        Count::Fast(v) => panic!("expected u128 overflow → Count::Big, got Fast({v})"),
    }
}

#[test]
fn all_u64_fast_path_no_false_overflow() {
    // Sum stays within u128 — the fast path must return Ok, never spuriously
    // promote to BigUint.
    let (ls, rs) = (vec![1_000_000_000u128], vec![2_000_000_000u128]);
    let (left, right) = (child(&ls), child(&rs));
    let pairs = [pair(0, 0), pair(0, 0)];
    match compute_cell_count(&pairs, &left, &right) {
        Count::Fast(v) => assert_eq!(v, 2 * 1_000_000_000u128 * 2_000_000_000u128),
        Count::Big(b) => panic!("fast path spuriously overflowed: {b}"),
    }
}

#[test]
fn non_all_u64_takes_general_path_and_matches() {
    // A slot just over u64::MAX disables the fast path; the general u128
    // path must still compute the exact product (no overflow here).
    let big_count = u64::MAX as u128 + 1;
    let (ls, rs) = (vec![big_count], vec![2u128]);
    let (left, right) = (child(&ls), child(&rs)); // left: all_u64 == false
    assert!(!left.col.all_u64(), "general path must be selected");
    let pairs = [pair(0, 0)];
    match compute_cell_count(&pairs, &left, &right) {
        Count::Fast(v) => assert_eq!(v, big_count * 2),
        Count::Big(b) => panic!("unexpected overflow: {b}"),
    }
}

#[test]
fn all_u64_mixed_magnitudes_overflow_exact() {
    // Heterogeneous slots (the realistic case): some pairs tiny, some huge.
    // Exercises the BigUint accumulator across a mix and checks the exact
    // total. left[0]·right[0] = (2^64-1)^2 (huge), left[1]·right[1] = 7·11.
    let (ls, rs) = (vec![u64::MAX as u128, 7], vec![u64::MAX as u128, 11]);
    let (left, right) = (child(&ls), child(&rs));
    assert!(left.col.all_u64() && right.col.all_u64());
    // Two huge products overflow u128 → BigUint; add the small one.
    let pairs = [pair(0, 0), pair(0, 0), pair(1, 1)];
    let m = BigUint::from(u64::MAX);
    let expected = (&m * &m) * BigUint::from(2u32) + BigUint::from(77u32);
    match compute_cell_count(&pairs, &left, &right) {
        Count::Big(big) => assert_eq!(big, expected),
        Count::Fast(v) => panic!("expected overflow, got Fast({v})"),
    }
}

#[test]
fn all_u64_two_accumulator_combine_overflow() {
    // Two pairs = exactly one chunk: t0 and t1 each hold one (2^64-1)^2
    // product (≈ u128::MAX, both fit individually), but the final
    // `t0.checked_add(t1)` combine overflows → must fall to BigUint with the
    // exact total 2·(2^64-1)^2. Guards the two-accumulator combine branch.
    let (ls, rs) = (vec![u64::MAX as u128], vec![u64::MAX as u128]);
    let (left, right) = (child(&ls), child(&rs));
    let pairs = [pair(0, 0), pair(0, 0)];
    let m = BigUint::from(u64::MAX);
    let expected = (&m * &m) * BigUint::from(2u32);
    match compute_cell_count(&pairs, &left, &right) {
        Count::Big(big) => assert_eq!(big, expected, "combine-overflow miscounted"),
        Count::Fast(v) => panic!("expected combine overflow → Count::Big, got Fast({v})"),
    }
}

#[test]
fn all_u64_odd_remainder_no_overflow() {
    // Odd pair count exercises the chunks_exact remainder tail. 3 pairs,
    // small values, no overflow → exact Ok across the chunk + remainder.
    let (ls, rs) = (vec![3u128, 5, 7], vec![11u128, 13, 17]);
    let (left, right) = (child(&ls), child(&rs));
    let pairs = [pair(0, 0), pair(1, 1), pair(2, 2)];
    match compute_cell_count(&pairs, &left, &right) {
        Count::Fast(v) => assert_eq!(v, 3 * 11 + 5 * 13 + 7 * 17),
        Count::Big(b) => panic!("spurious overflow: {b}"),
    }
}

#[test]
fn all_u64_certificate_excludes_overflow_sentinel() {
    // Sanity: a `COUNT_OVERFLOW` slot must not be certified `all_u64` (it is
    // u128::MAX > u64::MAX), so such a column routes to the general path
    // where the sentinel is honored — never silently truncated by the
    // fast-path `as u64`.
    let slots = vec![5u128, COUNT_OVERFLOW];
    let c = child(&slots);
    assert!(!c.col.all_u64(), "COUNT_OVERFLOW slot must defeat the all_u64 cert");
}

#[test]
fn the_structural_u64_fold_is_the_exact_fold_or_refuses() {
    use crate::value::{CountRead, IntFold};
    let (ls, rs) = (vec![0u128, 3, u64::MAX as u128, 1], vec![7u128, 0, 2, u64::MAX as u128]);
    let exact = |pairs: &[crate::diagram::ChildPair]| {
        IntFold::fold(
            pairs.iter().copied(),
            |k| CountRead::Fast(ls[k.0 as usize]),
            |k| CountRead::Fast(rs[k.0 as usize]),
        )
    };
    // Every length from empty through an odd tail, zeros included, agrees
    // with the exact fold while the total fits `u128`.
    let all: Vec<_> = (0..4).flat_map(|l| (0..4).map(move |r| pair(l, r))).collect();
    for n in 0..=all.len() {
        let pairs = &all[..n];
        match (IntFold::fold_structural_u64(pairs, &ls, &rs), exact(pairs)) {
            (Some(v), Count::Fast(w)) => assert_eq!(v, w, "{n} pairs"),
            (None, Count::Big(_)) => {}
            (fast, slow) => panic!("{n} pairs: {fast:?} against {slow:?}"),
        }
    }
    // Four products of `(2^64-1)^2` leave `u128` in either lane.
    let big = [pair(2, 3), pair(2, 3), pair(2, 3), pair(2, 3)];
    assert_eq!(IntFold::fold_structural_u64(&big, &ls, &rs), None);
    assert!(matches!(exact(&big), Count::Big(_)));
}
