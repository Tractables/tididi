
use crate::diagram::ChildDecoder;
use super::*;
use crate::Engine;
use crate::diagram::WeightValue;
use crate::test_helpers::{pair, rat, CountVecExt};

#[test]
fn from_u128_promotes_exact_max_to_big() {
    match Count::from_u128(u128::MAX) {
        Count::Big(b) => assert_eq!(b, BigUint::from(u128::MAX)),
        Count::Fast(_) => panic!("u128::MAX must promote to Big"),
    }
}

#[test]
fn from_u128_below_max_stays_fast() {
    match Count::from_u128(u128::MAX - 1) {
        Count::Fast(v) => assert_eq!(v, u128::MAX - 1),
        Count::Big(_) => panic!("u128::MAX - 1 must stay Fast"),
    }
}

#[test]
fn push_big_value_is_visible_and_index_aligned_after_backfill() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Fast(3));
    cv.push_i(&eng, Count::Fast(5));
    let big_val = BigUint::from(u64::MAX) * BigUint::from(2u32);
    cv.push_i(&eng, Count::Big(big_val.clone()));
    match cv.get(2) {
        CountRead::Big(b) => assert_eq!(*b, big_val),
        CountRead::Fast(_) => panic!("expected Big read"),
    }
    assert!(cv.has_big());
    // Backfilled slots stay None; the big table is index-aligned (no
    // out-of-bounds panic reading any slot up to len()).
    assert!(cv.big_val(&eng, 0).is_none());
    assert!(cv.big_val(&eng, 1).is_none());
    assert_eq!(cv.big_val(&eng, 2), Some(big_val));
    assert_eq!(cv.len(), 3);
}

#[test]
fn all_u64_true_after_u64_range_pushes() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Fast(1));
    cv.push_i(&eng, Count::Fast(u64::MAX as u128));
    assert!(cv.all_u64());
}

#[test]
fn all_u64_cleared_by_over_u64_fast_push_and_never_returns() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Fast(1));
    assert!(cv.all_u64());
    cv.push_i(&eng, Count::Fast(u64::MAX as u128 + 1));
    assert!(!cv.all_u64());
    cv.push_i(&eng, Count::Fast(2));
    assert!(!cv.all_u64(), "certificate must not return to true");
}

#[test]
fn all_u64_cleared_by_big_push_and_never_returns() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Fast(1));
    assert!(cv.all_u64());
    cv.push_i(&eng, Count::Big(BigUint::from(7u32)));
    assert!(!cv.all_u64());
    cv.set_i(&eng, 0, Count::Fast(9));
    assert!(
        !cv.all_u64(),
        "sentinel slot must defeat the certificate permanently"
    );
}

#[test]
fn set_overwrites_big_with_fast_clears_big_slot() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 2);
    cv.set_i(&eng, 0, Count::Big(BigUint::from(99u32)));
    assert!(cv.big_val(&eng, 0).is_some());
    cv.set_i(&eng, 0, Count::Fast(5));
    assert!(matches!(cv.get(0), CountRead::Fast(5)));
    assert!(
        cv.big_val(&eng, 0).is_none(),
        "big slot must clear to None when overwritten by Fast"
    );
}

/// Mirrors `conjoin::cell::tests::collect_sink_pushes_charge_the_soft_budget`'s
/// install/reset hygiene: a tiny soft budget must trip the engine's
/// `Err(OperationError::OverBudget)` well before an unbudgeted push loop would
/// exhaust real memory.
#[test]
fn count_column_charges_soft_budget() {
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(64));
    let mut cv = CountVec::try_with_width(&eng, 0)
        .expect("width-0 allocation must not trip a 64-byte budget");
    let mut result = Ok(());
    for _ in 0..1024 {
        result = cv.push(&eng, Count::Fast(1));
        if result.is_err() {
            break;
        }
    }
    assert!(
        matches!(result, Err(crate::limits::OperationError::OverBudget)),
        "CountVec pushes bypass the apply soft budget"
    );
}

#[test]
fn try_clone_round_trips_fast_and_big() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Fast(3));
    cv.push_i(&eng, Count::Big(BigUint::from(123456789u64)));
    cv.push_i(&eng, Count::Fast(7));
    let clone = cv.clone_guarded(&eng);
    assert_eq!(clone.len(), cv.len());
    for i in 0..cv.len() {
        assert_eq!(clone.get(i).to_count(), cv.get(i).to_count());
        assert_eq!(clone.big_val(&eng, i), cv.big_val(&eng, i));
    }
    assert_eq!(clone.all_u64(), cv.all_u64());
}

/// Readers over a fixture column, mirroring how the ensure-walk adapters
/// hand `CountVec::get` closures to the fold.
fn col(eng: &Engine, values: Vec<Count>) -> CountVec {
    let mut cv = CountVec::with_width(eng, values.len());
    for (i, v) in values.into_iter().enumerate() {
        cv.set_i(eng, i, v);
    }
    cv
}

#[test]
fn int_fold_stays_fast_within_u128() {
    let eng = Engine::new();
    let left = col(&eng, vec![Count::Fast(3), Count::Fast(5)]);
    let right = col(&eng, vec![Count::Fast(7), Count::Fast(11)]);
    let pairs = [pair(0, 0), pair(1, 1)];
    match IntFold::fold(pairs.iter().copied(), |k| left.get(ChildDecoder::structural().node(k).idx()), |k| right.get(ChildDecoder::structural().node(k).idx())) {
        Count::Fast(v) => assert_eq!(v, 3 * 7 + 5 * 11),
        Count::Big(b) => panic!("spurious overflow: {b}"),
    }
}

#[test]
fn int_fold_overflow_repass_is_exact_and_mixed_magnitude() {
    let eng = Engine::new();
    // Slot 0: huge fast values whose product overflows u128 (pass-2
    // Fast×Fast checked_mul-fails sub-case). Slot 1: a Big child × a small
    // fast child (mixed sub-case). Slot 2: tiny (both-fast sub-case).
    let huge = u64::MAX as u128 + 7; // > u64, still fast-representable
    let big_child = BigUint::from(u128::MAX) * BigUint::from(3u32);
    let left = col(
        &eng,
        vec![
            Count::Fast(huge),
            Count::Big(big_child.clone()),
            Count::Fast(7),
        ],
    );
    let right = col(
        &eng,
        vec![Count::Fast(huge), Count::Fast(5), Count::Fast(11)],
    );
    let pairs = [pair(0, 0), pair(1, 1), pair(2, 2)];
    let expected = BigUint::from(huge) * BigUint::from(huge)
        + &big_child * BigUint::from(5u32)
        + BigUint::from(77u32);
    match IntFold::fold(pairs.iter().copied(), |k| left.get(ChildDecoder::structural().node(k).idx()), |k| right.get(ChildDecoder::structural().node(k).idx())) {
        Count::Big(b) => assert_eq!(b, expected),
        Count::Fast(v) => panic!("expected overflow → Big, got Fast({v})"),
    }
}

#[test]
fn int_fold_exact_max_total_promotes_to_big() {
    let eng = Engine::new();
    // (2^64+1)·(2^64−1) = 2^128−1 = u128::MAX exactly: pass 1 completes
    // without overflowing, but the total lands exactly on the sentinel — `from_u128` must
    // promote so the stored value stays unambiguous.
    let left = col(&eng, vec![Count::Fast((1u128 << 64) + 1)]);
    let right = col(&eng, vec![Count::Fast((1u128 << 64) - 1)]);
    let pairs = [pair(0, 0)];
    match IntFold::fold(pairs.iter().copied(), |k| left.get(ChildDecoder::structural().node(k).idx()), |k| right.get(ChildDecoder::structural().node(k).idx())) {
        Count::Big(b) => assert_eq!(b, BigUint::from(u128::MAX)),
        Count::Fast(v) => panic!("exact-max total must promote to Big, got Fast({v})"),
    }
}

#[test]
fn weight_fold_sums_products_exactly() {
    let q = |n: i64, d: i64| WeightValue::exact(rat(n, d));
    let left = [q(1, 2), q(3, 4)];
    let right = [q(1, 3), q(2, 5)];
    let pairs = [pair(0, 0), pair(1, 1)];
    let got = WeightFold::fold(
        pairs.iter().copied(),
        |k| std::borrow::Cow::Borrowed(&left[ChildDecoder::structural().node(k).idx()]),
        |k| std::borrow::Cow::Borrowed(&right[ChildDecoder::structural().node(k).idx()]),
        q(0, 1),
    );
    // 1/2·1/3 + 3/4·2/5 = 1/6 + 3/10 = 7/15
    let got = got.into_rational_opt().expect("expected Exact");
    assert_eq!(got, rat(7, 15));
}

/// The weighted scratch column reserves through the apply soft budget
/// exactly like the integer one, so a refusal at the ceiling is an `Err`
/// rather than an abort.
#[test]
fn weighted_column_alloc_charges_soft_budget() {
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(64));
    let zero = WeightValue::exact(rat(0, 1));
    let res = WeightFold::alloc_col(&eng, 4096, &zero);
    assert!(
        matches!(res, Err(crate::limits::OperationError::OverBudget)),
        "weighted column allocation bypasses the apply soft budget"
    );
}

/// Regression guard for the sparse side-table discipline: the overflow table
/// is keyed by slot, so `Fast` pushes after a `Big` push must add nothing to
/// it — there is no separate "fits the fast lane" flag, an absent entry encodes
/// it — and a read at a
/// fast-lane slot must resolve as fast-only, not panic.
#[test]
fn fast_push_after_big_leaves_side_table_sparse() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    cv.push_i(&eng, Count::Big(BigUint::from(42u32)));
    cv.push_i(&eng, Count::Fast(7));
    cv.push_i(&eng, Count::Fast(9));
    let (fast, big) = cv.clone_guarded(&eng).into_parts();
    assert_eq!(fast.len(), 3);
    assert_eq!(
        big.expect("big table exists").len(),
        1,
        "fast pushes must not add overflow entries",
    );
    assert_eq!(cv.big_val(&eng, 0), Some(BigUint::from(42u32)));
    assert!(
        cv.big_val(&eng, 2).is_none(),
        "fast-lane slot must read None, not panic"
    );
    match cv.get(1) {
        CountRead::Fast(v) => assert_eq!(v, 7),
        CountRead::Big(_) => panic!("expected Fast read"),
    }
}

/// The empty case must be the cheap case: a store that never overflows owns no
/// overflow table at all, so its side-table footprint is exactly zero bytes —
/// the property the sparse representation exists for (the dense column charged
/// `24 B × width` at the first overflow, whatever the overflow count).
#[test]
fn all_fast_store_owns_no_overflow_table() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 0);
    for v in 0..64u128 {
        cv.push_i(&eng, Count::Fast(v));
    }
    let (_fast, big) = cv.clone_guarded(&eng).into_parts();
    assert!(
        big.is_none(),
        "a store with no overflow must not allocate a table"
    );

    // One overflow in a wide store costs one entry, not one per slot.
    let mut wide = CountVec::with_width(&eng, 0);
    for v in 0..64u128 {
        wide.push_i(&eng, Count::Fast(v));
    }
    wide.push_i(&eng, Count::Big(BigUint::from(1u32) << 200usize));
    for v in 0..64u128 {
        wide.push_i(&eng, Count::Fast(v));
    }
    let (fast, big) = wide.clone_guarded(&eng).into_parts();
    assert_eq!(fast.len(), 129);
    let big = big.expect("the overflow slot needs a table");
    assert_eq!(
        big.len(),
        1,
        "sparse table holds one entry per OVERFLOW slot"
    );
    assert_eq!(
        wide.big_val(&eng, 64),
        Some(BigUint::from(1u32) << 200usize)
    );
}

#[test]
fn refused_big_set_preserves_the_previous_value() {
    let eng = Engine::new();
    let mut col = CountVec::try_with_width(&eng, 1).unwrap();
    col.set(&eng, 0, Count::Fast(7)).unwrap();
    eng.limits().refuse_nth_reserve(0);
    assert_eq!(col.set(&eng, 0, Count::Big(BigUint::from(1u32) << 200)), Err(crate::OperationError::OverBudget));
    eng.limits().grant_every_reserve();
    assert_eq!(col.get(0).to_count(), Count::Fast(7));
    assert!(col.all_u64());
    col.set(&eng, 0, Count::Big(BigUint::from(1u32) << 200)).unwrap();
    assert_eq!(col.get(0).to_count(), Count::Big(BigUint::from(1u32) << 200));
}

#[test]
fn refused_big_push_preserves_length_and_slot_alignment() {
    let eng = Engine::new();
    let mut col = CountVec::try_with_capacity(&eng, 2).unwrap();
    col.push(&eng, Count::Fast(7)).unwrap();
    eng.limits().refuse_nth_reserve(1);
    assert_eq!(col.push(&eng, Count::Big(BigUint::from(1u32) << 200)), Err(crate::OperationError::OverBudget));
    eng.limits().grant_every_reserve();
    assert_eq!(col.len(), 1);
    assert_eq!(col.get(0).to_count(), Count::Fast(7));
    assert!(col.all_u64());
    col.push(&eng, Count::Big(BigUint::from(1u32) << 200)).unwrap();
    assert_eq!(col.len(), 2);
    assert_eq!(col.get(1).to_count(), Count::Big(BigUint::from(1u32) << 200));
}

#[test]
fn query_and_streaming_folds_match_exact_products_across_storage_boundaries() {
    let eng = Engine::new();
    let values = vec![
        Count::Fast(0), Count::Fast(1), Count::Fast(u64::MAX as u128),
        Count::Fast(u64::MAX as u128 + 1), Count::Fast(u128::MAX - 1),
        Count::Big(BigUint::from(u128::MAX)),
        Count::Big(BigUint::from(1u32) << 180), Count::Big(BigUint::ZERO),
    ];
    let exact: Vec<BigUint> = values.iter().map(|v| match v {
        Count::Fast(v) => BigUint::from(*v),
        Count::Big(v) => v.clone(),
    }).collect();
    let column = col(&eng, values);
    let pairs: Vec<_> = (0..exact.len()).flat_map(|l| {
        (0..exact.len()).map(move |r| pair(l as u32, r as u32))
    }).collect();
    let sum: BigUint = exact.iter().sum();
    let expected_total = &sum * &sum;
    let as_big = |value| match value {
        Count::Fast(v) => BigUint::from(v),
        Count::Big(v) => v,
    };
    for left_marginal in [false, true] {
        for right_marginal in [false, true] {
            let left = StreamChild::<IntFold> { col: column.as_count_ref(), is_marginal: left_marginal };
            let right = StreamChild::<IntFold> { col: column.as_count_ref(), is_marginal: right_marginal };
            for (i, pair) in pairs.iter().enumerate() {
                let expected = &exact[i / exact.len()] * &exact[i % exact.len()];
                let streamed = IntFold::fold_cell(std::slice::from_ref(pair), &left, &right, &());
                let queried = IntFold::fold(std::iter::once(*pair),
                    |k| column.get(k.raw() as usize), |k| column.get(k.raw() as usize));
                assert_eq!(as_big(streamed), expected);
                assert_eq!(as_big(queried), expected);
            }
            assert_eq!(as_big(IntFold::fold_cell(&pairs, &left, &right, &())), expected_total);
        }
    }
}

#[test]
fn streaming_exact_fallback_handles_inline_counts_and_accumulation_overflow() {
    use crate::diagram::{ChildPair, EncodedChildRef, ValueRef};
    let eng = Engine::new();
    let large = BigUint::from(1u32) << 180usize;
    let column = col(&eng, vec![Count::Big(large.clone())]);
    let empty = CountVec::default();
    let left = StreamChild::<IntFold> { col: empty.as_count_ref(), is_marginal: true };
    let right = StreamChild::<IntFold> { col: column.as_count_ref(), is_marginal: true };
    let inline = [0, 7, crate::diagram::MARGINAL_INLINE_MAX];
    let pairs: Vec<_> = inline.into_iter().map(|n| ChildPair {
        left: EncodedChildRef::from_raw(ValueRef::Inline(n).to_raw().0),
        right: EncodedChildRef::from_raw(0),
    }).collect();
    assert_eq!(IntFold::fold_cell(&pairs, &left, &right, &()),
        Count::Big(large * (7u64 + crate::diagram::MARGINAL_INLINE_MAX as u64)));

    // Individual widening products fit u128, but their sum does not.
    let column = col(&eng, vec![Count::Fast(u64::MAX as u128)]);
    let side = StreamChild::<IntFold> { col: column.as_count_ref(), is_marginal: false };
    assert_eq!(IntFold::fold_cell(&[pair(0, 0); 3], &side, &side, &()),
        Count::Big(BigUint::from(u64::MAX).pow(2) * 3u32));
}
