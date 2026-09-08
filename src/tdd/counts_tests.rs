use super::*;

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
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Fast(3));
    cv.push_i(Count::Fast(5));
    let big_val = BigUint::from(u64::MAX) * BigUint::from(2u32);
    cv.push_i(Count::Big(big_val.clone()));
    assert_eq!(cv.fast_val(2), STREAM_OVERFLOW);
    match cv.get(2) {
        CountRead::Big(b) => assert_eq!(*b, big_val),
        CountRead::Fast(_) => panic!("expected Big read"),
    }
    assert!(cv.has_big());
    // Backfilled slots stay None; the big table is index-aligned (no
    // out-of-bounds panic reading any slot up to len()).
    assert!(cv.big_val(0).is_none());
    assert!(cv.big_val(1).is_none());
    assert_eq!(cv.big_val(2), Some(&big_val));
    assert_eq!(cv.len(), 3);
}

#[test]
fn all_u64_true_after_u64_range_pushes() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Fast(1));
    cv.push_i(Count::Fast(u64::MAX as u128));
    assert!(cv.all_u64());
}

#[test]
fn all_u64_cleared_by_over_u64_fast_push_and_never_returns() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Fast(1));
    assert!(cv.all_u64());
    cv.push_i(Count::Fast(u64::MAX as u128 + 1));
    assert!(!cv.all_u64());
    cv.push_i(Count::Fast(2));
    assert!(!cv.all_u64(), "certificate must not return to true");
}

#[test]
fn all_u64_cleared_by_big_push_and_never_returns() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Fast(1));
    assert!(cv.all_u64());
    cv.push_i(Count::Big(BigUint::from(7u32)));
    assert!(!cv.all_u64());
    cv.set_i(0, Count::Fast(9));
    assert!(!cv.all_u64(), "sentinel slot must defeat the certificate permanently");
}

#[test]
fn set_overwrites_big_with_fast_clears_big_slot() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(2);
    cv.set_i(0, Count::Big(BigUint::from(99u32)));
    assert!(cv.big_val(0).is_some());
    cv.set_i(0, Count::Fast(5));
    assert_eq!(cv.fast_val(0), 5);
    assert!(cv.big_val(0).is_none(), "big slot must clear to None when overwritten by Fast");
}

/// Mirrors `conjoin::cell::tests::collect_sink_pushes_charge_the_soft_budget`'s
/// install/reset hygiene: a tiny soft budget must trip `ApplyBudget`'s
/// `Err(ApplyError::OverBudget)` well before an unbudgeted push loop would
/// exhaust real memory.
#[test]
fn apply_budget_policy_trips_over_budget() {
    let _g = crate::tdd::limits::apply_limits().budget(Some(64)).apply();
    let mut cv = CountVec::<ApplyBudget>::try_with_width(0)
        .expect("width-0 allocation must not trip a 64-byte budget");
    let mut result = Ok(());
    for _ in 0..1024 {
        result = cv.push(Count::Fast(1));
        if result.is_err() {
            break;
        }
    }
    assert!(
        matches!(result, Err(crate::tdd::limits::ApplyError::OverBudget)),
        "CountVec<ApplyBudget> pushes bypass the apply soft budget"
    );
}

#[test]
fn try_clone_round_trips_fast_and_big() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Fast(3));
    cv.push_i(Count::Big(BigUint::from(123456789u64)));
    cv.push_i(Count::Fast(7));
    let clone = cv.clone_guarded();
    assert_eq!(clone.len(), cv.len());
    for i in 0..cv.len() {
        assert_eq!(clone.fast_val(i), cv.fast_val(i));
        assert_eq!(clone.big_val(i).cloned(), cv.big_val(i).cloned());
    }
    assert_eq!(clone.all_u64(), cv.all_u64());
}

fn pair(l: u32, r: u32) -> crate::tdd::types::InputPair {
    crate::tdd::types::InputPair {
        left: crate::tdd::types::LocalNodeIdx(l),
        right: crate::tdd::types::LocalNodeIdx(r),
    }
}

/// Readers over a fixture column, mirroring how the ensure-walk adapters
/// hand `CountVec::get` closures to the fold.
fn col(vals: Vec<Count>) -> CountVec<RecoveryPanic> {
    let mut cv = CountVec::<RecoveryPanic>::with_width(vals.len());
    for (i, v) in vals.into_iter().enumerate() {
        cv.set_i(i, v);
    }
    cv
}

#[test]
fn int_fold_stays_fast_within_u128() {
    let left = col(vec![Count::Fast(3), Count::Fast(5)]);
    let right = col(vec![Count::Fast(7), Count::Fast(11)]);
    let pairs = [pair(0, 0), pair(1, 1)];
    match IntFold::fold(pairs.iter().copied(), |k| left.get(k), |k| right.get(k)) {
        Count::Fast(v) => assert_eq!(v, 3 * 7 + 5 * 11),
        Count::Big(b) => panic!("spurious overflow: {b}"),
    }
}

#[test]
fn int_fold_overflow_repass_is_exact_and_mixed_magnitude() {
    // Slot 0: huge fast values whose product overflows u128 (pass-2
    // Fast×Fast checked_mul-fails sub-case). Slot 1: a Big child × a small
    // fast child (mixed sub-case). Slot 2: tiny (both-fast sub-case).
    let huge = u64::MAX as u128 + 7; // > u64, still fast-representable
    let big_child = BigUint::from(u128::MAX) * BigUint::from(3u32);
    let left = col(vec![
        Count::Fast(huge),
        Count::Big(big_child.clone()),
        Count::Fast(7),
    ]);
    let right = col(vec![Count::Fast(huge), Count::Fast(5), Count::Fast(11)]);
    let pairs = [pair(0, 0), pair(1, 1), pair(2, 2)];
    let expected = BigUint::from(huge) * BigUint::from(huge)
        + &big_child * BigUint::from(5u32)
        + BigUint::from(77u32);
    match IntFold::fold(pairs.iter().copied(), |k| left.get(k), |k| right.get(k)) {
        Count::Big(b) => assert_eq!(b, expected),
        Count::Fast(v) => panic!("expected overflow → Big, got Fast({v})"),
    }
}

#[test]
fn int_fold_exact_max_total_promotes_to_big() {
    // (2^64+1)·(2^64−1) = 2^128−1 = u128::MAX exactly: pass 1 completes
    // without overflowing, but the total IS the sentinel — from_u128 must
    // promote so the stored value stays unambiguous.
    let left = col(vec![Count::Fast((1u128 << 64) + 1)]);
    let right = col(vec![Count::Fast((1u128 << 64) - 1)]);
    let pairs = [pair(0, 0)];
    match IntFold::fold(pairs.iter().copied(), |k| left.get(k), |k| right.get(k)) {
        Count::Big(b) => assert_eq!(b, BigUint::from(u128::MAX)),
        Count::Fast(v) => panic!("exact-max total must promote to Big, got Fast({v})"),
    }
}

#[test]
fn weight_fold_sums_products_exactly() {
    use num_rational::BigRational;
    let q = |n: i64, d: i64| {
        WeightVal::exact(BigRational::new(n.into(), d.into()))
    };
    let left = [q(1, 2), q(3, 4)];
    let right = [q(1, 3), q(2, 5)];
    let pairs = [pair(0, 0), pair(1, 1)];
    let got = WeightFold::fold(
        pairs.iter().copied(),
        |k| std::borrow::Cow::Borrowed(&left[k]),
        |k| std::borrow::Cow::Borrowed(&right[k]),
        q(0, 1),
    );
    // 1/2·1/3 + 3/4·2/5 = 1/6 + 3/10 = 7/15
    let got = got.into_rational_opt().expect("expected Exact");
    assert_eq!(got, BigRational::new(7.into(), 15.into()));
}

/// A1 (weighted streaming fallible-allocation parity): the weighted
/// scratch column must reserve through the apply soft budget exactly like
/// the integer one — before stage 3 the weighted ensure walk allocated
/// with an infallible `vec![wzero; width]` that bypassed the budget (and
/// would abort rather than cooperatively recover at the ceiling).
#[test]
fn a1_weighted_column_alloc_charges_soft_budget() {
    use num_rational::BigRational;
    let _g = crate::tdd::limits::apply_limits().budget(Some(64)).apply();
    let zero = WeightVal::exact(BigRational::new(0.into(), 1.into()));
    let res = WeightFold::alloc_col::<ApplyBudget>(4096, &zero);
    assert!(
        matches!(res, Err(crate::tdd::limits::ApplyError::OverBudget)),
        "weighted column allocation bypasses the apply soft budget"
    );
}

/// Regression guard for the sparse side-table discipline: the overflow table
/// is keyed by slot, so `Fast` pushes after a `Big` push must add nothing to
/// it — an absent entry IS the "fits the fast lane" encoding — and a read at a
/// fast-lane slot must resolve as fast-only, not panic.
#[test]
fn fast_push_after_big_leaves_side_table_sparse() {
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    cv.push_i(Count::Big(BigUint::from(42u32)));
    cv.push_i(Count::Fast(7));
    cv.push_i(Count::Fast(9));
    let (fast, big) = cv.clone_guarded().into_parts();
    assert_eq!(fast.len(), 3);
    assert_eq!(
        big.expect("big table exists").len(), 1,
        "fast pushes must not add overflow entries",
    );
    assert_eq!(cv.big_val(0).cloned(), Some(BigUint::from(42u32)));
    assert!(cv.big_val(2).is_none(), "fast-lane slot must read None, not panic");
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
    let mut cv = CountVec::<RecoveryPanic>::with_width(0);
    for v in 0..64u128 {
        cv.push_i(Count::Fast(v));
    }
    let (_fast, big) = cv.clone_guarded().into_parts();
    assert!(big.is_none(), "a store with no overflow must not allocate a table");

    // One overflow in a wide store costs one entry, not one per slot.
    let mut wide = CountVec::<RecoveryPanic>::with_width(0);
    for v in 0..64u128 {
        wide.push_i(Count::Fast(v));
    }
    wide.push_i(Count::Big(BigUint::from(1u32) << 200usize));
    for v in 0..64u128 {
        wide.push_i(Count::Fast(v));
    }
    let (fast, big) = wide.clone_guarded().into_parts();
    assert_eq!(fast.len(), 129);
    let big = big.expect("the overflow slot needs a table");
    assert_eq!(big.len(), 1, "sparse table holds one entry per OVERFLOW slot");
    assert_eq!(wide.big_val(64).cloned(), Some(BigUint::from(1u32) << 200usize));
}
