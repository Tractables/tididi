use num_bigint::BigUint;
use super::*;
use crate::diagram::ValueRef;

/// A sum landing exactly on `u128::MAX` must come back as `Big`, never
/// `Small(u128::MAX)`: stored as a small count it IS the OVERFLOW sentinel,
/// with no `marginal_counts_big` entry, and the next `count_key_at` read of
/// that slot panics.
#[test]
fn sum_exactly_u128_max_routes_to_big() {
    let counts = vec![u128::MAX - 1, 1u128];
    let refs = [ValueRef::slot_raw(0), ValueRef::slot_raw(1)];
    match sum_marginal_counts(CountRef::new(&counts, None), &refs) {
        Count::Big(b) => assert_eq!(b, BigUint::from(u128::MAX)),
        Count::Fast(c) => panic!("sum {c} collides with the OVERFLOW sentinel"),
    }
}

/// One below the sentinel is still a plain small sum.
#[test]
fn sum_below_sentinel_stays_small() {
    let counts = vec![u128::MAX - 2, 1u128];
    let refs = [ValueRef::slot_raw(0), ValueRef::slot_raw(1)];
    match sum_marginal_counts(CountRef::new(&counts, None), &refs) {
        Count::Fast(c) => assert_eq!(c, u128::MAX - 1),
        Count::Big(b) => panic!("small sum must not promote to Big({b})"),
    }
}
