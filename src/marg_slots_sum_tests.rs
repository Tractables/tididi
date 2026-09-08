use super::*;
use crate::diagram::MargRef;

/// A sum landing exactly on `u128::MAX` must come back as `Big`, never
/// `Small(u128::MAX)`: stored as a small count it IS the OVERFLOW sentinel,
/// with no `marginal_counts_big` entry, and the next `count_key_at` read of
/// that slot panics.
#[test]
fn sum_exactly_u128_max_routes_to_big() {
    let counts = vec![u128::MAX - 1, 1u128];
    let refs = [MargRef::slot_raw(0), MargRef::slot_raw(1)];
    match sum_marginal_counts(&counts, None, &refs) {
        CountKey::Big(b) => assert_eq!(b, BigUint::from(u128::MAX)),
        CountKey::Small(c) => panic!("sum {c} collides with the OVERFLOW sentinel"),
    }
}

/// One below the sentinel is still a plain small sum.
#[test]
fn sum_below_sentinel_stays_small() {
    let counts = vec![u128::MAX - 2, 1u128];
    let refs = [MargRef::slot_raw(0), MargRef::slot_raw(1)];
    match sum_marginal_counts(&counts, None, &refs) {
        CountKey::Small(c) => assert_eq!(c, u128::MAX - 1),
        CountKey::Big(b) => panic!("small sum must not promote to Big({b})"),
    }
}
