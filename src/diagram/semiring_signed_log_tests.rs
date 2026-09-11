use super::{SignedLog, WeightVal};
use num_bigint::BigInt;
use num_rational::BigRational;

use crate::test_helpers::rat;

#[test]
fn from_rational_log10_roundtrip() {
    // 0.5 → log10 = -0.30103
    let s = SignedLog::from_rational(&rat(1, 2));
    assert_eq!(s.sign, 1);
    assert!((s.log10_abs() - (0.5f64).log10()).abs() < 1e-9);

    // 100 → log10 = 2
    let s = SignedLog::from_rational(&rat(100, 1));
    assert_eq!(s.sign, 1);
    assert!((s.log10_abs() - 2.0).abs() < 1e-9);

    // 1e-16 = 1/10^16 → log10 = -16
    let s = SignedLog::from_rational(&BigRational::new(
        BigInt::from(1),
        BigInt::from(10_000_000_000_000_000i64),
    ));
    assert_eq!(s.sign, 1);
    assert!((s.log10_abs() - (-16.0)).abs() < 1e-9);

    // negative value tracks sign
    let s = SignedLog::from_rational(&rat(-7, 4));
    assert_eq!(s.sign, -1);
    assert!((s.log10_abs() - (1.75f64).log10()).abs() < 1e-9);

    // zero
    let z = SignedLog::from_rational(&rat(0, 1));
    assert_eq!(z.sign, 0);
    assert_eq!(z.ln_abs, f64::NEG_INFINITY);
}

#[test]
fn signed_cancellation_and_subtract() {
    // +5 + (-5) → exact cancellation, sign 0
    let mut a = SignedLog::from_rational(&rat(5, 1));
    let neg5 = SignedLog::from_rational(&rat(-5, 1));
    a.add_assign(&neg5);
    assert_eq!(a.sign, 0);
    assert_eq!(a.ln_abs, f64::NEG_INFINITY);

    // +5 + (-3) → +2, ln_abs ≈ ln 2
    let mut a = SignedLog::from_rational(&rat(5, 1));
    let neg3 = SignedLog::from_rational(&rat(-3, 1));
    a.add_assign(&neg3);
    assert_eq!(a.sign, 1);
    assert!((a.ln_abs - (2.0f64).ln()).abs() < 1e-9);

    // (-5) + (+3) → -2
    let mut a = SignedLog::from_rational(&rat(-5, 1));
    let pos3 = SignedLog::from_rational(&rat(3, 1));
    a.add_assign(&pos3);
    assert_eq!(a.sign, -1);
    assert!((a.ln_abs - (2.0f64).ln()).abs() < 1e-9);

    // Same sign: +5 + +3 → +8
    let mut a = SignedLog::from_rational(&rat(5, 1));
    let pos3 = SignedLog::from_rational(&rat(3, 1));
    a.add_assign(&pos3);
    assert_eq!(a.sign, 1);
    assert!((a.ln_abs - (8.0f64).ln()).abs() < 1e-9);

    // add zero is a no-op
    let mut a = SignedLog::from_rational(&rat(5, 1));
    a.add_assign(&SignedLog::zero());
    assert_eq!(a.sign, 1);
    assert!((a.ln_abs - (5.0f64).ln()).abs() < 1e-9);
}

#[test]
fn mul_sign_rules() {
    let p = SignedLog::from_rational(&rat(2, 1));
    let n = SignedLog::from_rational(&rat(-3, 1));
    // + * - = -, magnitude 6
    let r = p.mul(&n);
    assert_eq!(r.sign, -1);
    assert!((r.ln_abs - (6.0f64).ln()).abs() < 1e-9);
    // - * - = +
    let r = n.mul(&n);
    assert_eq!(r.sign, 1);
    assert!((r.ln_abs - (9.0f64).ln()).abs() < 1e-9);
    // anything * 0 = 0
    let r = p.mul(&SignedLog::zero());
    assert_eq!(r.sign, 0);
}

#[test]
fn weightval_log_ops() {
    let mut a = WeightVal::Log(SignedLog::from_rational(&rat(1, 2)));
    let b = WeightVal::Log(SignedLog::from_rational(&rat(1, 3)));
    let p = a.mul(&b); // 1/6
    if let WeightVal::Log(s) = p {
        assert!((s.log10_abs() - (1.0f64 / 6.0).log10()).abs() < 1e-9);
    } else {
        panic!("expected Log");
    }
    a.add_assign(&b); // 1/2 + 1/3 = 5/6
    if let WeightVal::Log(s) = a {
        assert_eq!(s.sign, 1);
        assert!((s.log10_abs() - (5.0f64 / 6.0).log10()).abs() < 1e-9);
    } else {
        panic!("expected Log");
    }
}

/// A sum whose magnitudes cancel is the canonical zero, and stays a number
/// when something is added to it.
///
/// The cancellation here is not exact in the log domain: `ln 1` and
/// `ln(1 + 1e-17)` are different reals, but `e^(ln 1 - ln(1+1e-17))` rounds to
/// `1` in `f64`, so `ln(1 - 1)` = `-∞`. Left with the larger term's sign that
/// is not zero, and the next `add_assign` reaches `-∞ - -∞` = NaN.
#[test]
fn a_cancelled_sum_is_zero_and_adds() {
    let mut v = SignedLog { ln_abs: 0.0, sign: 1 };
    v.add_assign(&SignedLog { ln_abs: 1e-17, sign: -1 });
    assert_eq!(v, SignedLog::zero(), "the cancellation normalizes to the canonical zero");
    assert!(v.is_zero());

    v.add_assign(&SignedLog { ln_abs: 2.0, sign: 1 });
    assert_eq!(v, SignedLog { ln_abs: 2.0, sign: 1 }, "adding to a cancelled sum is the addend");

    // And the same value is absorbing under multiplication rather than NaN.
    let zero_by_sign = SignedLog { ln_abs: f64::NEG_INFINITY, sign: 1 };
    assert!(zero_by_sign.is_zero());
    assert_eq!(zero_by_sign.mul(&SignedLog { ln_abs: 3.0, sign: -1 }), SignedLog::zero());
    let mut acc = zero_by_sign;
    acc.add_assign(&zero_by_sign);
    assert_eq!(acc, SignedLog::zero(), "adding two underflowed magnitudes is zero, not NaN");
}
