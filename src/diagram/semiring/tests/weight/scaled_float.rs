use super::{same_value, weight_key, ScaledFloat, ScaledSum, SignedLog, WeightValue};
use num_bigint::BigInt;
use num_rational::BigRational;

use crate::test_helpers::rat;

/// A value's mantissa is in `[0.5, 1)` in magnitude, or it is the one zero.
fn assert_normal(v: &ScaledFloat) {
    if v.is_zero() {
        assert_eq!(v.mantissa().to_bits(), 0.0f64.to_bits(), "zero's mantissa is +0.0");
        assert_eq!(v.exponent(), 0, "zero's exponent is 0");
    } else {
        assert!((0.5..1.0).contains(&v.mantissa().abs()), "mantissa {} out of [0.5, 1)", v.mantissa());
    }
}

/// Deterministic floats over a few dozen binades, both signs.
fn floats(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let unit = (s >> 11) as f64 / (1u64 << 53) as f64;
            let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
            sign * (unit + 0.01) * 2f64.powi(((s >> 3) % 40) as i32 - 20)
        })
        .collect()
}

#[test]
fn from_rational_converts_and_normalizes() {
    for (r, x) in [(rat(1, 2), 0.5), (rat(100, 1), 100.0), (rat(-7, 4), -1.75), (rat(3, 10), 0.3), (rat(1, 3), 1.0 / 3.0)] {
        let v = ScaledFloat::from_rational(&r);
        assert_normal(&v);
        assert_eq!(v.to_f64(), x, "{r}");
    }
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10_000_000_000_000_000i64));
    let v = ScaledFloat::from_rational(&tiny);
    assert!((v.log10_abs() + 16.0).abs() < 1e-12);
    let z = ScaledFloat::from_rational(&rat(0, 1));
    assert!(z.is_zero());
    assert_normal(&z);
    assert_eq!(z.log10_abs(), f64::NEG_INFINITY);
}

#[test]
fn from_rational_reaches_past_the_range_of_a_float() {
    let big = BigInt::from(1) << 5000usize;
    let v = ScaledFloat::from_rational(&BigRational::new(big.clone(), BigInt::from(3)));
    assert_normal(&v);
    assert!((v.log10_abs() - (5000.0 * std::f64::consts::LOG10_2 - 3f64.log10())).abs() < 1e-12);
    assert_eq!(v.to_f64(), f64::INFINITY);
    let v = ScaledFloat::from_rational(&BigRational::new(BigInt::from(-3), big));
    assert!((v.log10_abs() - (3f64.log10() - 5000.0 * std::f64::consts::LOG10_2)).abs() < 1e-12);
    assert!(v.mantissa() < 0.0);
    assert_eq!(v.to_f64(), 0.0);
}

#[test]
fn from_f64_is_exact_for_subnormals_too() {
    for x in [1.0, -3.5, f64::MIN_POSITIVE, f64::MIN_POSITIVE / 8.0, -f64::from_bits(1), f64::MAX] {
        let v = ScaledFloat::from_f64(x);
        assert_normal(&v);
        assert_eq!(v.to_f64(), x);
    }
}

/// Within the range of an `f64`, a product or a sum is the `f64` one, bit
/// for bit: scaling a mantissa by a power of two is exact, so each operation
/// rounds where the `f64` operation does.
#[test]
fn within_range_the_operations_are_those_of_f64() {
    let xs = floats(7, 200);
    let ys = floats(11, 200);
    for (&x, &y) in xs.iter().zip(&ys) {
        let (a, b) = (ScaledFloat::from_f64(x), ScaledFloat::from_f64(y));
        let p = a.mul(&b);
        assert_normal(&p);
        assert_eq!(p.to_f64(), x * y, "{x} * {y}");
        let mut s = a;
        s.add_assign(&b);
        assert_normal(&s);
        assert_eq!(s.to_f64(), x + y, "{x} + {y}");
    }
    // A running sum and product, rounded at every step as `f64` rounds.
    let (mut sum, mut want) = (ScaledFloat::ZERO, 0.0);
    for &x in &xs {
        sum.add_assign(&ScaledFloat::from_f64(x));
        want += x;
    }
    assert_eq!(sum.to_f64(), want);
}

#[test]
fn signs_cancellation_and_zero() {
    let five = ScaledFloat::from_rational(&rat(5, 1));
    let mut a = five;
    a.add_assign(&ScaledFloat::from_rational(&rat(-5, 1)));
    assert!(a.is_zero());
    assert_normal(&a);
    a.add_assign(&ScaledFloat::from_f64(2.0));
    assert_eq!(a.to_f64(), 2.0, "adding to a cancelled sum is the addend");

    let mut a = five;
    a.add_assign(&ScaledFloat::from_rational(&rat(-3, 1)));
    assert_eq!(a.to_f64(), 2.0);
    let mut a = ScaledFloat::from_rational(&rat(-5, 1));
    a.add_assign(&ScaledFloat::from_rational(&rat(3, 1)));
    assert_eq!(a.to_f64(), -2.0);

    let n = ScaledFloat::from_f64(-3.0);
    assert_eq!(five.mul(&n).to_f64(), -15.0);
    assert_eq!(n.mul(&n).to_f64(), 9.0);
    let z = n.mul(&ScaledFloat::ZERO);
    assert!(z.is_zero());
    assert_normal(&z);
    let mut a = five;
    a.add_assign(&ScaledFloat::ZERO);
    assert_eq!(a, five);
}

/// Magnitudes far outside a float's range multiply and add like any other,
/// and a term below the larger one's last place leaves it as it is.
#[test]
fn wide_magnitudes() {
    let half = ScaledFloat::from_f64(0.5);
    let mut p = ScaledFloat::from_f64(1.0);
    for _ in 0..3000 {
        p = p.mul(&half);
    }
    assert_normal(&p);
    assert_eq!((p.mantissa(), p.exponent()), (0.5, -2999));
    assert!((p.log10_abs() + 3000.0 * std::f64::consts::LOG10_2).abs() < 1e-9);
    let mut twice = p;
    twice.add_assign(&p);
    assert_eq!((twice.mantissa(), twice.exponent()), (0.5, -2998));

    let mut big = ScaledFloat::from_f64(1.0);
    let two = ScaledFloat::from_f64(2.0);
    for _ in 0..2200 {
        big = big.mul(&two);
    }
    let mut s = big;
    s.add_assign(&p);
    assert_eq!(s, big, "2^-3000 under 2^2200's last place");
    let mut s = p;
    s.add_assign(&big);
    assert_eq!(s, big);
    // 2^64 apart: under half the last place, the larger is the rounded sum.
    let mut s = ScaledFloat::from_f64(1.0);
    s.add_assign(&ScaledFloat::from_f64(2f64.powi(-64)));
    assert_eq!(s.to_f64(), 1.0);
    // 2^52 apart: one last place, kept.
    let mut s = ScaledFloat::from_f64(1.0);
    s.add_assign(&ScaledFloat::from_f64(2f64.powi(-52)));
    assert_eq!(s.to_f64(), 1.0 + f64::EPSILON);
}

/// One number, one representation: values reached by different routes key
/// alike.
#[test]
fn equal_values_share_a_key() {
    let quarter = ScaledFloat::from_f64(0.5).mul(&ScaledFloat::from_f64(0.5));
    let one = quarter.mul(&ScaledFloat::from_f64(4.0));
    let mut also_one = ScaledFloat::from_f64(0.75);
    also_one.add_assign(&quarter);
    let a = WeightValue::scaled(one);
    let b = WeightValue::scaled(also_one);
    let c = WeightValue::scaled(ScaledFloat::from_rational(&rat(1, 1)));
    assert!(same_value(&a, &b) && same_value(&a, &c));
    assert!(weight_key(&a) == weight_key(&b) && weight_key(&b) == weight_key(&c));
    let mut zero = WeightValue::scaled(ScaledFloat::from_f64(-0.5));
    zero.add_assign(&WeightValue::scaled(ScaledFloat::from_f64(0.5)));
    assert!(zero.is_zero());
    assert!(same_value(&zero, &WeightValue::scaled(ScaledFloat::ZERO)));
    assert!(!same_value(&a, &WeightValue::exact(rat(1, 1))));
}

#[test]
fn weightval_scaled_ops() {
    let mut a = WeightValue::scaled(ScaledFloat::from_rational(&rat(1, 2)));
    let b = WeightValue::scaled(ScaledFloat::from_rational(&rat(1, 3)));
    let p = a.mul(&b);
    assert_eq!(p.as_scaled().unwrap().to_f64(), 0.5 * (1.0 / 3.0));
    a.add_assign(&b);
    assert_eq!(a.as_scaled().unwrap().to_f64(), 0.5 + 1.0 / 3.0);
    let scaled = a.mul_rational(&rat(6, 5));
    assert!((scaled.as_scaled().unwrap().to_f64() - 1.0).abs() < 1e-15);
    assert!(a.as_log().is_none() && a.clone().into_rational_opt().is_none());
}

#[test]
#[should_panic(expected = "mixed arithmetics")]
fn scaled_times_log_panics() {
    let a = WeightValue::scaled(ScaledFloat::from_f64(0.5));
    let _ = a.mul(&WeightValue::log(SignedLog::from_rational(&rat(1, 2))));
}

#[test]
#[should_panic(expected = "mixed arithmetics")]
fn exact_plus_scaled_panics() {
    let mut a = WeightValue::exact(rat(1, 2));
    a.add_assign(&WeightValue::scaled(ScaledFloat::from_f64(0.5)));
}

/// Scaled floats of either sign over `binades` binades, with zeros.
fn spread(seed: u64, n: usize, binades: u64) -> Vec<ScaledFloat> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            if s.is_multiple_of(17) {
                return ScaledFloat::ZERO;
            }
            let mut v = ScaledFloat::from_f64(if s & 2 == 0 { 1.0 } else { -1.0 } * (0.5 + (s >> 12) as f64 / (1u64 << 53) as f64));
            v.exponent += ((s >> 4) % binades) as i64 - (binades / 2) as i64;
            v
        })
        .collect()
}

/// The unnormalized product sum leaves what the normalized operations
/// leave, bit for bit: near and far exponents, cancellation to zero and
/// below `2^-64`, and gaps beyond its span.
#[test]
fn product_sum_matches_mul_and_add_assign() {
    for (seed, binades) in [(1u64, 4u64), (2, 60), (3, 200), (4, 3000), (5, 1)] {
        for len in [1usize, 2, 3, 7, 40, 300] {
            let a = spread(seed * 1000 + len as u64, len, binades);
            let b = spread(seed * 7777 + len as u64, len, binades);
            let mut want = ScaledFloat::ZERO;
            let mut got = ScaledSum::ZERO;
            for (x, y) in a.iter().zip(&b) {
                want.add_assign(&x.mul(y));
                got.add_product(x, y);
                let g = got.finish();
                assert_normal(&g);
                assert_eq!((g.mantissa().to_bits(), g.exponent()), (want.mantissa().to_bits(), want.exponent()),
                    "seed {seed}, {binades} binades, {len} terms");
            }
        }
    }
    // Exact cancellation, then a term far below the cancelled ones.
    let one = ScaledFloat::from_f64(1.0);
    let mut tiny = ScaledFloat::from_f64(0.75);
    tiny.exponent -= 2000;
    let terms = [(one, one), (ScaledFloat::from_f64(-1.0), one), (tiny, one), (ScaledFloat::from_f64(1.0 + f64::EPSILON), one),
        (ScaledFloat::from_f64(-1.0), one)];
    let mut want = ScaledFloat::ZERO;
    let mut got = ScaledSum::ZERO;
    for (x, y) in &terms {
        want.add_assign(&x.mul(y));
        got.add_product(x, y);
        let g = got.finish();
        assert_eq!((g.mantissa().to_bits(), g.exponent()), (want.mantissa().to_bits(), want.exponent()));
    }
}
