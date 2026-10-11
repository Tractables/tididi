//! The weighted-marginal value type: its two exact representations, its two
//! bounded-precision alternatives (signed log domain and scaled float), and
//! the dedup key over them.

use std::borrow::Cow;

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

// ── Bounded-precision signed log-domain weight (weighted marginal path) ───────────

/// A signed value represented by an `f64` log-magnitude and a separate sign.
///
/// Used by [`Arithmetic::SignedLog`](crate::diagram::Arithmetic::SignedLog).
/// Zero has sign `0` and log-magnitude `f64::NEG_INFINITY`; nonzero signs are
/// `-1` and `1`. Keeping the logarithm permits magnitudes beyond the range of an
/// ordinary `f64` weight, and the sign permits negative literal weights.
///
/// Arithmetic has bounded precision. In particular, cancellation of nearly
/// equal values can cause large relative error; use exact rational arithmetic
/// when the result must be exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignedLog {
    /// Natural log of the magnitude (`f64::NEG_INFINITY` when `sign == 0`).
    pub ln_abs: f64,
    /// Sign of the value: `-1`, `0`, or `+1`.
    pub sign: i8,
}

impl SignedLog {
    /// The additive identity (`sign == 0`, `ln_abs == -∞`).
    pub fn zero() -> Self {
        SignedLog { ln_abs: f64::NEG_INFINITY, sign: 0 }
    }

    /// Whether this value is zero.
    ///
    /// A zero is canonically `{ ln_abs: -∞, sign: 0 }`, but a magnitude that has
    /// underflowed to `-∞` under a sign is the same number, and every operation
    /// below reads zero through this predicate so that such a value behaves like
    /// one instead of producing `-∞ - -∞` = NaN.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.sign == 0 || self.ln_abs == f64::NEG_INFINITY
    }

    /// Convert an exact `BigRational` to signed log domain without overflow.
    pub fn from_rational(r: &BigRational) -> Self {
        use num_bigint::Sign;
        if r.numer().is_zero() {
            return Self::zero();
        }
        let sign: i8 = if r.numer().sign() == Sign::Minus { -1 } else { 1 };
        // ln|r| = ln(|numer|) - ln(denom); denom is always positive in a
        // normalized BigRational.
        let ln_abs = ln_bigint_abs(r.numer()) - ln_bigint_abs(r.denom());
        SignedLog { ln_abs, sign }
    }

    /// Multiply two signed-log values: add the log-magnitudes, multiply the signs.
    #[inline]
    #[must_use]
    pub fn mul(&self, o: &SignedLog) -> SignedLog {
        if self.is_zero() || o.is_zero() {
            // Zero absorbs, and taking it here keeps `-∞ + -∞` out of the sum.
            return SignedLog::zero();
        }
        let ln_abs = self.ln_abs + o.ln_abs;
        if ln_abs == f64::NEG_INFINITY {
            // The product underflowed the log domain's range; it is zero, and
            // only the canonical zero is closed under further arithmetic.
            return SignedLog::zero();
        }
        SignedLog { ln_abs, sign: self.sign * o.sign }
    }

    /// Signed log-sum-exp accumulate: self += o.
    pub fn add_assign(&mut self, o: &SignedLog) {
        // An accumulator whose magnitude underflowed is zero; write it as the
        // canonical zero so the result is one whatever the addend turns out to be.
        if self.is_zero() {
            *self = SignedLog::zero();
        }
        if o.is_zero() {
            return;
        }
        if self.sign == 0 {
            *self = *o;
            return;
        }
        // Past those two guards both magnitudes are finite below, so neither
        // branch can reach `-∞ - -∞`.
        if self.sign == o.sign {
            // Same sign: magnitudes add. ln(e^a + e^b) = hi + ln(1 + e^(lo-hi)).
            let (lo, hi) = if self.ln_abs < o.ln_abs {
                (self.ln_abs, o.ln_abs)
            } else {
                (o.ln_abs, self.ln_abs)
            };
            self.ln_abs = hi + (lo - hi).exp().ln_1p();
            // sign unchanged
        } else {
            // opposite signs: magnitudes subtract. result sign = larger mag's sign.
            if self.ln_abs == o.ln_abs {
                *self = SignedLog::zero();
                return; // exact cancellation
            }
            let (smaller, larger, larger_sign) = if self.ln_abs < o.ln_abs {
                (self.ln_abs, o.ln_abs, o.sign)
            } else {
                (o.ln_abs, self.ln_abs, self.sign)
            };
            // ln(e^larger - e^smaller) = larger + ln(1 - e^(smaller-larger))
            let diff = (smaller - larger).exp(); // in (0,1)
            self.ln_abs = larger + (-diff).ln_1p(); // ln(1 - diff)
            self.sign = larger_sign;
            if self.ln_abs == f64::NEG_INFINITY {
                // The magnitudes differed, but not by enough for `f64` to see:
                // `diff` rounded to 1 and the difference cancelled. That is a
                // zero, and it is written as the canonical one — leaving the
                // sign on a `-∞` magnitude is what made the next addition NaN.
                *self = SignedLog::zero();
            }
        }
    }

    /// log10 of |value| (for output formatting / validation).
    pub fn log10_abs(&self) -> f64 {
        self.ln_abs / std::f64::consts::LN_10
    }
}

/// ln of |a non-zero `BigInt`|, overflow-safe (handles thousands-of-bit integers).
fn ln_bigint_abs(n: &num_bigint::BigInt) -> f64 {
    use num_traits::ToPrimitive;
    let n = n.magnitude(); // &BigUint, absolute value
    let bits = n.bits(); // u64
    if bits <= 53 {
        return n.to_f64().unwrap().ln();
    }
    let shift = bits - 53;
    let top = n >> shift; // BigUint fitting in 53 bits
    let m = top.to_f64().unwrap(); // exact
    m.ln() + (shift as f64) * std::f64::consts::LN_2
}

// ── Bounded-precision scaled float (weighted marginal path) ─────────────────────

/// A signed value held as an `f64` mantissa and a binary exponent of its own:
/// `mantissa · 2^exponent`.
///
/// Used by [`Arithmetic::ScaledFloat`](crate::diagram::Arithmetic::ScaledFloat).
/// The mantissa's magnitude lies in `[0.5, 1)`, or the value is zero with
/// mantissa `0.0` and exponent `0`, so each number has one representation.
/// As with [`SignedLog`], the separate exponent permits magnitudes far beyond
/// the range of an `f64`, and the mantissa's sign permits negative literal
/// weights. A product costs one `f64` multiplication and a sum one `f64`
/// addition, where a [`SignedLog`] sum costs an `exp` and a `ln_1p`; each
/// operation rounds once, to the mantissa's 53 bits.
///
/// Arithmetic has bounded precision. In particular, cancellation of nearly
/// equal values can cause large relative error; use exact rational arithmetic
/// when the result must be exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaledFloat {
    mantissa: f64,
    exponent: i64,
}

/// `2^k` for `-1022 <= k <= 1023`.
#[inline]
fn pow2(k: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((1023 + k) as u64) << 52)
}

/// A nonzero normal `f64` as a mantissa of magnitude in `[0.5, 1)` and the
/// power of two it stands under.
#[inline]
fn frexp_normal(x: f64) -> (f64, i64) {
    const EXPONENT: u64 = 0x7ff << 52;
    let bits = x.to_bits();
    let e = ((bits & EXPONENT) >> 52) as i64;
    debug_assert!(e != 0 && e != 0x7ff, "frexp_normal of a zero, subnormal or non-finite float");
    (f64::from_bits(bits & !EXPONENT | 1022 << 52), e - 1022)
}

/// A nonzero `BigUint` as a float of its leading 64 bits and the power of two
/// they stand under.
fn top_bits(n: &num_bigint::BigUint) -> (f64, i64) {
    let shift = n.bits().saturating_sub(64);
    ((n >> shift).to_u64().expect("64 bits fit a u64") as f64, shift as i64)
}

impl ScaledFloat {
    /// The additive identity.
    pub const ZERO: ScaledFloat = ScaledFloat { mantissa: 0.0, exponent: 0 };

    /// Whether this value is zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.mantissa == 0.0
    }

    /// The finite float `x`, exactly.
    ///
    /// # Panics
    ///
    /// Panics if `x` is infinite or NaN.
    #[must_use]
    pub fn from_f64(x: f64) -> ScaledFloat {
        assert!(x.is_finite(), "ScaledFloat::from_f64 of a non-finite float");
        if x == 0.0 {
            return ScaledFloat::ZERO;
        }
        if x.abs() < f64::MIN_POSITIVE {
            // Subnormal: scale into the normal range first, exactly.
            let (mantissa, e) = frexp_normal(x * pow2(64));
            return ScaledFloat { mantissa, exponent: e - 64 };
        }
        let (mantissa, exponent) = frexp_normal(x);
        ScaledFloat { mantissa, exponent }
    }

    /// Convert an exact `BigRational`, whatever its magnitude: the quotient of
    /// the leading 64 bits of its numerator and denominator, so an integer
    /// ratio within 53 bits converts correctly rounded.
    #[must_use]
    pub fn from_rational(r: &BigRational) -> ScaledFloat {
        use num_bigint::Sign;
        if r.numer().is_zero() {
            return ScaledFloat::ZERO;
        }
        let (n, n_shift) = top_bits(r.numer().magnitude());
        let (d, d_shift) = top_bits(r.denom().magnitude());
        let mut v = ScaledFloat::from_f64(n / d);
        v.exponent += n_shift - d_shift;
        if r.numer().sign() == Sign::Minus {
            v.mantissa = -v.mantissa;
        }
        v
    }

    /// The mantissa: magnitude in `[0.5, 1)`, or `0.0` for zero.
    #[inline]
    #[must_use]
    pub fn mantissa(&self) -> f64 {
        self.mantissa
    }

    /// The power of two the mantissa stands under (`0` for zero).
    #[inline]
    #[must_use]
    pub fn exponent(&self) -> i64 {
        self.exponent
    }

    /// Product: the mantissas multiply and the exponents add.
    #[inline]
    #[must_use]
    pub fn mul(&self, o: &ScaledFloat) -> ScaledFloat {
        let m = self.mantissa * o.mantissa;
        if m == 0.0 {
            return ScaledFloat::ZERO;
        }
        // Both magnitudes lie in [0.5, 1), so the product's lies in [0.25, 1):
        // at most one doubling, which is exact, brings it back.
        let exponent = self.exponent + o.exponent;
        if m.abs() < 0.5 {
            ScaledFloat { mantissa: m * 2.0, exponent: exponent - 1 }
        } else {
            ScaledFloat { mantissa: m, exponent }
        }
    }

    /// Accumulate: `self += o`, on the larger exponent.
    #[inline]
    pub fn add_assign(&mut self, o: &ScaledFloat) {
        if o.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = *o;
            return;
        }
        let (hi, lo) = if self.exponent >= o.exponent { (*self, *o) } else { (*o, *self) };
        let d = hi.exponent - lo.exponent;
        if d > 64 {
            // The smaller term is below 2^-64 of the larger, under half its
            // last place: the rounded sum is the larger term.
            *self = hi;
            return;
        }
        // Scaling by 2^-d keeps the smaller mantissa normal (at least 2^-65),
        // so it is exact, and the sum rounds once. Both terms are multiples
        // of 2^-117, so a sum that does not cancel is normal.
        let m = hi.mantissa + lo.mantissa * pow2(-d);
        if m == 0.0 {
            *self = ScaledFloat::ZERO;
            return;
        }
        let (mantissa, e) = frexp_normal(m);
        *self = ScaledFloat { mantissa, exponent: hi.exponent + e };
    }

    /// The nearest `f64`: zero below the least positive one, infinite above
    /// the largest.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() || self.exponent < -1100 {
            return 0.0 * self.mantissa.signum();
        }
        if self.exponent > 1024 {
            return f64::INFINITY * self.mantissa.signum();
        }
        // Two normal scales, the first exact, so the result rounds once.
        let half = self.exponent / 2;
        self.mantissa * pow2(half) * pow2(self.exponent - half)
    }

    /// log10 of |value| (`-∞` for zero), whatever the magnitude.
    #[must_use]
    pub fn log10_abs(&self) -> f64 {
        if self.is_zero() {
            return f64::NEG_INFINITY;
        }
        self.mantissa.abs().log10() + self.exponent as f64 * std::f64::consts::LOG10_2
    }
}

/// The value of a weighted marginalization: exact, or bounded-precision
/// `SignedLog` under `Arithmetic::SignedLog` or `ScaledFloat` under
/// `Arithmetic::ScaledFloat`. The modes never mix in one run; mixed-mode ops
/// panic.
///
/// # The exact domain has two representations
///
/// [`WeightValue::ExactSmall`] holds an integer-valued weight inline in an
/// `i128`: no heap cell, a `Copy` payload, and `checked_mul`/`checked_add`
/// arithmetic. [`WeightValue::Exact`] holds everything else in a `BigRational`.
/// A `BigRational` heap-allocates every value, so an integer-valued weight
/// table would otherwise pay an allocation per multiply and per accumulate.
///
/// # Canonicalization invariant
///
/// An exact `WeightValue` is `ExactSmall` whenever its value is an integer that
/// fits an `i128`, and `Exact` otherwise, so no number is representable both
/// ways and equal values always intern to one slot. Every construction goes
/// through [`WeightValue::exact`] and every op re-canonicalizes its result. A
/// hand-built `WeightValue::Exact(v)` for a small `v` breaks this; `weight_key`
/// debug-asserts it.
///
/// The enum is `#[non_exhaustive]`. Build exact values with
/// [`WeightValue::exact`] and read them back with
/// [`as_rational`](WeightValue::as_rational), [`into_rational`](WeightValue::into_rational)
/// or [`into_rational_opt`](WeightValue::into_rational_opt) rather than by matching.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum WeightValue {
    /// Exact integer-valued weight, held inline in an `i128`. Canonical for
    /// every exact value that fits one (see the type-level invariant).
    #[non_exhaustive]
    ExactSmall(i128),
    /// Exact arbitrary-precision rational value. Only ever holds what the small
    /// variant cannot: a non-integer, or an integer wider than `i128`.
    #[non_exhaustive]
    Exact(BigRational),
    /// Bounded-precision signed log-domain value.
    #[non_exhaustive]
    Log(SignedLog),
    /// Bounded-precision scaled float.
    #[non_exhaustive]
    Scaled(ScaledFloat),
}

/// The small form of an exact rational, if it has one: `Some(n)` iff `r` is
/// integer-valued (denominator 1) and its numerator fits an `i128`. The one
/// definition of "representable in the small variant" — the canonicalization
/// invariant is exactly `small_of(r).is_none()` for every stored
/// [`WeightValue::Exact`].
#[inline]
fn small_of(r: &BigRational) -> Option<i128> {
    if r.is_integer() { r.numer().to_i128() } else { None }
}

/// An `i128` as the canonical integer-valued `BigRational` `n/1`. `new_raw` is
/// sound here for the same reason as in [`exact_mul`]: `n/1` is already in
/// num-rational's normal form (lowest terms, positive denominator), zero
/// included.
#[inline]
fn rational_of_small(n: i128) -> BigRational {
    BigRational::new_raw(num_bigint::BigInt::from(n), num_bigint::BigInt::one())
}

/// Exact arbitrary-precision product `a · b`, skipping fraction reduction when
/// both operands are integer-valued (denominator 1). The arbitrary-precision
/// half of [`WeightValue::mul`] — reached on an `i128` spill, a mixed-width pair,
/// or a fractional operand.
///
/// `Ratio::mul` runs three big-integer gcds and four divisions per multiply,
/// all against 1 when both denominators are 1.
///
/// # Soundness
///
/// `Ratio::new_raw(n, 1)` is already num-rational's canonical form (lowest
/// terms, positive denominator), zero included; the sign lives in the
/// numerator.
#[inline]
fn exact_mul(a: &BigRational, b: &BigRational) -> BigRational {
    if a.is_integer() && b.is_integer() {
        BigRational::new_raw(a.numer() * b.numer(), num_bigint::BigInt::one())
    } else {
        a * b
    }
}

/// Exact `acc += o`, skipping fraction reduction when both are integer-valued.
/// Same soundness argument as [`exact_mul`].
#[inline]
fn exact_add_assign(acc: &mut BigRational, o: &BigRational) {
    if acc.is_integer() && o.is_integer() {
        // Move the accumulator out so the addition can reuse its numerator's
        // limb buffer in place, and hand its denominator (the existing 1) back
        // to the result rather than allocating a second one.
        let (mut n, one) = std::mem::replace(acc, BigRational::zero()).into_raw();
        n += o.numer();
        *acc = BigRational::new_raw(n, one);
    } else {
        *acc += o;
    }
}

impl WeightValue {
    /// The one canonicalizing constructor for an exact weight: `ExactSmall`
    /// when the value is an integer fitting an `i128`, `Exact` otherwise.
    ///
    /// The one way to build an exact `WeightValue` from a `BigRational` (see the
    /// canonicalization invariant on the type). `r` must be in num-rational's
    /// normal form, as every `BigRational::new` and op result is.
    #[inline]
    #[must_use]
    pub fn exact(r: BigRational) -> WeightValue {
        match small_of(&r) {
            Some(n) => WeightValue::ExactSmall(n),
            None => WeightValue::Exact(r),
        }
    }

    /// This exact value as a `BigRational`, borrowing when it already is one and
    /// materializing `n/1` when it is small.
    ///
    /// # Panics
    ///
    /// Panics on a `Log` or `Scaled` value — neither carries an exact rational.
    #[must_use]
    pub fn as_rational(&self) -> Cow<'_, BigRational> {
        match self {
            WeightValue::ExactSmall(n) => Cow::Owned(rational_of_small(*n)),
            WeightValue::Exact(r) => Cow::Borrowed(r),
            WeightValue::Log(_) | WeightValue::Scaled(_) => panic!("WeightValue::as_rational: value is bounded-precision"),
        }
    }

    /// A log-domain value.
    pub fn log(s: SignedLog) -> WeightValue {
        WeightValue::Log(s)
    }

    /// The log-domain value inside, if this is one.
    ///
    /// The two exact representations answer `None`; read those with
    /// [`as_rational`](WeightValue::as_rational) or
    /// [`into_rational`](WeightValue::into_rational).
    pub fn as_log(&self) -> Option<&SignedLog> {
        match self {
            WeightValue::Log(s) => Some(s),
            _ => None,
        }
    }

    /// A scaled-float value.
    pub fn scaled(s: ScaledFloat) -> WeightValue {
        WeightValue::Scaled(s)
    }

    /// The scaled-float value inside, if this is one.
    pub fn as_scaled(&self) -> Option<&ScaledFloat> {
        match self {
            WeightValue::Scaled(s) => Some(s),
            _ => None,
        }
    }

    /// The arithmetic this value belongs to.
    pub(crate) fn arithmetic(&self) -> crate::diagram::Arithmetic {
        use crate::diagram::Arithmetic;
        match self {
            WeightValue::ExactSmall(_) | WeightValue::Exact(_) => Arithmetic::ExactRational,
            WeightValue::Log(_) => Arithmetic::SignedLog,
            WeightValue::Scaled(_) => Arithmetic::ScaledFloat,
        }
    }

    /// Consume an exact value into a `BigRational` (moves the big payload out
    /// rather than cloning it).
    ///
    /// # Panics
    ///
    /// Panics on a `Log` or `Scaled` value — neither carries an exact rational.
    #[must_use]
    pub fn into_rational(self) -> BigRational {
        match self {
            WeightValue::ExactSmall(n) => rational_of_small(n),
            WeightValue::Exact(r) => r,
            WeightValue::Log(_) | WeightValue::Scaled(_) => panic!("WeightValue::into_rational: value is bounded-precision"),
        }
    }

    /// `Some(exact rational)` for either exact representation, `None` for a
    /// `Log` or `Scaled` value — for callers that treat a bounded-precision
    /// mode as "not my domain" rather than as a bug.
    #[must_use]
    pub fn into_rational_opt(self) -> Option<BigRational> {
        match self {
            WeightValue::Log(_) | WeightValue::Scaled(_) => None,
            v => Some(v.into_rational()),
        }
    }

    /// True when this is the additive identity, in whichever representation.
    /// This is an arithmetic zero (weight cancellation included), never a
    /// structural UNSAT label.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            WeightValue::ExactSmall(n) => *n == 0,
            WeightValue::Exact(r) => r.numer().is_zero(),
            WeightValue::Log(s) => s.is_zero(),
            WeightValue::Scaled(s) => s.is_zero(),
        }
    }

    /// Take the exact value out, leaving the canonical zero behind. Lets the
    /// arbitrary-precision `add_assign` path reuse the accumulator's limb
    /// buffer instead of cloning it.
    #[inline]
    fn take_rational(&mut self) -> BigRational {
        std::mem::replace(self, WeightValue::ExactSmall(0)).into_rational()
    }

    /// Product. Two small operands multiply in `i128` (no allocation); on
    /// overflow, a mixed-width pair, or a fractional operand the same product
    /// is taken in arbitrary precision by `exact_mul` and re-canonicalized.
    /// Log*Log adds logs; Scaled*Scaled multiplies mantissas. Mixed modes
    /// panic.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `o` belong to different arithmetics.
    #[inline]
    #[must_use]
    pub fn mul(&self, o: &WeightValue) -> WeightValue {
        match (self, o) {
            // Hot path: integer-valued and narrow on both sides.
            (WeightValue::ExactSmall(a), WeightValue::ExactSmall(b)) => {
                if let Some(p) = a.checked_mul(*b) {
                    return WeightValue::ExactSmall(p);
                }
                // Overflow — spill to the arbitrary-precision path below.
            }
            (WeightValue::Log(a), WeightValue::Log(b)) => return WeightValue::Log(a.mul(b)),
            (WeightValue::Scaled(a), WeightValue::Scaled(b)) => return WeightValue::Scaled(a.mul(b)),
            (WeightValue::Log(_) | WeightValue::Scaled(_), _) | (_, WeightValue::Log(_) | WeightValue::Scaled(_)) => {
                panic!("WeightValue::mul: mixed arithmetics")
            }
            // Mixed width or fractional — arbitrary-precision path below.
            _ => {}
        }
        WeightValue::exact(exact_mul(&self.as_rational(), &o.as_rational()))
    }

    /// Accumulate `o` into `self`. Two small operands add in `i128` (no
    /// allocation); otherwise the same sum is taken in arbitrary precision by
    /// `exact_add_assign` and re-canonicalized — so a sum that cancels back
    /// into `i128` range demotes to the small representation, and the value
    /// zero is the identical `ExactSmall(0)` however it arose. Mixed modes
    /// panic.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `o` belong to different arithmetics.
    #[inline]
    pub fn add_assign(&mut self, o: &WeightValue) {
        match (&mut *self, o) {
            // Hot path: integer-valued and narrow on both sides.
            (WeightValue::ExactSmall(a), WeightValue::ExactSmall(b)) => {
                if let Some(s) = a.checked_add(*b) {
                    *a = s;
                    return;
                }
                // Overflow — spill to the arbitrary-precision path below.
            }
            (WeightValue::Log(a), WeightValue::Log(b)) => {
                a.add_assign(b);
                return;
            }
            (WeightValue::Scaled(a), WeightValue::Scaled(b)) => {
                a.add_assign(b);
                return;
            }
            (WeightValue::Log(_) | WeightValue::Scaled(_), _) | (_, WeightValue::Log(_) | WeightValue::Scaled(_)) => {
                panic!("WeightValue::add_assign: mixed arithmetics")
            }
            // Mixed width, or both already big — arbitrary-precision path below.
            _ => {}
        }
        let mut acc = self.take_rational();
        exact_add_assign(&mut acc, &o.as_rational());
        *self = WeightValue::exact(acc);
    }

    /// Multiply `self` by the exact rational `r`, converted to `self`'s own
    /// arithmetic first: exact for an exact value, log domain for a log value,
    /// scaled float for a scaled one.
    ///
    /// ```
    /// use num_rational::BigRational;
    /// use tididi::diagram::WeightValue;
    /// let six = WeightValue::exact(BigRational::from_integer(6.into()));
    /// let half = BigRational::new(1.into(), 2.into());
    /// assert_eq!(six.mul_rational(&half).into_rational(), BigRational::from_integer(3.into()));
    /// ```
    #[inline]
    #[must_use]
    pub fn mul_rational(&self, r: &BigRational) -> WeightValue {
        let scale = match self {
            WeightValue::Log(_) => WeightValue::Log(SignedLog::from_rational(r)),
            WeightValue::Scaled(_) => WeightValue::Scaled(ScaledFloat::from_rational(r)),
            _ => WeightValue::exact(r.clone()),
        };
        self.mul(&scale)
    }
}

/// Hashable dedup key for a `WeightValue` (used by the weighted intern table and
/// by slot-prune's value merge). The Log mantissa is keyed on `f64::to_bits` so
/// bit-identical logs collapse; a scaled float, whose representation is
/// unique, on its mantissa's bits and its exponent.
///
/// The two exact key variants are sound to derive `Eq`/`Hash` over *because* of
/// [`WeightValue`]'s canonicalization invariant: an i128-representable value is
/// always an `ExactSmall`, so no number can produce both an `ExactSmall` key and
/// an `Exact` key, and equal values therefore always collide onto one entry.
#[derive(Hash, Eq, PartialEq, Clone)]
pub(crate) enum WeightKey {
    /// Key for an exact integer value held in the small representation.
    ExactSmall(i128),
    /// Key for an exact rational value with no small form.
    Exact(BigRational),
    /// Key for a log value: `(ln_abs.to_bits(), sign)`.
    Log(u64, i8),
    /// Key for a scaled float: `(mantissa.to_bits(), exponent)`.
    Scaled(u64, i64),
}

/// Build a `WeightKey` for interning/dedup. This is the choke point where every
/// structural weight comparison happens, so it is also where the
/// canonicalization invariant is checked.
pub(crate) fn weight_key(v: &WeightValue) -> WeightKey {
    match v {
        WeightValue::ExactSmall(n) => WeightKey::ExactSmall(*n),
        WeightValue::Exact(r) => {
            debug_assert_canonical_big(r);
            WeightKey::Exact(r.clone())
        }
        WeightValue::Log(s) => WeightKey::Log(s.ln_abs.to_bits(), s.sign),
        WeightValue::Scaled(s) => WeightKey::Scaled(s.mantissa.to_bits(), s.exponent),
    }
}

/// Whether `a` and `b` would share one [`WeightKey`]: the same exact value,
/// bit-identical logs, or the same scaled float. For a comparison, this is
/// `weight_key(a) == weight_key(b)` without building either key, which
/// clones a big rational.
pub(crate) fn same_value(a: &WeightValue, b: &WeightValue) -> bool {
    match (a, b) {
        (WeightValue::ExactSmall(x), WeightValue::ExactSmall(y)) => x == y,
        (WeightValue::Exact(x), WeightValue::Exact(y)) => {
            debug_assert_canonical_big(x);
            debug_assert_canonical_big(y);
            x == y
        }
        (WeightValue::Log(x), WeightValue::Log(y)) => {
            x.ln_abs.to_bits() == y.ln_abs.to_bits() && x.sign == y.sign
        }
        (WeightValue::Scaled(x), WeightValue::Scaled(y)) => {
            x.mantissa.to_bits() == y.mantissa.to_bits() && x.exponent == y.exponent
        }
        // The canonical form puts every i128-representable value in the small
        // variant, so a small and a big value are never equal.
        _ => false,
    }
}

/// The canonicalization invariant a big exact value must meet.
#[inline]
fn debug_assert_canonical_big(r: &BigRational) {
    debug_assert!(
        small_of(r).is_none(),
        "canonicalization invariant: an i128-representable exact value must be \
         WeightValue::ExactSmall — a non-canonical WeightValue::Exact would key and hash \
         as a value distinct from its own small form, splitting one value across two \
         intern slots (build exact values with WeightValue::exact)"
    );
}

#[cfg(test)]
#[path = "tests/weight/mod.rs"]
mod tests;
