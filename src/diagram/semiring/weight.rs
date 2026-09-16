//! The weighted-marginal value type: its two exact representations, its
//! bounded-precision log-domain alternative, and the dedup key over both.

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

/// The value of a weighted marginalization: exact, or bounded-precision
/// `SignedLog` under `Arithmetic::SignedLog`. The two modes never mix in one
/// run; mixed-mode ops panic.
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
    /// Panics on a `Log` value — the log domain carries no exact rational.
    #[must_use]
    pub fn as_rational(&self) -> Cow<'_, BigRational> {
        match self {
            WeightValue::ExactSmall(n) => Cow::Owned(rational_of_small(*n)),
            WeightValue::Exact(r) => Cow::Borrowed(r),
            WeightValue::Log(_) => panic!("WeightValue::as_rational: value is in log mode"),
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


    /// Consume an exact value into a `BigRational` (moves the big payload out
    /// rather than cloning it).
    ///
    /// # Panics
    ///
    /// Panics on a `Log` value — the log domain carries no exact rational.
    #[must_use]
    pub fn into_rational(self) -> BigRational {
        match self {
            WeightValue::ExactSmall(n) => rational_of_small(n),
            WeightValue::Exact(r) => r,
            WeightValue::Log(_) => panic!("WeightValue::into_rational: value is in log mode"),
        }
    }

    /// `Some(exact rational)` for either exact representation, `None` for a
    /// `Log` value — for callers that treat log mode as "not my domain" rather
    /// than as a bug.
    #[must_use]
    pub fn into_rational_opt(self) -> Option<BigRational> {
        match self {
            WeightValue::Log(_) => None,
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
            WeightValue::Log(s) => s.sign == 0,
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
    /// Log*Log adds logs. Mixed modes panic.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `o` mix an exact and a `Log` value.
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
            (WeightValue::Log(_), _) | (_, WeightValue::Log(_)) => {
                panic!("WeightValue::mul: mixed Exact/Log modes")
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
    /// Panics if `self` and `o` mix an exact and a `Log` value.
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
            (WeightValue::Log(_), _) | (_, WeightValue::Log(_)) => {
                panic!("WeightValue::add_assign: mixed Exact/Log modes")
            }
            // Mixed width, or both already big — arbitrary-precision path below.
            _ => {}
        }
        let mut acc = self.take_rational();
        exact_add_assign(&mut acc, &o.as_rational());
        *self = WeightValue::exact(acc);
    }

    /// Multiply `self` by the exact scalar `corr`, converted to `self`'s own
    /// mode first so the multiply is a same-mode `mul`.
    #[inline]
    #[must_use]
    pub fn mul_correction(&self, corr: &BigRational) -> WeightValue {
        let cw = if matches!(self, WeightValue::Log(_)) {
            WeightValue::Log(SignedLog::from_rational(corr))
        } else {
            WeightValue::exact(corr.clone())
        };
        self.mul(&cw)
    }
}

/// Hashable dedup key for a `WeightValue` (used by the weighted intern table and
/// by slot-prune's value merge). The Log mantissa is keyed on `f64::to_bits` so
/// bit-identical logs collapse.
///
/// The two exact key variants are sound to derive `Eq`/`Hash` over *because* of
/// [`WeightValue`]'s canonicalization invariant: an i128-representable value is
/// always an `ExactSmall`, so no number can produce both an `ExactSmall` key and
/// an `Exact` key, and equal values therefore always collide onto one entry.
///
/// `#[non_exhaustive]` for the same reason [`WeightValue`] is: the key variants
/// track the value representations one for one, so the two must be free to grow
/// together.
#[derive(Hash, Eq, PartialEq, Clone)]

#[non_exhaustive]
pub(crate) enum WeightKey {
    /// Key for an exact integer value held in the small representation.
    #[non_exhaustive]
    ExactSmall(i128),
    /// Key for an exact rational value with no small form.
    #[non_exhaustive]
    Exact(BigRational),
    /// Key for a log value: `(ln_abs.to_bits(), sign)`.
    Log(u64, i8),
}

/// Build a `WeightKey` for interning/dedup. This is the choke point where every
/// structural weight comparison happens, so it is also where the
/// canonicalization invariant is checked.
pub(crate) fn weight_key(v: &WeightValue) -> WeightKey {
    match v {
        WeightValue::ExactSmall(n) => WeightKey::ExactSmall(*n),
        WeightValue::Exact(r) => {
            debug_assert!(
                small_of(r).is_none(),
                "canonicalization invariant: an i128-representable exact value must be \
                 WeightValue::ExactSmall — a non-canonical WeightValue::Exact would key and hash \
                 as a value distinct from its own small form, splitting one value across two \
                 intern slots (build exact values with WeightValue::exact)"
            );
            WeightKey::Exact(r.clone())
        }
        WeightValue::Log(s) => WeightKey::Log(s.ln_abs.to_bits(), s.sign),
    }
}


#[cfg(test)]
#[path = "tests/weight/mod.rs"]
mod tests;
