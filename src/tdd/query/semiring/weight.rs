//! The weighted-marg value type: its two exact representations, its
//! bounded-precision log-domain alternative, and the dedup key over both.

use std::borrow::Cow;

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

// ── Bounded-precision signed log-domain weight (weighted marg path) ───────────

/// Bounded-precision signed log-domain weight: sign ∈ {-1,0,+1}; `ln_abs` = ln|value|
/// (conventionally `f64::NEG_INFINITY` when sign==0). Used by the weighted marg path
/// under `weight_store::Precision::Log`
/// to bound per-op cost (vs `BigRational` digit growth).
///
/// Literal weights are typically many-digit decimals, and a weighted multiply
/// compounds their digits onto the numerator AND denominator of an exact
/// `BigRational`, so on a formula with thousands of variables the rationals
/// reach thousands of decimal digits and each mul/add/gcd becomes O(digits).
/// The log domain bounds every op to O(1) `f64` work, at a relative error near
/// the `f64` epsilon per operation. The sign is tracked
/// separately so genuine signed WMC (literal weight `-1`) is supported.
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
        SignedLog { ln_abs: self.ln_abs + o.ln_abs, sign: self.sign * o.sign }
    }

    /// Signed log-sum-exp accumulate: self += o.
    pub fn add_assign(&mut self, o: &SignedLog) {
        if o.sign == 0 {
            return;
        }
        if self.sign == 0 {
            *self = *o;
            return;
        }
        if self.sign == o.sign {
            // same sign: magnitudes add. ln(e^a + e^b) = hi + ln(1 + e^(lo-hi)).
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

/// Weighted-marg-path value: exact (default oracle) or bounded-precision
/// `SignedLog` (`weight_store::Precision::Log`). The two modes
/// never mix in one run; mixed-mode ops panic. Only the weighted marginalizing
/// path uses this type — the full-diagram `RationalSemiring`/`evaluate` path
/// stays on stock `BigRational`.
///
/// # The exact domain has two representations
///
/// [`WeightVal::ExactSmall`] holds an integer-valued weight inline in an
/// `i128`: no heap cell, a `Copy` payload, and `checked_mul`/`checked_add`
/// arithmetic. [`WeightVal::Exact`] holds everything else in a `BigRational`.
/// This matters because num-bigint heap-allocates *every* value (no small-size
/// optimization), so wherever a caller has rescaled its weight table to a
/// common denominator — making every hot value an integer-valued rational with
/// denominator 1 — the arbitrary-precision representation pays a malloc/free
/// per multiply and per accumulate for numbers that fit in two registers.
///
/// # Canonicalization invariant
///
/// **An exact `WeightVal` is `ExactSmall` whenever its value is an integer that
/// fits an `i128`, and `Exact` otherwise.** The two exact variants therefore
/// *partition* the value space: no number is representable both ways. Every
/// construction point goes through [`WeightVal::exact`] (or mints `ExactSmall`
/// directly) and every op re-canonicalizes its result — a product/sum that
/// overflows `i128` spills to `Exact`, and one that shrinks back into range
/// demotes to `ExactSmall`. This is what keeps `WeightKey`'s derived
/// `Eq`/`Hash` sound: a `WeightKey::Exact` and a `WeightKey::ExactSmall` can
/// never denote the same number, so equal values always intern to one slot.
/// Constructing `WeightVal::Exact(v)` by hand for a small `v` breaks it (the
/// interner would then hold two keys for one value); `weight_key` carries a
/// `debug_assert` for exactly that.
///
/// # Extensibility
///
/// This enum is `#[non_exhaustive]`, so a `match` on it from another crate
/// needs a wildcard arm. Which representation holds a given weight is a
/// performance decision — the split between the two exact variants is one, and
/// the log domain is another — and freezing the variant list would turn every
/// later representation into a breaking change over a fact about layout that
/// callers have no reason to read. Build exact values with
/// [`WeightVal::exact`] and read them back with
/// [`as_rational`](WeightVal::as_rational),
/// [`into_rational`](WeightVal::into_rational) or
/// [`into_rational_opt`](WeightVal::into_rational_opt) rather than by matching,
/// and no future variant can reach you.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum WeightVal {
    /// Exact integer-valued weight, held inline in an `i128`. Canonical for
    /// every exact value that fits one (see the type-level invariant).
    ExactSmall(i128),
    /// Exact arbitrary-precision rational value. Only ever holds what the small
    /// variant cannot: a non-integer, or an integer wider than `i128`.
    Exact(BigRational),
    /// Bounded-precision signed log-domain value.
    Log(SignedLog),
}

/// The small form of an exact rational, if it has one: `Some(n)` iff `r` is
/// integer-valued (denominator 1) and its numerator fits an `i128`. The ONE
/// definition of "representable in the small variant" — the canonicalization
/// invariant is exactly `small_of(r).is_none()` for every stored
/// [`WeightVal::Exact`].
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
/// half of [`WeightVal::mul`] — reached on an `i128` spill, a mixed-width pair,
/// or a fractional operand.
///
/// **Why.** `Ratio::mul` cross-reduces (`gcd(numerₐ, denom_b)`,
/// `gcd(denomₐ, numer_b)`), divides both pairs, and then `Ratio::new` reduces
/// the product again: three big-integer gcds and four big divisions per
/// multiply. When both denominators are 1 every one of those gcds is against 1
/// and every division is by 1 — the whole apparatus re-proves that a product of
/// integers is in lowest terms, on numerators thousands of bits long, and it
/// dominates the weighted profile. That is the common case whenever a caller
/// rescales its weight table to integer-valued weights: every value the
/// weighted fold builds is a `+`/`·` closure over those seeds, so it stays
/// integer-valued all the way to the output.
///
/// **Why it is sound.** `gcd(n, 1) = 1` for every `n`, and the denominator 1 is
/// positive, so `Ratio::new_raw(n, 1)` is ALREADY in num-rational's canonical
/// form (lowest terms, positive denominator) — the identical value `Ratio::new`
/// would return, including for `n = 0` (`0/1` is `reduce`'s own normal form for
/// zero). No invariant is bypassed, only the work of re-deriving one. Signs need
/// no special care: a `BigRational`'s sign lives in its `BigInt` numerator.
///
/// The whole-diagram [`RationalSemiring`] oracle below deliberately does NOT use
/// these helpers — it stays on stock num-rational ops so the weighted
/// differential batteries check this path against an independent implementation.
#[inline]
fn exact_mul(a: &BigRational, b: &BigRational) -> BigRational {
    if a.is_integer() && b.is_integer() {
        BigRational::new_raw(a.numer() * b.numer(), num_bigint::BigInt::one())
    } else {
        a * b
    }
}

/// Exact `acc += o`, skipping fraction reduction when both are integer-valued.
/// Same soundness argument as [`exact_mul`]; the generic `AddAssign` would add
/// the numerators (denominators already equal) and then pay a `reduce` — a
/// `gcd(sum, 1)` plus two divisions by 1.
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

impl WeightVal {
    /// The ONE canonicalizing constructor for an exact weight: `ExactSmall`
    /// when the value is an integer fitting an `i128`, `Exact` otherwise.
    ///
    /// Every site that builds an exact `WeightVal` from a `BigRational` must go
    /// through here. A hand-built `WeightVal::Exact(v)` for a small `v` breaks
    /// the canonicalization invariant documented on the type, and the interner
    /// would then hold two distinct keys for one value.
    ///
    /// The input must be in num-rational's normal form (which every
    /// `BigRational::new` / parsed weight / op result is): a hypothetical
    /// unreduced `4/2` would report `is_integer() == false` and stay big.
    #[inline]
    #[must_use]
    pub fn exact(r: BigRational) -> WeightVal {
        match small_of(&r) {
            Some(n) => WeightVal::ExactSmall(n),
            None => WeightVal::Exact(r),
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
            WeightVal::ExactSmall(n) => Cow::Owned(rational_of_small(*n)),
            WeightVal::Exact(r) => Cow::Borrowed(r),
            WeightVal::Log(_) => panic!("WeightVal::as_rational: value is in log mode"),
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
            WeightVal::ExactSmall(n) => rational_of_small(n),
            WeightVal::Exact(r) => r,
            WeightVal::Log(_) => panic!("WeightVal::into_rational: value is in log mode"),
        }
    }

    /// `Some(exact rational)` for either exact representation, `None` for a
    /// `Log` value — for callers that treat log mode as "not my domain" rather
    /// than as a bug.
    #[must_use]
    pub fn into_rational_opt(self) -> Option<BigRational> {
        match self {
            WeightVal::Log(_) => None,
            v => Some(v.into_rational()),
        }
    }

    /// True when this is the additive identity, in whichever representation.
    /// This is an arithmetic zero (weight cancellation included), never a
    /// structural UNSAT label.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            WeightVal::ExactSmall(n) => *n == 0,
            WeightVal::Exact(r) => r.numer().is_zero(),
            WeightVal::Log(s) => s.sign == 0,
        }
    }

    /// Take the exact value out, leaving the canonical zero behind. Lets the
    /// arbitrary-precision `add_assign` path reuse the accumulator's limb
    /// buffer instead of cloning it.
    #[inline]
    fn take_rational(&mut self) -> BigRational {
        std::mem::replace(self, WeightVal::ExactSmall(0)).into_rational()
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
    pub fn mul(&self, o: &WeightVal) -> WeightVal {
        match (self, o) {
            // Hot path: integer-valued and narrow on both sides.
            (WeightVal::ExactSmall(a), WeightVal::ExactSmall(b)) => {
                if let Some(p) = a.checked_mul(*b) {
                    return WeightVal::ExactSmall(p);
                }
                // Overflow — spill to the arbitrary-precision path below.
            }
            (WeightVal::Log(a), WeightVal::Log(b)) => return WeightVal::Log(a.mul(b)),
            (WeightVal::Log(_), _) | (_, WeightVal::Log(_)) => {
                panic!("WeightVal::mul: mixed Exact/Log modes")
            }
            // Mixed width or fractional — arbitrary-precision path below.
            _ => {}
        }
        WeightVal::exact(exact_mul(&self.as_rational(), &o.as_rational()))
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
    pub fn add_assign(&mut self, o: &WeightVal) {
        match (&mut *self, o) {
            // Hot path: integer-valued and narrow on both sides.
            (WeightVal::ExactSmall(a), WeightVal::ExactSmall(b)) => {
                if let Some(s) = a.checked_add(*b) {
                    *a = s;
                    return;
                }
                // Overflow — spill to the arbitrary-precision path below.
            }
            (WeightVal::Log(a), WeightVal::Log(b)) => {
                a.add_assign(b);
                return;
            }
            (WeightVal::Log(_), _) | (_, WeightVal::Log(_)) => {
                panic!("WeightVal::add_assign: mixed Exact/Log modes")
            }
            // Mixed width, or both already big — arbitrary-precision path below.
            _ => {}
        }
        let mut acc = self.take_rational();
        exact_add_assign(&mut acc, &o.as_rational());
        *self = WeightVal::exact(acc);
    }

    /// Multiply `self` by a correction scalar (e.g. a correction contributed
    /// by an external preprocessing/reduction step, such as variable
    /// elimination or forced-literal detection), building the scalar in
    /// `self`'s own mode (Exact vs Log) so the multiply is a same-mode `mul`
    /// (log mode: adds logs). This is the fold every driver correction site
    /// needs — build-scalar-in-my-mode, then multiply — collapsed from four
    /// independent copies at the downstream driver's correction sites.
    #[inline]
    #[must_use]
    pub fn mul_correction(&self, corr: &BigRational) -> WeightVal {
        let cw = if matches!(self, WeightVal::Log(_)) {
            WeightVal::Log(SignedLog::from_rational(corr))
        } else {
            WeightVal::exact(corr.clone())
        };
        self.mul(&cw)
    }
}

/// Hashable dedup key for a `WeightVal` (used by the weighted intern table and
/// by slot-prune's value merge). The Log mantissa is keyed on `f64::to_bits` so
/// bit-identical logs collapse.
///
/// The two exact key variants are sound to derive `Eq`/`Hash` over *because* of
/// [`WeightVal`]'s canonicalization invariant: an i128-representable value is
/// always an `ExactSmall`, so no number can produce both an `ExactSmall` key and
/// an `Exact` key, and equal values therefore always collide onto one entry.
///
/// `#[non_exhaustive]` for the same reason [`WeightVal`] is: the key variants
/// track the value representations one for one, so the two must be free to grow
/// together.
#[derive(Hash, Eq, PartialEq, Clone)]

#[non_exhaustive]
pub(crate) enum WeightKey {
    /// Key for an exact integer value held in the small representation.
    ExactSmall(i128),
    /// Key for an exact rational value with no small form.
    Exact(BigRational),
    /// Key for a log value: `(ln_abs.to_bits(), sign)`.
    Log(u64, i8),
}

/// Build a `WeightKey` for interning/dedup. This is the choke point where every
/// structural weight comparison happens, so it is also where the
/// canonicalization invariant is checked.

pub(crate) fn weight_key(v: &WeightVal) -> WeightKey {
    match v {
        WeightVal::ExactSmall(n) => WeightKey::ExactSmall(*n),
        WeightVal::Exact(r) => {
            debug_assert!(
                small_of(r).is_none(),
                "canonicalization invariant: an i128-representable exact value must be \
                 WeightVal::ExactSmall — a non-canonical WeightVal::Exact would key and hash \
                 as a value distinct from its own small form, splitting one value across two \
                 intern slots (build exact values with WeightVal::exact)"
            );
            WeightKey::Exact(r.clone())
        }
        WeightVal::Log(s) => WeightKey::Log(s.ln_abs.to_bits(), s.sign),
    }
}

/// A `WeightVal`-keyed map (intern table for the weighted marg path).

pub(crate) type WeightMap = FxHashMap<WeightKey, u32>;

#[cfg(test)]
#[path = "../semiring_signed_log_tests.rs"]
mod signed_log_tests;

#[cfg(test)]
#[path = "../semiring_exact_tests.rs"]
mod exact_tests;
