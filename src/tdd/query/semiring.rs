//! Generic bottom-up TDD evaluation parameterized by a semiring.
//!
//! `evaluate(&tdd, &sr)` performs the same bottom-up traversal as
//! `query::compute_node_counts`, but with all arithmetic delegated to a
//! `Semiring` impl. The production impl is `RationalSemiring` (exact
//! arbitrary-precision rational WMC, Track 4 PWMC).
//!
//! Note: `query::model_count` (the production path) uses a hybrid
//! u128/BigUint scheme that requires per-node overflow detection — its
//! *storage* stays specialized and does not fit cleanly in the semiring
//! abstraction here. The count discipline itself (the sentinel, the
//! exact-max promotion rule, the lazy `BigUint` side table) now lives in
//! `crate::tdd::counts` (`Count`/`CountVec`), with the fold-level
//! unification across this integer path and the weighted path. `Semiring`
//! remains the whole-diagram
//! `evaluate` oracle — a traversal-level trait, not a fold-level one.

use std::borrow::Cow;

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::vtree::{VarId, VtreeIdx};

use crate::tdd::types::*;

/// Commutative semiring over `Value`, with leaf values keyed by
/// `(VarId, LeafLabel)` so weight-table semirings (e.g. WMC) can
/// look up per-variable weights.
///
/// `&self` lets impls carry mutable state (weight tables, RNGs); for
/// stateless semirings the receiver is a unit struct.
///
/// `LeafLabel::Zero` is never passed to `leaf` — `evaluate` short-circuits
/// it to `zero()` directly.
pub trait Semiring {
    /// The semiring's carrier type.
    type Value: Clone;
    /// The additive identity.
    fn zero(&self) -> Self::Value;
    /// Value of leaf `label` for variable `var`. `LeafLabel::Zero` is never
    /// passed here — `evaluate` short-circuits it to `zero()`.
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value;
    /// Accumulate `other` into `acc` (the semiring `+`).
    fn add_assign(&self, acc: &mut Self::Value, other: &Self::Value);
    /// The semiring product of `a` and `b`.
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
}

/// Bottom-up evaluate the TDD under semiring `sr`. Returns the value of
/// the output node (or `sr.zero()` for the constant-zero TDD).
pub fn evaluate<S: Semiring>(tdd: &Tdd, sr: &S) -> S::Value {
    if tdd.is_zero() {
        return sr.zero();
    }

    let mut counts: Vec<Vec<S::Value>> = (0..tdd.vtree.num_nodes())
        .map(|i| vec![sr.zero(); tdd.effective_width(VtreeIdx(i as u32))])
        .collect();
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());

    for (t, var) in tdd.vtree.leaf_bottomup() {
        let ti = t.idx();
        for i in 0..LEAF_WIDTH {
            let label = LeafLabel::from_idx(i);
            counts[ti][i] = match label {
                LeafLabel::Zero => sr.zero(),
                _ => sr.leaf(var, label),
            };
        }
    }
    for (t, left, right) in tdd.vtree.internal_bottomup() {
        let ti = t.idx();
        let li = left.idx();
        let ri = right.idx();
        let left_marg = tdd.levels[li].is_marginal();
        let right_marg = tdd.levels[ri].is_marginal();
        for (i, pairs) in tdd.levels[ti].internal_inputs_iter() {
            let mut total = sr.zero();
            for pair in pairs {
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in evaluate"),
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in evaluate"),
                };
                let prod = sr.mul(
                    &counts[li][l],
                    &counts[ri][r],
                );
                sr.add_assign(&mut total, &prod);
            }
            counts[ti][i] = total;
        }
        // The vtree is a tree: a node has exactly ONE parent, so its column has
        // exactly one consumer and is dead the moment that parent's column is
        // complete. Free it here rather than carrying every level's values to
        // the end of the walk — the live set becomes the frontier, not the
        // whole diagram. `out_t` is the one column read after the walk (it is
        // the root under the output-at-root invariant, hence never a child
        // here, but the walk does not rely on that).
        for c in [li, ri] {
            if c != out_t {
                counts[c] = Vec::new();
            }
        }
    }

    counts[out_t][out_i].clone()
}

// ── Exact rational weighted model counting (Track 4 PWMC) ─────────────────────

/// Exact weighted model counting over `num_rational::BigRational`.
///
/// This evaluates the weighted sum in exact arbitrary-precision rational
/// arithmetic — the competition precision-category-A requirement for Track 4
/// (PWMC). It is the production on-ramp for weighted/algebraic counting:
/// compile the TDD without marginalization (Boolean structure intact), then
/// `evaluate` it under this semiring.
///
/// `w_pos[v]` / `w_neg[v]` are the literal weights of variable `v`. A free
/// variable (`One` leaf) contributes `w_pos[v] + w_neg[v]`. Weights may be
/// zero: a satisfiable instance can then have weighted value 0 (weight
/// cancellation) — this is NOT unsat, so callers must not treat a 0 result
/// as structural ⊥ (the zero-cancellation hazard lives only at the output
/// SAT/UNSAT label, never in this arithmetic).
#[derive(Clone)]
pub struct RationalSemiring {
    /// Positive-literal weight of each variable, indexed by `VarId`.
    pub w_pos: Vec<BigRational>,
    /// Negative-literal weight of each variable, indexed by `VarId`.
    pub w_neg: Vec<BigRational>,
}

impl RationalSemiring {
    /// Build from per-variable `(w_neg, w_pos)` literal weights. Zero and any
    /// nonneg/negative rational weight is permitted — exactness imposes no
    /// sign restriction.
    pub fn from_weights(weights: &[(BigRational, BigRational)]) -> Self {
        let mut w_neg = Vec::with_capacity(weights.len());
        let mut w_pos = Vec::with_capacity(weights.len());
        for (wn, wp) in weights {
            w_neg.push(wn.clone());
            w_pos.push(wp.clone());
        }
        RationalSemiring { w_pos, w_neg }
    }

    /// All variables uniform with weight 1 on each polarity. Then
    /// `evaluate(&tdd, &sr)` equals the (integer) model count of `tdd`,
    /// as an exact `BigRational` with denominator 1.
    pub fn unit(num_vars: usize) -> Self {
        RationalSemiring {
            w_pos: vec![BigRational::one(); num_vars],
            w_neg: vec![BigRational::one(); num_vars],
        }
    }
}

impl Semiring for RationalSemiring {
    type Value = BigRational;

    fn zero(&self) -> BigRational { BigRational::zero() }

    fn leaf(&self, v: VarId, label: LeafLabel) -> BigRational {
        let i = v.idx();
        match label {
            LeafLabel::Pos => self.w_pos[i].clone(),
            LeafLabel::Neg => self.w_neg[i].clone(),
            LeafLabel::One => &self.w_pos[i] + &self.w_neg[i],
            LeafLabel::Zero => BigRational::zero(),
        }
    }

    #[inline]
    fn add_assign(&self, acc: &mut BigRational, other: &BigRational) { *acc += other; }

    #[inline]
    fn mul(&self, a: &BigRational, b: &BigRational) -> BigRational { a * b }
}

// ── Bounded-precision signed log-domain weight (weighted marg path) ───────────

/// Bounded-precision signed log-domain weight: sign ∈ {-1,0,+1}; `ln_abs` = ln|value|
/// (conventionally `f64::NEG_INFINITY` when sign==0). Used by the weighted marg path
/// in log mode (driver-selected per track, see `weight_store::resolve_log_mode`)
/// to bound per-op cost (vs `BigRational` digit growth).
///
/// MCC weights are 16-digit decimals; weighted multiply compounds ~16 digits onto
/// the numerator AND denominator of an exact `BigRational`, so on a multi-kilovar
/// instance the rationals reach thousands of decimal digits and each mul/add/gcd
/// becomes O(digits). The log domain bounds every op to O(1) f64 work at ~1e-15
/// relative error — far under the MCC 1% WMC tolerance. The sign is tracked
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
/// `SignedLog` (log mode, see `weight_store::resolve_log_mode`). The two modes
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
/// optimization), so on the weighted canopy path — where a common-denominator
/// rescaling upstream makes every hot value an integer-valued rational with
/// denominator 1 — the arbitrary-precision representation paid a malloc/free
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
/// demotes to `ExactSmall`. This is what keeps [`WeightKey`]'s derived
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
/// dominates the weighted profile. That is the common case here: the weighted
/// canopy pipeline rescales its weight table to integer-valued weights
/// (`driver::canopy::integer_valued_weights`), and every value the weighted
/// fold builds is a `+`/`·` closure over those seeds, so it stays integer-valued
/// all the way to the leaf's output.
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
#[doc(hidden)]
#[non_exhaustive]
pub enum WeightKey {
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
#[doc(hidden)]
pub fn weight_key(v: &WeightVal) -> WeightKey {
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
#[doc(hidden)]
pub type WeightMap = FxHashMap<WeightKey, u32>;

#[cfg(test)]
#[path = "semiring_signed_log_tests.rs"]
mod signed_log_tests;

#[cfg(test)]
#[path = "semiring_exact_tests.rs"]
mod exact_tests;
