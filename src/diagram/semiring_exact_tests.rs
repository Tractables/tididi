//! The exact `WeightVal` ops must agree with num-rational's generic reducing
//! path — value, representation, AND canonical variant.
//!
//! Two things are pinned here, and both matter:
//!
//! * **Canonical rational form.** `Ratio`'s `PartialEq` compares by VALUE (it
//!   cross-multiplies), so `assert_eq!` alone would not notice a non-canonical
//!   `(numer, denom)`. Every case therefore also pins the pair against
//!   `.reduced()`, which is what a canonicality break would move.
//! * **Canonical variant.** The exact domain has two representations — an
//!   inline `i128` ([`WeightVal::ExactSmall`]) and a `BigRational`
//!   ([`WeightVal::Exact`]) — and the type's invariant is that a value sits in
//!   the small one *whenever* it fits. `WeightKey`'s derived `Eq`/`Hash` treats
//!   the two variants as distinct keys, so a value that fits an `i128` but is
//!   parked in `Exact` would intern as a second, non-colliding copy of itself.
//!   [`assert_canonical_variant`] pins that at every step.

use super::{small_of, WeightVal};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

/// An integer-valued rational built the way the fast path builds one — through
/// `new_raw`, with no reduce on the way in. Small enough for the `i128`
/// representation.
fn int_raw(n: i64) -> BigRational {
    BigRational::new_raw(BigInt::from(n), BigInt::one())
}

/// A big integer-valued rational: ~153 bits, so it is past the `i128` boundary
/// and exercises the arbitrary-precision representation (a stray gcd/div here
/// would be on real limbs, not a fast-path-able small).
fn big_raw(n: i64) -> BigRational {
    BigRational::new_raw(BigInt::from(n) * BigInt::from(10i64).pow(40), BigInt::one())
}

/// The value of an exact `WeightVal`, in whichever representation it is in.
fn exact(v: &WeightVal) -> BigRational {
    assert!(!matches!(v, WeightVal::Log(_)), "expected an exact WeightVal");
    v.as_rational().into_owned()
}

/// Value equality plus canonical form: same `(numer, denom)` as the reduced
/// representative of the same value.
fn assert_same_rational(got: &BigRational, want: &BigRational) {
    assert_eq!(got, want, "value differs");
    let canon = got.reduced();
    assert_eq!(got.numer(), canon.numer(), "numerator not in canonical form");
    assert_eq!(got.denom(), canon.denom(), "denominator not in canonical form");
    assert_eq!(got.numer(), want.reduced().numer(), "numerator differs from generic path");
    assert_eq!(got.denom(), want.reduced().denom(), "denominator differs from generic path");
}

/// The canonicalization invariant: small whenever representable, big otherwise.
fn assert_canonical_variant(v: &WeightVal) {
    match v {
        WeightVal::ExactSmall(_) => {}
        WeightVal::Exact(r) => assert!(
            small_of(r).is_none(),
            "value {r} fits an i128 but is parked in WeightVal::Exact — it would intern \
             under a different WeightKey than its own small form"
        ),
        WeightVal::Log(_) => panic!("expected an exact WeightVal"),
    }
}

/// Both pins at once: canonical rational form and canonical variant.
fn assert_exact_eq(got: &WeightVal, want: &BigRational) {
    assert_same_rational(&exact(got), want);
    assert_canonical_variant(got);
}

#[test]
fn integer_valued_mul_and_add_match_the_generic_ratio_ops() {
    // Mixed signs and zero, small (i128 representation) and big (arbitrary
    // precision), on both sides — so every variant pairing is covered.
    let operands: Vec<BigRational> = vec![
        int_raw(0),
        int_raw(1),
        int_raw(-1),
        int_raw(97),
        int_raw(-97),
        big_raw(1234567),
        big_raw(-1234567),
    ];

    for a in &operands {
        for b in &operands {
            // Reference: the generic num-rational ops on fully-`new`-constructed
            // (already reduced) operands — the path this change bypasses.
            let (ra, rb) = (a.reduced(), b.reduced());

            let product = WeightVal::exact(a.clone()).mul(&WeightVal::exact(b.clone()));
            assert_exact_eq(&product, &(&ra * &rb));

            let mut acc = WeightVal::exact(a.clone());
            acc.add_assign(&WeightVal::exact(b.clone()));
            let mut want = ra.clone();
            want += &rb;
            assert_exact_eq(&acc, &want);
        }
    }
}

#[test]
fn the_representation_is_small_exactly_when_the_value_fits_an_i128() {
    // Boundary walk: i128::MAX is representable, i128::MAX + 1 is not.
    let max = BigInt::from(i128::MAX);
    let min = BigInt::from(i128::MIN);
    for (n, want_small) in [
        (BigInt::zero(), true),
        (BigInt::from(-1), true),
        (max.clone(), true),
        (min.clone(), true),
        (&max + 1u32, false),
        (&min - 1u32, false),
        (&max * 2u32, false),
    ] {
        let v = WeightVal::exact(BigRational::from_integer(n.clone()));
        assert_eq!(
            matches!(v, WeightVal::ExactSmall(_)),
            want_small,
            "wrong representation chosen for {n}"
        );
        assert_canonical_variant(&v);
        assert_eq!(exact(&v), BigRational::from_integer(n));
    }
    // A fractional value never has a small form, however narrow.
    let half = BigRational::new(BigInt::one(), BigInt::from(2));
    assert!(matches!(WeightVal::exact(half), WeightVal::Exact(_)));
}

#[test]
fn spilling_out_of_i128_and_demoting_back_round_trips() {
    // Square up past the i128 boundary: the product must leave the small
    // representation and stay exact.
    let mut v = WeightVal::exact(BigRational::from_integer(BigInt::from(3)));
    let mut want = BigRational::from_integer(BigInt::from(3));
    for _ in 0..8 {
        v = v.mul(&v.clone());
        want = &want * &want;
    }
    assert!(matches!(v, WeightVal::Exact(_)), "3^256 must not fit the small representation");
    assert_exact_eq(&v, &want);

    // Land back inside the boundary by exact cancellation: the result must
    // DEMOTE, or it would key differently from the same value built directly.
    let target = BigRational::from_integer(BigInt::from(-7));
    let delta = &target - &want;
    v.add_assign(&WeightVal::exact(delta));
    assert!(matches!(v, WeightVal::ExactSmall(-7)), "a sum landing in range must demote");
    assert_exact_eq(&v, &target);

    // A multiply that overflows i128 spills; the same product taken in
    // arbitrary precision is the same value.
    let a = WeightVal::ExactSmall(i128::MAX / 3);
    let b = WeightVal::ExactSmall(5);
    let product = a.mul(&b);
    assert!(matches!(product, WeightVal::Exact(_)), "the product overflows i128");
    assert_exact_eq(&product, &BigRational::from_integer(BigInt::from(i128::MAX / 3) * 5u32));

    // …and an add that overflows i128 spills the same way.
    let mut acc = WeightVal::ExactSmall(i128::MAX);
    acc.add_assign(&WeightVal::ExactSmall(i128::MAX));
    assert!(matches!(acc, WeightVal::Exact(_)), "the sum overflows i128");
    assert_exact_eq(&acc, &BigRational::from_integer(BigInt::from(i128::MAX) * 2u32));
}

#[test]
fn add_that_cancels_to_zero_stays_canonical_zero() {
    // Big + big cancelling: the sum is the value zero and must come back in the
    // canonical small representation, indistinguishable from a zero that never
    // left it (the slot-allocation sentinel discipline upstream keys off the
    // VALUE, so the two spellings must not diverge).
    let mut acc = WeightVal::exact(big_raw(42));
    acc.add_assign(&WeightVal::exact(big_raw(-42)));
    assert!(exact(&acc).is_zero());
    assert!(matches!(acc, WeightVal::ExactSmall(0)), "cancelled zero must be the canonical zero");
    // `0/1` is num-rational's normal form for zero; a leftover denominator would
    // still compare equal by value but hash/print differently.
    let got = exact(&acc);
    assert_eq!(got.numer(), &BigInt::zero());
    assert_eq!(got.denom(), &BigInt::one());
    // Accumulating on from zero keeps working.
    acc.add_assign(&WeightVal::exact(int_raw(-5)));
    assert_exact_eq(&acc, &BigRational::from_integer(BigInt::from(-5)));

    // Small + small cancelling reaches the same zero, by the other path.
    let mut acc = WeightVal::exact(int_raw(97));
    acc.add_assign(&WeightVal::exact(int_raw(-97)));
    assert!(matches!(acc, WeightVal::ExactSmall(0)));
}

#[test]
fn a_fractional_operand_falls_through_to_the_generic_path() {
    let third = BigRational::new(BigInt::from(1), BigInt::from(3));
    let six = int_raw(6);

    // Fractional × integer, both orders: must still reduce — and the integer
    // result must land back in the small representation.
    let product = WeightVal::exact(third.clone()).mul(&WeightVal::exact(six.clone()));
    assert_exact_eq(&product, &BigRational::from_integer(BigInt::from(2)));
    assert!(matches!(product, WeightVal::ExactSmall(2)));
    let product = WeightVal::exact(six.clone()).mul(&WeightVal::exact(third.clone()));
    assert_exact_eq(&product, &BigRational::from_integer(BigInt::from(2)));

    // Fractional + fractional summing to an integer, and the mixed order.
    let mut acc = WeightVal::exact(third.clone());
    acc.add_assign(&WeightVal::exact(BigRational::new(BigInt::from(2), BigInt::from(3))));
    assert_exact_eq(&acc, &BigRational::one());

    let mut acc = WeightVal::exact(six.clone());
    acc.add_assign(&WeightVal::exact(third.clone()));
    assert_exact_eq(&acc, &BigRational::new(BigInt::from(19), BigInt::from(3)));
}

/// Deterministic PRNG (SplitMix64) so the randomized sequence below replays
/// identically on every run and on every machine.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A long mixed `mul`/`add_assign` chain, run in lockstep against a pure
/// `Ratio<BigInt>` reference computed with STOCK num-rational operators.
///
/// This is the spill/demote pin: the sequence deliberately crosses the `i128`
/// boundary in both directions many times — squaring and wide multiplies push
/// values out of the small representation, "land on a chosen small integer" and
/// "cancel to zero" steps pull them back in, and fractional multiplies drop
/// into the no-small-form case and back out again. Every step compares value,
/// canonical rational form, and canonical variant, so a wrong spill threshold, a
/// missing demote, or a sign/overflow slip in the checked ops fails here rather
/// than as a silently mis-interned weight in a compile.
#[test]
fn randomized_op_sequence_matches_a_pure_bigrational_reference() {
    let mut rng = SplitMix64(0x0D15_EA5E_5EED_0001);
    let mut acc = WeightVal::exact(BigRational::one());
    let mut want = BigRational::one();
    // Coverage counters — the assertion at the end proves the sequence actually
    // visited both representations rather than trivially staying small.
    let (mut saw_small, mut saw_big, mut saw_frac) = (0u32, 0u32, 0u32);

    for step in 0..5000u32 {
        // Keep the reference from running away: past a few thousand bits force
        // the demote step, so the walk keeps re-crossing the boundary instead of
        // spending the rest of the run in huge-integer arithmetic.
        let runaway = want.numer().bits() > 2048 || want.denom().bits() > 2048;
        let op = if runaway { 4 } else { rng.below(7) };
        match op {
            // × a narrow integer: walks the value up through the boundary a
            // factor at a time.
            0 => {
                let k = BigRational::from_integer(BigInt::from(rng.next_i64() >> 20));
                acc = acc.mul(&WeightVal::exact(k.clone()));
                want = &want * &k;
            }
            // × a wide integer (~126 bits): single-step crossings.
            1 => {
                let k = BigRational::from_integer(
                    BigInt::from(rng.next_i64()) * BigInt::from(rng.next_i64()),
                );
                acc = acc.mul(&WeightVal::exact(k.clone()));
                want = &want * &k;
            }
            // + a narrow integer.
            2 => {
                let k = BigRational::from_integer(BigInt::from(rng.next_i64() >> 8));
                acc.add_assign(&WeightVal::exact(k.clone()));
                want += k;
            }
            // Squaring: the fastest way out of the small representation.
            3 => {
                acc = acc.mul(&acc.clone());
                want = &want * &want;
            }
            // Land exactly on a chosen small integer — the demote round-trip.
            4 => {
                let target = BigRational::from_integer(BigInt::from(rng.next_i64()));
                let delta = &target - &want;
                acc.add_assign(&WeightVal::exact(delta.clone()));
                want += delta;
                assert!(
                    matches!(acc, WeightVal::ExactSmall(_)),
                    "step {step}: a sum landing inside i128 must demote to the small form"
                );
            }
            // Exact cancellation to true zero.
            5 => {
                let delta = -want.clone();
                acc.add_assign(&WeightVal::exact(delta.clone()));
                want += delta;
                assert!(
                    matches!(acc, WeightVal::ExactSmall(0)),
                    "step {step}: cancellation must land on the canonical zero"
                );
                // Re-seed so the walk does not stay absorbed at zero.
                let seed = BigRational::from_integer(BigInt::from(rng.next_i64() | 1));
                acc.add_assign(&WeightVal::exact(seed.clone()));
                want += seed;
            }
            // × a small fraction: no small form on that operand, so this
            // exercises the mixed small×fractional path and the demote back to
            // an integer when the denominator divides out.
            _ => {
                let p = (rng.next_i64() % 17).clamp(-16, 16);
                let q = (rng.next_i64() % 19).abs().max(1);
                let k = BigRational::new(BigInt::from(p), BigInt::from(q));
                acc = acc.mul(&WeightVal::exact(k.clone()));
                want = &want * &k;
            }
        }

        assert_same_rational(&exact(&acc), &want);
        assert_canonical_variant(&acc);
        match &acc {
            WeightVal::ExactSmall(_) => saw_small += 1,
            WeightVal::Exact(r) => {
                saw_big += 1;
                if !r.is_integer() {
                    saw_frac += 1;
                }
            }
            WeightVal::Log(_) => unreachable!(),
        }
    }

    assert!(saw_small > 100, "sequence never settled in the small form ({saw_small})");
    assert!(saw_big > 100, "sequence never spilled out of the small form ({saw_big})");
    assert!(saw_frac > 10, "sequence never held a fractional value ({saw_frac})");
}
