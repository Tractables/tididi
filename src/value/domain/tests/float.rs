//! A cell whose child columns are narrow is summed in plain floats, and the
//! sum is the scaled-float fold's, bit for bit.

use super::{plain_fold, Offsets, Plain, PLAIN_HI, PLAIN_LO};
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, ScaledFloat};
use crate::test_helpers::Lcg;
use crate::value::FloatFold;

/// A float at offset `d`, `m · 2^d` with `m` in `[0.5, 1)`, of random
/// fraction and sign; zero one time in eight.
fn float_at(rng: &mut Lcg, d: i64) -> f64 {
    if rng.below(8) == 0 {
        return 0.0;
    }
    let sign = if rng.coin() { 1u64 << 63 } else { 0 };
    f64::from_bits(sign | ((d + 1022) as u64) << 52 | rng.next_u64() >> 12)
}

/// `len` floats at offsets in `lo..=hi`.
fn column(rng: &mut Lcg, len: usize, lo: i64, hi: i64) -> Vec<f64> {
    (0..len)
        .map(|_| {
            let d = lo + rng.below((hi - lo + 1) as u64) as i64;
            float_at(rng, d)
        })
        .collect()
}

fn bits(v: ScaledFloat) -> (u64, i64) {
    (v.mantissa().to_bits(), v.exponent())
}

fn pair(left: u32, right: u32) -> ChildPair {
    ChildPair { left: EncodedChildRef::from_raw(left), right: EncodedChildRef::from_raw(right) }
}

/// The scaled-float fold of the same cell.
fn scaled_fold(pairs: &[ChildPair], l: &[f64], ls: i64, r: &[f64], rs: i64) -> ScaledFloat {
    let view = ChildDecoder::structural();
    FloatFold::fold(
        pairs.iter().copied(),
        |k| ScaledFloat::at_offset(l[view.index(k)], ls),
        |k| ScaledFloat::at_offset(r[view.index(k)], rs),
    )
}

/// Random cells of signed values, near both ends of the offsets the plain
/// sum takes and well inside them, under random scales.
#[test]
fn a_plain_sum_is_the_scaled_fold_bit_for_bit() {
    let mut rng = Lcg::new(0xf10a7);
    let view = ChildDecoder::structural();
    let ranges = [(-3, 3), (-200, 40), (PLAIN_LO / 2, PLAIN_LO / 2 + 30), (PLAIN_HI / 2 - 30, PLAIN_HI / 2), (-900, 900)];
    let (mut plain, mut outside) = (0, 0);
    for round in 0..500 {
        let (lo, hi) = ranges[round % ranges.len()];
        let (l, r) = (column(&mut rng, 40, lo, hi), column(&mut rng, 40, lo, hi));
        let (ls, rs) = (rng.below(4000) as i64 - 2000, rng.below(4000) as i64 - 2000);
        let left = Plain { vals: &l, at: Offsets::of(&l, ls) };
        let right = Plain { vals: &r, at: Offsets::of(&r, rs) };
        let n = 1 + rng.below(30) as usize;
        let pairs: Vec<ChildPair> = (0..n).map(|_| pair(rng.below(40) as u32, rng.below(40) as u32)).collect();
        if left.at.lo + right.at.lo < PLAIN_LO || left.at.hi + right.at.hi > PLAIN_HI {
            outside += 1;
            continue;
        }
        if let Some(got) = plain_fold(&pairs, left, view, right, view) {
            assert_eq!(bits(got), bits(scaled_fold(&pairs, &l, ls, &r, rs)), "round {round}");
            plain += 1;
        }
    }
    assert!(plain > 300 && outside > 50, "{plain} plain sums, {outside} cells outside the offsets");
}

/// Two products that cancel to below the normal floats: the plain sum
/// declines the cell, whose value the scaled fold holds exactly.
#[test]
fn a_sum_cancelled_below_the_normal_floats_is_declined() {
    let view = ChildDecoder::structural();
    let x = 2f64.powi(-510);
    let l = [x * (1.0 + f64::EPSILON), -x];
    let r = [x];
    let pairs = [pair(0, 0), pair(1, 0)];
    let left = Plain { vals: &l, at: Offsets::of(&l, 0) };
    let right = Plain { vals: &r, at: Offsets::of(&r, 0) };
    assert!(left.at.lo + right.at.lo >= PLAIN_LO);
    assert!(plain_fold(&pairs, left, view, right, view).is_none());
    let exact = scaled_fold(&pairs, &l, 0, &r, 0);
    assert_eq!(bits(exact), (0.5f64.to_bits(), -1071));
}
