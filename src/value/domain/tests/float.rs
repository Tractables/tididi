//! A cell whose child columns are narrow is summed in plain floats, each
//! column's floats centred by a power of two where the cell needs it, and
//! the sum is the scaled-float fold's, bit for bit.

use super::{plain_fold, plain_fold_centred, pow2, FloatColumn, Plain, PLAIN_HI, PLAIN_LO};
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, ScaledFloat};
use crate::test_helpers::Lcg;
use crate::value::{FloatFold, StreamChild};

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

/// A narrow child column read through its slots.
fn child(col: Plain<'_>) -> StreamChild<'_, FloatFold> {
    StreamChild { col: FloatColumn::Narrow(col), view: ChildDecoder::structural() }
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
/// sum takes, well inside them and past them on either side, under random
/// scales: every cell whose columns' offsets fit, as they are or centred,
/// is summed in plain floats.
#[test]
fn a_plain_sum_is_the_scaled_fold_bit_for_bit() {
    let mut rng = Lcg::new(0xf10a7);
    let view = ChildDecoder::structural();
    let ranges = [
        (-3, 3),
        (-200, 40),
        (PLAIN_LO / 2, PLAIN_LO / 2 + 30),
        (PLAIN_HI / 2 - 30, PLAIN_HI / 2),
        (-900, 900),
        (-1021, -500),
        (600, 1024),
        (-1021, 200),
    ];
    let (mut plain, mut centred, mut outside) = (0, 0, 0);
    for round in 0..800 {
        let (lo, hi) = ranges[round % ranges.len()];
        let (l, r) = (column(&mut rng, 40, lo, hi), column(&mut rng, 40, lo, hi));
        let (ls, rs) = (rng.below(4000) as i64 - 2000, rng.below(4000) as i64 - 2000);
        let (left, right) = (Plain::new(&l, ls), Plain::new(&r, rs));
        let n = 1 + rng.below(30) as usize;
        let pairs: Vec<ChildPair> = (0..n).map(|_| pair(rng.below(40) as u32, rng.below(40) as u32)).collect();
        let fits = left.at.lo + right.at.lo >= PLAIN_LO && left.at.hi + right.at.hi <= PLAIN_HI;
        let got = match fits {
            true => plain_fold(&pairs, left, view, right, view),
            false => plain_fold_centred(&pairs, &child(left), &child(right)),
        };
        match (got, fits) {
            (Some(got), _) => {
                assert_eq!(bits(got), bits(scaled_fold(&pairs, &l, ls, &r, rs)), "round {round}");
                match fits {
                    true => plain += 1,
                    false => centred += 1,
                }
            }
            (None, false) => outside += 1,
            (None, true) => {}
        }
    }
    assert!(plain > 300 && centred > 100 && outside > 50, "{plain} plain, {centred} centred, {outside} outside");
}

/// A column inside its half of the plain range is read as it is; one past
/// it, of offsets spanning less than the half, is centred in it; one wider
/// than the half is read as it is.
#[test]
fn a_column_away_from_the_middle_is_centred() {
    let mut rng = Lcg::new(0xce47);
    let (lo, hi) = (PLAIN_LO / 2, PLAIN_HI / 2);
    for (a, b, moved) in [(-500, 0, false), (-1000, -700, true), (700, 1024, true), (-1021, 0, false)] {
        let vals: Vec<f64> = (0..64).map(|i| float_at(&mut rng, if i % 2 == 0 { a } else { b })).collect();
        let plain = Plain::new(&vals, 17);
        assert_eq!(plain.centre != 0, moved, "{a}..{b}");
        let at = plain.centred();
        if moved {
            assert!(at.lo >= lo && at.hi <= hi, "{a}..{b}: {at:?}");
        }
        let by = pow2(plain.centre);
        for &x in &vals {
            assert_eq!(bits(ScaledFloat::at_offset(x * by, at.scale)), bits(ScaledFloat::at_offset(x, 17)));
            assert!(x == 0.0 || (x * by).is_normal());
        }
    }
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
    let (left, right) = (Plain::new(&l, 0), Plain::new(&r, 0));
    assert!(left.centre == 0 && right.centre == 0 && left.at.lo + right.at.lo >= PLAIN_LO);
    assert!(plain_fold(&pairs, left, view, right, view).is_none());
    let exact = scaled_fold(&pairs, &l, 0, &r, 0);
    assert_eq!(bits(exact), (0.5f64.to_bits(), -1071));
}
