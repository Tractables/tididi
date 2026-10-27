//! A scaled-float column reads back every value it was given, bit for bit,
//! narrow while the values fit one scale and wide after one does not.

use super::{FloatCol, Repr};
use crate::diagram::ScaledFloat;
use crate::Engine;

thread_local! {
    static WIDENED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count a widening, for [`widenings`].
pub(super) fn note_widened() {
    WIDENED.with(|n| n.set(n.get() + 1));
}

/// How many columns this thread has widened.
pub(crate) fn widenings() -> u64 {
    WIDENED.with(std::cell::Cell::get)
}

/// Whether the column holds whole values.
fn is_wide(col: &FloatCol) -> bool {
    matches!(col.repr, Repr::Wide(_))
}

/// A column of `values`, outside any engine's accounting.
pub(crate) fn of(values: &[ScaledFloat]) -> FloatCol {
    let eng = Engine::new();
    let mut col = FloatCol::default();
    for &v in values {
        col.push(&eng, v).expect("an unlimited engine reserves");
    }
    col
}

/// `±(0.5 + k/2^20) · 2^e`.
fn value(k: u64, e: i64, negative: bool) -> ScaledFloat {
    let m = 0.5 + k as f64 / (1u64 << 21) as f64;
    let v = ScaledFloat::from_f64(if negative { -m } else { m });
    ScaledFloat::from_f64(v.mantissa()).mul(&power(e))
}

/// `2^e`, whatever `e`.
fn power(e: i64) -> ScaledFloat {
    let mut p = ScaledFloat::from_f64(1.0);
    let step = ScaledFloat::from_f64(if e < 0 { 2f64.powi(-512) } else { 2f64.powi(512) });
    let mut left = e;
    while left.abs() >= 512 {
        p = p.mul(&step);
        left -= 512 * left.signum();
    }
    p.mul(&ScaledFloat::from_f64(2f64.powi(left as i32)))
}

fn bits(v: ScaledFloat) -> (u64, i64) {
    (v.mantissa().to_bits(), v.exponent())
}

#[test]
fn offsets_read_back_exactly_at_the_range_ends() {
    for scale in [-5000i64, -3, 0, 7, 4000] {
        for d in [-1021i64, -1020, -1, 0, 1, 1023, 1024] {
            for negative in [false, true] {
                let v = value(12345, scale + d, negative);
                let x = v.offset_from(scale).expect("within the range");
                assert!(x.is_normal(), "scale {scale}, offset {d}");
                assert_eq!(bits(ScaledFloat::at_offset(x, scale)), bits(v), "scale {scale}, offset {d}");
            }
        }
        for d in [-1023i64, -1022, 1025, 1026, 5000] {
            assert_eq!(value(1, scale + d, false).offset_from(scale), None, "scale {scale}, offset {d}");
        }
        assert_eq!(ScaledFloat::ZERO.offset_from(scale), Some(0.0));
    }
}

#[test]
fn a_column_stays_narrow_within_its_scale() {
    let eng = Engine::new();
    let values: Vec<ScaledFloat> = (0..200u64)
        .map(|i| match i % 7 {
            0 => ScaledFloat::ZERO,
            // The first nonzero value sets the scale at 100.
            _ if i == 1 => value(1, 100, false),
            _ => value(i * 977, 100 + (i as i64 * 37) % 2000 - 1000, i % 3 == 0),
        })
        .collect();
    let mut col = FloatCol::with_capacity(&eng, 4).unwrap();
    for &v in &values {
        col.push(&eng, v).unwrap();
    }
    assert!(!is_wide(&col));
    assert_eq!(col.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(bits(col.get(i)), bits(v), "slot {i}");
    }
}

#[test]
fn a_value_outside_the_scale_widens_the_column() {
    let eng = Engine::new();
    for (first, far) in [(0i64, 1025i64), (0, -1022), (-40_000, 9_000), (3, 3 + 5000)] {
        let values = [ScaledFloat::ZERO, value(5, first, false), value(9, first + 1, true), value(7, far, false), ScaledFloat::ZERO, value(3, first, true)];
        let mut col = FloatCol::with_capacity(&eng, 2).unwrap();
        for (i, &v) in values.iter().enumerate() {
            col.push(&eng, v).unwrap();
            assert_eq!(is_wide(&col), i >= 3, "{first}, {far}, after {i}");
        }
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(bits(col.get(i)), bits(v), "{first}, {far}, slot {i}");
        }
        assert_eq!(col.iter().map(bits).collect::<Vec<_>>(), values.iter().copied().map(bits).collect::<Vec<_>>());
        let mut set = FloatCol::zeros(&eng, values.len()).unwrap();
        for (i, &v) in values.iter().enumerate().rev() {
            set.set(&eng, i, v).unwrap();
        }
        assert!(is_wide(&set));
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(bits(set.get(i)), bits(v), "{first}, {far}, set slot {i}");
        }
    }
}

#[test]
fn a_refused_widening_is_an_error() {
    let eng = Engine::new();
    let mut col = FloatCol::with_capacity(&eng, 4).unwrap();
    col.push(&eng, value(1, 0, false)).unwrap();
    eng.limits().refuse_nth_reserve(0);
    let refused = col.push(&eng, value(1, 4000, false));
    eng.limits().grant_every_reserve();
    assert!(refused.is_err());
    assert_eq!(bits(col.get(0)), bits(value(1, 0, false)));
}
