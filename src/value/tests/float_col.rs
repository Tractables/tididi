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
    let pow2 = |n: i64| f64::from_bits(((n + 1023) as u64) << 52);
    let mut p = ScaledFloat::from_f64(1.0);
    let step = ScaledFloat::from_f64(if e < 0 { pow2(-512) } else { pow2(512) });
    let mut left = e;
    while left.abs() >= 512 {
        p = p.mul(&step);
        left -= 512 * left.signum();
    }
    p.mul(&ScaledFloat::from_f64(pow2(left)))
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

/// A fold that appends floats under a scale of its own takes a narrow
/// column whose scale is unset, as on a column of zeros, or the same, and
/// no other; a wide column it never takes.
#[test]
fn a_column_is_taken_under_its_own_scale_only() {
    let eng = Engine::new();
    let mut col = FloatCol::zeros(&eng, 2).unwrap();
    col.narrow_under(5).expect("a column of zeros takes any scale").push(0.75);
    assert!(col.narrow_under(6).is_none());
    col.narrow_under(5).expect("the column's own scale").push(0.5);
    let want = [ScaledFloat::ZERO, ScaledFloat::ZERO, value(1 << 19, 5, false), value(0, 5, false)];
    assert_eq!(col.iter().map(bits).collect::<Vec<_>>(), want.iter().copied().map(bits).collect::<Vec<_>>());
    let mut wide = of(&[value(5, 0, false), value(7, 5000, false)]);
    assert!(is_wide(&wide));
    assert!(wide.narrow_under(0).is_none());
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

/// The narrow column's room, or the wide one's.
fn capacity(col: &FloatCol) -> usize {
    match &col.repr {
        Repr::Narrow { vals, .. } => vals.capacity(),
        Repr::Wide(vals) => vals.capacity(),
    }
}

#[test]
fn offsets_pushed_together_are_those_pushed_one_at_a_time() {
    let eng = Engine::new();
    let normal = |e: u64, k: u64| f64::from_bits(e << 52 | k.wrapping_mul(0x1234_5678_9abc) & ((1 << 52) - 1));
    let mut xs = [0.0f64; 64];
    for (i, x) in (0u64..).zip(xs.iter_mut()) {
        *x = match i % 9 {
            0 => 0.0,
            1 => normal(2 + i % 4, i),
            2 => normal(1, i),
            3 => normal(2046, i),
            4 => normal(1 + i * 31 % 2046, i),
            _ => normal(1000 + i, i),
        };
    }
    let at = 100i64;
    for d in [0i64, 7, -900, 1020, -1021, 2045, -2045, 5000] {
        for live in [u64::MAX, 0x5555_5555_5555_5555, 0xf0f0_0000_0000_0f0f, 1 << 63, 0] {
            for room in [1usize, 40, 70] {
                let start = || {
                    let mut col = FloatCol::with_capacity(&eng, room).unwrap();
                    col.push(&eng, value(1, at, false)).unwrap();
                    col
                };
                let (mut one, mut all) = (start(), start());
                let (mut want, mut got) = ([u32::MAX; 64], [u32::MAX; 64]);
                let mut left = live;
                while left != 0 {
                    let i = left.trailing_zeros() as usize;
                    want[i] = one.push_offset(&eng, xs[i], at + d).unwrap();
                    left &= left - 1;
                }
                all.push_offsets(&eng, (&xs, live), at + d, &mut got).unwrap();
                let case = format!("offset {d}, live {live:x}, room {room}");
                assert_eq!(got, want, "{case}");
                assert_eq!((all.len(), is_wide(&all), capacity(&all)), (one.len(), is_wide(&one), capacity(&one)), "{case}");
                for s in 0..one.len() {
                    assert_eq!(bits(all.get(s)), bits(one.get(s)), "{case}, slot {s}");
                }
            }
        }
    }
}

#[test]
fn offsets_pushed_together_past_the_slot_cap_are_refused() {
    let eng = Engine::new();
    let mut col = FloatCol::with_capacity(&eng, 64).unwrap();
    col.push(&eng, value(1, 0, false)).unwrap();
    let xs = [0.75f64; 64];
    eng.limits().pin_slot_cap(Some(8));
    let mut slots = [0u32; 64];
    assert!(col.push_offsets(&eng, (&xs, 0x7f), 0, &mut slots).is_ok());
    assert_eq!(col.len(), 8);
    let refused = col.push_offsets(&eng, (&xs, 1), 0, &mut slots);
    assert!(matches!(refused, Err(crate::limits::OperationError::IndexOverflow)));
    eng.limits().pin_slot_cap(None);
}
