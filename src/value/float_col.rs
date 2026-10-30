//! The column a scaled-float sum-out folds into.

use crate::diagram::ScaledFloat;
use crate::limits::OperationError;
use crate::Engine;

/// One level's scaled-float values, in slot order.
///
/// A column holds each value as an `f64` times one power of two of its own
/// (its scale, the first nonzero value's exponent), 8 bytes a value, while
/// every value lies within `2^-1021` and `2^1024` of the scale
/// ([`ScaledFloat::offset_from`]), so that the `f64` is the value exactly;
/// a value outside that range turns the column into whole [`ScaledFloat`]s,
/// 16 bytes each. Either way it reads back every value it was given, bit
/// for bit.
#[derive(Debug, Default)]
pub(crate) struct FloatCol {
    repr: Repr,
}

#[derive(Debug)]
enum Repr {
    /// `vals[i] · 2^scale`; the scale is unset while every value is zero.
    Narrow { vals: Vec<f64>, scale: Option<i64> },
    /// The values themselves.
    Wide(Vec<ScaledFloat>),
}

impl Default for Repr {
    fn default() -> Self {
        Repr::Narrow { vals: Vec::new(), scale: None }
    }
}

/// A column's values as the fold reads them: the narrow floats and their
/// scale, or whole values.
#[derive(Clone, Copy)]
pub(crate) enum FloatSlice<'a> {
    Narrow(&'a [f64], i64),
    Wide(&'a [ScaledFloat]),
}

impl FloatSlice<'_> {
    /// The value at slot `i`.
    #[inline(always)]
    pub(crate) fn get(self, i: usize) -> ScaledFloat {
        match self {
            FloatSlice::Narrow(vals, scale) => ScaledFloat::at_offset(vals[i], scale),
            FloatSlice::Wide(vals) => vals[i],
        }
    }
}

impl FloatCol {
    /// An empty column with room for `cap` values reserved through the
    /// engine.
    pub(crate) fn with_capacity(eng: &Engine, cap: usize) -> Result<FloatCol, OperationError> {
        let mut vals = Vec::new();
        eng.limits().reserve_exact(&mut vals, cap)?;
        Ok(FloatCol { repr: Repr::Narrow { vals, scale: None } })
    }

    /// `width` zeros, reserved through the engine.
    pub(crate) fn zeros(eng: &Engine, width: usize) -> Result<FloatCol, OperationError> {
        let mut col = FloatCol::with_capacity(eng, width)?;
        col.resize_zeros(width);
        Ok(col)
    }

    /// Fill an empty narrow column with `width` zeros within its reserved capacity.
    pub(crate) fn resize_zeros(&mut self, width: usize) {
        if let Repr::Narrow { vals, .. } = &mut self.repr {
            debug_assert!(vals.is_empty() && vals.capacity() >= width);
            vals.resize(width, 0.0);
        }
    }

    /// The bytes one value takes in a column that stays narrow.
    pub(crate) const NARROW_BYTES: usize = std::mem::size_of::<f64>();

    /// The number of values.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.repr {
            Repr::Narrow { vals, .. } => vals.len(),
            Repr::Wide(vals) => vals.len(),
        }
    }

    /// The value at slot `i`.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> ScaledFloat {
        self.slice().get(i)
    }

    /// The values, for a row loop's reads.
    #[inline]
    pub(crate) fn slice(&self) -> FloatSlice<'_> {
        match &self.repr {
            Repr::Narrow { vals, scale } => FloatSlice::Narrow(vals, scale.unwrap_or(0)),
            Repr::Wide(vals) => FloatSlice::Wide(vals),
        }
    }

    /// The floats of a narrow column whose scale is or becomes `scale`, for
    /// a fold that appends each value as its float under that scale; `None`,
    /// the column unchanged, where it is wide or a nonzero value holds
    /// another scale.
    pub(crate) fn narrow_under(&mut self, scale: i64) -> Option<&mut Vec<f64>> {
        match &mut self.repr {
            Repr::Narrow { vals, scale: at } if at.is_none_or(|at| at == scale) => {
                *at = Some(scale);
                Some(vals)
            }
            _ => None,
        }
    }

    /// The values in slot order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = ScaledFloat> + '_ {
        let s = self.slice();
        (0..self.len()).map(move |i| s.get(i))
    }

    /// `v` as this column's narrow float, setting the scale at the first
    /// nonzero value; `None` where the column is wide or `v` lies outside
    /// its scale's range.
    #[inline(always)]
    fn narrow(scale: &mut Option<i64>, v: ScaledFloat) -> Option<f64> {
        match *scale {
            Some(s) => v.offset_from(s),
            None if v.is_zero() => Some(0.0),
            None => {
                *scale = Some(v.exponent());
                v.offset_from(v.exponent())
            }
        }
    }

    /// Append `v`, the room reserved through the engine.
    #[inline]
    pub(crate) fn push(&mut self, eng: &Engine, v: ScaledFloat) -> Result<(), OperationError> {
        if let Repr::Narrow { vals, scale } = &mut self.repr {
            if let Some(x) = Self::narrow(scale, v) {
                eng.limits().reserve(vals, 1)?;
                vals.push(x);
                return Ok(());
            }
            self.widen(eng, 1)?;
        }
        let Repr::Wide(vals) = &mut self.repr else { unreachable!("a column that took a wide value is wide") };
        eng.limits().reserve(vals, 1)?;
        vals.push(v);
        Ok(())
    }

    /// Append [`ScaledFloat::at_offset`] of `x` and `scale`, for a float
    /// `x` that is zero or normal, and return its slot, checked as
    /// [`next_slot`](crate::limits::Limits::next_slot) checks it before the
    /// column grows. A narrow column under a set scale with room for it
    /// takes `x` times the power of two between the scales where that
    /// float is normal, the float [`push`](Self::push) appends, without
    /// forming the value.
    #[inline(always)]
    pub(crate) fn push_offset(&mut self, eng: &Engine, x: f64, scale: i64) -> Result<u32, OperationError> {
        if let Repr::Narrow { vals, scale: Some(at) } = &mut self.repr {
            let (len, bits) = (vals.len(), x.to_bits());
            // A positive normal float's biased exponent, sign bit clear.
            let e = (bits >> 52) as i64;
            let d = scale.wrapping_sub(*at);
            if len < vals.capacity() && (1..=2046).contains(&e) && (1 - e..=2046 - e).contains(&d) {
                let slot = eng.limits().next_slot(len)?;
                // The biased exponent becomes `e + d`, within 1..=2046: as
                // `offset_from` adds the difference of exponents.
                vals.push(f64::from_bits(bits.wrapping_add((d as u64) << 52)));
                return Ok(slot);
            }
        }
        let slot = eng.limits().next_slot(self.len())?;
        self.push(eng, ScaledFloat::at_offset(x, scale))?;
        Ok(slot)
    }

    /// Set slot `i`, which exists, to `v`.
    pub(crate) fn set(&mut self, eng: &Engine, i: usize, v: ScaledFloat) -> Result<(), OperationError> {
        if let Repr::Narrow { vals, scale } = &mut self.repr {
            if let Some(x) = Self::narrow(scale, v) {
                vals[i] = x;
                return Ok(());
            }
            self.widen(eng, 0)?;
        }
        let Repr::Wide(vals) = &mut self.repr else { unreachable!("a column that took a wide value is wide") };
        vals[i] = v;
        Ok(())
    }

    /// Turn a narrow column into whole values, with room for its capacity
    /// and `more` values beside.
    #[cold]
    #[inline(never)]
    fn widen(&mut self, eng: &Engine, more: usize) -> Result<(), OperationError> {
        let Repr::Narrow { vals, scale } = &self.repr else { return Ok(()) };
        #[cfg(test)]
        tests::note_widened();
        let mut wide: Vec<ScaledFloat> = Vec::new();
        eng.limits().reserve_exact(&mut wide, vals.capacity().max(vals.len() + more))?;
        let s = scale.unwrap_or(0);
        wide.extend(vals.iter().map(|&x| ScaledFloat::at_offset(x, s)));
        let freed = vals.capacity() * Self::NARROW_BYTES;
        self.repr = Repr::Wide(wide);
        eng.limits().release_bytes(freed as u64);
        Ok(())
    }

    /// Give back the room past the values, released through the engine.
    pub(crate) fn shrink_to_fit(&mut self, eng: &Engine) {
        let freed = match &mut self.repr {
            Repr::Narrow { vals, .. } => {
                let room = vals.capacity();
                vals.shrink_to_fit();
                (room - vals.capacity()) * Self::NARROW_BYTES
            }
            Repr::Wide(vals) => {
                let room = vals.capacity();
                vals.shrink_to_fit();
                (room - vals.capacity()) * std::mem::size_of::<ScaledFloat>()
            }
        };
        eng.limits().release_bytes(freed as u64);
    }
}

#[cfg(test)]
#[path = "tests/float_col.rs"]
pub(crate) mod tests;
