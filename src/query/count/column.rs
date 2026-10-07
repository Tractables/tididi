//! Query-owned counts, widening a column only when a value needs it.

use crate::{Engine, OperationError};
use crate::diagram::{ChildPair, TddLevel, LEAF_WIDTH};
use crate::limits::Charged;
use crate::value::{Count, CountRead, CountVec, IntFold};
use crate::query::fold::Column;
use smallvec::SmallVec;

/// Storage operations used by the shared exact counting fold.
pub(crate) trait CountColumn: Column + Charged + Sized {
    /// Only retained counters can install a compact read plan.
    const PREPARED_READS: bool = false;
    fn try_with_width(eng: &Engine, width: usize) -> Result<Self, OperationError>;
    fn get(&self, i: usize) -> CountRead<'_>;
    fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError>;
    fn fold_structural(pairs: impl Iterator<Item = ChildPair>, left: &Self, right: &Self) -> Option<u128>;
    /// Fill the slots `range` of `col`, an internal level's column whose
    /// children `left` and `right` are both structural, in one pass while
    /// each total fits the fast storage; returns the first slot it did not
    /// fill, `range.end` when it filled them all. The default fills none.
    fn fill_structural(_level: &TddLevel, _left: &Self, _right: &Self, _col: &mut Self, range: std::ops::Range<usize>) -> usize {
        range.start
    }
}

impl Column for CountVec {
    fn width(&self) -> usize { self.len() }
}

impl CountColumn for CountVec {
    fn try_with_width(eng: &Engine, width: usize) -> Result<Self, OperationError> {
        CountVec::try_with_width(eng, width)
    }

    fn get(&self, i: usize) -> CountRead<'_> { CountVec::get(self, i) }

    fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError> {
        CountVec::set(self, eng, i, value)
    }

    fn fold_structural(pairs: impl Iterator<Item = ChildPair>, left: &Self, right: &Self) -> Option<u128> {
        let (left, right) = (left.as_count_ref(), right.as_count_ref());
        if left.all_u64() && right.all_u64() {
            IntFold::fold_structural_by(pairs,
                |k| left.fast_slice()[k.raw() as usize] as u64 as u128,
                |k| right.fast_slice()[k.raw() as usize] as u64 as u128)
        } else { None }
    }

    fn fill_structural(level: &TddLevel, left: &Self, right: &Self, col: &mut Self, range: std::ops::Range<usize>) -> usize {
        let (left, right) = (left.as_count_ref(), right.as_count_ref());
        if left.all_u64() && right.all_u64() {
            IntFold::fill_structural_u64(level, left.fast_slice(), right.fast_slice(), col, range)
        } else { range.start }
    }
}

/// Counts fit in u64 until a column widens; leaf-sized columns stay inline.
/// Wide columns share the exact overflow representation used by marginal stores.
pub(crate) enum QueryCounts {
    Narrow(SmallVec<[u64; LEAF_WIDTH]>),
    Wide(CountVec),
}

impl Default for QueryCounts {
    fn default() -> Self { Self::Narrow(SmallVec::new()) }
}

impl Charged for QueryCounts {
    fn charged_bytes(&self) -> u64 {
        match self { Self::Narrow(v) => v.charged_bytes(), Self::Wide(v) => v.charged_bytes() }
    }
}

impl Column for QueryCounts {
    fn width(&self) -> usize {
        match self { Self::Narrow(v) => v.len(), Self::Wide(v) => v.len() }
    }
}

impl CountColumn for QueryCounts {
    const PREPARED_READS: bool = true;

    #[inline(always)]
    fn try_with_width(eng: &Engine, width: usize) -> Result<Self, OperationError> {
        let mut v = if width <= LEAF_WIDTH {
            SmallVec::new()
        } else {
            let mut heap = Vec::new();
            eng.limits().reserve_exact(&mut heap, width)?;
            SmallVec::from_vec(heap)
        };
        v.resize(width, 0);
        Ok(Self::Narrow(v))
    }

    fn get(&self, i: usize) -> CountRead<'_> {
        match self { Self::Narrow(v) => CountRead::Fast(v[i] as u128), Self::Wide(v) => v.get(i) }
    }

    #[inline]
    fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError> {
        match self {
            Self::Wide(v) => v.set(eng, i, value),
            Self::Narrow(v) => {
                if let Count::Fast(x) = value && x <= u64::MAX as u128 {
                    v[i] = x as u64;
                    return Ok(());
                }
                self.promote_and_set(eng, i, value)
            }
        }
    }

    fn fold_structural(pairs: impl Iterator<Item = ChildPair>, left: &Self, right: &Self) -> Option<u128> {
        match (left, right) {
            (Self::Narrow(l), Self::Narrow(r)) => {
                let (l, r) = (l.as_slice(), r.as_slice());
                IntFold::fold_structural_by(pairs,
                    |k| l[k.raw() as usize] as u128, |k| r[k.raw() as usize] as u128)
            }
            (Self::Narrow(l), Self::Wide(r)) if r.as_count_ref().all_u64() => {
                let (l, r) = (l.as_slice(), r.as_count_ref().fast_slice());
                IntFold::fold_structural_by(pairs, |k| l[k.raw() as usize] as u128, |k| r[k.raw() as usize] as u64 as u128)
            }
            (Self::Wide(l), Self::Narrow(r)) if l.as_count_ref().all_u64() => {
                let (l, r) = (l.as_count_ref().fast_slice(), r.as_slice());
                IntFold::fold_structural_by(pairs, |k| l[k.raw() as usize] as u64 as u128, |k| r[k.raw() as usize] as u128)
            }
            (Self::Wide(l), Self::Wide(r)) => CountVec::fold_structural(pairs, l, r),
            _ => None,
        }
    }

    fn fill_structural(level: &TddLevel, left: &Self, right: &Self, col: &mut Self, range: std::ops::Range<usize>) -> usize {
        match (left, right, col) {
            (Self::Narrow(l), Self::Narrow(r), Self::Narrow(c)) => IntFold::fill_structural_narrow(level, l, r, c, range),
            (Self::Wide(l), Self::Wide(r), Self::Wide(c)) => CountVec::fill_structural(level, l, r, c, range),
            _ => range.start,
        }
    }
}

impl QueryCounts {
    /// Finish promotion before replacing any existing values.
    #[cold]
    #[inline(never)]
    fn promote_and_set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError> {
        let Self::Narrow(v) = self else { unreachable!("only narrow columns promote"); };
        let mut wide = CountVec::try_with_width(eng, v.len())?;
        for (j, &x) in v.iter().enumerate() { wide.set(eng, j, Count::Fast(x as u128))?; }
        wide.set(eng, i, value)?;
        eng.limits().discard(std::mem::replace(self, Self::Wide(wide)));
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/column.rs"]
mod tests;
