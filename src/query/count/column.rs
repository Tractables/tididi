//! Query-owned counts, widening a column only when a value needs it.

use crate::{Engine, OperationError};
use crate::diagram::{ChildPair, LEAF_WIDTH};
use crate::limits::Charged;
use crate::value::{Count, CountRead, CountVec, IntFold};
use crate::query::fold::Column;
use smallvec::SmallVec;

/// Storage operations used by the shared exact counting fold.
pub(crate) trait CountColumn: Column + Charged + Sized {
    fn try_with_width(eng: &Engine, width: usize) -> Result<Self, OperationError>;
    fn get(&self, i: usize) -> CountRead<'_>;
    fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError>;
    fn fold_structural(pairs: &[ChildPair], left: &Self, right: &Self) -> Option<u128>;
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

    fn fold_structural(pairs: &[ChildPair], left: &Self, right: &Self) -> Option<u128> {
        let (left, right) = (left.as_count_ref(), right.as_count_ref());
        if left.all_u64() && right.all_u64() {
            IntFold::fold_structural_u64(pairs, left.fast_slice(), right.fast_slice())
        } else { None }
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

    fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError> {
        match self {
            Self::Wide(v) => v.set(eng, i, value),
            Self::Narrow(v) => {
                if let Count::Fast(x) = value && x <= u64::MAX as u128 {
                    v[i] = x as u64;
                    return Ok(());
                }
                // Finish promotion before replacing any existing values.
                let mut wide = CountVec::try_with_width(eng, v.len())?;
                for (j, &x) in v.iter().enumerate() { wide.set(eng, j, Count::Fast(x as u128))?; }
                wide.set(eng, i, value)?;
                eng.limits().discard(std::mem::replace(self, Self::Wide(wide)));
                Ok(())
            }
        }
    }

    fn fold_structural(pairs: &[ChildPair], left: &Self, right: &Self) -> Option<u128> {
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
}

#[cfg(test)]
#[path = "tests/column.rs"]
mod tests;
