//! Query-owned counts, widening a column only when a value needs it.

use crate::{Engine, OperationError};
use crate::diagram::{ChildDecoder, ChildPair, ChildRef, CountOverflow, EncodedChildRef, NodeIdx, ValueRef, LEAF_WIDTH};
use crate::limits::Charged;
use crate::value::{Count, CountRead, CountVec, IntFold};
use crate::query::fold::Column;
use smallvec::SmallVec;

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

impl QueryCounts {
    pub(crate) fn try_with_width(eng: &Engine, width: usize) -> Result<Self, OperationError> {
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

    pub(crate) fn get(&self, i: usize) -> CountRead<'_> {
        match self { Self::Narrow(v) => CountRead::Fast(v[i] as u128), Self::Wide(v) => v.get(i) }
    }

    pub(crate) fn read(&self, view: ChildDecoder, r: EncodedChildRef) -> CountRead<'_> {
        match view.child(r) {
            ChildRef::Value(ValueRef::Inline(c)) => CountRead::Fast(c as u128),
            ChildRef::Node(NodeIdx(i)) | ChildRef::Value(ValueRef::Slot(i)) => self.get(i as usize),
        }
    }

    pub(crate) fn set(&mut self, eng: &Engine, i: usize, value: Count) -> Result<(), OperationError> {
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

    pub(crate) fn into_parts(self, eng: &Engine) -> Result<(Vec<u128>, Option<CountOverflow>), OperationError> {
        match self {
            Self::Wide(v) => Ok(v.into_parts()),
            Self::Narrow(v) => {
                let mut wide = Vec::new();
                eng.limits().reserve_exact(&mut wide, v.len())?;
                wide.extend(v.iter().map(|&x| x as u128));
                eng.limits().discard(v);
                Ok((wide, None))
            }
        }
    }

    pub(crate) fn fold_structural(pairs: &[ChildPair], left: &Self, right: &Self) -> Option<u128> {
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
            (Self::Wide(l), Self::Wide(r)) if l.as_count_ref().all_u64() && r.as_count_ref().all_u64() => {
                IntFold::fold_structural_u64(pairs, l.as_count_ref().fast_slice(), r.as_count_ref().fast_slice())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "tests/column.rs"]
mod tests;
