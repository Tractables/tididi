//! Cold storage for pair ranges that exceed the node word's 31-bit fields.

use super::arena::{ArenaGrowth, Untracked};
use crate::diagram::PairRange;
use crate::execution::pool::Scratch;
use crate::limits::{Charged, Limits, OperationError};

#[derive(Clone, Default)]
// Move the vector header out of levels that never need wide ranges.
#[allow(clippy::box_collection)]
pub(crate) struct RangeTable(Option<Box<Vec<PairRange>>>);

impl std::fmt::Debug for RangeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.deref_slice(), f)
    }
}

impl std::ops::Deref for RangeTable {
    type Target = [PairRange];
    fn deref(&self) -> &[PairRange] { self.0.as_deref().map_or(&[], Vec::as_slice) }
}

impl std::ops::IndexMut<usize> for RangeTable {
    fn index_mut(&mut self, i: usize) -> &mut PairRange {
        &mut self.0.as_mut().expect("a ranged node has a range table")[i]
    }
}

impl std::ops::Index<usize> for RangeTable {
    type Output = PairRange;
    fn index(&self, i: usize) -> &PairRange { &self.deref_slice()[i] }
}

impl PartialEq for RangeTable {
    fn eq(&self, other: &Self) -> bool { self.deref_slice() == other.deref_slice() }
}
impl Eq for RangeTable {}

impl From<Vec<PairRange>> for RangeTable {
    fn from(v: Vec<PairRange>) -> Self {
        if v.capacity() == 0 { Self::default() } else { Self(Some(Box::new(v))) }
    }
}

impl RangeTable {
    fn deref_slice(&self) -> &[PairRange] { self }

    pub(crate) fn capacity(&self) -> usize { self.0.as_deref().map_or(0, Vec::capacity) }

    pub(crate) fn clear(&mut self) {
        if let Some(v) = &mut self.0 { v.clear(); }
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        if self.is_empty() { self.0 = None; }
        else if let Some(v) = &mut self.0 { v.shrink_to_fit(); }
    }

    pub(crate) fn grow(&mut self, growth: &impl ArenaGrowth, additional: usize) -> Result<(), OperationError> {
        if additional <= self.capacity() - self.len() { return Ok(()); }
        if let Some(v) = &mut self.0 { return growth.grow(v, additional); }
        let mut v = Vec::new();
        growth.grow(&mut v, additional)?;
        growth.charge(std::mem::size_of::<Vec<PairRange>>() as u64)?;
        self.0 = Some(Box::new(v));
        Ok(())
    }

    pub(crate) fn reserve_exact(&mut self, lim: &Limits, additional: usize) -> Result<(), OperationError> {
        if additional <= self.capacity() - self.len() { return Ok(()); }
        if let Some(v) = &mut self.0 { return lim.reserve_exact(v, additional); }
        let mut v = Vec::new();
        lim.reserve_exact(&mut v, additional)?;
        lim.charge_bytes(std::mem::size_of::<Vec<PairRange>>() as u64)?;
        self.0 = Some(Box::new(v));
        Ok(())
    }

    pub(crate) fn push(&mut self, range: PairRange) {
        self.grow(&Untracked, 1).expect("out of memory storing a wide pair range");
        self.0.as_mut().unwrap().push(range);
    }

    pub(crate) fn try_clone_on(&self, lim: &Limits) -> Result<Self, OperationError> {
        let mut out = Self::default();
        out.reserve_exact(lim, self.len())?;
        if let Some(v) = &mut out.0 { v.extend_from_slice(self); }
        Ok(out)
    }

    pub(crate) fn retain(&mut self, cap: usize) -> u64 {
        if self.charged_bytes() > cap as u64 { self.0 = None; }
        self.charged_bytes()
    }
}

impl Charged for RangeTable {
    fn charged_bytes(&self) -> u64 {
        self.0.as_ref().map_or(0, |v| {
            v.charged_bytes() + std::mem::size_of::<Vec<PairRange>>() as u64
        })
    }
}

impl Scratch for RangeTable {
    fn release(&mut self) { self.0 = None; }
}

#[cfg(test)]
#[path = "tests/ranges.rs"]
mod tests;
