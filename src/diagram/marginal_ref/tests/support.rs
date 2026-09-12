//! The budget-tracked clone the count-vector round-trip tests copy through.

use super::*;

impl BigSide {
    /// Budget-tracked clone: reserves the entry count exactly before copying.
    pub(crate) fn try_clone<R: crate::limits::ReservePolicy>(
        &self,
        eng: &Engine,
    ) -> Result<Self, R::Err> {
        let mut entries: Vec<(u32, BigUint)> = Vec::new();
        R::reserve_exact(eng, &mut entries, self.entries.len())?;
        entries.extend(self.entries.iter().cloned());
        Ok(BigSide { entries })
    }
}
