//! The budget-tracked clone the count-vector round-trip tests copy through.

use super::*;

impl CountOverflow {
    /// Budget-tracked clone: reserves the entry count exactly before copying.
    pub(crate) fn try_clone(
        &self,
        eng: &Engine,
    ) -> Result<Self, crate::limits::OperationError> {
        let mut entries: Vec<(u32, BigUint)> = Vec::new();
        eng.limits().reserve_exact(&mut entries, self.entries.len())?;
        entries.extend(self.entries.iter().cloned());
        Ok(CountOverflow { entries })
    }
}
