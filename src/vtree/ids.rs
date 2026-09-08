//! Variable and vtree-node identifiers.

/// A 0-indexed variable identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VarId(pub u32);

/// Index into the vtree node array.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VtreeIdx(pub u32);

impl VtreeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl VarId {
    /// The variable number as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}
