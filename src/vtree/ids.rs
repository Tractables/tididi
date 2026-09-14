//! Variable and vtree-node identifiers.

/// A zero-based variable identifier, independent of its position in the vtree.
///
/// `VarId(0)` is the variable named by integer literals `1` and `-1`.
/// Resolve its leaf with [`Vtree::leaf_of`](super::Vtree::leaf_of); a variable id
/// and a [`VtreeIdx`] are different index spaces.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VarId(pub u32);

/// An index identifying a node in one vtree's node array.
///
/// It may name a leaf or an internal node and is not a variable identifier.
/// Use [`Vtree::bottomup`](super::Vtree::bottomup) for traversal order, which can
/// differ from index order after a rotation.
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
