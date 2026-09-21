//! Group variables and choose a diagram's decomposition.
//!
//! A vtree is a binary tree with one Boolean variable at each leaf. An internal
//! node splits those variables into the two groups a diagram combines at that
//! level. The [grouping example](crate::guide::examples::vtrees) shows how the
//! choice affects storage.
//!
//! Start with [`Vtree::balanced`], or supply a leaf order to
//! [`Vtree::balanced_over`]. [`Vtree::join`] combines explicit groups;
//! [`Vtree::linear_from_order`] follows a variable order. See [`Vtree`] for
//! construction and traversal, and [`crate::restructure`] for changing a
//! compiled diagram's vtree.
//!
//! # Variable identifiers
//!
//! [`VarId`] identifies a variable independently of its position in the vtree.
//! A vtree can cover a sparse subset of ids: [`Vtree::num_leaves`] counts its
//! variables, while [`Vtree::num_vars`] gives the identifier-space size needed for a
//! weight table. [`Vtree::leaf_of`] returns `None` for an absent variable.
//!
//! # Traversal and storage
//!
//! [`VtreeIdx`] identifies a vtree node. Leaves occupy `0..num_leaves` and
//! internal nodes follow them. Rotations preserve indices but change edges,
//! so use [`Vtree::bottomup`] for child-before-parent order.
//!
//! [`Vtree::to_text`] and [`Vtree::from_text`] use the `.vtree` interchange
//! format; serialization preserves shape and variable labels, not node indices.


mod build;
pub(crate) mod graft;
mod node;
mod project;
pub(crate) mod rng;
mod topo;
mod validate;

pub use graft::GraftLayout;
pub use node::{Vtree, VtreeNode};
pub use topo::RotationKind;


/// The `.vtree` text codec, in both directions.
mod text;

pub(crate) mod rotate; // In-place vtree left/right rotations + topo fixup


/// A variable identifier, independent of its position in the vtree.
///
/// Variables are numbered from 1, like integer literals: `VarId(1)` is the
/// variable named by `1` and `-1`. Zero is not a variable. Resolve a
/// variable's leaf with [`Vtree::leaf_of`]; a variable id and a [`VtreeIdx`]
/// are different index spaces.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VarId(pub u32);

/// An index identifying a node in one vtree's node array.
///
/// It may name a leaf or an internal node and is not a variable identifier.
/// Use [`Vtree::bottomup`] for traversal order, which can
/// differ from index order after a rotation.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VtreeIdx(pub u32);

impl VtreeIdx {
    /// The index as a `usize`.
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl VarId {
    /// The zero-based array index of this variable: its number minus one.
    pub fn idx(self) -> usize {
        debug_assert!(self.0 >= 1, "variable numbers start at 1");
        (self.0 - 1) as usize
    }
}

/// Why a vtree could not be built, parsed, or checked.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VtreeError {
    /// `.vtree` text that does not describe a single tree.
    Text(String),
    /// Two of the trees being combined both carry this variable.
    OverlappingVariable(VarId),
    /// A structural invariant that does not hold (see [`Vtree::validate`](crate::vtree::Vtree::validate)),
    /// or a construction handed nothing to build from.
    Invalid(String),
    /// The variable-id space is wider than the node list can justify. The table
    /// mapping ids to leaves is indexed by variable id, so a single very large
    /// id sizes it however few leaves the tree has: two lines of `.vtree` text
    /// naming variable 4_000_000_000 would otherwise ask for 16 GB. Sparse ids
    /// stay legal; the bound grows with the node list, so a tree that really
    /// does carry millions of variables is unaffected.
    VariableSpaceTooLarge {
        /// The requested inclusive upper bound on variable IDs.
        num_vars: u32,
        /// The widest id space this node list may declare.
        max_num_vars: u32,
    },
}

impl std::fmt::Display for VtreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VtreeError::Text(msg) => write!(f, "malformed vtree text: {msg}"),
            VtreeError::OverlappingVariable(var) => write!(
                f,
                "variable {} is carried by more than one of the trees being combined",
                var.0
            ),
            VtreeError::Invalid(msg) => write!(f, "invalid vtree: {msg}"),
            VtreeError::VariableSpaceTooLarge { num_vars, max_num_vars } => write!(
                f,
                "variable-id space of {num_vars} exceeds the {max_num_vars} this node list allows",
            ),
        }
    }
}

impl std::error::Error for VtreeError {}

#[cfg(test)]
mod tests;
