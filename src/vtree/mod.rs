//! The variable tree, its orders, its text format, and graft and projection of
//! the tree itself.
//!
//! A vtree is a rooted binary tree whose leaves carry Boolean variables. Each
//! internal node partitions the variables of its subtree, and that partition is
//! what a diagram decomposes over: the nodes at level `t` denote sub-functions
//! `f(vars(t_L), vars(t_R))` as disjunctions of pairs. Diagram storage is
//! [`crate::diagram`]; rotating a compiled diagram to follow a rotated tree is
//! [`crate::restructure`].
//!
//! Entry points:
//!
//! - Constructors: [`Vtree::leaf`], [`Vtree::join`], [`Vtree::balanced`],
//!   [`Vtree::balanced_over`], [`Vtree::linear`], [`Vtree::linear_from_order`],
//!   [`Vtree::random`], [`Vtree::graft`], [`Vtree::project_to_vars`].
//! - Text: [`Vtree::from_text`] and [`Vtree::to_text`], the `.vtree`
//!   interchange format — a vtree written by another tool loads here unchanged.
//! - Reading the tree: [`Vtree::root`], [`Vtree::node`], [`Vtree::children`],
//!   [`Vtree::leaf_of`], [`Vtree::lca`], [`Vtree::sibling`].
//! - Traversal orders: [`Vtree::bottomup`], [`Vtree::leaf_bottomup`],
//!   [`Vtree::internal_bottomup`].
//! - Checking a hand-built tree: [`Vtree::validate`].
//!
//! ## Variable ids
//!
//! A vtree may cover a sparse subset of variable ids. [`Vtree::num_vars`] is
//! the id-space size (at least `max VarId + 1`); [`Vtree::num_leaves`] is the number of ids
//! the tree actually carries; [`Vtree::leaf_of`] is defined only for covered
//! ids. The two counts agree exactly when the leaves are `0..num_vars`.
//!
//! ## Node layout
//!
//! Nodes are stored in bottom-up level order: all leaves first (`0..num_leaves`),
//! then internal nodes (`num_leaves..n`). At construction that makes
//! `child.idx() < parent.idx()` hold for every edge — but it is a fact about a
//! freshly built tree, not an invariant: a rotation relinks nodes without
//! moving them, so on a rotated tree an edge may run the other way and the
//! array order is no longer topological. [`Vtree::bottomup`] is the authority:
//! Every traversal reads that order rather than `0..n`.


mod build;
pub(crate) mod graft;
mod node;
mod project;
pub(crate) mod rng;
mod topo;
mod validate;

pub use graft::GraftLayout;
pub use node::{Vtree, VtreeNode};
pub(crate) use topo::RotationKind;


/// The `.vtree` text codec, in both directions.
mod text;

pub(crate) mod rotate; // In-place vtree left/right rotations + topo fixup


/// A zero-based variable identifier, independent of its position in the vtree.
///
/// `VarId(0)` is the variable named by integer literals `1` and `-1`.
/// Resolve its leaf with [`Vtree::leaf_of`]; a variable id
/// and a [`VtreeIdx`] are different index spaces.
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
}

impl std::fmt::Display for VtreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VtreeError::Text(msg) => write!(f, "malformed vtree text: {msg}"),
            VtreeError::OverlappingVariable(var) => write!(
                f,
                "variable {} is carried by more than one of the trees being combined",
                var.0 + 1
            ),
            VtreeError::Invalid(msg) => write!(f, "invalid vtree: {msg}"),
        }
    }
}

impl std::error::Error for VtreeError {}

#[cfg(test)]
mod tests;
