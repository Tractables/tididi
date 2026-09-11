//! The variable tree, its orders, its text format, rotation and graft of the
//! tree itself.
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
//!   [`Vtree::balanced_over`], [`Vtree::linear`], [`Vtree::linear_over`],
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
//! the id space (`max VarId + 1`); [`Vtree::num_leaves`] is the number of ids
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
mod error;
pub(crate) mod graft;
mod ids;
mod node;
mod project;
mod topo;
mod validate;

pub use error::VtreeError;
pub use graft::GraftLayout;
pub use ids::{VarId, VtreeIdx};
pub use node::{Vtree, VtreeNode};
pub(crate) use topo::RotationKind;


/// The `.vtree` text codec, in both directions.
mod text;

pub(crate) mod rotate; // In-place vtree left/right rotations + topo fixup

pub use rotate::RotationInfo;

#[cfg(test)]
mod tests;
