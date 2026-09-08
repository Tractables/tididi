//! Variable tree (vtree): the structural backbone of a TDD.
//!
//! A vtree is a rooted binary tree whose leaves correspond to Boolean variables.
//! It governs how a TDD decomposes its Boolean function: each internal vtree node
//! `t` with children `t_L, t_R` defines a partition of variables, and the TDD
//! nodes at level `t` represent sub-functions `f(vars(t_L), vars(t_R))` as
//! disjunctions of input pairs `(left_child, right_child)`.
//!
//! This module is deliberately a copy of vitri's `src/vtree/` module, kept in
//! sync by hand: same public names, same accessor style, same semantics, so a
//! vtree built by vitri's construction heuristics carries over as `.vtree` text
//! (or as a node relabel) without translation. The heuristics that decide which
//! vtree to build live in vitri; this module owns the *structure* — topology,
//! traversal order, LCA, rotation, text I/O — and the programmatic constructors
//! ([`Vtree::leaf`], [`Vtree::join`], [`Vtree::balanced_over`],
//! [`Vtree::linear_over`], [`Vtree::graft`], [`Vtree::project_to_vars`]).
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
//! then internal nodes (`num_leaves..n`). This guarantees `child.idx() < parent.idx()`
//! for every edge at construction. A rotation relinks nodes without moving them,
//! so on a rotated tree the array order is no longer topological and
//! [`Vtree::bottomup`] is the authority: every traversal reads that order
//! rather than `0..n`.


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
pub use topo::RotationKind;


/// The `.vtree` text codec, in both directions.
mod text;

pub mod rotate; // In-place vtree left/right rotations + topo fixup

#[cfg(test)]
#[path = "vtree_tests.rs"]
mod tests;
