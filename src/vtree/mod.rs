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
//! [`Vtree::linear_from_order`], [`Vtree::graft`], [`Vtree::project_to_vars`]).
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
mod graft;
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

/// A literal: a variable with a polarity.
///
/// Lives lib-side (alongside `VarId`) so the pure-TDD layer (`clause_to_tdd`,
/// `apply_and_clause`) can accept `&[Literal]` slices without depending on the
/// CNF module. The CNF `Clause`/`CnfFormula` types build on it and re-export it.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Literal {
    /// The variable this literal refers to.
    pub var: VarId,
    /// `true` for a positive literal, `false` for a negated one.
    pub positive: bool,
}

impl Literal {
    /// Construct a literal over `var` with the given polarity.
    pub fn new(var: VarId, positive: bool) -> Self {
        Literal { var, positive }
    }

    /// The positive literal over `var`.
    pub fn pos(var: VarId) -> Self {
        Literal::new(var, true)
    }

    /// The negated literal over `var`.
    pub fn neg(var: VarId) -> Self {
        Literal::new(var, false)
    }

    /// This literal with its polarity flipped.
    #[must_use]
    pub fn negated(self) -> Self {
        Literal {
            var: self.var,
            positive: !self.positive,
        }
    }
}

/// Build a `Literal` from a signed **DIMACS** integer.
///
/// DIMACS variables are 1-based: `1` is the first variable (`VarId(0)`), `2` the
/// second, and so on; a negative value denotes a negated literal. The magnitude
/// is decremented to the 0-based [`VarId`] used internally — the same convention
/// as the CNF parser (`VarId(val.unsigned_abs() - 1)`).
///
/// # Panics
/// Panics on `0`, which is not a valid DIMACS literal (in the DIMACS format `0`
/// terminates a clause rather than naming a variable).
///
/// ```
/// use tididi::vtree::{Literal, VarId};
/// assert_eq!(Literal::from(1), Literal::pos(VarId(0)));
/// assert_eq!(Literal::from(-2), Literal::neg(VarId(1)));
/// ```
impl From<i32> for Literal {
    fn from(n: i32) -> Self {
        assert!(
            n != 0,
            "0 is not a DIMACS literal (it terminates a clause, not a variable)"
        );
        let var = VarId(n.unsigned_abs() - 1);
        if n > 0 {
            Literal::pos(var)
        } else {
            Literal::neg(var)
        }
    }
}

/// The `.vtree` text codec, in both directions.
mod text;

pub mod rotate; // In-place vtree left/right rotations + topo fixup

#[cfg(test)]
#[path = "vtree_tests.rs"]
mod tests;
