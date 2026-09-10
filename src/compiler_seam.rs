//! Clause-spine marking, mid-compile clustering, and the write half of the
//! level encoding, for a driver that builds a diagram clause by clause.
//!
//! These are hooks, not operations: the operations they steer are
//! [`crate::apply`] and [`crate::restructure`], and a caller that is not that
//! driver wants those modules instead.
//!
//! The documented traversal contract is read-only — a diagram a caller holds
//! came out of an operation or out of [`crate::diagram::TddBuilder`], so every
//! invariant an operation assumes holds without checking. The entries below are
//! the exceptions a clause-by-clause driver needs: the spine a clause
//! conjunction is planned from, the clustering a rotation mid-compile drives, a
//! hand-built marginal level, and the two whole-diagram edits that move a
//! subtree or change the tree a diagram is seated on. Each of the last three can
//! break an invariant, and the caller owes what the item's own documentation
//! states.
//!
//! Nothing here is covered by the crate's compatibility promise, none of it is
//! reachable through the documented modules, and all of it is hidden from the
//! documented API.

use std::sync::Arc;

use crate::diagram::{BigSide, Tdd, TddLevel};
use crate::engine::Engine;
use crate::vtree::{Vtree, VtreeIdx};

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;

/// A level holding per-node counts: `counts[i]` for node `i`, with `u128::MAX`
/// marking an overflow whose exact value is `big.get(i)`.
///
/// The level is ready to be handed to
/// [`TddBuilder::copy_level`](crate::diagram::TddBuilder::copy_level).
/// Marginality is downward-closed, so both of the level's children must
/// themselves be marginal or leaves in the diagram it is placed in; nothing
/// here checks that.
pub fn marginal_level(counts: Vec<u128>, big: Option<BigSide>) -> TddLevel {
    let mut level = TddLevel::new();
    level.become_marginal(counts, big);
    level
}

/// Move `other`'s levels below `t` into `f`, joining the two diagrams at the
/// vtree node `t`.
///
/// # Panics
///
/// If either diagram's level at `t` does not hold exactly one stored node.
pub fn splice_subtree(eng: &Engine, f: &mut Tdd, other: Tdd, t: VtreeIdx) {
    f.splice_subtree(eng, other, t);
}

/// Seat `f` on `vtree`, which must have the same number of nodes as the tree it
/// replaces so that every level stays addressable.
///
/// The two trees need not have the same shape: a rotation changes the shape
/// while leaving the nodes each in-flight diagram describes untouched, and
/// reseating those diagrams on the rotated tree is how a mid-compile rotation
/// reaches them. The caller owes the rest.
pub fn reseat_vtree(f: &mut Tdd, vtree: &Arc<Vtree>) {
    f.reseat_vtree(vtree);
}
