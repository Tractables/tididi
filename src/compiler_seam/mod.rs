//! The entry points the one driver that builds a diagram clause by clause
//! reaches the crate through, none of which the documented modules expose.
//!
//! The module is hidden from the crate's documentation. Nothing in it is
//! covered by the compatibility promise.

use std::sync::Arc;

use crate::diagram::{Tdd, TddBuilder, TddNodeId};
use crate::engine::Engine;
use crate::vtree::{Vtree, VtreeIdx};


pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;

/// Seat `b`'s diagram on `output` without the invariant walk
/// [`TddBuilder::finish`] runs.
///
/// For the driver's inner loops, which rebuild whole diagrams often enough
/// that a second walk of each one is a cost with no reader: those callers
/// derive every pair they push from a diagram that already held the
/// invariants, so the walk can only confirm what the construction established.
/// Every other caller — and every caller assembling pairs from something other
/// than an existing diagram — uses [`TddBuilder::finish`] and reads its error.
///
/// # Panics
///
/// A debug build runs the walk and panics on a violation, so a caller that has
/// the invariant wrong finds out under test rather than in an answer.
pub fn finish_unchecked(b: TddBuilder, output: TddNodeId) -> Tdd {
    b.finish_unchecked(output)
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
