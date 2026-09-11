//! The entry points the one driver that builds a diagram clause by clause
//! reaches the crate through, none of which the documented modules expose.
//!
//! The module is hidden from the crate's documentation. Nothing in it is
//! covered by the compatibility promise.

use std::sync::Arc;

use crate::diagram::{BigSide, Tdd, TddBuilder, TddLevel, TddNodeId, WeightStore};
use crate::engine::Engine;
use crate::vtree::{Vtree, VtreeIdx};

mod schedule;

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;
pub use schedule::{intra_batch_completions, marginalize_schedule};

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

/// [`Tdd::take_weights`] without the level scan, for a caller carrying the
/// store across an operation that rebuilds the diagram and putting it back on
/// the result.
///
/// The public detach refuses while any level still reads its values out of the
/// store, because handing the store away strands that level. A caller here
/// takes on the obligation instead: the store must go back onto the diagram
/// that replaces this one before anything reads it.
pub fn take_weights_unchecked(f: &mut Tdd) -> Option<WeightStore> {
    f.detach_weights()
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
