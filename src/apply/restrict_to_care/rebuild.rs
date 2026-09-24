//! Dropping the dead pairs of the operand in place.

use std::sync::Arc;

use crate::Engine;
use crate::limits::OperationError;
use crate::reduce::ReductionPlan;
use crate::diagram::{ChildDecoder, EncodedChildRef, Tdd, TddLevel, ZERO};
use crate::vtree::VtreeIdx;
use crate::apply::falsity::{propagate_false_nodes, rewrite_level_pairs};

use super::Marking;

impl Marking {
    /// Drop every pair the marks call dead, then every pair naming a node left
    /// with no pairs, and prune what nothing references any more.
    ///
    /// The operand is rewritten in place: nodes keep their indices, marginal
    /// levels and weights are untouched, and no pair is added. The output
    /// becomes the `ZERO` sentinel when its node loses every pair.
    pub(super) fn rebuild(self, eng: &Engine, mut f: Tdd) -> Result<Tdd, OperationError> {
        let lim = eng.limits();
        lim.check_stop()?;
        let mut gate = lim.gate();
        let vtree = Arc::clone(&f.vtree);
        // A child reference names a dead node: `ZERO`, or a structural node
        // the marks did not keep. Marginal children and leaf labels never die.
        let dead = |cv: VtreeIdx, child: &TddLevel, side: EncodedChildRef| {
            if child.is_marginal() { return false; }
            let node = ChildDecoder::structural().node(side);
            node == ZERO || (!vtree.node(cv).is_leaf() && !self.alive[cv.idx()][node.idx()])
        };
        for (v, left, right) in vtree.internal_bottomup() {
            if f.levels[v.idx()].is_marginal() { continue; }
            let [level, left_level, right_level] = f.levels
                .get_disjoint_mut([v.idx(), left.idx(), right.idx()])
                .expect("a parent and its children are distinct levels");
            let before = level.live_pairs();
            gate.poll(before as u64)?;
            let masks = &self.pair_alive[v.idx()];
            rewrite_level_pairs(level, |i, k, pair| {
                // A node with more than 64 pairs carries no per-pair marks.
                let mask = masks[i];
                let masked_out = mask != u64::MAX && k < 64 && (mask >> k) & 1 == 0;
                if masked_out || dead(left, left_level, pair.left) || dead(right, right_level, pair.right) {
                    None
                } else {
                    Some(pair)
                }
            });
            if level.live_pairs() != before {
                f.try_invalidate(eng, v)?;
            }
        }
        gate.flush()?;
        propagate_false_nodes(&mut f);
        // The nodes the dropped pairs no longer name are unreachable now.
        eng.reduce(&mut f, ReductionPlan::Prune)?;
        Ok(f)
    }
}

#[cfg(test)]
#[path = "tests/rebuild.rs"]
mod tests;
