//! Dropping the dead pairs of the operand in place.

use std::sync::Arc;

use crate::Engine;
use crate::limits::OperationError;
use crate::reduce::ReductionPlan;
use crate::diagram::{ChildDecoder, EncodedChildRef, NodeIdx, Tdd, TddLevel, ZERO};
use crate::vtree::VtreeIdx;
use crate::apply::falsity::{propagate_false_nodes, rewrite_live_level_pairs};

use super::Marking;
use super::pairs::PairRow;

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
        let keep = |v: VtreeIdx, a: usize| self.keep.get(v.idx()).is_some_and(|row| row[a]);
        let dead = |cv: VtreeIdx, child: &TddLevel, side: EncodedChildRef| {
            if child.is_marginal() { return false; }
            let node = ChildDecoder::structural().node(side);
            node == ZERO || (!vtree.node(cv).is_leaf() && !self.alive[cv.idx()][node.idx()])
        };
        // Whether a node the marks keep alive lost every pair. A pair that
        // stays names live nodes only, so where none did, every node left
        // empty is one no pair names any more and the falsity sweep would
        // find nothing to drop. The walk's marks never leave one (a live
        // product keeps a live pair, whose children are live products).
        let mut lost = false;
        for (v, left, right) in vtree.internal_bottomup() {
            if f.levels[v.idx()].is_marginal() { continue; }
            let [level, left_level, right_level] = f.levels
                .get_disjoint_mut([v.idx(), left.idx(), right.idx()])
                .expect("a parent and its children are distinct levels");
            let before = level.live_pairs();
            gate.poll(before as u64)?;
            let marks = &self.pair_alive;
            // The node whose marks `row` holds: one look-up a node, not a pair.
            let mut row: Option<(usize, bool, PairRow<'_>)> = None;
            let alive = &self.alive[v.idx()];
            let mut kept_one = false;
            // A node the marks did not keep alive has no kept or marked pair:
            // it is emptied unread, and the prune drops it.
            rewrite_live_level_pairs(lim, level, |i| alive[i], |i, k, count, pair| {
                let (kept, marked) = match row {
                    Some((at, kept, marked)) if at == i => (kept, marked),
                    _ => {
                        let entry = (i, keep(v, i), marks.row(v, NodeIdx(i as u32), count));
                        row = Some(entry);
                        (entry.1, entry.2)
                    }
                };
                let out = if !(kept || marked.contains(k))
                    || dead(left, left_level, pair.left) || dead(right, right_level, pair.right) {
                    None
                } else {
                    Some(pair)
                };
                kept_one = (k > 0 && kept_one) || out.is_some();
                if k + 1 == count && !kept_one && alive[i] {
                    lost = true;
                }
                out
            })?;
            if level.live_pairs() != before {
                f.try_invalidate(eng, v)?;
            }
        }
        gate.flush()?;
        if lost {
            propagate_false_nodes(lim, &mut f)?;
        }
        // The nodes the dropped pairs no longer name are unreachable now.
        eng.reduce(&mut f, ReductionPlan::Prune)?;
        Ok(f)
    }
}

#[cfg(test)]
#[path = "tests/rebuild.rs"]
mod tests;
