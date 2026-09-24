//! Splice one diagram's subtree into another on the same vtree.

use crate::Engine;
use crate::diagram::{return_levels, ChildPair, NodeKind, PoolSlot, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

impl Tdd {
    /// Replace the subtree under `t`'s right child with `other`'s, and pair the
    /// two roots at `t`.
    ///
    /// Both diagrams use the same vtree. Move `other`'s right-subtree levels
    /// into `self`, then join their constrained roots with one pair at `t`.
    /// Weight stores are combined; unused levels return to the engine's pool.
    ///
    /// # Safety
    ///
    /// Both operands must share the same vtree allocation and compatible weight
    /// configuration. The merge level `t` must be internal, with exactly one
    /// structural, single-pair node in each operand. The left operand must be
    /// independent of `t`'s right subtree and the right operand independent of
    /// its left subtree; their wrapper pairs must represent those free sides
    /// as true. Any levels above `t` retained from `self` must still reference
    /// the merged node correctly. The result must satisfy the storage and
    /// determinism invariants of [`TddBuilder`](crate::diagram::TddBuilder).
    /// Invalid references can cause out-of-bounds reads in later operations.
    ///
    /// # Panics
    ///
    /// If either diagram's level at `t` does not hold exactly one stored node.
    pub unsafe fn splice_subtree_unchecked(&mut self, eng: &Engine, mut other: Tdd, t: VtreeIdx) {
        assert_eq!(
            self.levels[t.idx()].slot_count(), 1,
            "splice_subtree: the left diagram has width {} at the merge point",
            self.levels[t.idx()].slot_count(),
        );
        assert_eq!(
            other.levels[t.idx()].slot_count(), 1,
            "splice_subtree: the right diagram has width {} at the merge point",
            other.levels[t.idx()].slot_count(),
        );
        let sole_pair = |level: &TddLevel| match level.nodes()[0].kind() {
            NodeKind::Inline(pair) => pair,
            other => panic!("splice_subtree: the merge point holds {other:?}"),
        };
        let left_ptr = sole_pair(&self.levels[t.idx()]).left;
        let right_ptr = sole_pair(&other.levels[t.idx()]).right;

        let right_child = self.vtree.children(t).1;
        for idx in self.vtree.subtree(right_child) {
            std::mem::swap(&mut self.levels[idx.idx()], &mut other.levels[idx.idx()]);
        }

        self.levels[t.idx()].clear();
        self.levels[t.idx()].push_internal_node(&[ChildPair::new(left_ptr, right_ptr)]);

        return_levels(eng, PoolSlot::First, std::mem::take(&mut other.levels).into_vec());

        if let Some(rw) = other.detach_weights() {
            match self.detach_weights() {
                Some(mut lw) => {
                    lw.absorb(rw);
                    self.weights = Some(lw);
                }
                None => self.weights = Some(rw),
            }
        }
    }
}
