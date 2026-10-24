//! Splice one diagram's subtree into another on the same vtree.

use crate::Engine;
use crate::diagram::{return_levels, ChildPair, NodeKind, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

impl Tdd {
    /// Replace the subtree under `t`'s right child with `other`'s, and pair the
    /// two roots at `t`.
    ///
    /// Both diagrams use the same vtree. Move `other`'s right-subtree levels
    /// into `self`, then join their constrained roots with one pair at `t`.
    /// Weight stores are combined, and so is the reduction work either
    /// diagram still owes; unused levels return to the engine's pool.
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
        let sole_pair = |level: &TddLevel| match level.node(0).kind() {
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

        // The moved levels keep what `other`'s reductions still owe them, and
        // its pair fusion records; every other level but `t` keeps this
        // diagram's.
        let mut fused = self.dirty.take_fused();
        let theirs = other.dirty.take_fused();
        let bit = |words: &[u64], i: usize| words.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 == 1);
        for idx in self.vtree.subtree(right_child).chain([t]) {
            let i = idx.idx();
            let set = idx != t && bit(&theirs, i);
            if set && fused.len() <= i / 64 {
                fused.resize(i / 64 + 1, 0);
            }
            if let Some(word) = fused.get_mut(i / 64) {
                *word = *word & !(1 << (i % 64)) | u64::from(set) << (i % 64);
            }
        }
        self.dirty.merge_under(other.take_worklists());
        self.dirty.restore_fused(fused, &[]);
        return_levels(eng, std::mem::take(&mut other.levels).into_vec());

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

#[cfg(test)]
#[path = "tests/splice.rs"]
mod tests;
