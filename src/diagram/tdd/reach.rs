//! Which nodes of a diagram are reachable.

use crate::diagram::ZERO;
use crate::vtree::VtreeIdx;

use super::Tdd;

impl Tdd {
    /// Allocate an all-false `[vtree_idx][local_idx]` reachability matrix sized to
    /// each level's effective width.
    fn empty_reach_matrix(&self) -> Vec<Vec<bool>> {
        (0..self.vtree.num_nodes())
            .map(|i| vec![false; self.effective_width(VtreeIdx(i as u32))])
            .collect()
    }

    /// Top-down reachability propagation over a pre-seeded root set. Every root
    /// node must already be marked `true` in `reachable`; on return every node
    /// reachable from those roots is marked. Single source of truth for the
    /// traversal shared by [`reachable_nodes`] (output-seeded) and
    /// [`reachable_from_root_level`] (root-level-seeded).
    fn propagate_reachability(&self, reachable: &mut [Vec<bool>]) {
        for (t, left_vtree, right_vtree) in self.vtree.internal_bottomup().rev() {
            // Marg-side refs are bit-30-tagged slot indices (or, post-Phase-B,
            // inline counts). Decode before indexing the child reachability
            // vector: a slot ref masks to its bare index; an inline-count ref
            // has no child node, so it marks nothing.
            let left_view = self.levels[left_vtree.idx()].side_view();
            let right_view = self.levels[right_vtree.idx()].side_view();
            let level = self.level(t);
            for (i, node) in level.nodes.iter().enumerate() {
                if !reachable[t.idx()][i] {
                    continue;
                }
                for pair in level.pairs_of(node) {
                    if pair.left != ZERO
                        && let Some(s) = left_view.child(pair.left).index() {
                            reachable[left_vtree.idx()][s] = true;
                        }
                    if pair.right != ZERO
                        && let Some(s) = right_view.child(pair.right).index() {
                            reachable[right_vtree.idx()][s] = true;
                        }
                }
            }
        }
    }

    /// Which nodes `output` reaches, as `[vtree index][local index]` over
    /// `effective_width`; all false for ⊥. A minimized diagram reaches every
    /// stored node.
    pub fn reachable_nodes(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        if self.is_zero() {
            return reachable;
        }
        reachable[self.output.vtree.idx()][self.output.local.idx()] = true;
        self.propagate_reachability(&mut reachable);
        reachable
    }

    /// Reachability seeded from every node at the vtree root level, not just the
    /// single `output`. The ray classification runs mid-compile, where the root
    /// level can hold several live candidate nodes that are not yet joined into
    /// one output; seeding only from `output` would then mis-classify those as
    /// dead. Shares `propagate_reachability` with [`reachable_nodes`](Self::reachable_nodes). For a
    /// `ZERO` (UNSAT) diagram the root level is empty, so the result is all-false.
    #[cfg(test)]
    pub(crate) fn reachable_from_root_level(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        for slot in reachable[self.vtree.root().idx()].iter_mut() {
            *slot = true;
        }
        self.propagate_reachability(&mut reachable);
        reachable
    }
}
