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
    /// reachable from those roots is marked.
    fn propagate_reachability(&self, reachable: &mut [Vec<bool>]) {
        for (t, left_vtree, right_vtree) in self.vtree.internal_bottomup().rev() {
            // A marginal-side ref decodes to a slot index, or to an inline
            // count that names no child node and marks nothing.
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
}

// Test support.
impl Tdd {
    /// Reachability seeded from every node at the vtree root level, not just
    /// `output`, for a diagram whose root level still holds several live
    /// candidates. All false for ⊥, whose root level is empty.
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
