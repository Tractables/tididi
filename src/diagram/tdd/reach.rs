//! Which nodes of a diagram are reachable.

use crate::diagram::ZERO;
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

use super::Tdd;

impl Tdd {
    /// Allocate an all-false `[vtree_idx][local_idx]` reachability matrix sized to
    /// each level's effective width.
    pub(super) fn empty_reach_matrix(&self) -> Vec<Vec<bool>> {
        (0..self.vtree.num_nodes())
            .map(|i| vec![false; self.reference_slot_count(VtreeIdx(i as u32))])
            .collect()
    }

    /// An all-`fill` table with one entry per reference slot of every level.
    pub(crate) fn slot_table<T: Clone>(&self, lim: &Limits, fill: T) -> Result<Vec<Vec<T>>, OperationError> {
        let mut rows = Vec::new();
        lim.reserve_exact(&mut rows, self.levels.len())?;
        for i in 0..self.levels.len() {
            let mut row = Vec::new();
            lim.try_resize(&mut row, self.reference_slot_count(VtreeIdx(i as u32)), fill.clone())?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// Top-down reachability propagation over a pre-seeded root set. Every root
    /// node must already be marked `true` in `reachable`; on return every node
    /// reachable from those roots is marked.
    pub(crate) fn propagate_reachability(&self, reachable: &mut [Vec<bool>]) {
        for (t, left_vtree, right_vtree) in self.vtree.internal_bottomup().rev() {
            // A marginal-side ref decodes to a slot index, or to an inline
            // count that names no child node and marks nothing.
            let left_view = self.levels[left_vtree.idx()].child_decoder();
            let right_view = self.levels[right_vtree.idx()].child_decoder();
            let level = self.level(t);
            for (i, node) in level.nodes().iter().enumerate() {
                if !reachable[t.idx()][i] {
                    continue;
                }
                for pair in level.pairs_iter_of(&node) {
                    if pair.left != ZERO.into()
                        && let Some(s) = left_view.child(pair.left).index() {
                            reachable[left_vtree.idx()][s] = true;
                        }
                    if pair.right != ZERO.into()
                        && let Some(s) = right_view.child(pair.right).index() {
                            reachable[right_vtree.idx()][s] = true;
                        }
                }
            }
        }
    }

    /// Which nodes `output` reaches, as `[vtree index][local index]` over
    /// `reference_slot_count`; all false for ⊥. A minimized diagram reaches every
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
