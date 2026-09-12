//! The worklist seeds and the wide reachability the reduction tests drive a
//! diagram with.

use super::*;

impl Tdd {
    /// Reachability seeded from every node at the vtree root level, not just
    /// `output`, for a diagram whose root level still holds several live
    /// candidates. All false for ⊥, whose root level is empty.
    pub(crate) fn reachable_from_root_level(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        for slot in reachable[self.vtree.root().idx()].iter_mut() {
            *slot = true;
        }
        self.propagate_reachability(&mut reachable);
        reachable
    }

    /// The twin-contraction worklist, for a test that asserts on what a rewrite
    /// seeded.
    pub(crate) fn contract_worklist(&self) -> &[u32] {
        &self.dirty.contract
    }

    /// Drive the twin-contraction worklist directly, for a test that wants a
    /// sweep to start from exactly `levels`.
    pub(crate) fn seed_contract_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.contract.clear();
        self.dirty.contract.extend(levels);
    }

    /// The same for the leaf-contraction worklist.
    pub(crate) fn seed_leaf_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.leaf_contract.clear();
        self.dirty.leaf_contract.extend(levels);
    }
}
