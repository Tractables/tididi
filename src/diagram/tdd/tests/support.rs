//! The wide reachability the reduction tests drive a diagram with.

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
}
