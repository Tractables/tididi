//! Restricting a vtree to a subset of its variables.

use super::build::{push_internal, push_leaf};
use super::{VarId, Vtree, VtreeIdx, VtreeNode};

impl Vtree {
    /// Restrict this vtree to a subset of its variables, renumbering them.
    ///
    /// `local_of(v)` returns `Some(local_var)` for a variable to keep (with its
    /// id in the projected vtree's `0..num_local` space) and `None` for one to
    /// drop. Kept leaves survive verbatim; an internal node whose subtree keeps
    /// variables on only one side is spliced out (replaced by that side), and
    /// one that keeps nothing disappears, so the result keeps the original
    /// vtree's variable grouping and execution context. O(nodes of `self`).
    ///
    /// `num_local` is the result's id space (`num_vars()`). When the ids
    /// `local_of` yields are exactly `0..num_local`, the result has
    /// `num_leaves() == num_local`; ids it skips are uncovered.
    ///
    /// Returns `None` when `num_local` is zero or `local_of` keeps no variable
    /// at all (there is no such thing as an empty vtree).
    ///
    /// # Panics
    ///
    /// Panics if `local_of` yields an id greater than `num_local`, or the same
    /// id for two variables.
    pub fn project_to_vars<F>(&self, local_of: F, num_local: u32) -> Option<Vtree>
    where
        F: Fn(VarId) -> Option<VarId>,
    {
        if num_local == 0 {
            return None;
        }
        // `new_of[old.idx()]` = the surviving node that old node `old` maps to,
        // in the fresh (pre-reindex) node array. A spliced-out internal node maps
        // to its single surviving child, so parents see one contiguous skeleton.
        let mut new_of: Vec<Option<VtreeIdx>> = vec![None; self.nodes.len()];
        let mut nodes: Vec<VtreeNode> = Vec::with_capacity(2 * num_local as usize);

        // `bottomup()` walks the side `topo` array, so children are always
        // visited before their parent even after rotations.
        for t in self.bottomup() {
            match &self.nodes[t.idx()] {
                VtreeNode::Leaf { var, .. } => {
                    if let Some(local) = local_of(*var) {
                        new_of[t.idx()] = Some(push_leaf(&mut nodes, local));
                    }
                }
                VtreeNode::Internal { left, right, .. } => {
                    new_of[t.idx()] = match (new_of[left.idx()], new_of[right.idx()]) {
                        (Some(l), Some(r)) => Some(push_internal(&mut nodes, l, r)),
                        // Exactly one side survives: splice this node out.
                        (Some(x), None) | (None, Some(x)) => Some(x),
                        (None, None) => None,
                    };
                }
            }
        }

        let root = new_of[self.root.idx()]?;
        Some(Self::from_nodes(nodes, root, num_local).expect("the surviving skeleton is one tree")
            .with_context(std::sync::Arc::clone(&self.context)))
    }
}
