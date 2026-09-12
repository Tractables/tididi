//! The from-scratch order the rotation tests check the localized fixup
//! against.

use super::*;

impl TopoOrder {
    /// Recompute everything from the tree by an iterative postorder,
    /// `O(num_nodes)` — the oracle the rotation tests check the localized
    /// fixup against.
    pub(in crate::vtree) fn rebuild(&mut self, nodes: &[VtreeNode], root: VtreeIdx) {
        let n = nodes.len();
        self.order.clear();
        self.order.reserve(n);
        self.internal.clear();
        self.leaves.clear();
        if self.pos.len() != n {
            self.pos.resize(n, 0);
        }

        // Iterative postorder via explicit stack: push (idx, visited_yet).
        // First pop pushes children; second pop emits the node.
        let mut stack: Vec<(VtreeIdx, bool)> = Vec::with_capacity(n);
        stack.push((root, false));
        while let Some((idx, done)) = stack.pop() {
            if done {
                self.pos[idx.idx()] = self.order.len() as u32;
                self.order.push(idx);
                if nodes[idx.idx()].is_leaf() {
                    self.leaves.push(idx);
                } else {
                    self.internal.push(idx);
                }
            } else {
                stack.push((idx, true));
                if let VtreeNode::Internal { left, right, .. } = nodes[idx.idx()] {
                    // Push right first so left is popped first → left visited
                    // before right (the bottom-up order used elsewhere).
                    stack.push((right, false));
                    stack.push((left, false));
                }
            }
        }
        debug_assert_eq!(self.order.len(), n, "bottom-up order missed nodes (disconnected vtree?)");
    }
}
