//! The from-scratch order the rotation tests check the localized fixup
//! against.

use super::*;

impl TopoOrder {
    /// Recompute everything from the tree by an iterative postorder,
    /// `O(num_nodes)`, listing the variable leaves first, then the regions,
    /// then the internal nodes: the oracle the rotation tests check the
    /// localized fixup against.
    pub(in crate::vtree) fn rebuild(&mut self, nodes: &[VtreeNode], root: VtreeIdx) {
        let n = nodes.len();
        let (mut leaves, mut regions, mut internal) = (Vec::new(), Vec::new(), Vec::new());
        // Iterative postorder via explicit stack: push (idx, visited_yet).
        // First pop pushes children; second pop emits the node.
        let mut stack: Vec<(VtreeIdx, bool)> = Vec::with_capacity(n);
        stack.push((root, false));
        while let Some((idx, done)) = stack.pop() {
            if done {
                let node = &nodes[idx.idx()];
                match (node.is_leaf(), node.is_region()) {
                    (true, false) => leaves.push(idx),
                    (true, true) => regions.push(idx),
                    _ => internal.push(idx),
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
        self.order = [leaves, regions, internal].concat();
        debug_assert_eq!(self.order.len(), n, "bottom-up order missed nodes (disconnected vtree?)");
        self.pos.resize(n, 0);
        for (pos, &t) in self.order.iter().enumerate() {
            self.pos[t.idx()] = pos as u32;
        }
    }
}
