//! The bottom-up topological order beside the node list, and its upkeep.

use super::rotate;
use super::{VarId, Vtree, VtreeIdx, VtreeNode};

/// Which rotation direction a `fixup_topo_after_rotate` call corresponds to —
/// selects which grandchild subtree may violate children-before-parents after
/// the rotation.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum RotationKind {
    /// A left rotation.
    Left,
    /// A right rotation.
    Right,
}

impl Vtree {
    /// Every node once, children before parents — the order a bottom-up pass
    /// over the tree must visit them in. Reverse it (the iterator is
    /// double-ended) for a top-down pass. This is the maintained topological
    /// order, not `0..num_nodes`: after a rotation the two disagree, and only
    /// this one is still topological.
    pub fn bottomup(&self) -> impl DoubleEndedIterator<Item = VtreeIdx> + ExactSizeIterator + '_ {
        self.topo.iter().copied()
    }

    /// Bottom-up traversal of leaf nodes only, yielding each leaf node index
    /// with its variable.
    /// Walks the cached `leaf_topo` slice; preserves topological order of leaves.
    pub fn leaf_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VarId)> + ExactSizeIterator + '_ {
        self.leaf_topo.iter().map(move |&t| {
            let var = match self.node(t) {
                VtreeNode::Leaf { var, .. } => *var,
                _ => unreachable!("leaf_topo entry is not a leaf"),
            };
            (t, var)
        })
    }

    /// Bottom-up traversal of internal nodes only, yielding each node index with
    /// its left and right child indices.
    /// Walks the cached `internal_topo` slice; remains valid after rotations.
    pub fn internal_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + ExactSizeIterator + '_
    {
        self.internal_topo.iter().map(move |&t| match self.node(t) {
            VtreeNode::Internal { left, right, .. } => (t, *left, *right),
            _ => unreachable!("internal_topo entry is not internal"),
        })
    }

    /// The internal nodes in bottom-up order as a slice — the same sequence
    /// [`Vtree::internal_bottomup`] yields, for a caller that indexes into it.
    pub(crate) fn internal_topo_slice(&self) -> &[VtreeIdx] {
        &self.internal_topo
    }

    /// Recompute `topo`, `topo_pos`, `internal_topo`, `leaf_topo` from the
    /// current parent/child structure by a full `O(n_nodes)` iterative
    /// postorder — the oracle the rotation tests check the localized
    /// `fixup_topo_after_rotate` against.
    #[cfg(test)]
    pub(crate) fn rebuild_topo(&mut self) {
        let n = self.nodes.len();
        self.topo.clear();
        self.topo.reserve(n);
        self.internal_topo.clear();
        self.leaf_topo.clear();
        if self.topo_pos.len() != n {
            self.topo_pos.resize(n, 0);
        }

        // Iterative postorder via explicit stack: push (idx, visited_yet).
        // First pop pushes children; second pop emits the node.
        let mut stack: Vec<(VtreeIdx, bool)> = Vec::with_capacity(n);
        stack.push((self.root, false));
        while let Some((idx, done)) = stack.pop() {
            if done {
                self.topo_pos[idx.idx()] = self.topo.len() as u32;
                self.topo.push(idx);
                if self.nodes[idx.idx()].is_leaf() {
                    self.leaf_topo.push(idx);
                } else {
                    self.internal_topo.push(idx);
                }
            } else {
                stack.push((idx, true));
                if let VtreeNode::Internal { left, right, .. } = self.nodes[idx.idx()] {
                    // Push right first so left is popped first → left visited
                    // before right (matches the bottom-up order used elsewhere).
                    stack.push((right, false));
                    stack.push((left, false));
                }
            }
        }
        debug_assert_eq!(
            self.topo.len(),
            n,
            "topo missed nodes (disconnected vtree?)"
        );
    }

    /// Localized topo update after a single rotation. O(subtree) instead of
    /// `rebuild_topo`'s `O(n_nodes)`; used by the rotation search hot loop.
    /// See `vtree::rotate` module documentation for the proof.
    pub(crate) fn fixup_topo_after_rotate(&mut self, info: &rotate::RotationInfo, kind: RotationKind) {
        let w_pos = self.topo_pos[info.w_idx.idx()] as usize;
        // The single new children-before-parents constraint introduced by a
        // rotation:
        //   Left rotation  v=(A,w),w=(B,C) → v=(w,C),w=(A,B): need A < w.
        //   Right rotation v=(w,C),w=(A,B) → v=(A,w),w=(B,C): need C < w.
        let misplaced_root = match kind {
            RotationKind::Left => info.a_idx,
            RotationKind::Right => info.c_idx,
        };
        // By root-last, topo_pos[misplaced_root] is the maximum topo position
        // over the entire misplaced subtree. When it precedes w the existing
        // topo already satisfies the new constraint and only the filtered
        // views need refreshing.
        let m_end = self.topo_pos[misplaced_root.idx()] as usize;
        if m_end >= w_pos {
            debug_assert!(
                m_end > w_pos,
                "misplaced_root and w cannot share a topo position"
            );

            // Slice [w_pos ..= m_end] currently starts with w (at w_pos) and ends
            // with the misplaced subtree's root (at m_end). After rotate_left(1),
            // w sits at m_end (one past every element of the misplaced subtree
            // that lay in the slice), and elements in (w_pos..=m_end] shift one
            // position to the left. This is a single contiguous memmove.
            //
            // Subtree contiguity is NOT required: even if non-misplaced elements
            // lie in (w_pos..m_end), the slice rotation preserves children-before-
            // parents for every edge in the post-rotation tree. The full proof is
            // in the `vtree::rotate` module doc.
            self.topo[w_pos..=m_end].rotate_left(1);

            for (offset, &node) in self.topo[w_pos..=m_end].iter().enumerate() {
                self.topo_pos[node.idx()] = (w_pos + offset) as u32;
            }
        }
        self.refresh_filtered_topo();
    }

    /// Refilter `internal_topo` and `leaf_topo` from the current `topo`.
    /// `O(n_nodes)`; called by `fixup_topo_after_rotate` to keep the filtered
    /// views consistent without rebuilding the full topo array.
    pub(crate) fn refresh_filtered_topo(&mut self) {
        self.internal_topo.clear();
        self.leaf_topo.clear();
        for &t in &self.topo {
            if self.nodes[t.idx()].is_leaf() {
                self.leaf_topo.push(t);
            } else {
                self.internal_topo.push(t);
            }
        }
    }

    /// Bottom-up topological order over all nodes (children before parents) as
    /// a slice — the same sequence [`Vtree::bottomup`] yields.
    #[inline]
    pub(crate) fn bottomup_topo(&self) -> &[VtreeIdx] {
        &self.topo
    }

    /// Topological position of `idx` (0 = first in bottom-up order, root = last).
    /// A rank comparator: `topo_pos(a) < topo_pos(b)` whenever `a` is a proper
    /// descendant of `b`.
    #[inline]
    pub(crate) fn topo_pos(&self, idx: VtreeIdx) -> u32 {
        self.topo_pos[idx.idx()]
    }
}
