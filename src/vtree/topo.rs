//! The bottom-up topological order beside the node list, and its upkeep.

use super::rotate;
use super::{VarId, Vtree, VtreeIdx, VtreeNode};

/// Which rotation direction a fixup call corresponds to — selects which
/// grandchild subtree may violate children-before-parents after the rotation.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum RotationKind {
    /// A left rotation.
    Left,
    /// A right rotation.
    Right,
}

/// The maintained bottom-up order of a vtree's nodes, with its inverse and the
/// two filtered views.
///
/// The node list itself is only topological at construction — a rotation
/// relinks nodes without moving them — so this, not `0..num_nodes`, is what
/// every bottom-up traversal reads. It is derived state:
/// [`fixup_after_rotate`](Self::fixup_after_rotate),
/// [`refresh_filtered`](Self::refresh_filtered) and the test-only `rebuild`
/// below are its only mutators, and each takes the node list to re-derive
/// from, so the order can never be edited into disagreement with the tree.
///
/// **Root-last**: for every node `t`, `pos(t)` is the *maximum* of `pos(d)`
/// over `d ∈ {t} ∪ descendants(t)` — each subtree's root sits at the latest
/// position among its members. `rebuild`'s strict postorder establishes it and
/// `fixup_after_rotate` preserves it; the fixup's O(1) case test depends on it.
///
/// **Subtree contiguity is not guaranteed**: after a sequence of rotations a
/// subtree's members may occupy a non-contiguous set of positions. Enumerate a
/// subtree by walking links, never by slicing a position range.
#[derive(Clone, Debug, Default)]
pub(super) struct TopoOrder {
    /// Every node once, children before parents.
    order: Vec<VtreeIdx>,
    /// Inverse of `order`: the position of each node. Inherits root-last.
    pos: Vec<u32>,
    /// `order` filtered to internal nodes.
    internal: Vec<VtreeIdx>,
    /// `order` filtered to leaves.
    leaves: Vec<VtreeIdx>,
}

impl TopoOrder {
    /// The bottom-up order over every node.
    #[inline]
    pub(super) fn all(&self) -> &[VtreeIdx] { &self.order }

    /// The internal nodes, in bottom-up order.
    #[inline]
    pub(super) fn internal(&self) -> &[VtreeIdx] { &self.internal }

    /// The leaves, in bottom-up order.
    #[inline]
    pub(super) fn leaves(&self) -> &[VtreeIdx] { &self.leaves }

    /// The position of `idx` in the bottom-up order (0 first, root last).
    #[inline]
    pub(super) fn pos(&self, idx: VtreeIdx) -> u32 { self.pos[idx.idx()] }

    /// The order as it stands for a freshly built tree whose node list is
    /// already in bottom-up layout: position is index.
    pub(super) fn identity(nodes: &[VtreeNode]) -> Self {
        let mut o = TopoOrder {
            order: (0..nodes.len() as u32).map(VtreeIdx).collect(),
            pos: (0..nodes.len() as u32).collect(),
            internal: Vec::new(),
            leaves: Vec::new(),
        };
        o.refresh_filtered(nodes);
        o
    }

    /// Localized repair after a single rotation: `O(subtree)` where
    /// `rebuild` is `O(num_nodes)`, which is what makes the
    /// rotation search loop affordable. See the `vtree::rotate` module
    /// documentation for the proof.
    pub(super) fn fixup_after_rotate(
        &mut self,
        nodes: &[VtreeNode],
        info: &rotate::RotationInfo,
        kind: RotationKind,
    ) {
        let w_pos = self.pos[info.w_idx.idx()] as usize;
        // The single new children-before-parents constraint a rotation introduces:
        //   Left rotation  v=(A,w),w=(B,C) → v=(w,C),w=(A,B): need A < w.
        //   Right rotation v=(w,C),w=(A,B) → v=(A,w),w=(B,C): need C < w.
        let misplaced_root = match kind {
            RotationKind::Left => info.a_idx,
            RotationKind::Right => info.c_idx,
        };
        // By root-last, the misplaced root's position is the maximum over its
        // whole subtree. When it precedes w the order already satisfies the new
        // constraint and only the filtered views need refreshing.
        let m_end = self.pos[misplaced_root.idx()] as usize;
        if m_end >= w_pos {
            debug_assert!(m_end > w_pos, "misplaced_root and w cannot share a position");

            // The slice [w_pos ..= m_end] starts with w and ends with the
            // misplaced subtree's root. After rotate_left(1), w sits at m_end
            // (one past every element of that subtree lying in the slice) and
            // everything in (w_pos..=m_end] shifts one position left — a single
            // contiguous memmove.
            //
            // Subtree contiguity is not required: even with non-misplaced
            // elements in (w_pos..m_end), the shift preserves
            // children-before-parents for every edge of the post-rotation tree.
            // The full proof is in the `vtree::rotate` module doc.
            self.order[w_pos..=m_end].rotate_left(1);
            for (offset, &node) in self.order[w_pos..=m_end].iter().enumerate() {
                self.pos[node.idx()] = (w_pos + offset) as u32;
            }
        }
        self.refresh_filtered(nodes);
    }

    /// Refilter the internal and leaf views from the current order,
    /// `O(num_nodes)` — what a rotation fixup owes after moving a node between
    /// positions without changing the full order's membership.
    pub(super) fn refresh_filtered(&mut self, nodes: &[VtreeNode]) {
        self.internal.clear();
        self.leaves.clear();
        for &t in &self.order {
            if nodes[t.idx()].is_leaf() {
                self.leaves.push(t);
            } else {
                self.internal.push(t);
            }
        }
    }

    /// Whether the order covers exactly `n` nodes with a consistent inverse.
    #[inline]
    pub(super) fn covers(&self, n: usize) -> bool {
        self.order.len() == n && self.pos.len() == n
    }
}

/// A set of vtree levels in bottom-up order, minted by
/// [`Vtree::bottom_up_subset`].
///
/// A bottom-up pass over a subset of levels is only well defined if every
/// level is visited after its children — a caller-supplied `&[VtreeIdx]` can
/// silently violate that and read a stale child column. The order is
/// established once, by the tree that owns it, and the type carries the proof
/// from there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BottomUpSubset(Vec<VtreeIdx>);

impl BottomUpSubset {
    /// The levels, children before parents.
    #[must_use]
    pub(crate) fn levels(&self) -> &[VtreeIdx] {
        &self.0
    }
}

impl Vtree {
    /// Order `levels` children-before-parents, dropping repeats.
    ///
    /// The one way to build a [`BottomUpSubset`]: the caller names the levels
    /// it cares about in any order, and the tree — which owns the topological
    /// order — puts them in one a bottom-up pass may follow.
    #[must_use]
    pub(crate) fn bottom_up_subset(&self, levels: impl IntoIterator<Item = VtreeIdx>) -> BottomUpSubset {
        let mut v: Vec<VtreeIdx> = levels.into_iter().collect();
        v.sort_unstable_by_key(|&t| self.topo.pos(t));
        v.dedup();
        BottomUpSubset(v)
    }
}

impl Vtree {
    /// Every node once, children before parents — the order a bottom-up pass
    /// over the tree must visit them in. Reverse it (the iterator is
    /// double-ended) for a top-down pass. This is the maintained topological
    /// order, not `0..num_nodes`: after a rotation the two disagree, and only
    /// this one is still topological.
    pub fn bottomup(&self) -> impl DoubleEndedIterator<Item = VtreeIdx> + ExactSizeIterator + '_ {
        self.topo.all().iter().copied()
    }

    /// Bottom-up traversal of the leaves, each with its variable.
    pub fn leaf_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VarId)> + ExactSizeIterator + '_ {
        self.topo.leaves().iter().map(move |&t| {
            let var = match self.node(t) {
                VtreeNode::Leaf { var, .. } => *var,
                _ => unreachable!("a leaf view entry is not a leaf"),
            };
            (t, var)
        })
    }

    /// Bottom-up traversal of the internal nodes, each with its two children.
    pub fn internal_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + ExactSizeIterator + '_
    {
        self.topo.internal().iter().map(move |&t| match self.node(t) {
            VtreeNode::Internal { left, right, .. } => (t, *left, *right),
            _ => unreachable!("an internal view entry is not internal"),
        })
    }

    /// The internal nodes in bottom-up order as a slice — the same sequence
    /// [`Vtree::internal_bottomup`] yields, for a caller that indexes into it.
    pub fn internal_bottomup_slice(&self) -> &[VtreeIdx] {
        self.topo.internal()
    }

    /// Repair the bottom-up order after one rotation, in `O(subtree)`.
    pub(super) fn fixup_topo_after_rotate(&mut self, info: &rotate::RotationInfo, kind: RotationKind) {
        self.topo.fixup_after_rotate(&self.nodes, info, kind);
    }

    /// Bottom-up topological order over all nodes (children before parents) as
    /// a slice — the same sequence [`Vtree::bottomup`] yields.
    #[inline]
    pub fn bottomup_slice(&self) -> &[VtreeIdx] {
        self.topo.all()
    }

    /// Topological position of `idx` (0 = first in bottom-up order, root =
    /// last). A rank comparator: `topo_pos(a) < topo_pos(b)` whenever `a` is a
    /// proper descendant of `b`.
    #[inline]
    pub fn topo_pos(&self, idx: VtreeIdx) -> u32 {
        self.topo.pos(idx)
    }
}

// Test support.
impl TopoOrder {
    /// Recompute everything from the tree by an iterative postorder,
    /// `O(num_nodes)` — the oracle the rotation tests check the localized
    /// fixup against.
    #[cfg(test)]
    pub(super) fn rebuild(&mut self, nodes: &[VtreeNode], root: VtreeIdx) {
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
