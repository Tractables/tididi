//! Vtree rotation primitives.
//!
//! A **left rotation** at internal node `v` promotes `v`'s right child `w`:
//!
//! ```text
//! before:     v_idx               After:      v_idx  (now w_new)
//!            / \                             / \
//!           A   w_idx                  w_idx   C
//!              / \                      / \
//!             B   C                   A   B
//! ```
//!
//! A **right rotation** at internal node `v` is the inverse — it promotes `v`'s
//! left child `w`:
//!
//! ```text
//! before:     v_idx               After:      v_idx  (now w_new)
//!            / \                             / \
//!         w_idx C                           A   w_idx
//!         / \                                  / \
//!        A   B                               B   C
//! ```
//!
//! Both operations are O(1) pointer surgery on `(v_idx, w_idx)`. Raw `VtreeIdx`
//! values for nodes never change after construction; the side `topo` list on
//! `Vtree` is updated locally via `Vtree::fixup_topo_after_rotate`, so
//! subsequent traversals (LCA, bottom-up iteration) remain correct.
//!
//! Returns `None` only when the rotation is structurally impossible (`v` or
//! `w` is a leaf).
//!
//! # Topo properties preserved by rotations
//!
//! `Vtree::topo` is required to satisfy *children-before-parents* (every
//! parent appears after both its children). The rotation primitives + topo
//! fixup additionally preserve a stronger property that is **load-bearing**
//! for `Vtree::fixup_topo_after_rotate`:
//!
//! ## Root-last property
//!
//! For every vtree node `t`,
//!
//! ```text
//! topo_pos[t] == max( topo_pos[d]  for d in {t} ∪ descendants(t) )
//! ```
//!
//! Each subtree's root sits at the latest topo position among its members.
//! This is *not* asserted after every operation, but it holds inductively
//! through any legal sequence of rebuilds + rotations:
//!
//! - **Base**: a full rebuild of the order produces strict postorder, in which
//!   each subtree root is visited after all of its descendants. Property
//!   holds trivially.
//! - **Pointer-only rotation**: pointer surgery only edits the parent/child
//!   links of `v` and `w`. The descendant *sets* of subtrees `A`, `B`, `C`
//!   (and of any node not in `v`'s subtree) are unchanged, so their
//!   max-position witness is unchanged. The property may be temporarily
//!   broken for `v` and `w` themselves until the topo fixup runs.
//! - **Order fixup**: relocates `w` past the misplaced subtree's segment as
//!   a single block (`topo[w_pos..=m_end].rotate_left(1)`). It does not
//!   reorder anything outside that slice, and within the slice it shifts
//!   every non-`w` element left by exactly one position. So the
//!   max-position witness of every subtree (including `w`'s new subtree
//!   and the misplaced subtree) updates consistently with the shift, and
//!   the property is restored for `w` and `v`.
//!
//! `TopoOrder::fixup_after_rotate` relies on it to decide in O(1) whether any
//! reordering is needed.
//!
//! Subtree contiguity is not preserved: after rotations a subtree's members may
//! occupy a non-contiguous range of `topo` positions, so nothing may index a
//! subtree by position range.

use super::{RotationKind, Vtree, VtreeIdx, VtreeNode};

/// Information about a completed rotation, sufficient to undo it or restructure
/// a diagram. Field naming follows the **left-rotation** geometry; right rotation
/// stores the same fields but with the corresponding subtrees.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub(crate) struct RotationInfo {
    /// Outer node index (parent before & after rotation).
    pub(crate) v_idx: VtreeIdx,
    /// Inner node index (the promoted/demoted child — same idx before & after).
    pub(crate) w_idx: VtreeIdx,
    /// Left rotation: was v's left child. Right rotation: was w's left child.
    pub(crate) a_idx: VtreeIdx,
    /// Left rotation: was w's left child. Right rotation: was w's right child.
    pub(crate) b_idx: VtreeIdx,
    /// Left rotation: was w's right child. Right rotation: was v's right child.
    pub(crate) c_idx: VtreeIdx,
}

/// A rotation whose bottom-up order repair is still owed.
///
/// Handed out by the pointer-only rotations, which relink nodes without
/// touching the order. Every path out of that state goes through this token:
/// [`commit`](Self::commit) repairs the order for the rotated tree,
/// [`revert`](Self::revert) puts the pointers back so the order that is still
/// in place is the right one again. It is `#[must_use]` and
/// debug-asserts if dropped unsettled, so no `&Vtree` reachable by a later read
/// can be left with an order that disagrees with its links.
#[must_use = "a pointer-only rotation owes the bottom-up order a commit, revert, or abandon"]
pub(crate) struct PendingTopo {
    info: RotationInfo,
    kind: RotationKind,
    settled: bool,
}

impl PendingTopo {
    fn new(info: RotationInfo, kind: RotationKind) -> Self {
        PendingTopo { info, kind, settled: false }
    }

    /// What the rotation did — readable while the repair is still owed, since
    /// the pointers are already in their new shape.
    pub(crate) fn info(&self) -> RotationInfo {
        self.info
    }

    /// Keep the rotation: repair the bottom-up order for the new shape.
    pub(crate) fn commit(mut self, vtree: &mut Vtree) -> RotationInfo {
        vtree.fixup_topo_after_rotate(&self.info, self.kind);
        self.settled = true;
        self.info
    }

    /// Drop the rotation: put the links back, which makes the untouched order
    /// correct again.
    pub(crate) fn revert(mut self, vtree: &mut Vtree) {
        unrotate_pointers(vtree, &self.info, self.kind);
        self.settled = true;
    }
}

impl Drop for PendingTopo {
    fn drop(&mut self) {
        debug_assert!(
            self.settled,
            "a pointer-only rotation was dropped without committing, reverting, or abandoning it: \
             the vtree's bottom-up order no longer matches its links",
        );
    }
}

/// The children of a node as a rotation of `kind` sees them: the child it
/// promotes (the right child of a left rotation, the left child of a right
/// one) and the other.
#[inline]
fn promoted_first(kind: RotationKind, left: VtreeIdx, right: VtreeIdx) -> (VtreeIdx, VtreeIdx) {
    match kind {
        RotationKind::Left => (right, left),
        RotationKind::Right => (left, right),
    }
}

/// An internal node whose child on the side a rotation of `kind` promotes is
/// `promoted` and whose other child is `other`.
#[inline]
fn internal_with(kind: RotationKind, promoted: VtreeIdx, other: VtreeIdx, parent: Option<VtreeIdx>) -> VtreeNode {
    match kind {
        RotationKind::Left => VtreeNode::Internal { left: other, right: promoted, parent },
        RotationKind::Right => VtreeNode::Internal { left: promoted, right: other, parent },
    }
}

/// Rotate at `v`, pointer surgery only: a left rotation promotes `v`'s right
/// child, a right rotation its left child. Returns `None` if `v` or the
/// promoted child is a leaf, and otherwise a [`PendingTopo`] the caller must
/// settle.
///
/// The two directions are mirror images, so one body serves both: with `w`
/// the promoted child, `x` the other child of `v`, and `y`, `z` the children
/// of `w` on the promoted side and the other side, the rotation gives `v` the
/// children `(y, w)` and `w` the children `(z, x)`, each named on the promoted
/// side first. Read left to right, the three subtrees are `(x, z, y)` for a
/// left rotation and `(y, z, x)` for a right one, which is how
/// [`RotationInfo`] names them.
///
/// The bottom-up order is left stale: the rotation search probes a rotation
/// through restructure, minimize and size, none of which read the order, and
/// usually reverts it. The returned token is `#[must_use]` and debug-asserts on
/// drop, so the order can only be left stale by a caller that says so.
pub(crate) fn rotate_pointers(vtree: &mut Vtree, v: VtreeIdx, kind: RotationKind) -> Option<PendingTopo> {
    let VtreeNode::Internal { left, right, parent: v_parent } = vtree.nodes[v.idx()] else {
        return None;
    };
    let (w, x) = promoted_first(kind, left, right);
    let VtreeNode::Internal { left, right, .. } = vtree.nodes[w.idx()] else {
        return None;
    };
    let (y, z) = promoted_first(kind, left, right);

    vtree.nodes[v.idx()] = internal_with(kind, y, w, v_parent);
    vtree.nodes[w.idx()] = internal_with(kind, z, x, Some(v));
    Vtree::set_parent(&mut vtree.nodes, x, w);
    Vtree::set_parent(&mut vtree.nodes, y, v);

    let (a_idx, c_idx) = match kind {
        RotationKind::Left => (x, y),
        RotationKind::Right => (y, x),
    };
    Some(PendingTopo::new(
        RotationInfo { v_idx: v, w_idx: w, a_idx, b_idx: z, c_idx },
        kind,
    ))
}

/// Undo a rotation of `kind`, pointer surgery only: the inverse of
/// [`rotate_pointers`] on its own [`RotationInfo`]. The bottom-up order is
/// left as it was before the rotation, which is why this is reachable only
/// through [`PendingTopo::revert`] (and the round-trip oracle in the rotation
/// tests).
fn unrotate_pointers(vtree: &mut Vtree, info: &RotationInfo, kind: RotationKind) {
    let RotationInfo { v_idx, w_idx, a_idx, b_idx, c_idx } = *info;
    let v_parent = vtree.nodes[v_idx.idx()].parent();
    let (x, y) = match kind {
        RotationKind::Left => (a_idx, c_idx),
        RotationKind::Right => (c_idx, a_idx),
    };

    vtree.nodes[v_idx.idx()] = internal_with(kind, w_idx, x, v_parent);
    vtree.nodes[w_idx.idx()] = internal_with(kind, y, b_idx, Some(v_idx));
    Vtree::set_parent(&mut vtree.nodes, x, v_idx);
    Vtree::set_parent(&mut vtree.nodes, y, w_idx);
}

// Test support.
impl PendingTopo {
    /// Discard the obligation because the caller is about to overwrite the
    /// order by other means. Only the rebuild-equivalence test needs this: it
    /// rotates and then recomputes the whole order from scratch.
    #[cfg(test)]
    pub(crate) fn abandon(mut self) -> RotationInfo {
        self.settled = true;
        self.info
    }
}

#[cfg(test)]
mod tests;
