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
//! `Vtree` is updated locally via `Vtree::fixup_topo_after_rotate` (the
//! convenience wrappers below also do this) so subsequent traversals (LCA,
//! bottom-up iteration) remain correct.
//!
//! Returns `None` only when the rotation is structurally impossible (`v` or
//! `w` is a leaf). There is no longer a topological-order applicability check
//! because the topo update is decoupled from node identity.
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
//! The case-detection check in `TopoOrder::fixup_after_rotate`
//! (`if topo_pos[misplaced_root] < topo_pos[w] { return; }`) relies on
//! `topo_pos[misplaced_root]` being the maximum position over the entire
//! misplaced subtree — i.e. on this property — to decide in O(1) whether
//! any reordering is needed.
//!
//! ## Subtree contiguity is *not* preserved
//!
//! After a sequence of rotations + fixups, a subtree's members may occupy
//! a non-contiguous range of positions in `topo` (an element from a sibling
//! subtree can sit "between" two members of the same subtree). This is
//! intentional: enforcing contiguity would require shifting unrelated
//! elements during fixups. No consumer of the order, its inverse, or the two
//! filtered views in this codebase indexes a subtree by position range — they
//! all walk parent pointers / child links or do rank comparisons by position. Children-before-parents and root-last are
//! sufficient for every consumer.
//!
//! Correctness of the slice rotation does **not** depend on contiguity: the
//! shift preserves children-before-parents for every edge in the new tree
//! (each edge either has both endpoints inside the slice — both shift — or
//! both endpoints outside — neither shifts — or one endpoint inside and the
//! shift never inverts the integer ordering between them). See the
//! `TopoOrder::fixup_after_rotate` body for the per-case argument.

use super::{RotationKind, Vtree, VtreeIdx, VtreeNode};

/// Information about a completed rotation, sufficient to undo it or restructure
/// a diagram. Field naming follows the **left-rotation** geometry; right rotation
/// stores the same fields but with the corresponding subtrees.
#[derive(Clone, Copy, Debug)]
pub struct RotationInfo {
    /// Outer node index (parent before & after rotation).
    pub v_idx: VtreeIdx,
    /// Inner node index (the promoted/demoted child — same idx before & after).
    pub w_idx: VtreeIdx,
    /// Left rotation: was v's left child. Right rotation: was w's left child.
    pub a_idx: VtreeIdx,
    /// Left rotation: was w's left child. Right rotation: was w's right child.
    pub b_idx: VtreeIdx,
    /// Left rotation: was w's right child. Right rotation: was v's right child.
    pub c_idx: VtreeIdx,
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
        match self.kind {
            RotationKind::Left => unrotate_left_pointers(vtree, &self.info),
            RotationKind::Right => unrotate_right_pointers(vtree, &self.info),
        }
        self.settled = true;
    }

    /// Discard the obligation because the caller is about to overwrite the
    /// order by other means. Only the rebuild-equivalence test needs this: it
    /// rotates and then recomputes the whole order from scratch.
    #[cfg(test)]
    pub(crate) fn abandon(mut self) -> RotationInfo {
        self.settled = true;
        self.info
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

/// Left-rotate, pointer surgery only. Returns `None` if `v` or its right child
/// is a leaf, and otherwise a [`PendingTopo`] the caller must settle.
///
/// The bottom-up order is left stale on purpose: the rotation search probes a
/// rotation through restructure + minimize + size — none of which read the
/// order — and usually reverts it, so repairing the order on every probe is
/// wasted work. The returned token is what makes that safe: it is `#[must_use]`
/// and debug-asserts on drop, so the order can only be left stale by a caller
/// that says so.
pub(crate) fn rotate_left_pointers(vtree: &mut Vtree, v: VtreeIdx) -> Option<PendingTopo> {
    let (a, w, v_parent) = match vtree.nodes[v.idx()] {
        VtreeNode::Internal { left, right, parent } => (left, right, parent),
        VtreeNode::Leaf { .. } => return None,
    };
    let (b, c) = match vtree.nodes[w.idx()] {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => return None,
    };

    // v_idx becomes w_new: children = (w_idx=v_new, C).
    vtree.nodes[v.idx()] = VtreeNode::Internal { left: w, right: c, parent: v_parent };
    // w_idx becomes v_new: children = (A, B).
    vtree.nodes[w.idx()] = VtreeNode::Internal { left: a, right: b, parent: Some(v) };
    Vtree::set_parent(&mut vtree.nodes, a, w);
    Vtree::set_parent(&mut vtree.nodes, c, v);

    Some(PendingTopo::new(
        RotationInfo { v_idx: v, w_idx: w, a_idx: a, b_idx: b, c_idx: c },
        RotationKind::Left,
    ))
}

/// Left-rotate the vtree at node `v`, promoting `v`'s right child `w`.
///
/// Returns `None` if `v` or its right child is a leaf, and succeeds otherwise:
/// topo order is tracked separately from node identity, so no rotation can
/// break it. Repairs the topo order in place afterwards
/// (`fixup_topo_after_rotate`), which touches only the nodes the
/// rotation moved.
pub fn rotate_left(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    Some(rotate_left_pointers(vtree, v)?.commit(vtree))
}

/// Undo a left rotation, pointer surgery only. The bottom-up order is left as
/// it was before the rotation, which is why this is reachable only through
/// [`PendingTopo::revert`] (and the test-only round-trip oracle below).
fn unrotate_left_pointers(vtree: &mut Vtree, info: &RotationInfo) {
    let RotationInfo { v_idx, w_idx, a_idx, b_idx, c_idx } = *info;
    let v_parent = vtree.nodes[v_idx.idx()].parent();

    vtree.nodes[v_idx.idx()] = VtreeNode::Internal { left: a_idx, right: w_idx, parent: v_parent };
    vtree.nodes[w_idx.idx()] = VtreeNode::Internal { left: b_idx, right: c_idx, parent: Some(v_idx) };
    Vtree::set_parent(&mut vtree.nodes, a_idx, v_idx);
    Vtree::set_parent(&mut vtree.nodes, c_idx, w_idx);
}

/// Undo a left rotation given its `RotationInfo`. Equivalent to a right rotation
/// at `v_idx` for trees that came from a left rotation.
///
/// Round-trip oracle for the rotation tests only: production
/// restructuring undoes a rotation through the pointer-only
/// [`unrotate_left_pointers`] plus its own topo bookkeeping, so this
/// convenience wrapper exists purely so a test can assert `rotate ∘ unrotate ==
/// identity`. `cfg(test)` keeps it out of the shipped library.
#[cfg(test)]
pub fn unrotate_left(vtree: &mut Vtree, info: &RotationInfo) {
    unrotate_left_pointers(vtree, info);
    // unrotate_left ≡ right rotation on the post-left-rot tree. The
    // RotationInfo's a/b/c happen to match the right-rotation conventions
    // (right rot's `a` is the post-left-rot's `w.left` = original `a`,
    // similarly for b and c), so we can pass `info` straight through.
    vtree.fixup_topo_after_rotate(info, RotationKind::Right);
}

/// Right-rotate, pointer surgery only. Mirror of [`rotate_left_pointers`]; see
/// its doc for the stale-order contract.
pub(crate) fn rotate_right_pointers(vtree: &mut Vtree, v: VtreeIdx) -> Option<PendingTopo> {
    let (w, c, v_parent) = match vtree.nodes[v.idx()] {
        VtreeNode::Internal { left, right, parent } => (left, right, parent),
        VtreeNode::Leaf { .. } => return None,
    };
    let (a, b) = match vtree.nodes[w.idx()] {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => return None,
    };

    // v_idx stays as v with children (A, w_idx=w_new).
    vtree.nodes[v.idx()] = VtreeNode::Internal { left: a, right: w, parent: v_parent };
    // w_idx becomes w_new: children = (B, C).
    vtree.nodes[w.idx()] = VtreeNode::Internal { left: b, right: c, parent: Some(v) };
    Vtree::set_parent(&mut vtree.nodes, a, v);
    Vtree::set_parent(&mut vtree.nodes, c, w);

    Some(PendingTopo::new(
        RotationInfo { v_idx: v, w_idx: w, a_idx: a, b_idx: b, c_idx: c },
        RotationKind::Right,
    ))
}

/// Right-rotate the vtree at node `v`, promoting `v`'s left child `w`.
///
/// Returns `None` if `v` or its left child is a leaf, and succeeds otherwise.
/// Repairs the topo order in place afterwards, as [`rotate_left`] does.
pub fn rotate_right(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    Some(rotate_right_pointers(vtree, v)?.commit(vtree))
}

/// Undo a right rotation, pointer surgery only. Mirror of
/// [`unrotate_left_pointers`].
fn unrotate_right_pointers(vtree: &mut Vtree, info: &RotationInfo) {
    let RotationInfo { v_idx, w_idx, a_idx, b_idx, c_idx } = *info;
    let v_parent = vtree.nodes[v_idx.idx()].parent();

    vtree.nodes[v_idx.idx()] = VtreeNode::Internal { left: w_idx, right: c_idx, parent: v_parent };
    vtree.nodes[w_idx.idx()] = VtreeNode::Internal { left: a_idx, right: b_idx, parent: Some(v_idx) };
    Vtree::set_parent(&mut vtree.nodes, a_idx, w_idx);
    Vtree::set_parent(&mut vtree.nodes, c_idx, v_idx);
}

/// Undo a right rotation given its `RotationInfo`.
///
/// Test-only round-trip oracle, exactly like [`unrotate_left`] — see its note.
#[cfg(test)]
pub fn unrotate_right(vtree: &mut Vtree, info: &RotationInfo) {
    unrotate_right_pointers(vtree, info);
    // unrotate_right ≡ left rotation on the post-right-rot tree. The
    // RotationInfo's a/b/c match left-rotation conventions on this side too.
    vtree.fixup_topo_after_rotate(info, RotationKind::Left);
}

#[cfg(test)]
#[path = "rotate_tests.rs"]
mod tests;
