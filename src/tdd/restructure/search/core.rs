//! Rotation-kind dispatch and the per-rotation helpers the mid-compile
//! marginal-clustering pass ([`cluster`](super::cluster)) builds on.
//!
//! `RotKind` + the rotate/unrotate/restructure wrappers, the per-level
//! pair-count helper, the marginal-level guard, and the subtree allow-mask.
//! `cluster` pulls these in via `use super::core::*`.

use crate::vtree::{RotationKind, Vtree, VtreeIdx, VtreeNode};
use crate::vtree::rotate::{
    rotate_left_pointers, rotate_right_pointers,
    unrotate_left_pointers, unrotate_right_pointers,
    RotationInfo,
};
use crate::tdd::types::{Tdd, TddLevel};
use crate::tdd::restructure::rotate::{
    restructure_after_left_rotation_bounded, restructure_after_right_rotation_bounded,
    RestructureScratch,
};

// ─── Shared utilities ─────────────────────────────────────────────────────

/// Sum of input pairs across one level's internal nodes — the per-level
/// component of `tdd.size()`. Used to compute size deltas locally over
/// `{v_idx, w_idx}` (Rotation Locality) without
/// re-summing every other level.
pub(super) fn level_pair_count(level: &TddLevel) -> usize {
    (0..level.nodes.len())
        .map(|i| if level.nodes[i].is_internal() { level.pair_count_at(i) } else { 0 })
        .sum()
}

/// Local rotation kind. Mirrors `vtree::RotationKind` but with `#[repr(u8)]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum RotKind { Left = 0, Right = 1 }

impl RotKind {
    #[inline]
    pub(super) fn as_rotation_kind(self) -> RotationKind {
        match self {
            RotKind::Left => RotationKind::Left,
            RotKind::Right => RotationKind::Right,
        }
    }
}

// Kind-dispatch wrappers around the four left/right primitives we call.
// Inlined, so the compiler collapses the match away — they exist purely to
// remove repeated `match kind { Left => ..._left, Right => ..._right }`
// blocks from the higher-level cluster pass.

#[inline]
pub(super) fn rotate_pointers_kind(vt: &mut Vtree, v: VtreeIdx, kind: RotKind) -> Option<RotationInfo> {
    match kind {
        RotKind::Left => rotate_left_pointers(vt, v),
        RotKind::Right => rotate_right_pointers(vt, v),
    }
}

#[inline]
pub(super) fn unrotate_pointers_kind(vt: &mut Vtree, info: &RotationInfo, kind: RotKind) {
    match kind {
        RotKind::Left => unrotate_left_pointers(vt, info),
        RotKind::Right => unrotate_right_pointers(vt, info),
    }
}

#[inline]
pub(super) fn restructure_kind_bounded(
    tdd: &mut Tdd,
    info: &RotationInfo,
    kind: RotKind,
    scratch: &mut RestructureScratch,
    bound: usize,
) -> Option<(TddLevel, TddLevel)> {
    match kind {
        RotKind::Left => restructure_after_left_rotation_bounded(tdd, info, scratch, bound),
        RotKind::Right => restructure_after_right_rotation_bounded(tdd, info, scratch, bound),
    }
}

/// Build an allow-mask for `subtree(root)`: every internal node in
/// `subtree(root)` (root included) is marked true. Used by the mid-compile
/// rotation pass to confine the search to the post-order frontier's
/// fully-compiled region. Including root is safe: the parent level has no
/// compiled data yet, and rotation at root preserves 1-to-1 node
/// correspondence at v_idx (rotation locality).
pub(super) fn subtree_allow_mask(vtree: &Vtree, root: VtreeIdx) -> Vec<bool> {
    let mut mask = vec![false; vtree.num_nodes()];
    let mut stack: Vec<VtreeIdx> = vec![root];
    while let Some(n) = stack.pop() {
        if let VtreeNode::Internal { left, right, .. } = *vtree.node(n) {
            mask[n.idx()] = true;
            stack.push(left);
            stack.push(right);
        }
    }
    mask
}

/// Returns true if a rotation level is marginal in a way that blocks the rotation.
///
/// `v`/`w` have their PAIRS iterated/rebuilt during restructure, and once a node
/// is marginalized its children no longer exist as levels (collapsed to counts) —
/// so the rotation that would split it is ill-defined; a `v`/`w`-marginal rotation
/// is genuinely unhandled and ALWAYS blocks.
///
/// `a`,`b`,`c` (grandchildren) are referenced only as bare node indices, so a
/// child/grandchild-only-marginal rotation is the "rotate the PARENT of a
/// marginalized subtree" case. It is structurally sound and count-safe — the
/// marginal-context full expansion in `rotate.rs` keeps the full cell/outer
/// multiset instead of sharing/deduping, so `#F` is preserved exactly — so it is
/// ALWAYS allowed (no flag, no guard). This is what lets the cluster pass rotate
/// through marginalized nodes.
pub(super) fn any_rotation_level_marginal(tdd: &Tdd, info: &RotationInfo) -> bool {
    // `v`/`w` marginal: pairs would be iterated and the marginalized node would
    // have to be decomposed into children that no longer exist. Always blocks.
    if tdd.levels[info.v_idx.idx()].is_marginal()
        || tdd.levels[info.w_idx.idx()].is_marginal()
    {
        return true;
    }
    // a/b/c (grandchild) marginal is the parent-of-marginal case: count-safe via
    // the marginal-context full expansion, so the rotation always proceeds.
    false
}
