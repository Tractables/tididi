use super::*;
use crate::test_helpers::{rotate_left, rotate_right};

mod roundtrip;

/// Undo a left rotation given its `RotationInfo`: the pointer surgery, then
/// the order repair of the right rotation it amounts to, so a test can assert
/// `rotate ∘ unrotate == identity`.
fn unrotate_left(vtree: &mut Vtree, info: &RotationInfo) {
    unrotate_pointers(vtree, info, RotationKind::Left);
    // unrotate_left ≡ right rotation on the post-left-rot tree. The
    // RotationInfo's a/b/c happen to match the right-rotation conventions
    // (right rot's `a` is the post-left-rot's `w.left` = original `a`,
    // similarly for b and c), so we can pass `info` straight through.
    vtree.fixup_topo_after_rotate(info, RotationKind::Right);
}

/// Undo a right rotation given its `RotationInfo`, the mirror of
/// [`unrotate_left`].
fn unrotate_right(vtree: &mut Vtree, info: &RotationInfo) {
    unrotate_pointers(vtree, info, RotationKind::Right);
    // unrotate_right ≡ left rotation on the post-right-rot tree. The
    // RotationInfo's a/b/c match left-rotation conventions on this side too.
    vtree.fixup_topo_after_rotate(info, RotationKind::Left);
}

/// Recompute the bottom-up order from the current links: the oracle the
/// localized fixup is checked against.
fn rebuild_topo(vtree: &mut Vtree) {
    vtree.topo.rebuild(&vtree.nodes, vtree.root);
}
