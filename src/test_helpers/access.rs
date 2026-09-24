//! What a test needs from crate-private state and the library does not run:
//! infallible fixture builders, an engine that stops at once, whole-tree
//! vtree rotations, and re-homing a diagram at a child of its root.

use num_bigint::BigUint;

use crate::Engine;
use crate::limits::{LimitConfig, StopDecision};
use crate::value::{Count, CountVec};
use crate::vtree::rotate::rotate_pointers;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{Tdd, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Count-column fixture builders that expect every reservation to succeed.
pub(crate) trait CountVecExt {
    /// `push`, infallible.
    fn push_i(&mut self, eng: &Engine, c: Count);
    /// `try_clone`, infallible.
    fn clone_guarded(&self, eng: &Engine) -> Self;
    /// The big table's entry for slot `i`, `None` when the table has none
    /// there (no table yet, or a plain fast slot).
    fn big_val(&self, eng: &Engine, i: usize) -> Option<BigUint>;
}

impl CountVecExt for CountVec {
    fn push_i(&mut self, eng: &Engine, c: Count) {
        self.push(eng, c).expect("test allocation succeeds")
    }

    fn clone_guarded(&self, eng: &Engine) -> Self {
        self.try_clone(eng).expect("test allocation succeeds")
    }

    fn big_val(&self, eng: &Engine, i: usize) -> Option<BigUint> {
        let (_, big) = self.clone_guarded(eng).into_parts();
        big.and_then(|b| b.get(i).cloned())
    }
}

/// A fresh engine whose schedule stops the first operation that asks it —
/// the preemption the deadline tests assert, without a wall clock.
#[must_use]
pub(crate) fn stopping_engine() -> Engine {
    let engine = Engine::new();
    let _prior = engine.limits().install(LimitConfig::none().with_stop_callback(Some(crate::limits::StopCallback::new(|_, _| StopDecision::Stop))));
    engine
}

/// Left-rotate the vtree at node `v`, promoting `v`'s right child, and repair
/// the bottom-up order. `None` if `v` or its right child is a leaf.
///
/// The rotation search rotates through `rotate_pointers` and commits or
/// reverts the pending order itself; this whole-rotation form is what the
/// tests are written against.
pub(crate) fn rotate_left(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    Some(rotate_pointers(vtree, v, RotationKind::Left)?.commit(vtree))
}

/// Right-rotate the vtree at node `v`, promoting `v`'s left child, and repair
/// the bottom-up order. `None` if `v` or its left child is a leaf.
pub(crate) fn rotate_right(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    Some(rotate_pointers(vtree, v, RotationKind::Right)?.commit(vtree))
}

/// Re-home a diagram that depends only on variables under one child of its
/// root so that it is rooted at that child — a genuinely low-rooted diagram.
///
/// Building and applying always root at the vtree root, so re-homing is the
/// only way to reach the differing-root operand shape a tightly-rooted segment
/// would take. The root level must hold a single identity pair, the `g ∧ ⊤`
/// shape a single-region function compiles to.
pub fn reroot_to_child(t: &Tdd, left_child: bool) -> Tdd {
    let root = t.output.vtree;
    let (lc, rc) = match *t.vtree.node(root) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => panic!("reroot_to_child: the root must be internal"),
    };
    let pairs = t.levels[root.idx()].pairs_of_idx(t.output.local.idx());
    assert_eq!(pairs.len(), 1, "reroot_to_child expects the single-region g ∧ ⊤ shape");
    let p = pairs[0];
    let (child, local) = if left_child { (lc, p.left) } else { (rc, p.right) };
    Tdd::from_levels_unchecked(
        t.vtree.clone(),
        t.levels.clone().into_vec(),
        TddNodeId { vtree: child, local: t.levels[child.idx()].child_decoder().node(local) },
    )
}
