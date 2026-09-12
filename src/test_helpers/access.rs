//! What a test needs from crate-private state and the library does not run:
//! infallible fixture builders, an engine that stops at once, and whole-tree
//! vtree rotations.

use num_bigint::BigUint;

use crate::engine::Engine;
use crate::limits::{LimitConfig, RecoveryPanic, StopDecision};
use crate::value::{unwrap_infallible, Count, CountVec};
use crate::vtree::rotate::rotate_pointers;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::vtree::{Vtree, VtreeIdx};

/// The budget-checked `CountVec` operations under [`RecoveryPanic`], where
/// they cannot fail, for building a fixture on an engine with nothing armed.
pub(crate) trait CountVecExt {
    /// `push`, infallible.
    fn push_i(&mut self, eng: &Engine, c: Count);
    /// `try_clone`, infallible.
    fn clone_guarded(&self, eng: &Engine) -> Self;
    /// The big table's entry for slot `i`, `None` when the table has none
    /// there (no table yet, or a plain fast slot).
    fn big_val(&self, eng: &Engine, i: usize) -> Option<BigUint>;
}

impl CountVecExt for CountVec<RecoveryPanic> {
    fn push_i(&mut self, eng: &Engine, c: Count) {
        unwrap_infallible(self.push(eng, c))
    }

    fn clone_guarded(&self, eng: &Engine) -> Self {
        unwrap_infallible(self.try_clone(eng))
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
