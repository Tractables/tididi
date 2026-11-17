//! The proof that two vtrees have one layout, which a diagram needs to move
//! from one to the other without touching its levels.

use std::sync::Arc;

use super::Vtree;

/// Two vtrees of one home with the same leaf set, so the same nodes at the
/// same indices with the same variables. Only `Vtree::leaf_sets` makes one;
/// `Tdd::reseat` takes it.
#[derive(Debug)]
pub(crate) struct SameLayout {
    a: Arc<Vtree>,
    b: Arc<Vtree>,
}

impl SameLayout {
    pub(super) fn new(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Self {
        Self { a: Arc::clone(a), b: Arc::clone(b) }
    }

    /// The vtree of the two that is not `from`.
    ///
    /// # Panics
    ///
    /// Panics if `from` is neither.
    pub(crate) fn other(&self, from: &Arc<Vtree>) -> &Arc<Vtree> {
        if Arc::ptr_eq(from, &self.a) {
            &self.b
        } else {
            assert!(Arc::ptr_eq(from, &self.b), "a layout proof for another vtree");
            &self.a
        }
    }
}
