//! Search for a better vtree shape while preserving a compiled function.
//!
//! [`Tdd::rotation_search`] tries local tree
//! rotations and keeps improvements according to a caller-supplied
//! [`RotationObjective`], such as [`MinimizePairs`]. [`RotationSearchConfig`] bounds the number of sweeps,
//! and [`RotationSearchStats`] reports the work performed.
//!
//! A changed vtree belongs to the resulting diagram. Build subsequent operands
//! using that diagram's [`Tdd::vtree`] so they share its allocation. The search
//! method's example shows both the objective and continued use of the result.

pub(crate) mod cluster;
mod probe;
pub(crate) mod local;

#[cfg(test)]
mod tests;

pub use local::{MinimizePairs, RotationObjective, RotationSearchConfig, RotationSearchStats};

pub(crate) use local::rotation_search_on;

use crate::diagram::Tdd;
use crate::limits::OperationError;

impl crate::Engine {
    /// Run [`Tdd::rotation_search`](crate::Tdd::rotation_search) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors, [`OperationError::Stopped`] on
    /// cancellation, or [`OperationError::OverBudget`] on allocation refusal.
    ///
    /// Trial storage is outside the byte budget and output cap. Stops are polled
    /// once per pivot; cancellation retains accepted rotations and leaves the
    /// diagram canonical and count-correct. An opening reduction failure leaves
    /// the valid partial result described by [`crate::Engine::minimize`].
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, OperationError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }
}

/// Keep one private vtree across probes, restoring shared identity if none was accepted.
pub(super) struct SearchTree<'a> {
    pub(super) tdd: &'a mut Tdd,
    pub(super) original: Option<std::sync::Arc<crate::Vtree>>,
}

impl<'a> SearchTree<'a> {
    /// Detach a shared tree once before probing any pivots.
    pub(super) fn new(tdd: &'a mut Tdd) -> Self {
        use std::sync::Arc;
        let original = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        Arc::make_mut(&mut tdd.vtree);
        Self { tdd, original }
    }
}

impl Drop for SearchTree<'_> {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() { self.tdd.vtree = original; }
    }
}

