//! Search for a better vtree shape while preserving a compiled function.
//!
//! [`Tdd::rotation_search`] tries local tree
//! rotations and keeps improvements according to a caller-supplied
//! [`RotationObjective`], such as [`MinimizePairs`]. [`RotationSearchConfig`] bounds the number of sweeps,
//! and [`RotationSearchStats`] reports the work performed.
//!
//! [`Engine::rotation_search_with`](crate::Engine::rotation_search_with) takes
//! the decision rule as well: [`Greedy`] is what the plain search uses, while
//! [`Tabu`] and [`Annealing`] keep a worsening move so the search can leave a
//! local minimum. [`RotationSearchConfig::neighborhood`] widens a sweep from
//! one rotation to two or three at a time, and
//! [`Engine::rotation_multistart`](crate::Engine::rotation_multistart) runs
//! several searches from perturbed copies and keeps the smallest result.
//! [`Tdd::rotate_if`] is the primitive underneath all of them: apply
//! rotations, look at what they did, and keep or undo them.
//!
//! A changed vtree belongs to the resulting diagram. Build subsequent operands
//! using that diagram's [`Tdd::vtree`] so they share its allocation. The search
//! method's example shows both the objective and continued use of the result.

pub(crate) mod cluster;
pub(crate) mod local;
mod multistart;
mod policy;
mod probe;

#[cfg(test)]
mod tests;

pub use local::{
    MinimizePairs, Neighborhood, RotationObjective, RotationSearchConfig, RotationSearchStats,
};
pub use multistart::{MultistartConfig, MultistartStats};
pub use policy::{AcceptancePolicy, Annealing, Greedy, Tabu};
pub use probe::{RotationMove, RotationProbe};

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
    /// A trial builds two levels before it can score them; that storage is
    /// charged to the byte budget while it is live and given back when the
    /// trial commits or reverts, so a rotation too wide for the budget is
    /// refused rather than taken. The output cap does not apply. Stops are
    /// polled once per pivot; cancellation retains accepted rotations and
    /// leaves the diagram canonical and count-correct. An opening reduction
    /// failure leaves the valid partial result described by
    /// [`crate::Engine::minimize`].
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, OperationError> {
        self.rotation_search_with(tdd, objective, &mut Greedy, config)
    }

    /// [`rotation_search`](Self::rotation_search) with the decision rule
    /// given, instead of the [`Greedy`] one it defaults to.
    ///
    /// The objective says what a sequence of rotations costs; the policy says
    /// what to do about the cost. A policy that keeps a worsening sequence
    /// searches past the local minimum a descent stops at, and the search
    /// rewinds to the best diagram it passed through before returning — so the
    /// result is never worse than the descent's, at the price of the extra
    /// sweeps.
    ///
    /// # Errors
    ///
    /// [`OperationError::UnboundedSearch`] when `policy` may keep a worsening
    /// sequence and `config.max_inner_pairs` has no bound: those rebuilds are
    /// the ones that grow, and without a bound a probe can ask for more memory
    /// than the host has. Otherwise the errors of
    /// [`rotation_search`](Self::rotation_search).
    pub fn rotation_search_with<O: RotationObjective, A: AcceptancePolicy>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        policy: &mut A,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, OperationError> {
        crate::restructure::search::local::rotation_search_on(self, tdd, objective, policy, config)
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

