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
//! [`Engine::rotate_pool_if`](crate::Engine::rotate_pool_if) and
//! [`Engine::pool_search`](crate::Engine::pool_search) do the same for
//! several diagrams that share one vtree, and can also swap a node's
//! children, which changes the order of the leaves.
//!
//! A changed vtree belongs to the resulting diagram. Build subsequent operands
//! using that diagram's [`Tdd::vtree`] so they share its allocation. The search
//! method's example shows both the objective and continued use of the result.

pub(crate) mod cluster;
pub(crate) mod local;
mod multistart;
mod policy;
mod pool;
mod probe;

#[cfg(test)]
mod tests;

pub use local::{
    MinimizePairs, Neighborhood, RotationObjective, RotationSearchConfig, RotationSearchStats,
};
pub use multistart::{MultistartConfig, MultistartStats};
pub use policy::{AcceptancePolicy, Annealing, Greedy, Tabu};
pub use pool::{PoolMove, PoolProbe, PoolRestarts, PoolSearchConfig, PoolSearchStats};
pub use probe::{RotationMove, RotationProbe};
pub(in crate::restructure) use probe::{ProbeRule, probe_moves};

use crate::diagram::Tdd;
use crate::vtree::Vtree;
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
}

/// Keep one private vtree across probes, restoring shared identity if none was
/// accepted.
///
/// Every probe of a diagram's vtree goes through here: the tree is detached
/// from its sharers once, before the first probe, and put back when the
/// search ends with nothing kept, so a declined trial leaves not just the
/// shape but the allocation as it was.
///
/// A projection keeps its link while it moves, stale but for its leaves. A
/// search that kept a move ends by seating the diagram on the projection of
/// the home the moves imply (`Vtree::lifted_home`), so its
/// counts stay over a home's variables; one that kept nothing makes no home.
/// A home's root is not a home while it moves; it gets an index of its new
/// shape when the search ends with a move kept, and its own index back when
/// none is kept and no clone of the diagram shares the tree.
pub(super) struct SearchTree<'a> {
    pub(super) tdd: &'a mut Tdd,
    eng: &'a crate::Engine,
    original: Option<std::sync::Arc<crate::Vtree>>,
    kept: bool,
    was_home: bool,
    /// The index of a home's root held by this diagram alone, set aside
    /// while the tree moves and put back if no move is kept.
    index: crate::vtree::RootIndex,
}

impl<'a> SearchTree<'a> {
    /// Detach a shared tree once before probing any pivots.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel)
    /// naming a region: a kept move would take the tree off its home, which
    /// a region cannot leave.
    pub(super) fn new(eng: &'a crate::Engine, tdd: &'a mut Tdd) -> Result<Self, crate::OperationError> {
        use std::sync::Arc;
        if let Some(region) = tdd.vtree.regions().next() {
            return Err(crate::OperationError::MarginalLevel(region));
        }
        let original = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        let was_home = tdd.vtree.is_home();
        let index = Arc::get_mut(&mut tdd.vtree).map(Vtree::take_index).unwrap_or_default();
        Vtree::make_mut(&mut tdd.vtree);
        Ok(Self { tdd, eng, original, kept: false, was_home, index })
    }

    /// [`probe_moves`](probe::probe_moves) on the private tree; a kept
    /// sequence is what makes the private tree the diagram's own. On a
    /// projection, a diagram with a summed-out level keeps no move: the seat
    /// on the new home moves structural levels only.
    fn probe_moves<R: probe::ProbeRule>(
        &mut self,
        eng: &crate::Engine,
        moves: &[RotationMove],
        rule: &mut R,
        scratch: &mut crate::restructure::scratch::RestructureScratch,
        bound: usize,
    ) -> Result<bool, OperationError> {
        if self.tdd.vtree.home().is_some() && self.tdd.has_marginal_level() {
            return Ok(false);
        }
        let kept = probe::probe_moves(eng, self.tdd, moves, rule, scratch, bound)?;
        if kept {
            self.original = None;
            self.kept = true;
        }
        Ok(kept)
    }
}

impl Drop for SearchTree<'_> {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() {
            self.tdd.vtree = original;
        } else if self.kept && self.was_home {
            Vtree::become_home(&mut self.tdd.vtree);
        } else if self.kept {
            crate::restructure::transport::seat_on_lifted_home(self.eng, self.tdd);
        } else if let Some(own) = std::sync::Arc::get_mut(&mut self.tdd.vtree) {
            own.put_index(std::mem::take(&mut self.index));
        }
    }
}

