//! Minimize diagrams or run selected reduction passes.
//!
//! [`Tdd::minimize`] uses the default [`ReductionPlan`]; [`Tdd::reduce`] accepts
//! a plan explicitly. Pruning removes unreachable nodes, twin contraction
//! merges nodes with the same parent context, and pair fusion combines
//! contributions at marginal boundaries.

pub(crate) mod prune;
pub(crate) mod contract;
pub(crate) mod slot_prune; // post-tagger marginal-slot compaction
mod driver;
pub(crate) use driver::restore_marginal_invariants;

use crate::execution::pool::{Drain, Pool, Pools};

use self::contract::scratch::ContractScratch;
use crate::value::slots::RefSlotScratch;

/// Every buffer one engine's reductions reuse between calls.
#[derive(Default)]
pub(crate) struct ReduceScratch {
    /// `prune_unreachable`'s flat reachability/remap array.
    prune_remap: Pool<Vec<u32>>,
    /// `prune_unreachable`'s per-level offsets into `prune_remap`, for the
    /// walk over the whole diagram.
    prune_level_base: Pool<Vec<usize>>,
    /// The levels the seeded walk of `prune_unreachable` descended into.
    prune_visits: Pool<Vec<self::prune::Visit>>,
    /// The ascending run the seeded walk remaps an unchanged child level
    /// through.
    prune_identity: Pool<Vec<u32>>,
    /// `prune_value_slots`'s per-store slot bookkeeping.
    slot_prune_slots: Pool<RefSlotScratch>,
    /// `prune_value_slots`'s slot remap array.
    slot_prune_remap: Pool<Vec<u32>>,
    /// Twin contraction's working set, shared with the content-twin merge.
    contract: Pool<ContractScratch>,
}

impl Pools for ReduceScratch {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        visit(&self.prune_remap);
        visit(&self.prune_level_base);
        visit(&self.prune_visits);
        visit(&self.prune_identity);
        visit(&self.slot_prune_slots);
        visit(&self.slot_prune_remap);
        visit(&self.contract);
    }
}

/// The reduction passes to run, with a content-twin policy only for a full pass.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReductionPlan<'a> {
    /// Prune unreachable nodes and orphaned marginal slots.
    Prune,
    /// Contract inner-node twins.
    Contract,
    /// Prune, contract inner and leaf twins, then apply the content-twin policy.
    Full(ContentTwinPolicy<'a>),
}

impl Default for ReductionPlan<'_> {
    fn default() -> Self { Self::Full(ContentTwinPolicy::Fresh) }
}

/// Whether eligible content-twin scans run and retain their adaptive schedule.
#[derive(Debug, Default)]
#[non_exhaustive]
pub enum ContentTwinPolicy<'a> {
    /// Omit the content-twin scan.
    Skip,
    /// Run eligible scans without retaining a schedule between calls.
    #[default]
    Fresh,
    /// Carry the scan schedule across successive diagrams. A weighted diagram
    /// is scanned on every call regardless of the scheduled size.
    Adaptive(&'a mut ContentTwinSchedule),
}

/// Reuse the content-twin scan schedule across successive diagrams.
///
/// Pass the same schedule to [`ContentTwinPolicy::Adaptive`] when repeatedly
/// reducing a growing diagram. Eligible small and weighted diagrams are scanned on
/// every call. For larger unweighted diagrams, a scan schedules the next one
/// at four times its input size; falling below the size threshold resets it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContentTwinSchedule {
    /// Node count at which a skipped (above-cap) scan is re-attempted.
    /// 0 = scan on the next above-cap call.
    pub(crate) next_scan_at_nodes: u64,
}

use crate::Engine;
use self::prune::PruneScope;
use crate::limits::OperationError;
use crate::diagram::Tdd;

impl Engine {
    /// Minimize a diagram under this engine's resource limits.
    ///
    /// See [`Tdd::minimize`] for canonical form and preservation guarantees.
    /// Uses the full default [`ReductionPlan`]; on refusal the diagram remains
    /// well-formed and count-correct at the last completed pass boundary.
    pub fn minimize(&self, f: &mut Tdd) -> Result<(), OperationError> {
        self.reduce(f, ReductionPlan::default())
    }

    /// Run the selected reduction passes under this engine's resource limits.
    ///
    /// See [`Tdd::reduce`] for plan semantics and preservation guarantees.
    /// Allocation and stop refusals leave the diagram at the last completed pass
    /// boundary, where its count is still readable and preserved.
    pub fn reduce(&self, f: &mut Tdd, plan: ReductionPlan<'_>) -> Result<(), OperationError> {
        self.reduce_scoped(f, plan, PruneScope::Whole)
    }

    /// [`reduce`](Self::reduce) with the scope of its prune chosen by the
    /// caller. Only an operation that knows how the diagram it just built
    /// became unreachable in places may narrow it; see [`PruneScope`].
    pub(crate) fn reduce_scoped(
        &self,
        f: &mut Tdd,
        plan: ReductionPlan<'_>,
        scope: PruneScope,
    ) -> Result<(), OperationError> {
        driver::Reduction::new(self, f).run(plan, scope)
    }
}

#[cfg(test)]
mod tests;
