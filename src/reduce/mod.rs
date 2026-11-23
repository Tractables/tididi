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
mod trim;
pub(crate) use driver::restore_marginal_invariants;
pub(crate) use trim::trim;

use crate::execution::pool::{Drain, Pool, Pools};

use self::contract::scratch::ContractScratch;
use crate::value::slots::RefSlotScratch;

/// Every buffer one engine's reductions reuse between calls.
#[derive(Default)]
pub(crate) struct ReduceScratch {
    /// `prune_unreachable`'s marks, new indices and walk.
    prune: Pool<self::prune::PruneScratch>,
    /// `drop_clause_orphans`'s marks, new indices and levels to read.
    clause_drop: Pool<self::prune::DropScratch>,
    /// `prune_value_slots`'s per-store slot bookkeeping.
    slot_prune_slots: Pool<RefSlotScratch>,
    /// `prune_value_slots`'s slot remap array.
    slot_prune_remap: Pool<Vec<u32>>,
    /// Twin contraction's working set, shared with the content-twin merge.
    contract: Pool<ContractScratch>,
    /// Per vtree node and side, the node that refused the last leaf
    /// contraction there (`contract::contract_leaf`).
    leaf_refused: Pool<Vec<u32>>,
}

impl Pools for ReduceScratch {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        visit(&self.prune);
        visit(&self.clause_drop);
        visit(&self.slot_prune_slots);
        visit(&self.slot_prune_remap);
        visit(&self.contract);
        visit(&self.leaf_refused);
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
///
/// With a marginal level, a scan merges nodes that no later rewrite splits,
/// so the form a diagram built by successive conjunctions reaches depends on
/// where scans ran between them.
#[derive(Debug, Default)]
#[non_exhaustive]
pub enum ContentTwinPolicy<'a> {
    /// Omit the content-twin scan.
    Skip,
    /// Run eligible scans without retaining a schedule between calls.
    #[default]
    Fresh,
    /// Carry the scan schedule across successive diagrams. A weighted diagram
    /// with a marginal level is scanned on every call regardless of the
    /// scheduled size; a diagram with no marginal level is never scanned.
    Adaptive(&'a mut ContentTwinSchedule),
    /// As `Adaptive`, but a scan the schedule calls for on a diagram of at
    /// least `min_pairs` pairs, counted after the prune and contraction that
    /// precede the scan, stops once its first pass has merged the bottom
    /// levels holding a thirty-second of the pairs, unless the duplicates it
    /// found there held at least `min_yield` of those levels' pairs. A
    /// stopped scan keeps those merges, runs the prune and contraction they
    /// need, scans no other level, and sets the schedule as a finished scan
    /// would. Under `min_pairs` the scan is `Adaptive`'s.
    /// [`ContentTwinSchedule::last_sample_yield`] reports the yield measured.
    Sampled {
        /// The schedule, as for `Adaptive`.
        schedule: &'a mut ContentTwinSchedule,
        /// The share of the sampled pairs the duplicates must hold for the
        /// scan to go on.
        min_yield: f64,
        /// The fewest pairs a diagram holds for its scan to be sampled.
        min_pairs: usize,
    },
}

/// Reuse the content-twin scan schedule across successive diagrams.
///
/// Pass the same schedule to [`ContentTwinPolicy::Adaptive`] when repeatedly
/// reducing a growing diagram. Eligible small and weighted diagrams are scanned on
/// every call. For larger unweighted diagrams, a scan schedules the next one
/// at 1.5 to 4 times its input size, sooner the larger the share of nodes
/// it removed; falling below the size threshold resets it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContentTwinSchedule {
    /// Node count at which a skipped (above-cap) scan is re-attempted.
    /// 0 = scan on the next above-cap call.
    pub(crate) next_scan_at_nodes: u64,
    /// What the last sampled scan measured.
    pub(crate) last_sample_yield: Option<f64>,
}

impl ContentTwinSchedule {
    /// The share of the sampled pairs that duplicates held in the last scan
    /// run through this schedule, if that scan was
    /// [`Sampled`](ContentTwinPolicy::Sampled); `None` after an unsampled
    /// scan or a reduction that scanned nothing. With no pairs sampled it is
    /// 0.
    #[must_use]
    pub fn last_sample_yield(&self) -> Option<f64> {
        self.last_sample_yield
    }
}

/// The nodes a prune may remove that a diagram knows of without a walk
/// ([`Engine::unnamed_nodes`]): those no node of their parent level names,
/// and the root level's other than the output, with the pairs they hold.
/// None of them is reached from the output, and there are none exactly when
/// a prune removes nothing; the nodes reached only through them are not
/// counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Unnamed {
    /// The nodes.
    pub nodes: usize,
    /// Their pairs.
    pub pairs: usize,
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

    /// The nodes of `f` no node of their parent level names ([`Unnamed`]),
    /// where `f` keeps the levels that may hold one: after a conjunction
    /// without summing out, a prune or a contraction, and while no edit
    /// forgot them. `None` where it does not, on a diagram with a marginal
    /// level, on one whose output is not at the root, and where the read
    /// may read as many pairs as a prune's walk from the root, which a prune
    /// then takes.
    ///
    /// A read reads the parent levels of the levels not counted since they
    /// last changed, one unit of work a pair, and keeps each level's count
    /// for the next read, and the levels found holding such a node for the
    /// next prune or conjunction; a conjunction without summing out keeps
    /// the counts of the levels it takes from an operand with their parent.
    ///
    /// # Errors
    ///
    /// `OverBudget` if the scratch the read takes is refused, and the stop
    /// the read polls for; `f` is unchanged then.
    pub fn unnamed_nodes(&self, f: &mut Tdd) -> Result<Option<Unnamed>, OperationError> {
        let Some(loose) = f.dirty.loose() else {
            return Ok(None);
        };
        if f.is_zero() || f.is_one() || f.has_marginal_level() || !prune::below_root_walk_applies(f) {
            return Ok(None);
        }
        let uncounted: Vec<u32> = loose.entries().filter(|e| e.1.is_none()).map(|e| e.0).collect();
        if !uncounted.is_empty() {
            if prune::settle_reads_all(f, &uncounted) {
                return Ok(None);
            }
            let held = prune::count_unnamed(self, f, &uncounted)?;
            let loose = f.dirty.loose_levels_mut().expect("listed above");
            for (t, unnamed) in held {
                loose.insert_counted(t, Some(unnamed));
            }
            for t in uncounted {
                if loose.unnamed(t).is_none() {
                    loose.remove(t);
                }
            }
        }
        let mut unnamed = Unnamed::default();
        let root = &f.levels[f.vtree.root().idx()];
        for k in (0..root.node_count()).filter(|&k| k != f.output.local.idx()) {
            unnamed.nodes += 1;
            unnamed.pairs += root.pair_count_at(k);
        }
        for (_, held) in f.dirty.loose().expect("listed above").entries() {
            let held = held.expect("counted above");
            unnamed.nodes += held.nodes;
            unnamed.pairs += held.pairs;
        }
        Ok(Some(unnamed))
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
        if f.vtree.num_nodes() == 0 {
            return Ok(());
        }
        driver::Reduction::new(self, f).run(plan, scope)
    }

    /// End a negation of a canonical operand: prune below the root and
    /// certify the result canonical, with no contraction. `shrink` trims the
    /// level arrays as a full reduction does. `old` is the old output's slot
    /// where the complement was built beside it.
    pub(crate) fn certify_complement(&self, f: &mut Tdd, shrink: bool, old: Option<crate::diagram::NodeIdx>) -> Result<(), OperationError> {
        driver::Reduction::new(self, f).certify_complement(shrink, old)
    }
}

#[cfg(test)]
mod tests;
