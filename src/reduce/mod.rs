//! Minimize diagrams or run selected reduction passes.
//!
//! [`Tdd::minimize`] uses the default [`ReductionPlan`]; [`Tdd::reduce`] accepts
//! a plan explicitly. Pruning removes unreachable nodes, twin contraction
//! merges nodes with the same parent context, and pair fusion combines
//! contributions at marginal boundaries.

mod prune;
pub(crate) mod contract;
pub(crate) mod slot_prune; // post-tagger marginal-slot compaction
mod content_twins;


use crate::limits::pool::Pool;

use self::contract::content_twin::ContentTwinScratch;
use self::contract::scratch::ContractScratch;
use crate::value::slots::RefSlotScratch;

/// Every buffer one engine's reductions reuse between calls.
#[derive(Default)]
pub(crate) struct ReduceScratch {
    /// `prune_unreachable`'s flat reachability/remap array.
    prune_remap: Pool<Vec<u32>>,
    /// `prune_unreachable`'s per-level offsets into `prune_remap`.
    prune_level_base: Pool<Vec<usize>>,
    /// `prune_value_slots`'s per-store slot bookkeeping.
    slot_prune_slots: Pool<RefSlotScratch>,
    /// `prune_value_slots`'s slot remap array.
    slot_prune_remap: Pool<Vec<u32>>,
    /// Twin contraction's working set.
    contract: Pool<ContractScratch>,
    /// Content-twin canonicalization's working set.
    content_twin: Pool<ContentTwinScratch>,
}

impl ReduceScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.prune_remap.drain();
        self.prune_level_base.drain();
        self.slot_prune_slots.drain();
        self.slot_prune_remap.drain();
        self.contract.drain();
        self.content_twin.drain();
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
    /// Carry the scan schedule across successive diagrams.
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
use self::contract::contract_leaf::contract_leaf_twins;
use self::contract::contract_all_twins;
use self::prune::prune_unreachable;
use crate::limits::OperationError;
use crate::diagram::Tdd;

/// Snapshot per-level `is_marginal` flags so a later
/// `assert_no_demarginalization` can detect a violation of invariant 5 (marginality is
/// permanent — see `test_helpers::check::marginal`) and name the offending pass.
#[cfg(debug_assertions)]
fn snapshot_marginal_flags(tdd: &Tdd) -> Vec<bool> {
    tdd.levels.iter().map(|l| l.is_marginal()).collect()
}

/// Assert no level present in `before` (as marginal) has become structural.
/// Panics naming the level and the `pass` that violated invariant 5.
#[cfg(debug_assertions)]
fn assert_no_demarginalization(tdd: &Tdd, before: &[bool], pass: &str) {
    for (i, &was_marginal) in before.iter().enumerate() {
        if was_marginal && !tdd.levels[i].is_marginal() {
            panic!(
                "vtree level {i} became structural during {pass}; marginalization is permanent"
            );
        }
    }
}

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
        let eng = self;
        let _op = eng.limits().begin_operation();
        let content_twins = match plan {
            ReductionPlan::Contract => return contract_all_twins(eng, f),
            ReductionPlan::Prune => {
                // Prune removes nodes, which can create twins in a shrunk level's
                // children; `prune_unreachable` seeds the contract worklists with
                // those levels so a later contraction pass covers them.
                prune_unreachable(eng, f)?;
                // Pairs the prune removed may have orphaned marginal count slots.
                crate::reduce::slot_prune::prune_value_slots(eng, f);
                return Ok(());
            }
            ReductionPlan::Full(policy) => policy,
        };

        // Invariant 5 guard: snapshot the marginal flags before the structural
        // passes so `assert_no_demarginalization` can name the offending pass.
        #[cfg(debug_assertions)]
        let marginal_before = snapshot_marginal_flags(f);

        prune_unreachable(eng, f)?;
        #[cfg(debug_assertions)]
        assert_no_demarginalization(f, &marginal_before, "prune");

        contract_twins_and_leaves(eng, f)?;
        #[cfg(debug_assertions)]
        assert_no_demarginalization(f, &marginal_before, "contract+leaf");

        // Content-twin scanning includes compaction of orphaned value slots.
        match content_twins {
            ContentTwinPolicy::Skip => {},
            ContentTwinPolicy::Fresh => content_twins::scan_if_due(eng, f, None)?,
            ContentTwinPolicy::Adaptive(probe) => content_twins::scan_if_due(eng, f, Some(probe))?,
        }

        // Release the doubling overshoot a rebuilt pair arena leaves behind.
        // `shrink_arrays` only acts when capacity exceeds 4x the length, so levels
        // without slack pay nothing.
        for level in &mut f.levels {
            level.shrink_arrays();
        }

        Ok(())
    }
}

/// Twin contraction, then leaf-twin contraction, then twin contraction again
/// if the leaf pass fired. Both passes drain a worklist, so a clean diagram
/// costs one empty check each.
fn contract_twins_and_leaves(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    contract_all_twins(eng, tdd)?;
    // Leaf labels are implicit indices, not stored nodes, so inner-node twin
    // contraction cannot reach them; the leaf rewrite can mint inner twins.
    if contract_leaf_twins(eng, tdd)? {
        contract_all_twins(eng, tdd)?;
    }
    Ok(())
}


#[cfg(test)]
mod tests;
