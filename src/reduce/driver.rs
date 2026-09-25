//! Reduction order and the follow-up work required by each rewrite.

use crate::{Engine, Tdd, OperationError};
use crate::diagram::{Pass, WeightStore};
use crate::vtree::{Vtree, VtreeIdx};
use super::{ReductionPlan, ContentTwinPolicy, ContentTwinSchedule};
use super::prune::{PruneScope, prune_unreachable};
use super::contract::{contract_all_twins, contract_leaf::contract_leaf_twins};
use super::contract::content_twin::merge_content_equal_nodes;
use super::contract::pair_fusion::fuse_pairs_at_parents;
use super::slot_prune::prune_value_slots;

const CONTENT_SCAN_MAX_NODES: u64 = 131_072;

/// The driver consumes rewrite reports; kernels own their scratch and local
/// worklists. A content merge needs pruning and contraction, a leaf rewrite
/// needs contraction, and equal value slots can require a content rescan.
/// The sibling fixed point stays inside contraction's top-down sweep.
pub(super) struct Reduction<'a> {
    eng: &'a Engine,
    tdd: &'a mut Tdd,
}

impl<'a> Reduction<'a> {
    pub(super) fn new(eng: &'a Engine, tdd: &'a mut Tdd) -> Self { Self { eng, tdd } }

    pub(super) fn run(&mut self, plan: ReductionPlan<'_>, scope: PruneScope) -> Result<(), OperationError> {
        let _op = self.eng.limits().enter()?;
        let whole = matches!(scope, PruneScope::Whole);
        let policy = match plan {
            ReductionPlan::Contract => return contract_all_twins(self.eng, self.tdd),
            ReductionPlan::Prune => {
                prune_unreachable(self.eng, self.tdd, scope)?;
                prune_value_slots(self.eng, self.tdd);
                return Ok(());
            }
            ReductionPlan::Full(policy) => {
                // A certified diagram with nothing owed is at the fixpoint of
                // every pass below, so there is nothing to run.
                if self.tdd.levels.is_canonical(self.tdd.output) && self.tdd.dirty.is_empty() {
                    return Ok(());
                }
                policy
            }
        };
        #[cfg(debug_assertions)]
        let marginal_before: Vec<_> = self.tdd.levels.iter().map(|l| l.is_marginal()).collect();
        prune_unreachable(self.eng, self.tdd, scope)?;
        #[cfg(debug_assertions)]
        self.assert_marginal_flags(&marginal_before, "prune");
        self.contract()?;
        #[cfg(debug_assertions)]
        self.assert_marginal_flags(&marginal_before, "contract+leaf");
        let scanned = match policy {
            ContentTwinPolicy::Skip => false,
            ContentTwinPolicy::Fresh => self.scan_if_due(&mut ContentTwinSchedule::default())?,
            ContentTwinPolicy::Adaptive(schedule) => self.scan_if_due(schedule)?,
        };
        // A content-twin scan ends with the slot prune. Without one, the slots
        // that prune and contraction orphaned at the boundary stores are still
        // there.
        if !scanned && self.tdd.has_marginal_level() {
            prune_value_slots(self.eng, self.tdd);
        }
        let mut structural = true;
        for level in &mut self.tdd.levels {
            structural &= !level.is_marginal();
            level.shrink_arrays();
        }
        if whole {
            // The passes above are the fixpoint of everything a worklist can
            // ask for. What the lists still hold is what the passes' own
            // rewrites pushed after the sweep had taken its list, plus
            // content-twin entries that only the scan reads.
            for pass in [Pass::Contract, Pass::LeafContract, Pass::ContentTwin] {
                self.tdd.dirty.clear(pass);
            }
            if structural {
                self.tdd.levels.certify(self.tdd.output);
            }
        }
        Ok(())
    }

    /// Leaf rewrites can create inner twins; clean passes drain empty worklists.
    fn contract(&mut self) -> Result<(), OperationError> {
        contract_all_twins(self.eng, self.tdd)?;
        if contract_leaf_twins(self.eng, self.tdd)? {
            contract_all_twins(self.eng, self.tdd)?;
        }
        Ok(())
    }

    /// Always scan small or weighted marginal diagrams. Large unweighted
    /// diagrams use the schedule's fourfold growth threshold. Returns whether
    /// a scan ran.
    fn scan_if_due(&mut self, schedule: &mut ContentTwinSchedule) -> Result<bool, OperationError> {
        if !self.tdd.has_marginal_level() { return Ok(false); }
        let before: u64 = self.tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
        if self.tdd.weights.is_some() || before <= CONTENT_SCAN_MAX_NODES || before >= schedule.next_scan_at_nodes {
            self.content_twins()?;
            let after: u64 = self.tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
            schedule.next_scan_at_nodes = if after > CONTENT_SCAN_MAX_NODES { before.saturating_mul(4) } else { 0 };
            return Ok(true);
        }
        Ok(false)
    }

    /// Slot compaction reports value merges at marginal levels. They may
    /// make their parents' nodes equal, so include those boundaries in a rescan.
    fn compact_for_rescan(&mut self) {
        let merged = prune_value_slots(self.eng, self.tdd);
        self.tdd.dirty.requeue(Pass::ContentTwin, merged);
    }

    /// Each productive round removes at least one node. Its rewrites mark the
    /// levels to revisit; only the first round scans the whole diagram, so
    /// what the slot prune before it reports is not kept.
    pub(super) fn content_twins(&mut self) -> Result<(), OperationError> {
        prune_value_slots(self.eng, self.tdd);
        self.tdd.dirty.clear(Pass::ContentTwin);
        let mut next: Option<rustc_hash::FxHashSet<u32>> = None;
        loop {
            self.eng.limits().check_stop()?;
            if next.as_ref().is_some_and(|levels| levels.is_empty()) { break; }
            let merged = merge_content_equal_nodes(self.eng, self.tdd, next.take())?;
            if merged == 0 { break; }
            prune_unreachable(self.eng, self.tdd, PruneScope::Whole)?;
            self.contract()?;
            self.compact_for_rescan();
            next = Some(self.tdd.dirty.take(Pass::ContentTwin).into_iter().collect());
        }
        self.tdd.dirty.clear(Pass::ContentTwin);
        Ok(())
    }

    /// Marginalization creates pair-fusion work at the affected parents and
    /// orphans slots. Log arithmetic skips fusion because it cannot sum exactly.
    fn after_marginalize(&mut self, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
        if !self.tdd.weights().is_some_and(WeightStore::is_log) {
            let mut parents: Vec<_> = levels.iter().filter_map(|&level| vtree.node(level).parent()).collect();
            parents.sort_unstable();
            parents.dedup();
            fuse_pairs_at_parents(self.eng, self.tdd, &parents)?;
            #[cfg(debug_assertions)]
            crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(self.tdd, Some(&parents), "marginalize_levels");
        }
        prune_value_slots(self.eng, self.tdd);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn assert_marginal_flags(&self, before: &[bool], pass: &str) {
        for (i, &was_marginal) in before.iter().enumerate() {
            assert!(!was_marginal || self.tdd.levels[i].is_marginal(),
                "vtree level {i} became structural during {pass}; marginalization is permanent");
        }
    }
}

/// Restore marginal boundary invariants after a completed batch of level folds.
pub(crate) fn restore_marginal_invariants(eng: &Engine, tdd: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    Reduction::new(eng, tdd).after_marginalize(levels, vtree)
}
