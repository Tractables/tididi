//! Reduction order and the follow-up work required by each rewrite.

use crate::{Engine, Tdd, OperationError};
use crate::diagram::{NodeIdx, Pass, WeightStore};
use crate::vtree::{Vtree, VtreeIdx};
use super::{ReductionPlan, ContentTwinPolicy, ContentTwinSchedule};
use super::prune::{PruneScope, below_root_walk_applies, prune_complement, prune_unreachable};
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
    /// The levels' first marginal level, once known: no pass changes a
    /// level's kind, so it holds across them all.
    first_marginal: Option<Option<VtreeIdx>>,
}

impl<'a> Reduction<'a> {
    /// The passes mark the levels they change, so a count kept at the close
    /// before them holds on a level they did not mark.
    pub(super) fn new(eng: &'a Engine, tdd: &'a mut Tdd) -> Self {
        tdd.levels.hold_counts(tdd.levels.is_closed());
        let first_marginal = tdd.levels.known_first_marginal();
        Self { eng, tdd, first_marginal }
    }

    fn has_marginal_level(&mut self) -> bool {
        let levels = &self.tdd.levels;
        self.first_marginal.get_or_insert_with(|| levels.first_marginal()).is_some()
    }

    /// [`contract_all_twins`] with the first marginal level known to it.
    fn contract_all_twins(&mut self) -> Result<(), OperationError> {
        if let Some(first) = self.first_marginal {
            self.tdd.levels.keep_first_marginal(first);
        }
        contract_all_twins(self.eng, self.tdd)
    }

    pub(super) fn run(&mut self, plan: ReductionPlan<'_>, scope: PruneScope) -> Result<(), OperationError> {
        let done = self.run_passes(plan, scope);
        // A pass that left a level's live pairs for the close at the end
        // (`TddLevel::leave_live_pairs`) may be followed by one refused
        // before that close.
        if done.is_err() {
            self.tdd.levels.forget_left_pairs();
        } else if let Some(first) = self.first_marginal {
            self.tdd.levels.keep_first_marginal(first);
        }
        done
    }

    fn run_passes(&mut self, plan: ReductionPlan<'_>, scope: PruneScope) -> Result<(), OperationError> {
        let _op = self.eng.limits().enter()?;
        let whole = matches!(scope, PruneScope::Whole);
        let policy = match plan {
            ReductionPlan::Contract => {
                // Read here, not lazily by the sweep, whose first merge
                // forgets it: the result keeps it for the next reduction.
                self.has_marginal_level();
                self.contract_all_twins()?;
                self.tdd.close_marked_levels();
                return Ok(());
            }
            ReductionPlan::Prune => {
                prune_unreachable(self.eng, self.tdd, scope)?;
                prune_value_slots(self.eng, self.tdd)?;
                self.tdd.close_marked_levels();
                return Ok(());
            }
            ReductionPlan::Full(policy) => {
                // A certified diagram with nothing owed is at the fixpoint of
                // every pass below, so there is nothing to run.
                if self.tdd.levels.is_canonical(self.tdd.output) && self.tdd.dirty.is_empty() {
                    return Ok(());
                }
                // Read before the passes, which forget it: every full
                // reduction reads it below.
                self.has_marginal_level();
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
            ContentTwinPolicy::Fresh => self.scan_if_due(&mut ContentTwinSchedule::default(), None)?,
            ContentTwinPolicy::Adaptive(schedule) => self.scan_if_due(schedule, None)?,
            ContentTwinPolicy::Sampled { schedule, min_yield, min_pairs } => self.scan_if_due(schedule, Some((min_yield, min_pairs)))?,
        };
        // A content-twin scan ends with the slot prune. Without one, the slots
        // that prune and contraction orphaned at the boundary stores are still
        // there.
        if !scanned && self.has_marginal_level() {
            prune_value_slots(self.eng, self.tdd)?;
        }
        let mut structural = true;
        for level in &mut self.tdd.levels {
            structural &= !level.is_marginal();
            level.shrink_arrays();
        }
        // The passes above are the fixpoint of everything a worklist can ask
        // for. What the lists still hold is what the passes' own rewrites
        // pushed after the sweep had taken its list, plus content-twin
        // entries that only the scan reads.
        self.finish(structural, whole);
        Ok(())
    }

    /// Close the levels the passes marked, and on a `structural` diagram list
    /// no level loose: the prune left every node reachable, and a contraction
    /// only merges twins, whose children are the same. At the `fixpoint` of
    /// every pass, clear the worklists and certify a structural diagram.
    fn finish(&mut self, structural: bool, fixpoint: bool) {
        self.tdd.close_marked_levels();
        if structural {
            #[cfg(debug_assertions)]
            super::prune::debug_assert_all_reached(self.tdd);
            self.tdd.dirty.set_loose(Some(Vec::new()));
        }
        if fixpoint {
            for pass in [Pass::Contract, Pass::LeafContract, Pass::ContentTwin] {
                self.tdd.dirty.clear(pass);
            }
            if structural {
                self.tdd.levels.certify(self.tdd.output);
            }
        }
    }

    /// Prune a negation of a canonical diagram and certify the result, with no
    /// contraction.
    ///
    /// Negation completes every level with the cells no node covers, the
    /// class whose residual is false, and complements at the root, so the
    /// levels below the root keep their nodes. Complement maps residuals
    /// one to one: two nodes with equal residuals after it had equal
    /// residuals before, which a canonical operand rules out, inner and leaf
    /// twins alike. The one class per level whose residual was true dies at
    /// the root and is all the prune removes: with `old`, the old output's
    /// slot beside its complement, the prune reads and drops only those
    /// nodes ([`prune_complement`]). `shrink` trims the level arrays as a
    /// full reduction does.
    pub(super) fn certify_complement(&mut self, shrink: bool, old: Option<NodeIdx>) -> Result<(), OperationError> {
        let _op = self.eng.limits().enter()?;
        match old {
            Some(old) if below_root_walk_applies(self.tdd) && !self.tdd.has_marginal_level() && old != self.tdd.output.local => {
                prune_complement(self.eng, self.tdd, old)?;
            }
            _ => prune_unreachable(self.eng, self.tdd, PruneScope::BelowRoot)?,
        }
        if shrink {
            for level in &mut self.tdd.levels {
                level.shrink_arrays();
            }
        }
        self.finish(true, true);
        Ok(())
    }

    /// Leaf rewrites can create inner twins; clean passes drain empty worklists.
    fn contract(&mut self) -> Result<(), OperationError> {
        self.contract_all_twins()?;
        if contract_leaf_twins(self.eng, self.tdd)? {
            self.contract_all_twins()?;
        }
        Ok(())
    }

    /// Always scan small or weighted marginal diagrams. Large unweighted
    /// diagrams use the schedule's fourfold growth threshold. Returns whether
    /// a scan, stopped after its sample or not, ran.
    fn scan_if_due(&mut self, schedule: &mut ContentTwinSchedule, sample: Option<(f64, usize)>) -> Result<bool, OperationError> {
        schedule.last_sample_yield = None;
        if !self.has_marginal_level() { return Ok(false); }
        let before: u64 = self.tdd.levels.iter().map(|l| l.nodes().len() as u64).sum();
        if self.tdd.weights.is_some() || before <= CONTENT_SCAN_MAX_NODES || before >= schedule.next_scan_at_nodes {
            schedule.last_sample_yield = self.content_twins(sample)?;
            let after: u64 = self.tdd.levels.iter().map(|l| l.nodes().len() as u64).sum();
            schedule.next_scan_at_nodes = if after > CONTENT_SCAN_MAX_NODES { before.saturating_mul(4) } else { 0 };
            return Ok(true);
        }
        Ok(false)
    }

    /// Each productive round removes at least one node. Its rewrites mark the
    /// levels to revisit; only the first round scans the whole diagram, so
    /// what the slot prune before it reports is not kept. With `sample`'s
    /// `(min_yield, min_pairs)` on a diagram of at least `min_pairs` pairs
    /// the first round samples its yield ([`ContentTwinPolicy::Sampled`]);
    /// a round that stops there is the last. Returns the sample's yield.
    pub(super) fn content_twins(&mut self, sample: Option<(f64, usize)>) -> Result<Option<f64>, OperationError> {
        prune_value_slots(self.eng, self.tdd)?;
        let min_yield = sample.and_then(|(min_yield, min_pairs)| (self.tdd.pair_count() >= min_pairs).then_some(min_yield));
        self.tdd.dirty.clear(Pass::ContentTwin);
        let mut next: Option<rustc_hash::FxHashSet<u32>> = None;
        let mut sampled = None;
        loop {
            self.eng.limits().check_stop()?;
            if next.as_ref().is_some_and(|levels| levels.is_empty()) { break; }
            let first = next.is_none();
            let (merged, found) = merge_content_equal_nodes(self.eng, self.tdd, next.take(), min_yield.filter(|_| first))?;
            sampled = sampled.or(found);
            if merged == 0 { break; }
            prune_unreachable(self.eng, self.tdd, PruneScope::Whole)?;
            self.contract()?;
            // A value merge at a marginal level may make its parents' nodes
            // equal: those boundaries are scanned again.
            let merged = prune_value_slots(self.eng, self.tdd)?;
            if min_yield.zip(found).is_some_and(|(min, found)| found < min) { break; }
            self.tdd.dirty.requeue(Pass::ContentTwin, merged);
            next = Some(self.tdd.dirty.take(Pass::ContentTwin).into_iter().collect());
        }
        self.tdd.dirty.clear(Pass::ContentTwin);
        Ok(sampled)
    }

    #[cfg(debug_assertions)]
    fn assert_marginal_flags(&self, before: &[bool], pass: &str) {
        for (i, &was_marginal) in before.iter().enumerate() {
            assert!(!was_marginal || self.tdd.levels[i].is_marginal(),
                "vtree level {i} became structural during {pass}; marginalization is permanent");
        }
    }
}

/// Restore marginal boundary invariants after a completed batch of level
/// folds. Marginalization creates pair-fusion work at the affected parents
/// and orphans slots. Bounded-precision arithmetic skips fusion because it
/// cannot sum exactly.
pub(crate) fn restore_marginal_invariants(eng: &Engine, tdd: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    if !tdd.weights().is_some_and(WeightStore::is_bounded) {
        let mut parents: Vec<_> = levels.iter().filter_map(|&level| vtree.node(level).parent()).collect();
        parents.sort_unstable();
        parents.dedup();
        fuse_pairs_at_parents(eng, tdd, &parents)?;
        #[cfg(debug_assertions)]
        crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(tdd, Some(&parents), "marginalize_levels");
    }
    prune_value_slots(eng, tdd)?;
    tdd.close_marked_levels();
    Ok(())
}
