//! Reduction order and the follow-up work required by each rewrite.

use crate::{Engine, Tdd, OperationError};
use crate::diagram::{NodeIdx, Pass, Weights};
use crate::vtree::{Vtree, VtreeIdx};
use super::{Close, ReductionPlan, ContentTwinPolicy, ContentTwinSchedule};
use super::prune::{PruneScope, below_root_walk_applies, prune_complement, prune_unreachable};
use super::contract::{contract_all_twins_known, contract_leaf::contract_leaf_twins};
use super::contract::content_twin::merge_content_equal_nodes;
use super::contract::pair_fusion::fuse_pairs_at_parents;
use super::slot_prune::prune_value_slots;

const CONTENT_SCAN_MAX_NODES: u64 = 131_072;
/// A scan that removed a share `y` of the nodes sets the next one at
/// `1 + SCAN_GROWTH_PER_YIELD / y` times its input size, clamped to
/// `SCAN_GROWTH_MIN..=SCAN_GROWTH_MAX`. Fitted on model-counting runs:
/// a later scan that saves less than it costs followed a first scan with
/// a 19% yield, a paying one a 22% yield.
const SCAN_GROWTH_PER_YIELD: f64 = 0.27;
const SCAN_GROWTH_MIN: f64 = 1.5;
const SCAN_GROWTH_MAX: f64 = 4.0;

/// The driver consumes rewrite reports; kernels own their scratch and local
/// worklists. A content merge needs pruning and contraction, a leaf rewrite
/// needs contraction on a diagram with a summed-out level, and equal value
/// slots can require a content rescan.
/// The sibling fixed point stays inside contraction's top-down sweep.
pub(super) struct Reduction<'a> {
    eng: &'a Engine,
    tdd: &'a mut Tdd,
    /// The seed of a closing reduction; `None` in a plain one.
    seed: Option<&'a [VtreeIdx]>,
    /// The leaves leaf contraction found twins at in a closing reduction,
    /// their parents' levels left as they were for the close to drop.
    recorded: Vec<VtreeIdx>,
    /// The levels' first marginal level, read at most once: no pass changes
    /// a level's kind, so it holds across them all, and each read scans
    /// every level.
    first_marginal: Option<Option<VtreeIdx>>,
    /// The variables some pair names true at entry.
    #[cfg(debug_assertions)]
    named_one: Vec<crate::vtree::VarId>,
}

impl<'a> Reduction<'a> {
    /// The passes mark the levels they change, so a count kept at the close
    /// before them holds on a level they did not mark.
    pub(super) fn new(eng: &'a Engine, tdd: &'a mut Tdd, close: Close<'a>) -> Self {
        let seed = match close {
            Close::Plain => None,
            Close::Settle(seed) => Some(seed),
        };
        tdd.levels.hold_counts(tdd.levels.is_closed());
        let first_marginal = tdd.levels.known_first_marginal();
        Self {
            eng,
            tdd,
            seed,
            recorded: Vec::new(),
            first_marginal,
            #[cfg(debug_assertions)]
            named_one: Vec::new(),
        }
    }

    pub(super) fn run(&mut self, plan: ReductionPlan<'_>, scope: PruneScope) -> Result<(), OperationError> {
        #[cfg(debug_assertions)]
        if self.seed.is_some() {
            self.named_one = oracle::named_one(self.tdd);
        }
        let vtree = std::sync::Arc::clone(&self.tdd.vtree);
        let done = self.run_passes(plan, scope);
        // A pass that left a level's live pairs for the close at the end
        // (`TddLevel::leave_live_pairs`) may be followed by one refused
        // before that close.
        if done.is_err() {
            self.tdd.levels.forget_left_pairs();
            if !self.recorded.is_empty() {
                let vtree = &self.tdd.vtree;
                let parents: Vec<u32> = self.recorded.iter().map(|&leaf| vtree.node(leaf).parent().expect("a recorded leaf has a parent").0).collect();
                self.tdd.dirty.requeue(Pass::LeafContract, parents);
            }
        } else if let Some(first) = self.first_marginal {
            // A settle that trims the vtree rebuilds the levels, which moves
            // the first marginal level's index; with none there is none after.
            if first.is_none() || std::sync::Arc::ptr_eq(&vtree, &self.tdd.vtree) {
                self.tdd.levels.keep_first_marginal(first);
            }
        }
        done
    }

    fn run_passes(&mut self, plan: ReductionPlan<'_>, scope: PruneScope) -> Result<(), OperationError> {
        let _op = self.eng.limits().enter()?;
        // A certified diagram with nothing owed is at the fixpoint of every
        // pass below, so there is nothing to run.
        if self.tdd.levels.is_canonical(self.tdd.output)
            && self.tdd.dirty.is_empty()
            && self.seed.is_none_or(<[VtreeIdx]>::is_empty)
        {
            return Ok(());
        }
        let policy = match plan {
            ReductionPlan::Contract => {
                // Read here, not lazily by the sweep, whose first merge
                // forgets it: the result keeps it for the next reduction.
                // Read here, not lazily by the sweep, whose first merge
                // forgets it: the result keeps it for the next reduction.
                let marginal = self.marginal();
                contract_all_twins_known(self.eng, self.tdd, Some(marginal))?;
                self.tdd.close_marked_levels();
                return self.settle();
            }
            ReductionPlan::Prune => {
                prune_unreachable(self.eng, self.tdd, scope)?;
                if self.marginal() {
                    prune_value_slots(self.eng, self.tdd)?;
                }
                self.tdd.close_marked_levels();
                return self.settle();
            }
            ReductionPlan::Full(policy) => {
                // Read before the passes, which forget it: every full
                // reduction reads it below.
                self.marginal();
                policy
            }
        };
        #[cfg(debug_assertions)]
        let marginal_before: Vec<_> = self.tdd.levels.iter().map(|l| l.is_marginal()).collect();
        // With no level loose under a root holding the output alone, every
        // node is reachable: a walk would read nothing, yet be charged as one
        // from the output.
        let pruned = !self.tdd.is_zero()
            && self.tdd.dirty.loose().is_some_and(|loose| loose.is_empty())
            && below_root_walk_applies(self.tdd)
            && !self.marginal()
            && self.tdd.levels[self.tdd.vtree.root().idx()].slot_count() == 1;
        if !pruned {
            prune_unreachable(self.eng, self.tdd, scope)?;
        }
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
        if !scanned && self.marginal() {
            self.prune_slots()?;
        }
        let structural = !self.marginal();
        for level in &mut self.tdd.levels {
            level.shrink_arrays();
        }
        debug_assert_eq!(structural, !self.tdd.has_marginal_level());
        // The passes above are the fixpoint of everything a worklist can ask
        // for. What the lists still hold is what the passes' own rewrites
        // pushed after the sweep had taken its list, plus content-twin
        // entries that only the scan reads.
        self.finish(structural)
    }

    /// Close the levels the passes marked, settle a closing reduction, clear
    /// the worklists, and on a `structural` diagram list no level loose and
    /// certify it: the prune left every node reachable, and a contraction
    /// only merges twins, whose children are the same.
    fn finish(&mut self, structural: bool) -> Result<(), OperationError> {
        self.tdd.close_marked_levels();
        let before = std::sync::Arc::clone(&self.tdd.vtree);
        self.settle()?;
        if !structural && !std::sync::Arc::ptr_eq(&before, &self.tdd.vtree) && self.tdd.has_marginal_level() {
            self.passes_after_trim()?;
        }
        if self.tdd.vtree.num_nodes() == 0 {
            return Ok(());
        }
        if structural {
            #[cfg(debug_assertions)]
            super::prune::debug_assert_all_reached(self.tdd);
            self.tdd.dirty.set_loose(Some(Vec::new()));
        }
        self.tdd.dirty.clear_lists();
        if structural {
            self.tdd.levels.certify(self.tdd.output);
            #[cfg(debug_assertions)]
            super::contract::contract_leaf::assert_no_full_twin_side(self.tdd);
        }
        Ok(())
    }

    /// Run the passes again, every level owed, on a summed diagram the close
    /// trimmed onto a smaller projection: the trim can leave a node two pairs
    /// over the same children, a fusion group no pass has seen, and the
    /// fusion can fill a region, which the close of that run drops.
    #[cold]
    #[inline(never)]
    fn passes_after_trim(&mut self) -> Result<(), OperationError> {
        let vtree = std::sync::Arc::clone(&self.tdd.vtree);
        let internal = vtree.internal_bottomup_slice().iter().copied();
        self.tdd.dirty = crate::diagram::Dirty::default().seeded(&vtree, internal, Some(self.eng))?;
        self.eng.reduce(self.tdd, ReductionPlan::default())
    }

    /// Drop the recorded leaves and the seed's in a closing reduction
    /// ([`super::trim::settle`]).
    fn settle(&mut self) -> Result<(), OperationError> {
        let Some(seed) = self.seed else { return Ok(()) };
        if !seed.is_empty() {
            self.eng.limits().reserve_exact(&mut self.recorded, seed.len())?;
            self.recorded.extend_from_slice(seed);
        }
        #[cfg(debug_assertions)]
        let dropped = oracle::at_entry(self.tdd, &self.recorded, seed, &self.named_one);
        super::trim::settle(self.eng, self.tdd, &mut self.recorded, self.first_marginal.map(|first| first.is_some()))?;
        #[cfg(debug_assertions)]
        oracle::at_exit(self.tdd, dropped.as_deref(), &self.named_one);
        Ok(())
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
    /// nodes ([`prune_complement`]). The level arrays are then trimmed as a
    /// full reduction trims them.
    pub(super) fn certify_complement(&mut self, old: Option<NodeIdx>) -> Result<(), OperationError> {
        let _op = self.eng.limits().enter()?;
        match old {
            Some(old) if below_root_walk_applies(self.tdd) && !self.marginal() && old != self.tdd.output.local => {
                prune_complement(self.eng, self.tdd, old)?;
            }
            _ => prune_unreachable(self.eng, self.tdd, PruneScope::BelowRoot)?,
        }
        for level in &mut self.tdd.levels {
            level.shrink_arrays();
        }
        self.finish(true)
    }

    fn marginal(&mut self) -> bool {
        let levels = &self.tdd.levels;
        self.first_marginal.get_or_insert_with(|| levels.first_marginal()).is_some()
    }

    /// A closing reduction on a home with no summed-out level records the
    /// leaf sides in place of writing them.
    fn contract(&mut self) -> Result<(), OperationError> {
        let marginal = self.marginal();
        contract_all_twins_known(self.eng, self.tdd, Some(marginal))?;
        let record = self.seed.is_some() && !marginal && self.tdd.vtree.on_home();
        // Leaf contraction leaves no inner twins unless a level is summed out.
        if contract_leaf_twins(self.eng, self.tdd, record.then_some(&mut self.recorded))? && marginal {
            contract_all_twins_known(self.eng, self.tdd, Some(marginal))?;
        }
        Ok(())
    }

    /// The slot prune, and the passes each value merge owes at the merged
    /// level's parent, until a prune merges no values: the merge repeats
    /// references there, which can make fusion groups and leaf twins at the
    /// parent and twins at its other child. Each merged level is also listed
    /// for the content-twin scan.
    fn prune_slots(&mut self) -> Result<(), OperationError> {
        loop {
            let merged = prune_value_slots(self.eng, self.tdd)?;
            if merged.is_empty() {
                return Ok(());
            }
            let vtree = std::sync::Arc::clone(&self.tdd.vtree);
            let parents = merged.iter().filter_map(|&v| vtree.node(VtreeIdx(v)).parent()).map(|p| p.0);
            for p in parents.clone() {
                self.tdd.dirty.forget_fused(p);
            }
            self.tdd.dirty.requeue(Pass::Contract, parents.clone());
            self.tdd.dirty.requeue(Pass::LeafContract, parents);
            self.tdd.dirty.requeue(Pass::ContentTwin, merged);
            self.contract()?;
        }
    }

    /// Always scan small or weighted marginal diagrams. Large unweighted
    /// diagrams use the schedule's growth threshold, which each scan sets
    /// from its yield. Returns whether a scan, stopped after its sample or
    /// not, ran.
    fn scan_if_due(&mut self, schedule: &mut ContentTwinSchedule, sample: Option<(f64, usize)>) -> Result<bool, OperationError> {
        schedule.last_sample_yield = None;
        if !self.marginal() { return Ok(false); }
        let before: u64 = self.tdd.levels.iter().map(|l| l.nodes().len() as u64).sum();
        if self.tdd.weight_table().is_some() || before <= CONTENT_SCAN_MAX_NODES || before >= schedule.next_scan_at_nodes {
            schedule.last_sample_yield = self.content_twins(sample)?;
            let after: u64 = self.tdd.levels.iter().map(|l| l.nodes().len() as u64).sum();
            schedule.next_scan_at_nodes = if after > CONTENT_SCAN_MAX_NODES {
                let removed = before.saturating_sub(after).max(1) as f64;
                let growth = (1.0 + SCAN_GROWTH_PER_YIELD * before as f64 / removed).clamp(SCAN_GROWTH_MIN, SCAN_GROWTH_MAX);
                (before as f64 * growth) as u64
            } else { 0 };
            return Ok(true);
        }
        Ok(false)
    }

    /// Each productive round removes at least one node. Its rewrites mark the
    /// levels to revisit; only the first round scans the whole diagram, so
    /// what the slot prune before it marks is not kept. With `sample`'s
    /// `(min_yield, min_pairs)` on a diagram of at least `min_pairs` pairs
    /// the first round samples its yield ([`ContentTwinPolicy::Sampled`]);
    /// a round that stops there is the last. Returns the sample's yield.
    // Inlined into `scan_if_due` with link-time optimization, its merge
    // loop compiled to about 3% more instructions on a pedigree network.
    #[inline(never)]
    pub(super) fn content_twins(&mut self, sample: Option<(f64, usize)>) -> Result<Option<f64>, OperationError> {
        self.prune_slots()?;
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
            self.prune_slots()?;
            if min_yield.zip(found).is_some_and(|(min, found)| found < min) { break; }
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

/// The debug checks of a closing reduction, over every pair of every node.
#[cfg(debug_assertions)]
mod oracle {
    use crate::Tdd;
    use crate::diagram::{ChildSide, ONE_LEAF_IDX};
    use crate::reduce::contract::contract_leaf::is_full_twin_side;
    use crate::vtree::{VarId, VtreeIdx, VtreeNode};

    fn names_one(tdd: &Tdd, parent: VtreeIdx, side: ChildSide) -> (bool, bool) {
        let level = &tdd.levels[parent.idx()];
        let (mut some, mut all) = (false, true);
        for i in 0..level.slot_count() {
            for p in level.pairs_iter_of_idx(i) {
                let one = side.pick(p.left, p.right).raw() == ONE_LEAF_IDX.0;
                (some, all) = (some || one, all && one);
            }
        }
        (some, all)
    }

    /// The variables of the leaves some pair names true.
    pub(super) fn named_one(tdd: &Tdd) -> Vec<VarId> {
        let vtree = &tdd.vtree;
        let mut vars = Vec::new();
        for t in vtree.bottomup() {
            let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
            if tdd.levels[t.idx()].is_marginal() {
                continue;
            }
            for (c, side) in [(left, ChildSide::Left), (right, ChildSide::Right)] {
                if vtree.node(c).is_leaf() && !tdd.levels[c.idx()].is_marginal() && names_one(tdd, t, side).0 {
                    vars.push(vtree.leaf_var(c));
                }
            }
        }
        vars.sort_unstable();
        vars
    }

    /// Check the leaves `settle` is given, and return their variables when
    /// it will drop them.
    pub(super) fn at_entry(tdd: &Tdd, dropped: &[VtreeIdx], seed: &[VtreeIdx], at_reduce: &[VarId]) -> Option<Vec<VarId>> {
        let vtree = &tdd.vtree;
        if !vtree.on_home() || tdd.has_marginal_level() || vtree.num_nodes() == 0 {
            return None;
        }
        let parent = |leaf: VtreeIdx| {
            let p = vtree.node(leaf).parent().expect("a dropped leaf has a parent");
            (p, ChildSide::of(vtree, p, leaf))
        };
        for &leaf in &dropped[..dropped.len() - seed.len()] {
            let (p, side) = parent(leaf);
            assert!(is_full_twin_side(tdd, p, side), "recorded leaf {} is not a full twin side of level {}", leaf.0, p.0);
        }
        for &leaf in seed {
            let (p, side) = parent(leaf);
            assert!(names_one(tdd, p, side).1, "seed leaf {} is not named true by every pair of level {}", leaf.0, p.0);
        }
        let mut vars: Vec<VarId> = dropped.iter().map(|&leaf| vtree.leaf_var(leaf)).collect();
        vars.sort_unstable();
        vars.dedup();
        if !tdd.is_zero() {
            let mut expected: Vec<VarId> = seed.iter().map(|&leaf| vtree.leaf_var(leaf)).chain(at_reduce.iter().copied()).collect();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(named_one(tdd), expected, "the leaves named true at the close are not the seed and those named at entry");
        }
        Some(vars)
    }

    pub(super) fn at_exit(tdd: &Tdd, dropped: Option<&[VarId]>, at_reduce: &[VarId]) {
        let Some(dropped) = dropped else { return };
        for var in named_one(tdd) {
            assert!(at_reduce.contains(&var), "the close names {var:?} true, which no pair named at entry");
            assert!(!dropped.contains(&var), "the close kept {var:?}, which it was to drop");
        }
    }
}

/// Restore marginal boundary invariants after a completed batch of level
/// folds. Marginalization creates pair-fusion work at the affected parents
/// and orphans slots. Bounded-precision arithmetic skips fusion because it
/// cannot sum exactly.
pub(crate) fn restore_marginal_invariants(eng: &Engine, tdd: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    if !tdd.weights().is_some_and(Weights::is_bounded) {
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
