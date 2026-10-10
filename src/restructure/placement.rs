//! Place levels and connect their roots after an operation validates its layout.
//!
//! Two assemblers own the destination storage while an operation fills it.
//! [`CopyPlacement`] copies structural levels out of a source that stays as it
//! is, the way an embedding does; [`MovePlacement`] moves whole parts, levels
//! and weight columns alike, the way a graft does. Both keep local node
//! indices through the transfer and prune what the new joins leave unused.

use std::sync::Arc;

use crate::{Engine, OperationError};
use crate::diagram::{
    Assembly, ChildPair, ChildSide, Dirty, LevelView, MarginalStorage, NodeIdx, Tdd, TddBuildError,
    TddLevel, TddNodeId, WeightStore, WeightValue, LEAF_WIDTH, ONE_LEAF_IDX, try_take_levels,
};
use crate::execution::pool::PoolGuard;
use crate::vtree::{Vtree, VtreeIdx};

/// Drop the nodes the joins made in `result` that nothing refers to.
fn prune(eng: &Engine, result: &mut Tdd) -> Result<(), OperationError> {
    eng.reduce(result, crate::reduce::ReductionPlan::Prune)
}

/// The true reference for an unconstrained child already placed bottom-up.
#[inline]
fn true_node(vtree: &Vtree, child: VtreeIdx) -> NodeIdx {
    if vtree.node(child).is_leaf() { ONE_LEAF_IDX } else { NodeIdx(0) }
}

/// Make `level` the level of internal node `at` in a diagram constant true
/// under it: one node, true on both sides. `level` is empty.
#[inline]
pub(crate) fn push_free_level(level: &mut TddLevel, vtree: &Vtree, at: VtreeIdx) {
    let (left, right) = vtree.children(at);
    level.push_internal_node(&[ChildPair::new(true_node(vtree, left), true_node(vtree, right))]);
}

/// The joins that lift every reference of one child of `at` through a free
/// sibling: one node per slot of the carried child, at the same index.
struct PassThrough {
    carries: VtreeIdx,
    carries_leaf: bool,
    one: NodeIdx,
    width: usize,
    free_side: ChildSide,
}

impl PassThrough {
    fn new(vtree: &Vtree, assembly: &Assembly<'_>, at: VtreeIdx, free_side: ChildSide) -> Self {
        let (left, right) = vtree.children(at);
        let (free, carries) = if free_side == ChildSide::Left { (left, right) } else { (right, left) };
        let carries_leaf = vtree.node(carries).is_leaf();
        let width = if carries_leaf { LEAF_WIDTH } else { assembly.level(carries).slot_count() };
        Self { carries, carries_leaf, one: true_node(vtree, free), width, free_side }
    }

    /// The `(left, right)` children of each join, in slot order.
    fn pairs(&self) -> impl Iterator<Item = (NodeIdx, NodeIdx)> + '_ {
        (0..self.width).map(move |i| {
            let child = NodeIdx(i as u32);
            if self.free_side == ChildSide::Left { (self.one, child) } else { (child, self.one) }
        })
    }
}

/// The levels a placement built or changed, bottom-up, each once: the only
/// ones its seat closes when every level it copied or moved was closed in
/// its source. The list is the engine's, checked out for the placement.
struct Changed<'a> {
    levels: PoolGuard<'a, Vec<VtreeIdx>>,
    sources_closed: bool,
}

impl<'a> Changed<'a> {
    fn new(eng: &'a Engine, vtree: &Vtree) -> Result<Self, OperationError> {
        let mut levels = eng.scratch.placed.checkout(eng);
        eng.limits().reserve_exact(&mut levels, vtree.num_nodes())?;
        Ok(Self { levels, sources_closed: true })
    }

    /// Record level `t`, built or changed in place; a placement works a
    /// level at a time, so a repeat follows its first record.
    #[inline]
    fn push(&mut self, t: VtreeIdx) {
        if self.levels.last() != Some(&t) {
            self.levels.push(t);
        }
    }

    /// Record a level taken from `source` unchanged.
    #[inline]
    fn take_from(&mut self, source: &Tdd) {
        self.sources_closed &= source.levels.is_closed();
    }

    /// What the seat closes: the recorded levels, or every level when one
    /// taken unchanged was not known closed.
    fn to_close(&self) -> Option<&[VtreeIdx]> {
        self.sources_closed.then_some(&self.levels[..])
    }
}

/// Bounded structural copies into a fresh destination, under the engine's
/// limits.
pub(super) struct CopyPlacement<'a> {
    eng: &'a Engine,
    vtree: &'a Arc<Vtree>,
    assembly: Assembly<'a>,
    prune: bool,
    changed: Changed<'a>,
}

impl<'a> CopyPlacement<'a> {
    pub(super) fn new(eng: &'a Engine, vtree: &'a Arc<Vtree>) -> Result<Self, OperationError> {
        let changed = Changed::new(eng, vtree)?;
        Ok(Self { eng, vtree, assembly: Assembly::new(eng, vtree)?, prune: false, changed })
    }

    /// Copy a structural level without its source's weight configuration.
    #[inline]
    pub(super) fn copy_level(&mut self, source: &Tdd, from: VtreeIdx, to: VtreeIdx) -> Result<(), OperationError> {
        let view = LevelView::unweighted(source.level(from))
            .expect("embedding requires structural levels");
        self.changed.take_from(source);
        self.assembly.replace_level(self.eng, to, view)
    }

    /// [`copy_level`](Self::copy_level) onto a destination node whose children
    /// are the source node's, swapped: every pair is read the other way round.
    pub(super) fn copy_level_mirrored(&mut self, source: &Tdd, from: VtreeIdx, to: VtreeIdx) -> Result<(), OperationError> {
        self.copy_level(source, from, to)?;
        let (levels, _) = self.assembly.parts_mut();
        levels[to.idx()].swap_sides();
        self.changed.push(to);
        Ok(())
    }

    /// The true reference for an unconstrained child already placed bottom-up.
    #[inline]
    pub(super) fn true_node(&self, child: VtreeIdx) -> NodeIdx {
        true_node(self.vtree, child)
    }

    /// The slots placed at `at` so far.
    #[inline]
    pub(super) fn slot_count(&self, at: VtreeIdx) -> usize {
        self.assembly.level(at).slot_count()
    }

    /// Lift all references from one child through a join with a free sibling.
    #[inline]
    pub(super) fn pass_through(&mut self, at: VtreeIdx, free_side: ChildSide) -> Result<(), OperationError> {
        let through = PassThrough::new(self.vtree, &self.assembly, at, free_side);
        // A leaf contributes all three implicit labels. Prune any wrappers the
        // copied parent does not use; keep their indices until then.
        self.prune |= through.carries_leaf;
        for (left, right) in through.pairs() {
            self.join(at, left, right)?;
        }
        Ok(())
    }

    /// Add a connecting node.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> Result<NodeIdx, OperationError> {
        self.changed.push(at);
        self.assembly.push(self.eng, at, &[ChildPair::new(left, right)])
    }

    /// Seat the chosen root reference and drop what the joins left unused.
    ///
    /// Copies of a valid diagram's levels joined through true nodes are valid
    /// storage by construction, so only debug builds check them.
    pub(super) fn finish(self, local: NodeIdx) -> Result<Tdd, OperationError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let mut result = self.assembly.finish_asserted(output, self.changed.to_close())?;
        if self.prune {
            prune(self.eng, &mut result)?;
        }
        Ok(result)
    }
}

/// Whole parts moved into a destination, levels and weight columns alike,
/// under graft assembly's untracked contract.
pub(super) struct MovePlacement<'a> {
    eng: &'a Engine,
    vtree: &'a Arc<Vtree>,
    assembly: Assembly<'a>,
    /// Whether a join sits over a marginal level, which is what leaves
    /// references to repair.
    prune: bool,
    /// Whether a pass-through sits over a marginal level. Its nodes are told
    /// apart by value slots alone, and slots the prune merges leave twins
    /// behind, which a contraction after it removes.
    contract: bool,
    changed: Changed<'a>,
}

impl<'a> MovePlacement<'a> {
    pub(super) fn new(eng: &'a Engine, vtree: &'a Arc<Vtree>, weights: Option<WeightStore>) -> Result<Self, OperationError> {
        let changed = Changed::new(eng, vtree)?;
        let levels = try_take_levels(eng, vtree.num_nodes())?;
        let assembly = Assembly::from_levels(eng, Arc::clone(vtree), levels, weights);
        Ok(Self { eng, vtree, assembly, prune: false, contract: false, changed })
    }

    /// Move every level and its weighted column through a checked placement map.
    /// The caller has checked that stored values retain their interpretation.
    pub(super) fn move_part(&mut self, source: &mut Tdd, map: &[VtreeIdx]) {
        debug_assert_eq!(source.vtree().num_nodes(), map.len());
        self.changed.take_from(source);
        let (levels, weights) = self.assembly.parts_mut();
        // One mutable access to the source's levels for the whole move: each
        // access forgets what the storage knows of them.
        let source_levels = &mut source.levels[..];
        for (from, &to) in map.iter().enumerate() {
            MarginalStorage::new(&mut levels[to.idx()], weights.as_mut(), to.idx())
                .move_from(&mut source_levels[from], source.weights.as_mut(), from);
        }
    }

    /// Exchange the two sides of every pair of each moved level `plan`
    /// mirrors ([`TddLevel::swap_sides`]): the level of the same functions
    /// over its image, whose children are the source node's swapped. Its own
    /// inverse, so a refused placement undoes it before
    /// [`move_back`](Self::move_back).
    ///
    /// [`TddLevel::swap_sides`]: crate::diagram::TddLevel
    pub(super) fn mirror(&mut self, plan: &crate::restructure::embed::Plan) {
        let (levels, _) = self.assembly.parts_mut();
        for (s, &mirrored) in plan.mirrored.iter().enumerate() {
            if mirrored {
                levels[plan.embedding.levels[s].idx()].swap_sides();
            }
        }
    }

    /// Record level `at`, changed in place by [`mirror`](Self::mirror), for
    /// the seat to close; called bottom-up with the levels built.
    pub(super) fn changed_at(&mut self, at: VtreeIdx) {
        self.changed.push(at);
    }

    /// Make an internal level marginal with no values: a level under a
    /// marginal parent, whose values the parent's subsume.
    pub(super) fn subsume(&mut self, at: VtreeIdx) {
        self.changed.push(at);
        let (levels, weights) = self.assembly.parts_mut();
        MarginalStorage::new(&mut levels[at.idx()], weights.as_mut(), at.idx()).install_weights(Vec::new());
    }

    /// Multiply every value of a weight-marginal level by `factor`.
    pub(super) fn scale(&mut self, at: VtreeIdx, factor: &WeightValue) {
        let (_, weights) = self.assembly.parts_mut();
        if let Some(values) = weights.as_mut().and_then(|w| w.level_vals_mut(at.idx())) {
            for value in values.iter_mut() {
                *value = value.mul(factor);
            }
        }
    }

    /// The true reference for an unconstrained child already placed bottom-up.
    #[inline]
    pub(super) fn true_node(&self, child: VtreeIdx) -> NodeIdx {
        true_node(self.vtree, child)
    }

    /// Build the level of `at`, a node no placed variable is under.
    pub(super) fn free(&mut self, at: VtreeIdx) {
        push_free_level(&mut self.assembly.parts_mut().0[at.idx()], self.vtree, at);
    }

    /// Lift all references from one child through a join with a free sibling.
    /// Hands back a join's refusal.
    pub(super) fn pass_through(&mut self, at: VtreeIdx, free_side: ChildSide) -> Result<(), OperationError> {
        let through = PassThrough::new(self.vtree, &self.assembly, at, free_side);
        self.prune |= through.carries_leaf;
        self.contract |= self.assembly.level(through.carries).is_marginal();
        for (left, right) in through.pairs() {
            self.join(at, left, right)?;
        }
        Ok(())
    }

    /// Lift a leaf child through a join with a free sibling, one node per
    /// label of `labels` in that order: the labels the level reading the
    /// lifted references names, which determinism keeps within `{One}` or
    /// within `{Pos, Neg}`. A node per label of the leaf would put `One` and
    /// a literal on one level. Hands back a join's refusal.
    pub(super) fn pass_over_leaf(&mut self, at: VtreeIdx, free_side: ChildSide, labels: &[NodeIdx]) -> Result<(), OperationError> {
        let (left, right) = self.vtree.children(at);
        let one = true_node(self.vtree, if free_side == ChildSide::Left { left } else { right });
        for &label in labels {
            let (l, r) = if free_side == ChildSide::Left { (one, label) } else { (label, one) };
            self.join(at, l, r)?;
        }
        Ok(())
    }

    /// The first reference the level at `at` holds on `side`, if it holds
    /// any.
    pub(super) fn first_ref(&self, at: VtreeIdx, side: ChildSide) -> Option<NodeIdx> {
        let (_, mut pairs) = self.assembly.level(at).internal_inputs_iter().next()?;
        let pair = pairs.next()?;
        let decoder = crate::diagram::ChildDecoder::structural();
        Some(decoder.node(if side == ChildSide::Left { pair.left } else { pair.right }))
    }

    /// Add a connecting node, recording any marginal boundary it introduces.
    /// Its storage grows through the engine's limits; a refusal is handed
    /// back, for the caller to undo the placement.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> Result<NodeIdx, OperationError> {
        let (l, r) = self.vtree.children(at);
        self.prune |= self.assembly.level(l).is_marginal() || self.assembly.level(r).is_marginal();
        self.changed.push(at);
        self.assembly.parts_mut().0[at.idx()].push_node(self.eng.limits(), &[ChildPair::new(left, right)])
    }

    /// Seat the chosen root reference without the repairs of
    /// [`finish`](Self::finish), for structural parts: a pass-through over a
    /// leaf keeps the wrappers nothing above reads. The result owes the
    /// contraction passes `carried`, what the moved levels owed where they
    /// came from. Newly built levels have no twins. Hands the placement back
    /// when the result's worklists are refused.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub(super) fn seat(self, local: NodeIdx, carried: Dirty) -> Result<Tdd, (OperationError, Self)> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let Self { eng, vtree, assembly, prune, contract, changed } = self;
        match assembly.finish_with_or_return(output, carried, &[], changed.to_close()) {
            Ok(tdd) => Ok(tdd),
            Err((e, assembly)) => Err((e, Self { eng, vtree, assembly, prune, contract, changed })),
        }
    }

    /// Move the structural levels [`move_part`](Self::move_part) placed back
    /// into `source`, through the same map.
    pub(super) fn move_back(mut self, source: &mut Tdd, map: &[VtreeIdx]) {
        let (levels, _) = self.assembly.parts_mut();
        let source_levels = &mut source.levels[..];
        for (from, &to) in map.iter().enumerate() {
            source_levels[from] = std::mem::take(&mut levels[to.idx()]);
        }
    }

    /// Check that the destination store covers every placed level's column.
    pub(super) fn check_weights(&mut self) -> Result<(), TddBuildError> {
        let (levels, weights) = self.assembly.parts_mut();
        match weights {
            Some(weights) => weights.check_levels(self.vtree, levels),
            None => Ok(()),
        }
    }

    /// Seat the chosen root reference and repair the boundaries introduced
    /// here. Call [`check_weights`](Self::check_weights) first.
    pub(super) fn finish(self, local: NodeIdx) -> Result<Tdd, OperationError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let mut result = self.assembly.finish_changed(output, self.changed.to_close())?;
        if self.prune {
            for (leaf, _) in result.vtree.leaf_bottomup() {
                if result.levels[leaf.idx()].is_weight_marginal() {
                    crate::marginal::canonicalize_weighted_leaf_refs(
                        &[leaf.idx()], &result.vtree, &mut result.levels, result.weights.as_ref());
                }
            }
            crate::diagram::inline_small_marginal_refs(&mut result, None);
        }
        if self.contract {
            self.eng.reduce(&mut result, crate::reduce::ReductionPlan::Full(crate::reduce::ContentTwinPolicy::Skip))?;
        } else if self.prune {
            prune(self.eng, &mut result)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "tests/placement.rs"]
mod tests;
