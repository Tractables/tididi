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
    Assembly, ChildPair, ChildSide, LevelView, MarginalStorage, NodeIdx, Tdd, TddNodeId,
    WeightStore, LEAF_WIDTH, ONE_LEAF_IDX, try_take_levels,
};
use crate::vtree::{Vtree, VtreeIdx};
use super::GraftError;

/// Drop the nodes the joins made in `result` that nothing refers to.
fn prune(eng: &Engine, result: &mut Tdd) -> Result<(), OperationError> {
    eng.reduce(result, crate::reduce::ReductionPlan::Prune)
}

/// Bounded structural copies into a fresh destination, under the engine's
/// limits.
pub(super) struct CopyPlacement<'a> {
    eng: &'a Engine,
    vtree: &'a Arc<Vtree>,
    assembly: Assembly<'a>,
    prune: bool,
}

impl<'a> CopyPlacement<'a> {
    pub(super) fn new(eng: &'a Engine, vtree: &'a Arc<Vtree>) -> Result<Self, OperationError> {
        Ok(Self { eng, vtree, assembly: Assembly::new(eng, vtree)?, prune: false })
    }

    /// Copy a structural level without its source's weight configuration.
    #[inline]
    pub(super) fn copy_level(&mut self, source: &Tdd, from: VtreeIdx, to: VtreeIdx) -> Result<(), OperationError> {
        let view = LevelView::unweighted(source.level(from))
            .expect("embedding requires structural levels");
        self.assembly.replace_level(self.eng, to, view)
    }

    /// [`copy_level`](Self::copy_level) onto a destination node whose children
    /// are the source node's, swapped: every pair is read the other way round.
    pub(super) fn copy_level_mirrored(&mut self, source: &Tdd, from: VtreeIdx, to: VtreeIdx) -> Result<(), OperationError> {
        self.copy_level(source, from, to)?;
        let (levels, _) = self.assembly.parts_mut();
        levels[to.idx()].swap_sides();
        Ok(())
    }

    /// The true reference for an unconstrained child already placed bottom-up.
    #[inline]
    pub(super) fn true_node(&self, child: VtreeIdx) -> NodeIdx {
        if self.vtree.node(child).is_leaf() { ONE_LEAF_IDX } else { NodeIdx(0) }
    }

    /// Lift all references from one child through a join with a free sibling.
    #[inline]
    pub(super) fn pass_through(&mut self, at: VtreeIdx, free_side: ChildSide) -> Result<(), OperationError> {
        let (left, right) = self.vtree.children(at);
        let (free, carries) = if free_side == ChildSide::Left { (left, right) } else { (right, left) };
        let one = self.true_node(free);
        let carries_leaf = self.vtree.node(carries).is_leaf();
        let width = if carries_leaf { LEAF_WIDTH } else { self.assembly.level(carries).slot_count() };
        // A leaf contributes all three implicit labels. Prune any wrappers the
        // copied parent does not use; keep their indices until then.
        self.prune |= carries_leaf;
        for i in 0..width {
            let child = NodeIdx(i as u32);
            let (left, right) = if free_side == ChildSide::Left { (one, child) } else { (child, one) };
            self.join(at, left, right)?;
        }
        Ok(())
    }

    /// Add a connecting node.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> Result<NodeIdx, OperationError> {
        self.assembly.push(self.eng, at, &[ChildPair::new(left, right)])
    }

    /// Seat the chosen root reference and drop what the joins left unused.
    ///
    /// Copies of a valid diagram's levels joined through true nodes are valid
    /// storage by construction, so only debug builds check them.
    pub(super) fn finish(self, local: NodeIdx) -> Result<Tdd, OperationError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let mut result = self.assembly.finish_asserted(output)?;
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
}

impl<'a> MovePlacement<'a> {
    pub(super) fn new(eng: &'a Engine, vtree: &'a Arc<Vtree>, weights: Option<WeightStore>) -> Result<Self, OperationError> {
        let levels = try_take_levels(eng, vtree.num_nodes())?;
        let assembly = Assembly::from_levels(eng, Arc::clone(vtree), levels, weights);
        Ok(Self { eng, vtree, assembly, prune: false })
    }

    /// Move every level and its weighted column through a checked placement map.
    /// The caller has checked that stored values retain their interpretation.
    pub(super) fn move_part(&mut self, source: &mut Tdd, map: &[VtreeIdx]) {
        debug_assert_eq!(source.vtree().num_nodes(), map.len());
        let (levels, weights) = self.assembly.parts_mut();
        for (from, &to) in map.iter().enumerate() {
            MarginalStorage::new(&mut levels[to.idx()], weights.as_mut(), to.idx())
                .move_from(&mut source.levels[from], source.weights.as_mut(), from);
        }
    }

    /// Add a connecting node, recording any marginal boundary it introduces.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> NodeIdx {
        let (l, r) = self.vtree.children(at);
        self.prune |= self.assembly.level(l).is_marginal() || self.assembly.level(r).is_marginal();
        self.assembly.parts_mut().0[at.idx()].push_internal_node(&[ChildPair::new(left, right)])
    }

    /// Seat the chosen root reference and repair the boundaries introduced here.
    pub(super) fn finish(mut self, local: NodeIdx) -> Result<Tdd, GraftError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let (levels, weights) = self.assembly.parts_mut();
        if let Some(weights) = weights {
            weights.check_levels(self.vtree, levels).map_err(GraftError::DestinationWeights)?;
        }
        let mut result = self.assembly.finish(output)?;
        if self.prune {
            for (leaf, _) in result.vtree.leaf_bottomup() {
                if result.levels[leaf.idx()].is_weight_marginal() {
                    crate::marginal::canonicalize_weighted_leaf_refs(
                        &[leaf.idx()], &result.vtree, &mut result.levels, result.weights.as_ref());
                }
            }
            crate::diagram::inline_small_marginal_refs(&mut result, None);
            prune(self.eng, &mut result)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "tests/placement.rs"]
mod tests;
