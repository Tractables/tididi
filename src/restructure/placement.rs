//! Place levels and connect their roots after an operation validates its layout.

use std::sync::Arc;

use crate::{Engine, OperationError};
use crate::diagram::{
    Assembly, ChildPair, LevelView, MarginalStorage, NodeIdx, Tdd, TddNodeId,
    WeightStore, LEAF_WIDTH, ONE_LEAF_IDX, take_levels,
};
use crate::vtree::{Vtree, VtreeIdx};
use super::GraftError;

/// Own destination storage, its weight columns, and the repair owed by new joins.
/// Layout validation belongs to the caller; local node indices survive placement.
/// `COPY` selects bounded structural copies at compile time; moves retain graft
/// assembly's untracked contract. Both run required pruning under the engine.
pub(super) struct Placement<'a, const COPY: bool> {
    eng: &'a Engine,
    vtree: &'a Arc<Vtree>,
    assembly: Assembly<'a>,
    prune: bool,
}

impl<'a> Placement<'a, true> {
    pub(super) fn copying(eng: &'a Engine, vtree: &'a Arc<Vtree>) -> Result<Self, OperationError> {
        Ok(Self { eng, vtree, assembly: Assembly::new(eng, vtree)?,
            prune: false })
    }

    /// Copy a structural level without its source's weight configuration.
    #[inline]
    pub(super) fn copy_level(&mut self, source: &Tdd, from: VtreeIdx, to: VtreeIdx) -> Result<(), OperationError> {
        let view = LevelView::unweighted(source.level(from))
            .expect("embedding requires structural levels");
        self.assembly.replace_level(self.eng, to, view)
    }

    /// The true reference for an unconstrained child already placed bottom-up.
    #[inline]
    pub(super) fn true_node(&self, child: VtreeIdx) -> NodeIdx {
        if self.vtree.node(child).is_leaf() { ONE_LEAF_IDX } else { NodeIdx(0) }
    }

    /// Lift all references from one child through a join with a free sibling.
    #[inline]
    pub(super) fn pass_through(&mut self, at: VtreeIdx, free_is_left: bool) -> Result<(), OperationError> {
        let (left, right) = self.vtree.children(at);
        let (free, carries) = if free_is_left { (left, right) } else { (right, left) };
        let one = self.true_node(free);
        let carries_leaf = self.vtree.node(carries).is_leaf();
        let width = if carries_leaf { LEAF_WIDTH } else { self.assembly.level(carries).slot_count() };
        // A leaf contributes all three implicit labels. Prune any wrappers the
        // copied parent does not use; keep their indices until then.
        self.prune |= carries_leaf;
        for i in 0..width {
            let child = NodeIdx(i as u32);
            let (left, right) = if free_is_left { (one, child) } else { (child, one) };
            self.join(at, left, right)?;
        }
        Ok(())
    }
}

impl<'a> Placement<'a, false> {
    pub(super) fn moving(eng: &'a Engine, vtree: &'a Arc<Vtree>, weights: Option<WeightStore>) -> Self {
        let assembly = Assembly::from_levels(eng, Arc::clone(vtree), take_levels(eng, vtree.num_nodes()), weights);
        Self { eng, vtree, assembly, prune: false }
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
}

impl<const COPY: bool> Placement<'_, COPY> {
    /// Add a connecting node, recording any marginal boundary it introduces.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> Result<NodeIdx, OperationError> {
        let pair = ChildPair::new(left, right);
        if COPY {
            self.assembly.push(self.eng, at, &[pair])
        } else {
            let (l, r) = self.vtree.children(at);
            self.prune |= self.assembly.level(l).is_marginal() || self.assembly.level(r).is_marginal();
            Ok(self.assembly.parts_mut().0[at.idx()].push_internal_node(&[pair]))
        }
    }

    /// Seat the chosen root reference and repair the boundaries introduced here.
    pub(super) fn finish(mut self, local: NodeIdx) -> Result<Tdd, GraftError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        let mut result = if COPY {
            self.assembly.finish_checked(output).map_err(OperationError::from)?
        } else {
            let (levels, weights) = self.assembly.parts_mut();
            if let Some(weights) = weights {
                weights.check_levels(self.vtree, levels).map_err(GraftError::DestinationWeights)?;
            }
            self.assembly.finish_untracked(output)
        };
        if !COPY && self.prune {
            for (leaf, _) in result.vtree.leaf_bottomup() {
                if result.levels[leaf.idx()].is_weight_marginal() {
                    crate::marginal::canonicalize_apply_leaf_refs(
                        &[leaf.idx()], &result.vtree, &mut result.levels, result.weights.as_ref());
                }
            }
            crate::diagram::tag_all_marginal_side_slots(&mut result, None);
        }
        if self.prune {
            self.eng.reduce(&mut result, crate::reduce::ReductionPlan::Prune)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "tests/placement.rs"]
mod tests;
