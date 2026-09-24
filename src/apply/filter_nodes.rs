//! Removing chosen internal nodes from a diagram.
//!
//! The predicate is asked about every stored structural internal node first,
//! so an operand that loses nothing is never copied. Otherwise one bottom-up
//! sweep over every stored level writes the survivors into a fresh diagram.
//! A level is read once, after both of its children: a pair survives when
//! both of its sides do, and a node survives when it was kept and one of its
//! pairs survives. Each level's survivors are numbered in their original
//! order, so the remapped pair lists keep the operand's order. Marginal levels
//! are copied whole and leaf references are unchanged.

use crate::Engine;
use crate::diagram::{Assembly, ChildDecoder, ChildPair, ChildSide, EncodedChildRef, NodeIdx, Tdd, TddNodeId, ValueRef, WeightStore, ZERO};
use crate::limits::OperationError;
use crate::reduce::ReductionPlan;
use crate::vtree::VtreeIdx;

/// What [`Tdd::filter_nodes`] removed besides the rejected nodes.
///
/// The counts cover every stored node the predicate kept, reachable from the
/// output or not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FilterStats {
    /// Pairs removed from kept nodes because a side was false: the `ZERO`
    /// sentinel, an inline count of zero, or a node that was rejected or
    /// emptied.
    pub pairs_dropped: u64,
    /// Kept nodes that lost every pair and were removed in turn.
    pub emptied_nodes: u64,
}

/// Outcome of [`Tdd::filter_nodes`].
#[derive(Debug)]
pub enum FilterOutcome {
    /// Every node was kept, or the operand was already false. Nothing was
    /// rebuilt; the operand is the result.
    Unchanged,
    /// The operand without the removed nodes.
    Filtered {
        /// The rebuilt diagram, after the requested reduction.
        tdd: Tdd,
        /// What the sweep removed.
        stats: FilterStats,
    },
    /// The output node was rejected or emptied.
    Unsatisfiable {
        /// False on the operand's vtree, retaining its weight configuration.
        tdd: Tdd,
        /// What the sweep removed.
        stats: FilterStats,
    },
}

impl FilterOutcome {
    /// What the filter removed besides the rejected nodes; zero when
    /// [`Unchanged`](FilterOutcome::Unchanged).
    #[must_use]
    pub fn stats(&self) -> FilterStats {
        match self {
            FilterOutcome::Unchanged => FilterStats::default(),
            FilterOutcome::Filtered { stats, .. } | FilterOutcome::Unsatisfiable { stats, .. } => *stats,
        }
    }
}

/// A removed node in a level's remap row: rejected or emptied.
const DEAD: u32 = u32::MAX;

impl Engine {
    /// Run [`Tdd::filter_nodes`] with this engine's allocation, cancellation
    /// and output limits. The callback's own work is not bounded by them.
    ///
    /// # Errors
    ///
    /// Cancellation, allocation refusal and the output-node cap return
    /// [`OperationError::Stopped`], [`OperationError::OverBudget`] and
    /// [`OperationError::OutputCap`], respectively. A level that would exceed
    /// the addressable width returns [`OperationError::IndexOverflow`].
    pub fn filter_nodes(&self, f: &Tdd, keep: impl FnMut(TddNodeId) -> bool) -> Result<FilterOutcome, OperationError> {
        self.filter_nodes_with(f, keep, ReductionPlan::Prune)
    }

    /// [`filter_nodes`](Self::filter_nodes) with the reduction that finishes a
    /// filtered result chosen explicitly.
    ///
    /// Before `plan` runs, the kept nodes of each level are numbered in their
    /// original order and every node's surviving pairs keep their order.
    /// Unreachable nodes and the orphans the removals create are still stored
    /// at that point. [`ReductionPlan::Prune`] removes them;
    /// a full plan also makes a structural result canonical. `plan` does not
    /// run on an unchanged or unsatisfiable outcome.
    ///
    /// # Errors
    ///
    /// As [`filter_nodes`](Self::filter_nodes), including refusals from
    /// `plan`.
    pub fn filter_nodes_with(
        &self,
        f: &Tdd,
        keep: impl FnMut(TddNodeId) -> bool,
        plan: ReductionPlan<'_>,
    ) -> Result<FilterOutcome, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        if f.is_zero() { return Ok(FilterOutcome::Unchanged); }
        let Some(mut remap) = ask(self, f, keep)? else { return Ok(FilterOutcome::Unchanged) };
        let (assembly, stats) = sweep(self, f, &mut remap)?;
        let root = f.output;
        let local = if f.vtree.node(root.vtree).is_leaf() || f.levels[root.vtree.idx()].is_marginal() {
            root.local
        } else {
            NodeIdx(remap[root.vtree.idx()][root.local.idx()])
        };
        release(self, remap);
        if local == NodeIdx(DEAD) {
            drop(assembly);
            return Ok(FilterOutcome::Unsatisfiable { tdd: crate::build::constant_like(self, f, false)?, stats });
        }
        let mut tdd = assembly.finish(TddNodeId { vtree: root.vtree, local })?;
        self.reduce(&mut tdd, plan)?;
        Ok(FilterOutcome::Filtered { tdd, stats })
    }
}

/// Ask `keep` about every stored structural internal node, bottom-up by level
/// and by index within a level, and return one row per level with [`DEAD`]
/// at each rejected node. `None` when every node was kept.
fn ask(eng: &Engine, f: &Tdd, mut keep: impl FnMut(TddNodeId) -> bool) -> Result<Option<Vec<Vec<u32>>>, OperationError> {
    let lim = eng.limits();
    let mut remap: Vec<Vec<u32>> = Vec::new();
    lim.reserve_exact(&mut remap, f.levels.len())?;
    remap.resize_with(f.levels.len(), Vec::new);
    for (t, _, _) in f.vtree.internal_bottomup() {
        let level = &f.levels[t.idx()];
        if !level.is_marginal() { lim.try_resize(&mut remap[t.idx()], level.nodes.len(), 0)?; }
    }
    let mut poll = lim.gate();
    let mut rejected = false;
    for (t, _, _) in f.vtree.internal_bottomup() {
        let level = &f.levels[t.idx()];
        if level.is_marginal() { continue; }
        for (i, _) in level.nodes.iter().enumerate() {
            poll.poll(1)?;
            if !keep(TddNodeId { vtree: t, local: NodeIdx(i as u32) }) {
                remap[t.idx()][i] = DEAD;
                rejected = true;
            }
        }
    }
    poll.flush()?;
    if rejected { return Ok(Some(remap)); }
    release(eng, remap);
    Ok(None)
}

/// Hand the remap rows' charge back to the engine.
fn release(eng: &Engine, mut remap: Vec<Vec<u32>>) {
    let lim = eng.limits();
    for row in remap.drain(..) { lim.discard(row); }
    lim.discard(remap);
}

/// Write the surviving nodes into a new assembly, completing `remap` with each
/// kept node's new index or [`DEAD`].
///
/// Marginal levels are copied whole, with their weighted columns, and keep
/// their slots, so references into them pass through; a reference to a leaf
/// passes through too. The operand's weight configuration and each structural
/// level's value-reference markers come along.
fn sweep<'e>(eng: &'e Engine, f: &Tdd, remap: &mut [Vec<u32>]) -> Result<(Assembly<'e>, FilterStats), OperationError> {
    let lim = eng.limits();
    let mut out = Assembly::new(eng, &f.vtree)?;
    *out.parts_mut().1 = f.weights.as_ref().map(WeightStore::empty_like);
    for (i, source) in f.levels.iter().enumerate() {
        let t = VtreeIdx(i as u32);
        if source.is_marginal() { out.replace_level(eng, t, f.level_view(t))?; }
    }
    let mut stats = FilterStats::default();
    let mut poll = lim.gate();
    let mut pairs = Vec::new();
    let mut emitted = 0u64;
    for (t, left, right) in f.vtree.internal_bottomup() {
        let source = &f.levels[t.idx()];
        if source.is_marginal() { continue; }
        let side = |child: VtreeIdx, raw: EncodedChildRef, remap: &[Vec<u32>]| -> Option<EncodedChildRef> {
            if f.levels[child.idx()].is_marginal() {
                (ChildDecoder::marginal().value(raw) != ValueRef::Inline(0)).then_some(raw)
            } else if f.vtree.node(child).is_leaf() {
                Some(raw)
            } else {
                let new = remap[child.idx()][ChildDecoder::structural().node(raw).idx()];
                (new != DEAD).then(|| NodeIdx(new).into())
            }
        };
        for i in 0..source.nodes.len() {
            poll.poll(1)?;
            if remap[t.idx()][i] == DEAD { continue; }
            pairs.clear();
            for pair in source.pairs_of_idx(i) {
                let kept = if pair.left == ZERO.into() || pair.right == ZERO.into() {
                    None
                } else {
                    side(left, pair.left, &*remap).zip(side(right, pair.right, &*remap))
                };
                match kept {
                    Some((l, r)) => lim.try_push(&mut pairs, ChildPair::new(l, r))?,
                    None => stats.pairs_dropped += 1,
                }
            }
            if pairs.is_empty() {
                remap[t.idx()][i] = DEAD;
                stats.emptied_nodes += 1;
                continue;
            }
            remap[t.idx()][i] = out.push(eng, t, &pairs)?.0;
            emitted += 1;
        }
        lim.level_done(emitted)?;
        let level = &mut out.parts_mut().0[t.idx()];
        for side in [ChildSide::Left, ChildSide::Right] {
            level.set_has_value_refs(side, source.has_value_refs(side));
        }
    }
    poll.flush()?;
    lim.discard(pairs);
    Ok((out, stats))
}

#[cfg(test)]
#[path = "tests/filter_nodes/mod.rs"]
mod tests;
