//! Place levels and connect their roots after an operation validates its layout.
//!
//! [`MovePlacement`] owns the destination storage while the graft fills it:
//! it moves whole parts and keeps local node indices through the transfer.

use std::sync::Arc;

use crate::{Engine, OperationError};
use crate::diagram::{
    Assembly, ChildPair, ChildSide, EncodedNode, ImplicitLevel, MarginalStorage, NodeIdx, Tdd,
    TddLevel, TddNodeId, LEAF_WIDTH, ONE_LEAF_IDX, floor,
    stored_levels_forced, try_take_levels,
};
use crate::limits::Limits;
use crate::execution::pool::PoolGuard;
use crate::vtree::{Vtree, VtreeIdx};


/// The true reference for an unconstrained child already placed bottom-up.
#[inline]
pub(crate) fn true_node(vtree: &Vtree, child: VtreeIdx) -> NodeIdx {
    if vtree.node(child).is_leaf() { ONE_LEAF_IDX } else { NodeIdx(0) }
}

/// Make `level` the level of internal node `at` in a diagram constant true
/// under it: one node, true on both sides. `level` is empty.
#[inline]
pub(crate) fn push_free_level(level: &mut TddLevel, vtree: &Vtree, at: VtreeIdx) {
    let (left, right) = vtree.children(at);
    level.push_internal_node(&[ChildPair::new(true_node(vtree, left), true_node(vtree, right))]);
}

/// The labels of a leaf that `level`, its parent, names on `side`, one bit
/// each: the One label alone, which the first pair shows (determinism), or
/// the literals the pairs name, read until both are found; none on a level
/// with no pair.
pub(crate) fn named_labels(level: &TddLevel, side: ChildSide) -> u8 {
    use crate::diagram::{NEG_LEAF_IDX, POS_LEAF_IDX};
    let (one, both) = (1 << ONE_LEAF_IDX.0, 1 << POS_LEAF_IDX.0 | 1 << NEG_LEAF_IDX.0);
    let decoder = crate::diagram::ChildDecoder::structural();
    let mut named = 0u8;
    for (_, pairs) in level.internal_inputs_iter() {
        for pair in pairs {
            named |= 1 << decoder.node(side.pick(pair.left, pair.right)).0;
            if named & one != 0 {
                return one;
            }
            if named == both {
                return both;
            }
        }
    }
    named
}

/// The joins that lift every reference of one child of `at` through a free
/// sibling: one node per slot of the carried child, at the same index.
pub(crate) struct PassThrough {
    pub(crate) carries: VtreeIdx,
    pub(crate) carries_leaf: bool,
    pub(crate) one: NodeIdx,
    pub(crate) width: usize,
    pub(crate) free_side: ChildSide,
}

impl PassThrough {
    /// The joins at `at`, whose child on `free_side` is free, the carried
    /// internal child's level having `slots(child)` slots.
    pub(crate) fn new(vtree: &Vtree, at: VtreeIdx, free_side: ChildSide, slots: impl FnOnce(VtreeIdx) -> usize) -> Self {
        let (left, right) = vtree.children(at);
        let (free, carries) = free_side.pick((left, right), (right, left));
        let carries_leaf = vtree.node(carries).is_var_leaf();
        let width = if carries_leaf { LEAF_WIDTH } else { slots(carries) };
        Self { carries, carries_leaf, one: true_node(vtree, free), width, free_side }
    }

    /// The `(left, right)` children of join `i`.
    #[inline]
    fn pair(&self, i: usize) -> (NodeIdx, NodeIdx) {
        let child = NodeIdx(i as u32);
        self.free_side.pick((self.one, child), (child, self.one))
    }

    /// The `(left, right)` children of each join, in slot order.
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (NodeIdx, NodeIdx)> + '_ {
        (0..self.width).map(move |i| self.pair(i))
    }

    /// Make `level`, empty, the joins' level, closed: over an internal
    /// carried child as [`build`](Self::build) makes it, over a leaf one
    /// join per label of `labels`, in order.
    pub(crate) fn fill(&self, lim: &Limits, level: &mut TddLevel, labels: Option<&[NodeIdx]>) -> Result<(), OperationError> {
        match labels {
            None if self.width > 0 => self.build(lim, level)?,
            None => {}
            Some(labels) => {
                for &label in labels {
                    let (l, r) = self.free_side.pick((self.one, label), (label, self.one));
                    level.push_node(lim, &[ChildPair::new(l, r)])?;
                }
            }
        }
        level.close();
        level.close_held_pairs();
        Ok(())
    }

    /// Make `level`, empty, the joins' level over an internal carried
    /// child, as one join a slot leaves it: a node a slot, each holding its
    /// pair inline, grown and charged as the joins grow and charge it.
    /// Where nothing bounds memory, a level the seat's close would hold as
    /// the description of its pairs is held so here, its node arena's
    /// growth charged and nothing stored.
    pub(crate) fn build(&self, lim: &Limits, level: &mut TddLevel) -> Result<(), OperationError> {
        debug_assert!(!self.carries_leaf && level.nodes().is_empty() && level.pairs.is_empty());
        let width = self.width;
        if width >= floor().max(2) && lim.memory_unbounded() && !stored_levels_forced() {
            let (left, right) = self.pair(0);
            let first = (i64::from(left.0), i64::from(right.0));
            let step = self.free_side.pick((0, 1), (1, 0));
            if let Some(described) = ImplicitLevel::one_step(width, first, step) {
                let held = level.nodes.imply();
                let node_capacity = lim.charge_as_pushed::<EncodedNode>(held, width)?;
                let capacity = level.pairs.capacity();
                level.pairs.describe(described, capacity, node_capacity);
                return Ok(());
            }
        }
        let nodes = level.nodes.stored_mut();
        lim.reserve_as_pushed(nodes, width)?;
        nodes.extend((0..width).map(|i| {
            let (left, right) = self.pair(i);
            EncodedNode::inline(ChildPair::new(left, right))
        }));
        Ok(())
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

/// Whole parts moved into a destination, levels and weight columns alike.
/// The level array and every level it builds are charged to the engine's
/// limits; the moved levels are not charged again.
pub(super) struct MovePlacement<'a> {
    eng: &'a Engine,
    vtree: &'a Arc<Vtree>,
    assembly: Assembly<'a>,
    changed: Changed<'a>,
}

impl<'a> MovePlacement<'a> {
    pub(super) fn new(eng: &'a Engine, vtree: &'a Arc<Vtree>) -> Result<Self, OperationError> {
        let changed = Changed::new(eng, vtree)?;
        let levels = try_take_levels(eng, vtree.num_nodes())?;
        let assembly = Assembly::from_levels(eng, Arc::clone(vtree), levels);
        Ok(Self { eng, vtree, assembly, changed })
    }

    /// Move every level through a checked placement map.
    /// The caller has checked that stored values retain their interpretation.
    pub(super) fn move_part(&mut self, source: &mut Tdd, map: &[VtreeIdx]) {
        debug_assert_eq!(source.vtree().num_nodes(), map.len());
        self.changed.take_from(source);
        let levels = self.assembly.levels_mut();
        // One mutable access to the source's levels for the whole move: each
        // access forgets what the storage knows of them.
        let source_levels = &mut source.levels[..];
        for (from, &to) in map.iter().enumerate() {
            MarginalStorage::new(&mut levels[to.idx()])
                .move_from(&mut source_levels[from]);
        }
    }

    /// Add a connecting node. Its storage grows through the engine's limits;
    /// a refusal is handed back, for the caller to undo the placement.
    #[inline]
    pub(super) fn join(&mut self, at: VtreeIdx, left: NodeIdx, right: NodeIdx) -> Result<NodeIdx, OperationError> {
        self.changed.push(at);
        self.assembly.levels_mut()[at.idx()].push_node(self.eng.limits(), &[ChildPair::new(left, right)])
    }

    /// Seat the chosen root reference.
    pub(super) fn finish(self, local: NodeIdx) -> Result<Tdd, OperationError> {
        let output = TddNodeId { vtree: self.vtree.root(), local };
        self.assembly.finish_changed(output, self.changed.to_close())
    }
}
