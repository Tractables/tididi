//! Levels an operation writes where a diagram is unconstrained (free levels,
//! and pass-through levels that carry a child across a free sibling), and
//! the labels a parent names of a leaf.

use crate::OperationError;
use crate::diagram::{
    ChildPair, ChildSide, EncodedNode, ImplicitLevel, NodeIdx, TddLevel, LEAF_WIDTH, ONE_LEAF_IDX, floor,
    stored_levels_forced,
};
use crate::limits::Limits;
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
