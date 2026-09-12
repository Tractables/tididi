//! What [`TddBuilder::finish`](crate::diagram::TddBuilder::finish) refuses, and why.

use super::primitives::{InputPair, NodeIdx, TddNodeId};
use crate::vtree::VtreeIdx;

/// Why [`TddBuilder::finish`](crate::diagram::TddBuilder::finish) rejected a
/// diagram assembled level by level.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TddBuildError {
    /// `levels.len()` is not the vtree's node count.
    LevelCountMismatch {
        /// `vtree.num_nodes()`.
        expected: usize,
        /// `levels.len()`.
        found: usize,
    },
    /// A leaf level stores nodes or pairs.
    NonEmptyLeafLevel(VtreeIdx),
    /// A stored node is a leaf label, which only leaf levels denote (implicitly).
    LeafNodeStored {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: NodeIdx,
    },
    /// A stored node has no pairs; no stored node may compute false.
    EmptyNode {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: NodeIdx,
    },
    /// A pair side has bit 31 set (the `ZERO` sentinel, or a corrupt word).
    ReservedBitSet {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: NodeIdx,
        /// The offending pair.
        pair: InputPair,
    },
    /// A pair side is out of range for the child level it refers to.
    ChildIndexOutOfRange {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: NodeIdx,
        /// The offending pair.
        pair: InputPair,
        /// The child vtree node whose level was indexed (says which side).
        child: VtreeIdx,
    },
    /// A marginal count slot holds the overflow sentinel but the side table
    /// has no value for it.
    OverflowWithoutValue {
        /// The marginal level.
        level: VtreeIdx,
        /// The slot.
        slot: usize,
    },
    /// A marginal level has a structural (non-leaf, non-marginal) child.
    MarginalNotDownwardClosed {
        /// The marginal level.
        level: VtreeIdx,
        /// Its structural child.
        child: VtreeIdx,
    },
    /// A level is weight-marginal, but the diagram has no weight store to hold
    /// its values: none was attached to the
    /// [`TddBuilder`](crate::diagram::TddBuilder) with its `set_weights`, or
    /// [`Tdd::take_weights`](crate::Tdd::take_weights) would leave none.
    WeightedLevelWithoutStore {
        /// The weight-marginal level.
        level: VtreeIdx,
    },
    /// `output` is not a node of the root level (nor the `ZERO` sentinel).
    BadOutput(TddNodeId),
}

impl std::fmt::Display for TddBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LevelCountMismatch { expected, found } => {
                write!(f, "{found} levels for a vtree with {expected} nodes")
            }
            Self::NonEmptyLeafLevel(t) => write!(f, "leaf level {} stores nodes", t.idx()),
            Self::WeightedLevelWithoutStore { level } => write!(
                f,
                "level {} is weight-marginal, but a diagram assembled here has no weight store",
                level.idx()
            ),
            Self::LeafNodeStored { level, node } => {
                write!(
                    f,
                    "level {} node {} is a leaf label",
                    level.idx(),
                    node.idx()
                )
            }
            Self::EmptyNode { level, node } => {
                write!(f, "level {} node {} has no pairs", level.idx(), node.idx())
            }
            Self::ReservedBitSet { level, node, pair } => write!(
                f,
                "level {} node {} pair ({}, {}) has bit 31 set",
                level.idx(),
                node.idx(),
                pair.left.0,
                pair.right.0
            ),
            Self::ChildIndexOutOfRange {
                level,
                node,
                pair,
                child,
            } => write!(
                f,
                "level {} node {} pair ({}, {}) indexes past the end of child level {}",
                level.idx(),
                node.idx(),
                pair.left.0,
                pair.right.0,
                child.idx()
            ),
            Self::OverflowWithoutValue { level, slot } => write!(
                f,
                "marginal level {} slot {slot} is marked overflowed but has no exact value",
                level.idx()
            ),
            Self::MarginalNotDownwardClosed { level, child } => write!(
                f,
                "marginal level {} has structural child {}",
                level.idx(),
                child.idx()
            ),
            Self::BadOutput(id) => write!(
                f,
                "output ({}, {}) is not a node of the root level",
                id.vtree.idx(),
                id.local.0
            ),
        }
    }
}

impl std::error::Error for TddBuildError {}
