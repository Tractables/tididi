//! A level's nodes: the arena that holds them, and the view a reader takes
//! them through.

use crate::diagram::primitives::{EncodedNode, NodeIdx};
use crate::limits::Charged;
use super::TddLevel;

/// The nodes of a level, indexed by [`NodeIdx`]. Empty on leaf and marginal
/// levels.
///
/// A reader takes a node through its level, by value
/// ([`TddLevel::node`], [`TddLevel::nodes`]); code that builds a level's
/// nodes or changes them in place takes the vector
/// ([`stored_mut`](Self::stored_mut)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeArena {
    stored: Vec<EncodedNode>,
}

impl NodeArena {
    /// The number of nodes.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.stored.len()
    }

    /// Whether the arena holds no nodes.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.stored.is_empty()
    }

    /// The arena's capacity, which the level pool and the meters read.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.stored.capacity()
    }

    /// The stored nodes, to read as a slice where a level's nodes are known
    /// to be stored: by the code that built them or changes them in place.
    #[inline]
    pub(crate) fn stored(&self) -> &[EncodedNode] {
        &self.stored
    }

    /// The stored nodes, to build or change in place.
    #[inline]
    pub(crate) fn stored_mut(&mut self) -> &mut Vec<EncodedNode> {
        &mut self.stored
    }

    /// Drop every node, keeping the capacity, as [`Vec::clear`] does.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.stored.clear();
    }

    /// Drop the capacity past the arena's length, as
    /// [`Vec::shrink_to_fit`] does.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        self.stored.shrink_to_fit();
    }
}

impl From<Vec<EncodedNode>> for NodeArena {
    #[inline]
    fn from(stored: Vec<EncodedNode>) -> Self {
        NodeArena { stored }
    }
}

impl Charged for NodeArena {
    #[inline]
    fn charged_bytes(&self) -> u64 {
        self.stored.charged_bytes()
    }
}

/// The nodes of a level, as [`TddLevel::nodes`] gives them: node `i` is
/// [`get`](Self::get)`(i)`, and [`iter`](Self::iter) reads them in index
/// order. Nodes are read by value.
#[derive(Clone, Copy, Debug)]
pub struct Nodes<'a> {
    level: &'a TddLevel,
}

impl<'a> Nodes<'a> {
    /// The number of nodes: the level's slot count on a structural level, 0
    /// on a leaf or marginal one.
    #[inline]
    pub fn len(self) -> usize {
        self.level.nodes.len()
    }

    /// Whether the level holds no nodes.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.level.nodes.is_empty()
    }

    /// Node `i`, or `None` past the last.
    #[inline]
    pub fn get(self, i: usize) -> Option<EncodedNode> {
        self.level.nodes.stored.get(i).copied()
    }

    /// The nodes in index order.
    #[inline]
    pub fn iter(self) -> NodesIter<'a> {
        NodesIter { stored: self.level.nodes.stored.iter() }
    }
}

impl<'a> IntoIterator for Nodes<'a> {
    type Item = EncodedNode;
    type IntoIter = NodesIter<'a>;
    #[inline]
    fn into_iter(self) -> NodesIter<'a> {
        self.iter()
    }
}

/// The nodes of a level in index order, by value ([`Nodes::iter`]).
#[derive(Clone, Debug)]
pub struct NodesIter<'a> {
    stored: std::slice::Iter<'a, EncodedNode>,
}

impl Iterator for NodesIter<'_> {
    type Item = EncodedNode;

    #[inline]
    fn next(&mut self) -> Option<EncodedNode> {
        self.stored.next().copied()
    }

    #[inline]
    fn fold<B, F: FnMut(B, EncodedNode) -> B>(self, init: B, f: F) -> B {
        self.stored.copied().fold(init, f)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<EncodedNode> {
        self.stored.nth(n).copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stored.size_hint()
    }
}

impl ExactSizeIterator for NodesIter<'_> {}

impl std::iter::FusedIterator for NodesIter<'_> {}

impl TddLevel {
    /// Node `i` of a structural level.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not below [`slot_count`](Self::slot_count) on a
    /// structural level, or on a leaf or marginal level, which hold no
    /// nodes.
    #[inline]
    #[track_caller]
    pub fn node(&self, i: usize) -> EncodedNode {
        self.nodes.stored[i]
    }

    /// The nodes of a structural level, in index order: node `i` is
    /// `nodes().get(i)`. Empty on a leaf or marginal level, which hold no
    /// nodes.
    #[inline]
    pub fn nodes(&self) -> Nodes<'_> {
        Nodes { level: self }
    }

    /// [`nodes`](Self::nodes) paired with each slot's index.
    #[inline]
    pub(crate) fn nodes_iter(&self) -> impl Iterator<Item = (NodeIdx, EncodedNode)> + '_ {
        self.nodes().iter().enumerate().map(|(i, n)| (NodeIdx(i as u32), n))
    }
}
