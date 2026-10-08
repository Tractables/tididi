//! A level's nodes: the arena that holds them, and the view a reader takes
//! them through.
//!
//! An implicit level (see [`ImplicitLevel`]) holds neither its pairs nor its
//! nodes: node `i` is implied by the description, its pair inline at one
//! pair a node, else the range of pairs `i · k .. (i + 1) · k`. Its node
//! arena is empty, and the description keeps the capacity the arena would
//! have, which the meters and the level pool read as a stored arena's.
//! Where a node's word does not fit the plain encoding (a range past 2^31
//! pairs) the words are stored beside the description.

use crate::diagram::primitives::{EncodedNode, NodeIdx};
use super::implicit::NodeCursor;
use super::{ImplicitLevel, TddLevel};

/// The stored nodes of a level, indexed by [`NodeIdx`]: every node of a
/// level that stores them, none of a leaf or marginal level or of an
/// implicit level, whose nodes are implied.
///
/// A reader takes a node through its level, by value
/// ([`TddLevel::node`], [`TddLevel::nodes`]); code that builds a level's
/// nodes or changes them in place takes the vector
/// ([`stored_mut`](Self::stored_mut)), on a level that stores them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeArena {
    stored: Vec<EncodedNode>,
}

impl NodeArena {
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

    /// Drop every stored node, keeping the capacity, as [`Vec::clear`] does.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.stored.clear();
    }

    /// Drop the capacity past the stored nodes, as
    /// [`Vec::shrink_to_fit`] does.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        self.stored.shrink_to_fit();
    }

    /// Drop the stored nodes and their allocation, for a level whose nodes
    /// its description implies; returns the capacity they had.
    #[inline]
    pub(crate) fn imply(&mut self) -> usize {
        std::mem::take(&mut self.stored).capacity()
    }
}

impl From<Vec<EncodedNode>> for NodeArena {
    #[inline]
    fn from(stored: Vec<EncodedNode>) -> Self {
        NodeArena { stored }
    }
}

/// The nodes of a level, as [`TddLevel::nodes`] gives them: node `i` is
/// [`get`](Self::get)`(i)`, and [`iter`](Self::iter) reads them in index
/// order. Nodes are read by value: an implicit level's are implied by its
/// description.
#[derive(Clone, Copy, Debug)]
pub struct Nodes<'a> {
    level: &'a TddLevel,
}

impl<'a> Nodes<'a> {
    /// The number of nodes: the level's slot count on a structural level, 0
    /// on a leaf or marginal one.
    #[inline]
    pub fn len(self) -> usize {
        self.level.node_count()
    }

    /// Whether the level holds no nodes.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Node `i`, or `None` past the last.
    #[inline]
    pub fn get(self, i: usize) -> Option<EncodedNode> {
        (i < self.len()).then(|| self.level.node(i))
    }

    /// The nodes in index order.
    #[inline]
    pub fn iter(self) -> NodesIter<'a> {
        let level = self.level;
        let implied = level.implied_by().map_or(0..0, |d| 0..d.nodes());
        NodesIter { stored: level.nodes.stored.iter(), implied, level, cursor: None }
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

/// The nodes of a level in index order, by value ([`Nodes::iter`]): a
/// stored level's read off its arena, an implicit level's implied by its
/// description.
#[derive(Clone, Debug)]
pub struct NodesIter<'a> {
    /// The stored nodes still to come; empty on an implicit level.
    stored: std::slice::Iter<'a, EncodedNode>,
    /// The implied nodes still to come; empty on a stored level.
    implied: std::ops::Range<usize>,
    level: &'a TddLevel,
    /// The implied nodes' first pairs, at one pair a node, stepped on from
    /// node to node.
    cursor: Option<Box<NodeCursor<'a>>>,
}

impl NodesIter<'_> {
    /// The implied node `i`, out of line, so that a stored level's read
    /// inlines where it is called.
    #[inline(never)]
    fn implied(&mut self, i: usize) -> EncodedNode {
        self.level.implied_by().expect("an implicit level").node_word_next(&mut self.cursor, i)
    }
}

impl Iterator for NodesIter<'_> {
    type Item = EncodedNode;

    #[inline]
    fn next(&mut self) -> Option<EncodedNode> {
        match self.stored.next() {
            Some(&node) => Some(node),
            None => self.implied.next().map(|i| self.implied(i)),
        }
    }

    /// The stored nodes folded as a slice's, then the implied ones in a
    /// loop of their own.
    #[inline]
    fn fold<B, F: FnMut(B, EncodedNode) -> B>(self, init: B, mut f: F) -> B {
        let acc = self.stored.copied().fold(init, &mut f);
        if self.implied.is_empty() {
            return acc;
        }
        let d = self.level.implied_by().expect("an implicit level");
        let mut cursor = self.cursor;
        self.implied.fold(acc, |acc, i| f(acc, d.node_word_next(&mut cursor, i)))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<EncodedNode> {
        let stored = self.stored.len();
        match self.stored.nth(n) {
            Some(&node) => Some(node),
            None => self.implied.nth(n - stored).map(|i| self.implied(i)),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.stored.len() + self.implied.len();
        (n, Some(n))
    }
}

impl ExactSizeIterator for NodesIter<'_> {}

impl std::iter::FusedIterator for NodesIter<'_> {}

impl TddLevel {
    /// Node `i` of a structural level: stored, or implied by the level's
    /// description.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not below [`slot_count`](Self::slot_count) on a
    /// structural level, or on a leaf or marginal level, which hold no
    /// nodes.
    #[inline]
    #[track_caller]
    pub fn node(&self, i: usize) -> EncodedNode {
        match self.nodes.stored.get(i) {
            Some(&node) => node,
            None => self.implied_node(i),
        }
    }

    /// Node `i` of a level whose description implies its nodes, out of
    /// line, so that a stored level's read inlines where it is called.
    #[inline(never)]
    #[track_caller]
    pub(crate) fn implied_node(&self, i: usize) -> EncodedNode {
        match self.implied_by() {
            Some(d) if i < d.nodes() => d.node_word(i),
            _ => panic!("node {i} of a level of {} nodes", self.node_count()),
        }
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

    /// The number of nodes, stored or implied.
    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        match self.nodes.stored.len() {
            0 => self.pairs.implicit().map_or(0, ImplicitLevel::nodes),
            n => n,
        }
    }

    /// The capacity of the node arena: on a level whose nodes are implied,
    /// the capacity it would have, which the level pool and the meters read
    /// as they would read the stored one's.
    #[inline]
    pub(crate) fn node_capacity(&self) -> usize {
        match self.implied_by() {
            Some(_) => self.pairs.node_capacity(),
            None => self.nodes.stored.capacity(),
        }
    }

    /// The description that implies the level's nodes, when it does: the
    /// level is implicit and stores no node.
    #[inline]
    pub(crate) fn implied_by(&self) -> Option<&ImplicitLevel> {
        if self.nodes.stored.is_empty() { self.pairs.implicit() } else { None }
    }

    /// Drop the node arena's capacity past its nodes, as
    /// [`Vec::shrink_to_fit`] does: on a level whose nodes are implied, the
    /// capacity it would have.
    pub(crate) fn shrink_nodes(&mut self) {
        match self.implied_by() {
            Some(d) => {
                let n = d.nodes();
                self.pairs.set_node_capacity(n);
            }
            None => self.nodes.shrink_to_fit(),
        }
    }

    /// Store the words of the nodes the level's description implies, at the
    /// capacity the arena would have, before the level is stored: nothing
    /// on a level that stores its nodes.
    pub(crate) fn store_implied_nodes(&mut self) {
        let Some(d) = self.implied_by() else { return };
        let mut stored = Vec::with_capacity(self.pairs.node_capacity().max(d.nodes()));
        stored.extend(self.nodes().iter());
        self.nodes = NodeArena::from(stored);
    }
}
