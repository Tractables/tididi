//! The `PairsIter` type for iterating a node's input pairs.

use super::primitives::ChildPair;

/// The input pairs of one node, as yielded by [`TddLevel::pairs_iter_of`]
/// and [`TddLevel::internal_inputs_iter`].
///
/// Yields owned [`ChildPair`]s in storage order, which carries no meaning
/// (a node is the set of its pairs). Implements [`ExactSizeIterator`], so
/// `len()` is the node's pair count.
///
/// [`TddLevel::pairs_iter_of`]: super::TddLevel::pairs_iter_of
/// [`TddLevel::internal_inputs_iter`]: super::TddLevel::internal_inputs_iter
#[derive(Clone)]
pub struct PairsIter<'a>(Inner<'a>);

#[derive(Clone)]
enum Inner<'a> {
    Empty,
    Inline(Option<ChildPair>),
    Slice(std::slice::Iter<'a, ChildPair>),
}

impl<'a> PairsIter<'a> {
    #[inline]
    pub(super) fn empty() -> Self {
        PairsIter(Inner::Empty)
    }

    #[inline]
    pub(super) fn inline(pair: ChildPair) -> Self {
        PairsIter(Inner::Inline(Some(pair)))
    }

    #[inline]
    pub(super) fn slice(pairs: &'a [ChildPair]) -> Self {
        PairsIter(Inner::Slice(pairs.iter()))
    }
}

impl std::fmt::Debug for PairsIter<'_> {
    /// How many pairs are still to come, which is all an iterator's state is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let remaining = match &self.0 {
            Inner::Empty => 0,
            Inner::Inline(opt) => usize::from(opt.is_some()),
            Inner::Slice(iter) => iter.len(),
        };
        f.debug_struct("PairsIter").field("remaining", &remaining).finish()
    }
}

impl<'a> Iterator for PairsIter<'a> {
    type Item = ChildPair;
    #[inline]
    fn next(&mut self) -> Option<ChildPair> {
        match &mut self.0 {
            Inner::Empty => None,
            Inner::Inline(opt) => opt.take(),
            Inner::Slice(iter) => iter.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.0 {
            Inner::Empty => 0,
            Inner::Inline(Some(_)) => 1,
            Inner::Inline(None) => 0,
            Inner::Slice(iter) => iter.len(),
        };
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for PairsIter<'a> {}
