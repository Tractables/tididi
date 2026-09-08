//! The `PairsIter` type for iterating a node's input pairs.

use super::primitives::InputPair;

/// The input pairs of one node, as yielded by [`TddLevel::pairs_iter_of`]
/// and [`TddLevel::internal_inputs_iter`].
///
/// Yields owned [`InputPair`]s in storage order, which carries no meaning
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
    Inline(Option<InputPair>),
    Slice(std::slice::Iter<'a, InputPair>),
}

impl<'a> PairsIter<'a> {
    #[inline]
    pub(super) fn empty() -> Self {
        PairsIter(Inner::Empty)
    }

    #[inline]
    pub(super) fn inline(pair: InputPair) -> Self {
        PairsIter(Inner::Inline(Some(pair)))
    }

    #[inline]
    pub(super) fn slice(pairs: &'a [InputPair]) -> Self {
        PairsIter(Inner::Slice(pairs.iter()))
    }
}

impl<'a> Iterator for PairsIter<'a> {
    type Item = InputPair;
    #[inline]
    fn next(&mut self) -> Option<InputPair> {
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
