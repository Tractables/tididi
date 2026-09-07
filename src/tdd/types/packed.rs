//! The `PairsIter` type for iterating a node's input pairs.

use super::primitives::InputPair;

/// Phase C iterator for `TddLevel::pairs_iter_of` / `pairs_iter_of_idx`.
///
/// Yields owned `InputPair`s, dispatching internally over three cases:
///
///   - `Empty`: leaf nodes (no pairs).
///   - `Inline(Some(pair))`: inline-encoded nodes (one pair).
///   - `Slice(iter)`: unpacked multi-pair nodes — wraps `slice::Iter`
///     and copies each element.
///
/// All variants yield `Copy` items; iteration cost is ~free.
#[derive(Clone)]
pub enum PairsIter<'a> {
    Empty,
    Inline(Option<InputPair>),
    Slice(std::slice::Iter<'a, InputPair>),
}

impl<'a> Iterator for PairsIter<'a> {
    type Item = InputPair;
    #[inline]
    fn next(&mut self) -> Option<InputPair> {
        match self {
            PairsIter::Empty => None,
            PairsIter::Inline(opt) => opt.take(),
            PairsIter::Slice(iter) => iter.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            PairsIter::Empty => 0,
            PairsIter::Inline(Some(_)) => 1,
            PairsIter::Inline(None) => 0,
            PairsIter::Slice(iter) => iter.len(),
        };
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for PairsIter<'a> {}
