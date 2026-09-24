//! Reading a level: pair views, decoding, remapping, per-node pair counts, and
//! the canonical sort a rewritten pair list is put back in.

use crate::diagram::EncodedChildRef;

use crate::diagram::marginal_ref::ChildDecoder;
use crate::diagram::PairsIter;
use crate::diagram::primitives::{ChildPair, EncodedNode, NodeKind};
use super::TddLevel;

impl TddLevel {
    /// Every node as `(local index, pairs)`.
    /// The index is the node's slot in `nodes`, so it is valid for arrays
    /// sized by `slot_count()`. Empty on a marginal level.
    pub fn internal_inputs_iter(&self) -> impl Iterator<Item = (usize, PairsIter<'_>)> + '_ {
        self.internal_inputs_range(0..self.nodes.len())
    }

    /// Iterate structural nodes in a valid slot range, retaining their level indices.
    #[inline]
    pub(crate) fn internal_inputs_range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = (usize, PairsIter<'_>)> + '_ {
        let start = range.start;
        self.nodes[range].iter().enumerate().map(move |(i, n)| (start + i, self.pairs_iter_of(n)))
    }

    /// A multi-pair node's pair-arena start and pair count, decoded from
    /// either the packed or the extended (side-table) encoding.
    fn multi_span(&self, node: &EncodedNode) -> (usize, usize) {
        match node.kind() {
            NodeKind::Multi { start, len } => (start as usize, len as usize),
            NodeKind::MultiRanged(idx) => {
                let e = &self.multi_pairs[idx as usize];
                (e.start as usize, e.len as usize)
            }
            other => panic!("multi_span on {other:?}"),
        }
    }

    /// A multi-pair node's pair-arena range.
    pub(crate) fn multi_range(&self, node: &EncodedNode) -> std::ops::Range<usize> {
        let (start, len) = self.multi_span(node);
        start..start + len
    }

    /// The pairs of `node`, which must describe a node of this level.
    ///
    /// The slice borrows both the level and `node`, because an inline pair lives in the node.
    ///
    /// ```compile_fail,E0597
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let diagram = Tdd::one(&vtree);
    /// let level = diagram.level(vtree.root());
    /// let pairs;
    /// {
    ///     let node = level.nodes()[0];
    ///     pairs = level.pairs_of(&node);
    /// }
    /// assert!(!pairs.is_empty()); // the copied node no longer exists
    /// ```
    pub fn pairs_of<'a>(&'a self, node: &'a EncodedNode) -> &'a [ChildPair] {
        match node.kind() {
            // Safety: EncodedNode is #[repr(C)] {a: u32, b: u32}.
            //         ChildPair is #[repr(C)] {left: EncodedChildRef(u32), right: EncodedChildRef(u32)}.
            //         For inline nodes, a == left.0 and b == right.0 by construction.
            //         Both types have identical {u32, u32} layout, so the cast is valid.
            NodeKind::Inline(_) => unsafe {
                std::slice::from_ref(&*(node as *const EncodedNode as *const ChildPair))
            },
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => &self.pairs[self.multi_range(node)],
        }
    }

    /// [`pairs_of`](Self::pairs_of) by node index; not valid on a marginal level.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not below `nodes().len()`, which on a marginal level
    /// is every `idx`.
    pub fn pairs_of_idx(&self, idx: usize) -> &[ChildPair] {
        // A debug_assert! rather than a check: this is a hot path, and callers
        // route around marginal levels.
        debug_assert!(
            !self.is_marginal(),
            "pairs_of_idx({idx}) called on marginal level (width={}, nodes.len()={}, pairs.len()={}). \
             Callers must guard via is_marginal() — marginal levels store model counts, \
             not pair structure.",
            self.slot_count(), self.nodes.len(), self.pairs.len(),
        );
        self.pairs_of(&self.nodes[idx])
    }

    /// [`pairs_iter_of`](Self::pairs_iter_of) by node index; not valid on a
    /// marginal level.
    pub(crate) fn pairs_iter_of_idx(&self, idx: usize) -> PairsIter<'_> {
        debug_assert!(
            !self.is_marginal(),
            "pairs_iter_of_idx({idx}) called on marginal level",
        );
        let d = &self.nodes[idx];
        self.pairs_iter_of(d)
    }

    /// Like [`pairs_of_idx`](Self::pairs_of_idx), but decodes marginal-side fields to the bare
    /// coordinates structural use wants ([`ChildDecoder::coord`]).
    ///
    /// With neither child marginal this is the zero-copy `pairs_of_idx`;
    /// otherwise it materializes a decoded copy into `scratch`.
    pub(crate) fn pairs_view_decoded<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<ChildPair>,
        left: ChildDecoder,
        right: ChildDecoder,
    ) -> &'a [ChildPair] {
        if !left.is_marginal() && !right.is_marginal() {
            return self.pairs_of_idx(idx);
        }
        scratch.clear();
        self.decode_pairs_into(idx, scratch, left, right);
        scratch.as_slice()
    }

    /// Append `idx`'s pairs, marginal-decoded, onto `out` (no clear). The
    /// caller pre-reserves `out` when the total is known.
    #[inline]
    pub(crate) fn decode_pairs_into(
        &self,
        idx: usize,
        out: &mut Vec<ChildPair>,
        left: ChildDecoder,
        right: ChildDecoder,
    ) {
        for p in self.pairs_iter_of_idx(idx) {
            out.push(ChildPair::new(EncodedChildRef::from_raw(left.coord(p.left)), EncodedChildRef::from_raw(right.coord(p.right))));
        }
    }

    /// The pairs of `node` as an iterator; the owned-item twin of
    /// [`pairs_of`](Self::pairs_of).
    #[inline]
    pub fn pairs_iter_of<'a>(&'a self, node: &'a EncodedNode) -> PairsIter<'a> {
        match node.kind() {
            NodeKind::Inline(pair) => PairsIter::inline(pair),
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => {
                let range = self.multi_range(node);
                PairsIter::slice(&self.pairs[range])
            }
        }
    }

    /// Get mutable access to a multi-pair node's pairs in the arena.
    /// Only valid for multi-pair nodes; panics on inline nodes.
    #[inline]
    pub(crate) fn pairs_mut(&mut self, idx: usize) -> &mut [ChildPair] {
        debug_assert!(self.nodes[idx].kind().pairs_in_arena(),
            "pairs_mut called on inline node");
        let range = self.multi_range(&self.nodes[idx]);
        &mut self.pairs[range]
    }

    /// Index-remap a multi-pair node's pairs in place: each side is rewritten
    /// through its lookup slice and [`ChildDecoder::remap`], which leaves a
    /// marginal side's inline values alone.
    ///
    /// Precondition (debug-asserted): `self.nodes[idx].is_multi()`; every
    /// structural coordinate looked up is within its remap slice.
    #[inline]
    pub(crate) fn pairs_remap_indexed(
        &mut self,
        idx: usize,
        left_remap: &[u32],
        right_remap: &[u32],
        left: ChildDecoder,
        right: ChildDecoder,
    ) {
        debug_assert!(self.nodes[idx].kind().pairs_in_arena(),
            "pairs_remap_indexed called on inline node");
        let range = self.multi_range(&self.nodes[idx]);
        for pair in &mut self.pairs[range] {
            pair.left = left.remap(pair.left, left_remap);
            pair.right = right.remap(pair.right, right_remap);
        }
    }

    /// Pair-arena start offset for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn multi_start_at(&self, idx: usize) -> usize {
        self.multi_span(&self.nodes[idx]).0
    }

    /// Pair count for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn multi_len_at(&self, idx: usize) -> usize {
        self.multi_span(&self.nodes[idx]).1
    }

    /// Pair-arena range for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn pair_range_at(&self, idx: usize) -> std::ops::Range<usize> {
        self.multi_range(&self.nodes[idx])
    }

    /// Number of pairs of the node at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not below `nodes().len()`.
    #[inline]
    pub fn pair_count_at(&self, idx: usize) -> usize {
        if matches!(self.nodes[idx].kind(), NodeKind::Inline(_)) { 1 } else { self.multi_len_at(idx) }
    }

    /// The pair count of every node in index order. Empty on a marginal
    /// level, which holds no nodes.
    #[inline]
    pub(crate) fn pair_counts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.nodes.len()).map(|i| self.pair_count_at(i))
    }

    /// The pairs held by this level's live nodes.
    pub(crate) fn live_pairs(&self) -> usize {
        self.pair_counts().sum()
    }

}

/// Sorting network for 3..=8 elements: a fixed sequence of conditional swaps
/// (`cswap!(a, b)` = "if `s[a] > s[b]`, swap them") on any indexable,
/// swappable container.
macro_rules! sorting_network {
    ($s:expr, $n:expr) => {{
        macro_rules! cswap {
            ($a:expr, $b:expr) => { if $s[$a] > $s[$b] { $s.swap($a, $b); } };
        }
        match $n {
            3 => { cswap!(0,1); cswap!(1,2); cswap!(0,1); }
            4 => { cswap!(0,1); cswap!(2,3); cswap!(0,2); cswap!(1,3); cswap!(1,2); }
            5 => {
                cswap!(0,1); cswap!(3,4); cswap!(2,4); cswap!(2,3);
                cswap!(0,3); cswap!(0,2); cswap!(1,4); cswap!(1,3); cswap!(1,2);
            }
            6 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(0,2); cswap!(1,3); cswap!(0,4); cswap!(1,5);
                cswap!(1,2); cswap!(3,5); cswap!(2,4); cswap!(3,4); cswap!(1,2);
            }
            7 => {
                // Green's construction (Knuth, The Art of Computer Programming,
                // volume 3; 16 comparators).
                cswap!(0,4); cswap!(1,5); cswap!(2,6);
                cswap!(0,2); cswap!(1,3); cswap!(4,6);
                cswap!(2,4); cswap!(3,5);
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(1,4); cswap!(3,6);
                cswap!(1,2); cswap!(3,4); cswap!(5,6);
            }
            8 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5); cswap!(6,7);
                cswap!(0,2); cswap!(1,3); cswap!(4,6); cswap!(5,7);
                cswap!(1,2); cswap!(5,6);
                cswap!(0,4); cswap!(3,7); cswap!(1,5); cswap!(2,6);
                cswap!(1,4); cswap!(3,6); cswap!(2,4); cswap!(3,5); cswap!(3,4);
            }
            _ => unreachable!("sorting_network called with n={}, expected 3..=8", $n),
        }
    }};
}

/// Sort a slice of input pairs in-place into ascending `(left, right)` order.
///
/// Sorting networks for ≤8 pairs, insertion sort for ≤24, and `sort_unstable`
/// for larger, after a check for an already sorted slice. Pair lists carry no
/// sorted invariant; this is for a rewrite that canonicalizes a list before
/// pushing the node.
#[inline]
pub(crate) fn sort_pairs(pairs: &mut [ChildPair]) {
    let n = pairs.len();
    if n < 2 { return; }
    if n == 2 {
        if pairs[0] > pairs[1] { pairs.swap(0, 1); }
        return;
    }
    // Already sorted is the common case.
    let mut sorted = true;
    for i in 1..n {
        if pairs[i - 1] > pairs[i] { sorted = false; break; }
    }
    if sorted { return; }
    match n {
        3..=8 => { sorting_network!(pairs, n); }
        9..=24 => {
            // Insertion sort: O(n²) with a low constant at this length.
            for i in 1..n {
                let key = pairs[i];
                let mut j = i;
                while j > 0 && pairs[j - 1] > key {
                    pairs[j] = pairs[j - 1];
                    j -= 1;
                }
                pairs[j] = key;
            }
        }
        _ => {
            // Sort via explicit u64 key `(left << 32) | right` instead of the
            // derived field-by-field Ord. The derive expands to a branchy
            // `left.cmp(&right) else right.cmp(...)`; the u64 form is a single
            // unsigned compare and lets the branchless partition in
            // `sort_unstable` kick in.
            pairs.sort_unstable_by_key(|p| ((p.left.0 as u64) << 32) | (p.right.0 as u64));
        }
    }
}

#[cfg(test)]
#[path = "tests/pairs/mod.rs"]
mod tests;
