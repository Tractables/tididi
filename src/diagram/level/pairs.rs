//! Reading a level: pair views, decoding, remapping, per-node pair counts, and
//! the canonical sort a rewritten pair list is put back in.

use crate::diagram::marginal_ref::SideView;
use crate::diagram::packed::PairsIter;
use crate::diagram::primitives::{
    InputPair, NodeIdx, TddNodeData,
    LEAF_BIT, MULTI_BIT, RANGE_SENTINEL,
};
use super::TddLevel;

impl TddLevel {
    /// Every node with pairs, as `(local index, pairs)`, skipping tombstones.
    /// The index is the node's slot in `nodes`, so it is valid for arrays
    /// sized by `width()`. Empty on a marginal level.
    pub fn internal_inputs_iter(&self) -> impl Iterator<Item = (usize, PairsIter<'_>)> + '_ {
        self.nodes.iter().enumerate().filter_map(|(i, n)| {
            if n.is_internal() { Some((i, self.pairs_iter_of(n))) } else { None }
        })
    }

    /// Resolve a multi-pair node's pair range, transparently handling both the
    /// normal (packed) and extended (side-table) encodings.
    #[inline(always)]
    pub(crate) fn multi_range(&self, node: &TddNodeData) -> std::ops::Range<usize> {
        debug_assert!(node.is_multi());
        if node.b == RANGE_SENTINEL {
            let e = &self.multi_pairs[(node.a & !MULTI_BIT) as usize];
            (e.start as usize)..((e.start + e.len) as usize)
        } else {
            let start = (node.a & !MULTI_BIT) as usize;
            start..start + node.b as usize
        }
    }

    /// The pairs of `node`, which must be a node of this level. Empty for a
    /// tombstone.
    #[inline(always)]
    pub fn pairs_of(&self, node: &TddNodeData) -> &[InputPair] {
        if node.is_leaf() { return &[]; }
        if node.is_multi() {
            &self.pairs[self.multi_range(node)]
        } else {
            // SAFETY: TddNodeData is #[repr(C)] {a: u32, b: u32}.
            //         InputPair is #[repr(C)] {left: NodeIdx(u32), right: NodeIdx(u32)}.
            //         For inline nodes, a == left.0 and b == right.0 by construction.
            //         Both types have identical {u32, u32} layout, so the cast is valid.
            unsafe { std::slice::from_ref(&*(node as *const TddNodeData as *const InputPair)) }
        }
    }

    /// [`pairs_of`](Self::pairs_of) by node index; not valid on a marginal level.
    #[inline(always)]
    pub fn pairs_of_idx(&self, idx: usize) -> &[InputPair] {
        // Unreachable in production: the structural check at
        // `conjoin_clause_into` entry and the per-operand marginal branches
        // in `apply_and` route around marginal levels before they reach here.
        // A debug_assert! rather than a check, to avoid a hot-path branch —
        // debug builds and tests keep the safety net.
        debug_assert!(
            !self.is_marginal(),
            "pairs_of_idx({idx}) called on marginal level (width={}, nodes.len()={}, pairs.len()={}). \
             Callers must guard via is_marginal() — marginal levels store model counts, \
             not pair structure.",
            self.width(), self.nodes.len(), self.pairs.len(),
        );
        let d = &self.nodes[idx];
        if d.b & LEAF_BIT != 0 { return &[]; }
        if d.a & MULTI_BIT != 0 {
            &self.pairs[self.multi_range(d)]
        } else {
            // SAFETY: same layout guarantee as in pairs_of.
            unsafe { std::slice::from_ref(&*(d as *const TddNodeData as *const InputPair)) }
        }
    }

    /// [`pairs_iter_of`](Self::pairs_iter_of) by node index; not valid on a
    /// marginal level.
    #[inline(always)]
    pub fn pairs_iter_of_idx(&self, idx: usize) -> PairsIter<'_> {
        debug_assert!(
            !self.is_marginal(),
            "pairs_iter_of_idx({idx}) called on marginal level",
        );
        let d = &self.nodes[idx];
        self.pairs_iter_of(d)
    }

    /// Like [`pairs_of_idx`](Self::pairs_of_idx), but decodes marginal-side fields to the bare
    /// coordinates structural use wants ([`SideView::coord`]).
    ///
    /// With neither child marginal this defers to the zero-copy
    /// `pairs_of_idx`, so the common path pays nothing; when a side is
    /// valued it materializes a decoded copy into `scratch`.
    #[inline(always)]
    pub(crate) fn pairs_view_decoded<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<InputPair>,
        left: SideView,
        right: SideView,
    ) -> &'a [InputPair] {
        if !left.is_marginal() && !right.is_marginal() {
            return self.pairs_of_idx(idx);
        }
        scratch.clear();
        self.decode_pairs_into(idx, scratch, left, right);
        scratch.as_slice()
    }

    /// Append `idx`'s pairs, marginal-decoded, onto `out` (no clear — callers
    /// append). The one decode loop shared by `pairs_view_decoded` (per-cell
    /// scratch) and the per-level decode arena of `conjoin::cell`'s per-column
    /// descriptor table. Caller pre-reserves `out` when the
    /// total is known (the pushes here are then realloc-free).
    #[inline]
    pub(crate) fn decode_pairs_into(
        &self,
        idx: usize,
        out: &mut Vec<InputPair>,
        left: SideView,
        right: SideView,
    ) {
        for p in self.pairs_iter_of_idx(idx) {
            out.push(InputPair { left: left.coord(p.left), right: right.coord(p.right) });
        }
    }

    /// The pairs of `node` as an iterator; the owned-item twin of
    /// [`pairs_of`](Self::pairs_of).
    #[inline]
    pub fn pairs_iter_of<'a>(&'a self, node: &'a TddNodeData) -> PairsIter<'a> {
        if node.is_leaf() {
            return PairsIter::empty();
        }
        if node.is_multi() {
            let range = self.multi_range(node);
            PairsIter::slice(&self.pairs[range])
        } else {
            // Inline node: a / b directly hold the pair fields.
            PairsIter::inline(InputPair {
                left: NodeIdx(node.a),
                right: NodeIdx(node.b),
            })
        }
    }

    /// Get mutable access to a multi-pair node's pairs in the arena.
    /// Only valid for multi-pair nodes; panics on inline nodes.
    #[inline]
    pub(crate) fn pairs_mut(&mut self, idx: usize) -> &mut [InputPair] {
        if self.nodes[idx].is_leaf() {
            return &mut [];
        }
        debug_assert!(self.nodes[idx].is_multi(),
            "pairs_mut called on inline node");
        let range = self.multi_range(&self.nodes[idx]);
        &mut self.pairs[range]
    }

    /// Index-remap a multi-pair node's pairs in place via two lookup
    /// slices. Transparently handles packed vs unpacked storage:
    ///
    /// - Packed: decode each u32 word to (left,right) in registers,
    ///   index into the remap slices, re-encode and write back to the
    ///   same word. Never materializes `InputPair` in memory.
    /// - Unpacked: standard slice rewrite via `pairs_mut`.
    ///
    /// Used by prune's bottom-up pair-rewrite, which therefore operates
    /// directly on the packed representation and never has to unpack first.
    ///
    /// Preconditions:
    /// - `self.nodes[idx].is_multi()` (debug-asserted)
    /// - For packed levels: `left_remap[i] < 2^bits_left` and
    ///   `right_remap[j] < 2^bits_right` for all values that will be
    ///   looked up (debug-asserted).
    ///   Prune satisfies this because the remap is monotone
    ///   non-increasing — new indices ≤ old indices ≤ original
    ///   per-side bounds.
    #[inline]
    pub(crate) fn pairs_remap_indexed(
        &mut self,
        idx: usize,
        left_remap: &[u32],
        right_remap: &[u32],
        left: SideView,
        right: SideView,
    ) {
        if self.nodes[idx].is_leaf() {
            return;
        }
        debug_assert!(self.nodes[idx].is_multi(),
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
        let n = &self.nodes[idx];
        debug_assert!(n.is_multi());
        if n.b == RANGE_SENTINEL {
            self.multi_pairs[(n.a & !MULTI_BIT) as usize].start as usize
        } else {
            (n.a & !MULTI_BIT) as usize
        }
    }

    /// Pair count for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn multi_len_at(&self, idx: usize) -> usize {
        let n = &self.nodes[idx];
        debug_assert!(n.is_multi());
        if n.b == RANGE_SENTINEL {
            self.multi_pairs[(n.a & !MULTI_BIT) as usize].len as usize
        } else {
            n.b as usize
        }
    }

    /// Pair-arena range for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn pair_range_at(&self, idx: usize) -> std::ops::Range<usize> {
        self.multi_range(&self.nodes[idx])
    }

    /// Number of pairs of the internal node at `idx`.
    #[inline]
    pub fn pair_count_at(&self, idx: usize) -> usize {
        let n = &self.nodes[idx];
        debug_assert!(n.is_internal());
        if n.is_inline() { 1 } else { self.multi_len_at(idx) }
    }

}

/// Sorting network for 3..=8 elements (optimal compare-swap counts).
///
/// Works on any indexable + swappable container. Each case is a hardcoded
/// sequence of conditional swaps (`cswap!(a, b)` = "if s[a] > s[b], swap
/// them"). Faster than general-purpose sort for small n because the comparison
/// sequence is known at compile time, enabling branch-free code generation.
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
                // Green's construction (Knuth TAOCP Vol 3, 16 comparators).
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
/// Optimized for the small pair counts typical in diagram nodes: uses sorting
/// networks for ≤8 pairs, insertion sort for ≤24, and `sort_unstable` for
/// larger. Checks if already sorted first, which is common after a conjunction.
///
/// This is a localized helper for the specific node-construction paths that
/// build a pair list in arbitrary order and must canonicalize it before pushing
/// the node — projection, conditioning and restriction. It is not part of the general
/// pair-storage contract: pair lists carry no globally-maintained sorted
/// invariant, the apply/conjoin hot path never calls this, and no data layout
/// assumes sorted order.
#[inline]
pub(crate) fn sort_pairs(pairs: &mut [InputPair]) {
    let n = pairs.len();
    if n < 2 { return; }
    if n == 2 {
        if pairs[0] > pairs[1] { pairs.swap(0, 1); }
        return;
    }
    // Check if already sorted (common after apply_and which builds sorted pairs).
    let mut sorted = true;
    for i in 1..n {
        if pairs[i - 1] > pairs[i] { sorted = false; break; }
    }
    if sorted { return; }
    match n {
        3..=8 => { sorting_network!(pairs, n); }
        9..=24 => {
            // Insertion sort for small-medium lists: O(n²) but low constant
            // factor, no recursion overhead, excellent cache behavior, and no
            // partitioning overhead to amortize at this length.
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
#[path = "../../pair_sort_tests.rs"]
mod pair_sort_tests;
