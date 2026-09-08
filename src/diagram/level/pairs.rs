//! Reading a level: pair views, decoding, remapping, and per-node pair counts.

use crate::diagram::marg::MargRef;
use crate::diagram::packed::PairsIter;
use crate::diagram::primitives::{
    InputPair, LocalNodeIdx, TddNodeData,
    LEAF_BIT, MULTI_BIT, EXT_SENTINEL,
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
        if node.b == EXT_SENTINEL {
            let e = &self.ext[(node.a & !MULTI_BIT) as usize];
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
            //         InputPair is #[repr(C)] {left: LocalNodeIdx(u32), right: LocalNodeIdx(u32)}.
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

    /// Slice view of a node's pairs, transparently handling packed levels.
    ///
    /// For unpacked levels, returns a direct borrow into `self.pairs` —
    /// zero copy, same semantics as `pairs_of_idx`. For packed levels,
    /// decodes pairs into the caller-provided `scratch` buffer and returns
    /// a borrow of that. The scratch is cleared first; the caller is
    /// responsible for keeping it alive for the duration of the returned
    /// slice (the borrow checker enforces this via the shared `'a`).
    ///
    /// Use this in hot paths that genuinely need slice semantics
    /// (random index, sub-slicing, sort/binsearch) but must work on both
    /// packed and unpacked levels. For sequential iteration, prefer
    /// `pairs_iter_of_idx`.
    #[inline(always)]
    pub(crate) fn pairs_view_into<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<InputPair>,
    ) -> &'a [InputPair] {
        debug_assert!(
            !self.is_marginal(),
            "pairs_view_into({idx}) called on marginal level",
        );
        let d = &self.nodes[idx];
        if d.is_leaf() {
            &[]
        } else if d.is_multi() {
            let range = self.multi_range(d);
            &self.pairs[range]
        } else {
            // Inline node: zero-cost pointer cast to a single-element slice.
            //
            // SAFETY: TddNodeData is #[repr(C)] {a: u32, b: u32};
            //         InputPair is #[repr(C)] {left: LocalNodeIdx(u32),
            //         right: LocalNodeIdx(u32)} — identical layout.
            //         For inline nodes the (a,b) fields hold (left,right)
            //         by construction.
            let _ = scratch; // scratch unused on this fast path
            unsafe { std::slice::from_ref(&*(d as *const TddNodeData as *const InputPair)) }
        }
    }

    /// Like `pairs_view_into`, but decodes marg-side fields to bare slot
    /// indices for structural use. `left_mask`/`right_mask` are
    /// `MARG_VALUE_MASK` when the corresponding child level is marginal,
    /// `u32::MAX` (identity) otherwise. When neither side needs decoding
    /// (both masks identity) this defers to the zero-copy `pairs_view_into`
    /// — the common non-marginal path pays nothing. When a side IS marginal,
    /// it materializes a decoded copy into `scratch` (gated, so the fast path
    /// stays a borrow). See `decode_marg_coord` for the per-field semantics.
    #[inline(always)]
    pub(crate) fn pairs_view_decoded<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<InputPair>,
        left_mask: u32,
        right_mask: u32,
    ) -> &'a [InputPair] {
        if left_mask == u32::MAX && right_mask == u32::MAX {
            return self.pairs_view_into(idx, scratch);
        }
        scratch.clear();
        self.decode_pairs_into(idx, scratch, left_mask, right_mask);
        scratch.as_slice()
    }

    /// Append `idx`'s pairs, marg-decoded, onto `out` (no clear — callers
    /// append). The ONE decode loop shared by `pairs_view_decoded` (per-cell
    /// scratch) and the per-level decode arena of `conjoin::cell`'s per-column
    /// descriptor table. Caller pre-reserves `out` when the
    /// total is known (the pushes here are then realloc-free).
    #[inline]
    pub(crate) fn decode_pairs_into(
        &self,
        idx: usize,
        out: &mut Vec<InputPair>,
        left_mask: u32,
        right_mask: u32,
    ) {
        for p in self.pairs_iter_of_idx(idx) {
            out.push(InputPair {
                left: LocalNodeIdx(crate::diagram::marg::decode_marg_coord(p.left.0, left_mask)),
                right: LocalNodeIdx(crate::diagram::marg::decode_marg_coord(p.right.0, right_mask)),
            });
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
                left: LocalNodeIdx(node.a),
                right: LocalNodeIdx(node.b),
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
        left_marg: bool,
        right_marg: bool,
    ) {
        if self.nodes[idx].is_leaf() {
            return;
        }
        debug_assert!(self.nodes[idx].is_multi(),
            "pairs_remap_indexed called on inline node");
        // A marg-side ref is slot-tagged (bit 30): mask before indexing the
        // child's remap, re-tag the compacted slot on write. Non-marg side
        // indexes verbatim.
        // Phase B: an inline ref (bit 30 clear) carries a bare count, not a slot
        // index — pass it through verbatim; only slot refs index the remap. On
        // the pure-slot path (Step A) every marg-side ref is a slot, so this is
        // behavior-preserving.
        let lf = |l: u32| -> u32 {
            if left_marg {
                match MargRef::from_raw(l) {
                    MargRef::Slot(s) => MargRef::slot_raw(left_remap[s as usize]),
                    MargRef::Inline(_) => {
                        l
                    }
                }
            } else {
                left_remap[l as usize]
            }
        };
        let rf = |r: u32| -> u32 {
            if right_marg {
                match MargRef::from_raw(r) {
                    MargRef::Slot(s) => MargRef::slot_raw(right_remap[s as usize]),
                    MargRef::Inline(_) => {
                        r
                    }
                }
            } else {
                right_remap[r as usize]
            }
        };
        let range = self.multi_range(&self.nodes[idx]);
        for pair in &mut self.pairs[range] {
            pair.left = LocalNodeIdx(lf(pair.left.0));
            pair.right = LocalNodeIdx(rf(pair.right.0));
        }
    }

    /// Pair-arena start offset for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn multi_start_at(&self, idx: usize) -> usize {
        let n = &self.nodes[idx];
        debug_assert!(n.is_multi());
        if n.b == EXT_SENTINEL {
            self.ext[(n.a & !MULTI_BIT) as usize].start as usize
        } else {
            (n.a & !MULTI_BIT) as usize
        }
    }

    /// Pair count for a multi-pair node at `idx` (normal or extended).
    #[inline]
    pub(crate) fn multi_len_at(&self, idx: usize) -> usize {
        let n = &self.nodes[idx];
        debug_assert!(n.is_multi());
        if n.b == EXT_SENTINEL {
            self.ext[(n.a & !MULTI_BIT) as usize].len as usize
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
