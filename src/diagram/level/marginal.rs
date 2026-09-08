//! Converting a level to its marginal form, and the marginal-side slot writer.

use crate::diagram::marg::{BigSide, MARG_OVERFLOW_TAG, ValueRef, marg_inline_max};
use super::TddLevel;

impl TddLevel {
    /// Inline-emit writer (end-of-apply tagger inner): for each bare marg-side
    /// slot ref, look up its child count and either INLINE it (bit-30 set) when
    /// small, or keep it a bare self-describing slot (bit-30 clear) when
    /// large/big-table.
    pub(crate) fn emit_marg_side_slots(
        &mut self,
        left_counts: Option<&[u128]>,
        right_counts: Option<&[u128]>,
    ) {
        // Inline rule: a marg-side ref is inlined whenever its count is
        // INLINABLE (≤ MARG_INLINE_MAX, not a u128::MAX overflow); a large or
        // overflow count stays a TAGGED SLOT. Counts need NOT be unique on the
        // child level — the count IS the anonymous identity of a marginal node,
        // so two slots sharing a count are interchangeable: count consumers read
        // the same value either way; the only structural use of a marg slot as a grid coordinate is the
        // invariant-forbidden marginal×marginal conjoin (marginal×identity is
        // pass-through, grid result discarded); and duplicate pairs are summed,
        // not deduped (see the `// No \`pairs.dedup()\`` notes in
        // conjoin_clause.rs / conjoin/sparse.rs), so collapsing two same-count
        // refs to one inline value preserves the total.
        // Rewrite one marg-side ref toward the inline OPTIMISATION. Under the
        // bit-30-clear==slot polarity bit 30 alone disambiguates — no marker or
        // self-describing flag needed:
        //   bit-31 set → ZERO sentinel, pass through.
        //   bit-30 set → already an inline count → idempotent pass-through.
        //   bit-30 clear → bare slot: resolve its count and INLINE it (set bit-30)
        //                  when small (and, under the gate, unique); else leave it
        //                  a bare slot — which is already a correct, self-describing
        //                  reference, so doing nothing is sound.
        fn emit_or_tag(raw: u32, counts: &[u128]) -> u32 {
            if raw & (1 << 31) != 0 {
                return raw; // ZERO sentinel
            }
            if raw & MARG_OVERFLOW_TAG != 0 {
                return raw; // already inline (bit-30 set) — idempotent
            }
            let slot = (raw & crate::diagram::marg::MARG_VALUE_MASK) as usize;
            if slot >= counts.len() {
                return raw; // OOB ⟹ keep as a bare slot
            }
            let c = counts[slot];
            // Inline whenever the count fits the inline width. Duplicates are
            // allowed: the count is the anonymous identity of a marginal node, and
            // duplicate pairs are summed (never deduped), so collapsing two
            // same-count refs to one inline value preserves the total.
            let inlinable = c != u128::MAX && c <= marg_inline_max() as u128;
            if inlinable {
                // Invariant: counts at marginalization are ≥ 1. Dead/UNSAT nodes
                // are zero-suppressed during apply and eliminated by prune_unreachable
                // before any count is taken; every surviving node therefore has at
                // least one model. Inline(0) is unreachable on any natural compile
                // path — only artificially-constructed TDDs (e.g. unit tests) can
                // produce it here.
                ValueRef::Inline(c as u32).to_raw().0 // INLINE: bit-30 set
            } else {
                raw // keep as a bare slot (bit-30 clear)
            }
        }
        for node in &mut self.nodes {
            if node.is_inline() {
                if let Some(lc) = left_counts {
                    node.a = emit_or_tag(node.a, lc);
                }
                if let Some(rc) = right_counts {
                    node.b = emit_or_tag(node.b, rc);
                }
            }
        }
        for p in &mut self.pairs {
            if let Some(lc) = left_counts {
                p.left.0 = emit_or_tag(p.left.0, lc);
            }
            if let Some(rc) = right_counts {
                p.right.0 = emit_or_tag(p.right.0, rc);
            }
        }
    }

    /// Replace this level's structure with per-node counts: `counts[i]` for
    /// node `i`, with `u128::MAX` marking an overflow whose exact value is
    /// `big.get(i)`.
    ///
    /// Marginality must stay downward-closed, so both child levels must
    /// already be marginal or leaves. This does not check; parents that refer
    /// to this level keep their indices, which remain valid as bare slot
    /// references (see [`resolve_marg_ref`](super::resolve_marg_ref)).
    pub fn make_marginal(&mut self, counts: Vec<u128>, big: Option<BigSide>) {
        self.nodes.clear();
        self.nodes.shrink_to_fit();
        self.pairs.clear();
        self.pairs.shrink_to_fit();
        self.ext.clear();
        self.ext.shrink_to_fit();
        // The node array is gone — its tombstone slots with it. Stale counter
        // would corrupt live_width() (width() is now marginal_counts.len()) and
        // make tombstone-aware readers index the empty node array (Tier 2).
        self.n_tombstones = 0;
        // The pair arena is gone, so its garbage accounting is too.
        self.dead_pairs = 0;
        // This level no longer has structural pairs, so the inline-emit markers
        // (which describe pair-field encoding) are meaningless — reset them.
        self.marg_flags = 0;
        self.marginal_counts = Some(counts);
        self.marginal_counts_big = big;
    }

    /// Weighted-mode analogue of [`make_marginal`](Self::make_marginal): clears the level's *pair*
    /// structure (`pairs`/`ext` — the O(width²) product grid, which is the
    /// memory win) and marks the level weight-marginal via the `MARG_WEIGHTED`
    /// flag. The per-node semiring values are stored by the caller in the
    /// external `WeightStore` (this level's `marginal_counts` stays `None`).
    ///
    /// Unlike [`make_marginal`](Self::make_marginal), `nodes` is KEPT (only pairs are freed): the
    /// integer path uses `marginal_counts.len()` as its width carrier, but the
    /// weighted store is external and `TddLevel` is at its size cap, so `width()`
    /// (= `nodes.len()` when `marginal_counts` is `None`) must keep reporting the
    /// real slot count. Marg-side refs are bare node-index slots (slot ≡ node
    /// index), so a parent's full-width refs stay in bounds and the `WeightStore`
    /// level (sized from `width()` before this call) matches `nodes.len()`.
    /// `n_tombstones` is left intact so `live_width()` stays correct.
    pub(crate) fn make_marginal_weighted(&mut self) {
        // Stash the slot count (= width, incl. tombstones) BEFORE clearing nodes.
        // Weight-marginal levels carry no `marginal_counts` width carrier, so
        // `width()` reads it back from `retired_marg_width` (repurposed: in
        // weighted mode this field is the LIVE slot count, not the integer arm's
        // retirement tally — slot_prune's `WeightFold::update_width` assigns it
        // the compacted store length where `IntFold::update_width` accumulates
        // freed slots, and the integer-mode retire metric is never consulted
        // here). This
        // lets us CLEAR nodes (like the integer `make_marginal`), so every
        // structural traversal that iterates `nodes`→`pairs_of` is a no-op on a
        // weight-marginal level instead of indexing the freed `pairs` and
        // panicking. The external `WeightStore` level (sized from `width()`
        // before this call) still matches this slot count, and parent marg-side
        // refs (bare node-index slots) stay in bounds.
        let n = self.nodes.len() as u32;
        self.make_marginal_weighted_with_slots(n);
    }

    /// As [`make_marginal_weighted`](Self::make_marginal_weighted) but with an explicit slot count (the streaming
    /// path remaps parent refs to compacted CELL indices, so the slot count is the
    /// number of alive cells, not `nodes.len()`).
    pub(crate) fn make_marginal_weighted_with_slots(&mut self, slots: u32) {
        self.retired_marg_width = slots;
        self.nodes.clear(); self.nodes.shrink_to_fit();
        self.pairs.clear(); self.pairs.shrink_to_fit();
        self.ext.clear(); self.ext.shrink_to_fit();
        self.dead_pairs = 0;
        self.marg_flags = Self::MARG_WEIGHTED;
        debug_assert!(self.marginal_counts.is_none());
    }
}
