//! `TddLevel` struct and its full impl.

use super::marg::{
    BigSide, MARG_OVERFLOW_TAG, MargRef, marg_inline_max,
};
use super::packed::PairsIter;
use super::primitives::{
    ExtMulti, InputPair, LocalNodeIdx, TddNodeData,
    LEAF_BIT, MULTI_BIT, EXT_SENTINEL,
};
// `types/marg.rs` already depends on the apply-side error/fallible-push
// primitives (`resolve_swapped_marg_side`) — this is the same established
// cross-dependency, not a new one, needed for `reencode_shrunk_multi`'s
// `ext` push.
use crate::tdd::transform::pairwise::conjoin::{try_push, ApplyError};

/// All t-nodes for a single vtree node t (one level of the TDD).
///
/// Single-pair nodes (the majority) store their pair inline in the node itself.
/// Multi-pair nodes reference a contiguous slice in the shared `pairs` arena.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct TddLevel {
    /// The t-nodes stored at this level, indexed by `LocalNodeIdx`; `width()` is its length.
    pub nodes: Vec<TddNodeData>,
    /// Shared arena for all internal nodes' input pairs at this level.
    pub pairs: Vec<InputPair>,
    /// Side table for extended multi-pair nodes (those whose `pair_start` or `pair_len`
    /// exceeds 2^31). Normally empty; only grows for pathological cases with huge
    /// product grids. See `TddNodeData` docs for the four-way encoding.
    pub ext: Vec<ExtMulti>,
    /// Marginal-ref state flags, packed. `marg_inlined_left` (bit 0): this
    /// level's `pairs[*].left` fields toward a marginal left child already hold
    /// INLINE MODEL COUNTS (bit-30 clear bare values), NOT fresh slot indices.
    /// Set in two places: (1) the apply pass-through path, which carries an
    /// already-inlined carrier field through verbatim; (2) the end-of-apply
    /// tagger after it emits this side. Checked by the tagger's emit arm to
    /// SKIP re-emitting (re-running `emit_or_tag` on an inline count would
    /// misread it as a slot → `counts[C]` corruption). The non-emit tag path
    /// (`| TAG`, idempotent) ignores this marker. Reset by `clear()` and
    /// `make_marginal`. `marg_inlined_right` (bit 1) mirrors for the right side.
    /// (Bit 2 is free.)
    ///
    /// Packed as a bitfield (not separate `bool`s) so the flags + `n_tombstones`
    /// fit `TddLevel`'s padding without crossing the 128 B / 2-cache-line
    /// boundary (see the size assert below). Access via the
    /// `marg_inlined_left()`/`set_marg_inlined_left(..)` style methods.
    pub marg_flags: u8,
    /// Number of tombstone slots in `nodes` — dead nodes the index-stable
    /// conjoin (Tier 2) leaves in place instead of compacting out. 0 on the
    /// dense path. `width()` still counts every slot (it is the index bound for
    /// flat-array allocation); `live_width()` subtracts this. Reset to 0 by
    /// `clear()` and after prune compaction (which physically removes them).
    pub n_tombstones: u32,
    /// Slots freed from this level's marginal store by `prune_marg_slots`
    /// (deep clears + boundary compaction). Monotone per level; reset only by
    /// `clear()`. Travels with the level through `mem::swap` (apply swaps whole
    /// levels between TDDs), so `Tdd::retired_marg_total()` — the sum over all
    /// levels — correctly follows the circuit lineage the gates already use for
    /// `total_nodes()`.
    ///
    /// Purpose: threshold-offset gating in the downstream compile driver. Each adaptive-minimize
    /// baseline records the `retired_marg_total()` at snapshot time; at gate
    /// comparison the difference (`collected_since`) is added to `total_nodes()`
    /// so that slot-pruning does not silently deflate the metric and delay
    /// minimize triggers. `total_nodes()` itself remains the honest
    /// surviving-circuit count.
    pub retired_marg_width: u32,
    /// Slots in `pairs` that no live node references any more.
    ///
    /// Twin contraction mints these: a merged union is appended at the arena
    /// tail (`concat_twin_pairs`), abandoning every source range, and the parent
    /// rewrite / duplicate resolution shrink pair lists in place, abandoning
    /// their tails. `compact_pairs_if_stale`
    /// reclaims them and resets this to 0.
    ///
    /// APPROXIMATE by design — it is only the sweep TRIGGER, so an over- or
    /// under-count shifts *when* the sweep runs, never which bytes it moves
    /// (the sweep derives liveness from `nodes`/`ext`, not from this counter).
    /// Not every garbage source feeds it: prune drops a node without accounting
    /// its range (see `prune.rs`), so prune-only garbage waits for a
    /// contraction-triggered sweep instead of triggering one. Reset to 0
    /// wherever the pair arena is replaced or dropped wholesale — an
    /// enumeration here would rot; the sites are grep-able as `dead_pairs = 0`.
    pub dead_pairs: u32,
    /// Marginal mode: level stores only per-node model counts, no structure.
    /// When Some, `nodes`/`pairs`/`ext` are empty; width = `marginal_counts.len()`.
    pub marginal_counts: Option<Vec<u128>>,
    /// Overflow storage for marginal counts exceeding u128 (sentinel
    /// `u128::MAX`). SPARSE — keyed by slot index, sized by the overflow set,
    /// not by the level's width; see [`BigSide`]. `None` and an empty table are
    /// equivalent to every reader (both mean "no slot overflowed") and both own
    /// zero heap.
    pub marginal_counts_big: Option<BigSide>,
}

/// `TddLevel` should stay compact — the hot sequential-scan stride depends on it.
/// `n_tombstones: u32` was placed among the bool fields to fit existing padding,
/// as was `dead_pairs: u32` (the pairs-arena garbage counter).
/// `retired_marg_width: u32` (metric-only retire counter) grew the struct to
/// 136 B (one extra 8-byte slot). The hot per-node / per-pair minimize loops
/// iterate a level's *heap-backed* `nodes`/`pairs` arenas, not the `TddLevel`
/// structs themselves, so only the O(levels) sweeps (shrink, `total_nodes`)
/// see the stride.
const _: () = assert!(
    std::mem::size_of::<TddLevel>() <= 160,
    "TddLevel grew past 160 B"
);

impl Default for TddLevel {
    /// Returns an empty level, identical to [`TddLevel::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl TddLevel {
    /// Bit positions in `marg_flags`. See the field doc.
    pub const MARG_INLINED_LEFT: u8 = 1 << 0;
    /// Right marg-child of a boundary parent is inline-encoded in the pair field
    /// (companion of [`MARG_INLINED_LEFT`](Self::MARG_INLINED_LEFT)).
    pub const MARG_INLINED_RIGHT: u8 = 1 << 1;
    // bit 2 free.
    /// Weighted/algebraic marginalization: this level has been marginalized in
    /// `--weighted` mode. Its per-node semiring values live in the external
    /// `WeightStore` side-table (indexed by vtree level), NOT in `marginal_counts`
    /// (which stays `None`). Keeps `TddLevel` at its 136 B size budget — adding a
    /// `Vec<BigRational>` field would overflow it. Never set on the integer `--mc`
    /// path, so `is_marginal()` stays byte-identical there.
    pub const MARG_WEIGHTED: u8 = 1 << 3;

    /// True if the left marg-child inline-encoding flag is set.
    #[inline(always)]
    pub fn marg_inlined_left(&self) -> bool {
        self.marg_flags & Self::MARG_INLINED_LEFT != 0
    }
    /// True if the right marg-child inline-encoding flag is set.
    #[inline(always)]
    pub fn marg_inlined_right(&self) -> bool {
        self.marg_flags & Self::MARG_INLINED_RIGHT != 0
    }
    /// Set or clear the left marg-child inline-encoding flag.
    #[inline(always)]
    pub fn set_marg_inlined_left(&mut self, v: bool) {
        if v { self.marg_flags |= Self::MARG_INLINED_LEFT }
        else { self.marg_flags &= !Self::MARG_INLINED_LEFT }
    }
    /// Set or clear the right marg-child inline-encoding flag.
    #[inline(always)]
    pub fn set_marg_inlined_right(&mut self, v: bool) {
        if v { self.marg_flags |= Self::MARG_INLINED_RIGHT }
        else { self.marg_flags &= !Self::MARG_INLINED_RIGHT }
    }

    /// Create an empty level with no nodes.
    pub fn new() -> Self {
        TddLevel {
            nodes: Vec::new(),
            pairs: Vec::new(),
            ext: Vec::new(),
            marg_flags: 0,
            n_tombstones: 0,
            retired_marg_width: 0,
            dead_pairs: 0,
            marginal_counts: None,
            marginal_counts_big: None,
        }
    }

    /// Reset to empty in place, preserving allocated buffer capacity.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.pairs.clear();
        self.ext.clear();
        self.marg_flags = 0;
        self.n_tombstones = 0;
        self.retired_marg_width = 0;
        self.dead_pairs = 0;
        self.marginal_counts = None;
        self.marginal_counts_big = None;
    }

    /// Slot count of the level: `marginal_counts.len()` on an integer-marginal
    /// level, `retired_marg_width` on a weight-marginal level, else `nodes.len()`
    /// (live + tombstone slots). Use for index bounds and flat-array sizing; use
    /// [`live_width`](Self::live_width) for reported node counts.
    pub fn width(&self) -> usize {
        if let Some(counts) = &self.marginal_counts {
            counts.len()
        } else if self.is_weight_marginal() {
            // Nodes are cleared on weight-marginal levels; the slot count lives
            // in `retired_marg_width` (set by `make_marginal_weighted`).
            self.retired_marg_width as usize
        } else {
            self.nodes.len()
        }
    }

    /// Number of *live* nodes — `width()` minus tombstone slots. Use for
    /// reporting node counts (`total_nodes`, `max_width`); use `width()` for
    /// index bounds and flat-array allocation (tombstone slots still occupy a
    /// position). Equal to `width()` on the dense path (`n_tombstones == 0`).
    pub fn live_width(&self) -> usize {
        self.width() - self.n_tombstones as usize
    }

    /// True iff any node at this level has >1 input pair. Recomputed on demand
    /// by scanning `nodes` (was formerly a cached `has_multi_pair` bool that
    /// every pair-list mutation had to keep in sync). A single-pair extended
    /// node (`is_multi()` but `multi_len == 1`) does NOT count — the check
    /// mirrors the old `pair_len >= 2` setter condition. Marginal levels have
    /// empty `nodes`, so this is `false` there (matching the old reset).
    #[inline]
    pub fn has_multi_pair(&self) -> bool {
        (0..self.nodes.len()).any(|i| {
            self.nodes[i].is_multi() && self.multi_len_at(i) >= 2
        })
    }

    /// True if this level is in marginal mode (no nodes/pairs, just counts).
    pub fn is_marginal(&self) -> bool {
        self.marginal_counts.is_some() || self.is_weight_marginal()
    }

    /// Weighted-marginal: structure cleared, per-node semiring values live in the
    /// external `WeightStore`. Mutually exclusive with integer-marginal
    /// (`marginal_counts.is_some()`) within a single compile.
    #[inline(always)]
    pub fn is_weight_marginal(&self) -> bool {
        self.marg_flags & Self::MARG_WEIGHTED != 0
    }

    /// Inline-emit writer (end-of-apply tagger inner): for each bare marg-side
    /// slot ref, look up its child count and either INLINE it (bit-30 set) when
    /// small, or keep it a bare self-describing slot (bit-30 clear) when
    /// large/big-table.
    pub fn emit_marg_side_slots(
        &mut self,
        left_counts: Option<&[u128]>,
        right_counts: Option<&[u128]>,
    ) {
        // Inline rule (#59): a marg-side ref is inlined whenever its count is
        // INLINABLE (≤ MARG_INLINE_MAX, not a u128::MAX overflow); a large or
        // overflow count stays a TAGGED SLOT. Counts need NOT be unique on the
        // child level — the count IS the anonymous identity of a marginal node,
        // so two slots sharing a count are interchangeable: count consumers read
        // the same value either way; the only structural use of a marg slot as a grid coordinate is the
        // invariant-forbidden marginal×marginal conjoin (marginal×identity is
        // pass-through, grid result discarded); and duplicate pairs are summed,
        // not deduped (see the `// No \`pairs.dedup()\`` notes in apply_clause.rs
        // / conjoin/sparse.rs), so collapsing two same-count refs to one
        // inline value preserves the total. This retires the old freq==1
        // unique-inlinable filter.
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
            let slot = (raw & super::marg::MARG_VALUE_MASK) as usize;
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
                MargRef::Inline(c as u32).to_raw() // INLINE: bit-30 set
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

    /// Trim retained slack in `nodes`, `pairs`, and `ext` when capacity exceeds
    /// `(eighths / 8)` × length AND absolute capacity is ≥ 1 Ki slots. Called
    /// after a level is finalized in apply to release the Vec-doubling
    /// overshoot from the per-cell `try_push` emit loop, and by
    /// [`compact_pairs_if_stale`](Self::compact_pairs_if_stale) once it has
    /// truncated the pairs arena — this is the level's ONE decision about
    /// returning slack to the allocator. The ratio trades
    /// peak savings against realloc-copies on hot levels that get re-grown
    /// soon. Fixed at 32 (= 4×); the `TIDIDI_SHRINK_RATIO_EIGHTHS` override
    /// (range 9..=64) was removed. Marginal levels (already shrunk by `make_marginal`)
    /// are skipped.
    #[inline]
    pub(crate) fn shrink_arrays(&mut self) {
        if self.marginal_counts.is_some() {
            return;
        }
        const MIN_SHRINK_CAP: usize = 1024;
        // cap > (32 / 8) * len  ⟺  8 * cap > 32 * len  ⟺  cap > 4 * len
        const SHRINK_RATIO_EIGHTHS: u64 = 32; // fixed at 4× (formerly TIDIDI_SHRINK_RATIO_EIGHTHS)
        let should_shrink = |cap: usize, len: usize| -> bool {
            cap >= MIN_SHRINK_CAP && (cap as u64) * 8 > SHRINK_RATIO_EIGHTHS * (len as u64)
        };
        if should_shrink(self.nodes.capacity(), self.nodes.len()) {
            self.nodes.shrink_to_fit();
        }
        if should_shrink(self.pairs.capacity(), self.pairs.len()) {
            self.pairs.shrink_to_fit();
        }
        if should_shrink(self.ext.capacity(), self.ext.len()) {
            self.ext.shrink_to_fit();
        }
    }

    /// Source-agnostic pair count. Use this for `pair_start` snapshots in
    /// the emit loop.
    #[inline]
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Source-agnostic pop, returns the last pair from the pairs arena.
    /// Used by `emit_product_node!`'s 1-pair inline path.
    #[inline]
    pub fn pop_pair(&mut self) -> Option<InputPair> {
        self.pairs.pop()
    }

    /// Length of the pair-arena tail starting at `start`.
    ///
    /// Pair lists are unordered sets and no operation requires a particular
    /// order (twin contraction is order-independent), so apply emit sites no
    /// longer sort the tail — they just count it via this. See the NOTE at the
    /// bottom of this file.
    #[inline]
    pub fn pair_tail_len(&self, start: usize) -> usize {
        self.pairs.len() - start
    }

    /// Convert this level to marginal mode given pre-computed per-node counts.
    ///
    /// Callers must satisfy the marginalization precondition: both children
    /// of the target vtree node must already be marginal (or be leaves).
    /// See `assert_can_make_marginal` for the soundness check and the
    /// `marginalization` WIP for the rationale. This raw entry point does
    /// NOT check — callers in production should go through
    /// `assert_can_make_marginal` first, or use a future guarded TDD-level
    /// entry point.
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
    pub fn make_marginal_weighted(&mut self) {
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
    pub fn make_marginal_weighted_with_slots(&mut self, slots: u32) {
        self.retired_marg_width = slots;
        self.nodes.clear(); self.nodes.shrink_to_fit();
        self.pairs.clear(); self.pairs.shrink_to_fit();
        self.ext.clear(); self.ext.shrink_to_fit();
        self.dead_pairs = 0;
        self.marg_flags = Self::MARG_WEIGHTED;
        debug_assert!(self.marginal_counts.is_none());
    }

    /// Iterate internal nodes, yielding `(local_index, pairs_iter)`, transparently
    /// handling both packed and unpacked levels via `PairsIter`.
    ///
    /// Tombstone-tolerant for free: tombstones report `is_internal() == false`
    /// (see `TOMBSTONE_B`), so they are filtered out. The yielded `i` stays the
    /// physical slot index, which is what flat count arrays sized by `width()`
    /// expect — so all model-count paths skip tombstones without change.
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

    /// Get the input pairs for a node. Returns `&[]` for leaf nodes.
    /// For inline nodes, returns a single-element slice via pointer cast (zero cost).
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

    /// Get the input pairs for a node by index. Returns `&[]` for leaf nodes.
    /// For inline nodes, returns a single-element slice via pointer cast (zero cost).
    #[inline(always)]
    pub fn pairs_of_idx(&self, idx: usize) -> &[InputPair] {
        // Unreachable in production: the structural check at
        // `try_apply_and_clause` entry (apply_inner.rs) and the per-operand
        // marginal branches in `apply_and` route around marginal levels
        // before they reach here. Downgraded to debug_assert! to avoid a
        // hot-path branch — debug builds and tests keep the safety net.
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

    /// Iterator-based pair accessor for a node by index. Yields owned
    /// `InputPair`s from `self.pairs` (caller pays a copy; `InputPair: Copy`
    /// keeps it cheap). Equivalent to `pairs_iter_of(nodes[idx])`.
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
    pub fn pairs_view_into<'a>(
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
    pub fn pairs_view_decoded<'a>(
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
    /// scratch) and the per-level prepared-operand arena
    /// (`conjoin::cell::PreparedC2`). Caller pre-reserves `out` when the
    /// total is known (the pushes here are then realloc-free).
    #[inline]
    pub fn decode_pairs_into(
        &self,
        idx: usize,
        out: &mut Vec<InputPair>,
        left_mask: u32,
        right_mask: u32,
    ) {
        for p in self.pairs_iter_of_idx(idx) {
            out.push(InputPair {
                left: LocalNodeIdx(super::marg::decode_marg_coord(p.left.0, left_mask)),
                right: LocalNodeIdx(super::marg::decode_marg_coord(p.right.0, right_mask)),
            });
        }
    }

    /// Same as `pairs_iter_of_idx` but takes a `&TddNodeData` directly.
    /// Mirror of `pairs_of` for iterator semantics.
    #[inline]
    pub fn pairs_iter_of<'a>(&'a self, node: &'a TddNodeData) -> PairsIter<'a> {
        if node.is_leaf() {
            return PairsIter::Empty;
        }
        if node.is_multi() {
            let range = self.multi_range(node);
            PairsIter::Slice(self.pairs[range].iter())
        } else {
            // Inline node: a / b directly hold the pair fields.
            PairsIter::Inline(Some(InputPair {
                left: LocalNodeIdx(node.a),
                right: LocalNodeIdx(node.b),
            }))
        }
    }

    /// Get mutable access to a multi-pair node's pairs in the arena.
    /// Only valid for multi-pair nodes; panics on inline nodes.
    #[inline]
    pub fn pairs_mut(&mut self, idx: usize) -> &mut [InputPair] {
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
    /// Used by prune's bottom-up pair-rewrite (`prune.rs:286`) so we
    /// can skip `unpack_all_levels` before prune — the prune pass now
    /// operates directly on the Phase F packed representation.
    ///
    /// Preconditions:
    /// - `self.nodes[idx].is_multi()` (debug-asserted)
    /// - For packed levels: `left_remap[i] < 2^bits_left` and
    ///   `right_remap[j] < 2^bits_right` for all values that will be
    ///   looked up (debug-asserted via `remap_range_in_place`).
    ///   Prune satisfies this because the remap is monotone
    ///   non-increasing — new indices ≤ old indices ≤ original
    ///   per-side bounds.
    #[inline]
    pub fn pairs_remap_indexed(
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
    pub fn multi_start_at(&self, idx: usize) -> usize {
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
    pub fn multi_len_at(&self, idx: usize) -> usize {
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
    pub fn pair_range_at(&self, idx: usize) -> std::ops::Range<usize> {
        self.multi_range(&self.nodes[idx])
    }

    /// Pair count for any internal node (1 for inline, actual count for multi).
    /// Handles normal and extended multi-pair encodings.
    #[inline]
    pub fn pair_count_at(&self, idx: usize) -> usize {
        let n = &self.nodes[idx];
        debug_assert!(n.is_internal());
        if n.is_inline() { 1 } else { self.multi_len_at(idx) }
    }

    /// Build a multi-pair node data from `(pair_start, pair_len)`, promoting to the
    /// extended encoding when either value doesn't fit in 31 bits and allocates an
    /// `ext` entry as needed.
    /// `pair_len == 1` should use inline instead; `pair_len == 0` is allowed (empty
    /// placeholder used by full.rs).
    ///
    /// # Panics
    ///
    /// Panics if `pair_len == 1` (that value aliases the `multi_extended` encoding;
    /// callers must use the inline path via `push_internal_node` instead).
    #[inline]
    pub fn encode_multi(&mut self, pair_start: usize, pair_len: usize) -> TddNodeData {
        assert!(pair_len != 1, "encode_multi: pair_len=1 aliases multi_extended encoding; use push_internal_node");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            TddNodeData::multi_pair(pair_start as u32, pair_len as u32)
        } else {
            let ext_idx = self.ext.len();
            debug_assert!(ext_idx < (1usize << 31), "too many extended nodes in a single level");
            self.ext.push(ExtMulti { start: pair_start as u64, len: pair_len as u64 });
            TddNodeData::multi_extended(ext_idx as u32)
        }
    }

    /// Update `pair_len` for a multi-pair node (used after in-place dedup shrinks
    /// the pair count). Caller must ensure `new_len` >= 2; use inline conversion
    /// for 1-pair results. Correctly handles extended nodes by updating the side table.
    ///
    /// # Panics
    ///
    /// Panics if `new_len < 2` (`new_len == 1` aliases the `multi_extended`
    /// encoding; convert to the inline or extended form instead).
    #[inline]
    pub fn set_pair_len(&mut self, node_idx: usize, new_len: u32) {
        assert!(new_len >= 2, "set_pair_len: new_len=1 aliases multi_extended; convert to inline or extended");
        let node = &mut self.nodes[node_idx];
        if node.is_multi_extended() {
            let ext_idx = (node.a & !MULTI_BIT) as usize;
            // Shrinking stays extended even if new_len now fits in u31 — the
            // ext slot is already allocated, and callers don't rely on form.
            self.ext[ext_idx].len = new_len as u64;
        } else {
            node.set_pair_len(new_len);
        }
    }

    /// Re-encode a multi-pair node at `node_idx` after an in-place rewrite has
    /// compacted its arena range `[start, start+old_len)` down to `new_len`
    /// live survivors sitting at the prefix `[start, start+new_len)`. Shared
    /// epilogue for `contract_leaf::rewrite_level` and
    /// `p_fusion::rebuild_parent_level` — both drive a write-cursor-behind-
    /// read-cursor rewrite over a node's own arena range and then need the
    /// exact same re-encode: shrink in place (`new_len >= 2`), inline the sole
    /// survivor (`new_len == 1` and it fits), or fall back to a length-1
    /// extended multi pointing at that one slot.
    ///
    /// Precondition: `new_len < old_len` (a strict shrink — an unchanged list
    /// is the caller's own early-out, not this helper's job) and `new_len >=
    /// 1` (emptying a node entirely goes through a different path).
    /// Debug-asserted; both callers establish the shrink themselves before
    /// calling.
    ///
    /// Returns the number of pair-arena slots this abandons — the caller's
    /// `dead_pairs` contribution, via whichever accounting style it already
    /// uses (`note_dead_pairs` immediately, or accumulated and applied once).
    ///
    /// ## Ext-slot reuse
    ///
    /// In the `new_len == 1`, can't-inline arm, reuse the node's OWN `ext`
    /// entry when it is already `is_multi_extended()` (no allocation) and
    /// only `try_push` a fresh `ExtMulti` when the node started life as a
    /// normal (packed) multi. Skipping this reuse would leak the node's prior
    /// `ext` entry: `ext` is append-only (never compacted), so an
    /// unconditionally-fresh push abandons the old slot as permanent garbage
    /// for the life of the level.
    ///
    /// # Errors
    ///
    /// `Err(ApplyError::OverBudget)` if the fresh `ext` push (the one
    /// allocating arm) cannot be reserved.
    #[inline]
    pub(crate) fn reencode_shrunk_multi(
        &mut self,
        node_idx: usize,
        start: usize,
        old_len: usize,
        new_len: usize,
    ) -> Result<usize, ApplyError> {
        debug_assert!(new_len < old_len, "reencode_shrunk_multi: not a shrink");
        debug_assert!(new_len >= 1, "reencode_shrunk_multi: emptying a node is a different path");
        if new_len >= 2 {
            self.set_pair_len(node_idx, new_len as u32);
            return Ok(old_len - new_len);
        }
        let surviving = self.pairs[start];
        if surviving.can_inline() {
            self.nodes[node_idx] = TddNodeData::inline(surviving);
            Ok(old_len) // an inline node owns no arena slot
        } else if self.nodes[node_idx].is_multi_extended() {
            let e = self.nodes[node_idx].ext_idx() as usize;
            self.ext[e] = ExtMulti { start: start as u64, len: 1 };
            Ok(old_len - 1)
        } else {
            let e = self.ext.len();
            try_push(&mut self.ext, ExtMulti { start: start as u64, len: 1 })?;
            self.nodes[node_idx] = TddNodeData::multi_extended(e as u32);
            Ok(old_len - 1)
        }
    }

    /// Rewrite a multi-pair node's arena start, handling both encodings.
    /// Companion of [`set_pair_len`](Self::set_pair_len); only the pairs-arena
    /// sweep needs it, and only ever to *lower* a start.
    ///
    /// A normal-multi node keeps its packed form: the new start is ≤ the old one,
    /// which already fit 31 bits, so the encoding cannot overflow. An extended
    /// node stays extended (its `ext` slot is already allocated).
    #[inline]
    fn set_multi_start(&mut self, node_idx: usize, new_start: usize) {
        let node = &mut self.nodes[node_idx];
        debug_assert!(node.is_multi());
        if node.is_multi_extended() {
            let ext_idx = (node.a & !MULTI_BIT) as usize;
            self.ext[ext_idx].start = new_start as u64;
        } else {
            debug_assert!(new_start < (1usize << 31), "set_multi_start: start overflows the packed encoding");
            node.a = (new_start as u32) | MULTI_BIT;
        }
    }

    /// Record `n` pair slots that just became unreachable garbage in this
    /// level's arena. See the [`dead_pairs`](Self::dead_pairs) field doc — the
    /// count is a trigger heuristic, so saturating at `u32::MAX` (or truncating
    /// an absurd `n`) can only move the sweep earlier or later, never corrupt it.
    #[inline]
    pub(crate) fn note_dead_pairs(&mut self, n: usize) {
        self.dead_pairs = self
            .dead_pairs
            .saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
    }

    /// Pair-arena slots owned by the node at `idx` — the ONE definition of a
    /// node's dead range: its pair count when the node is multi-encoded, 0 for
    /// the inline/leaf/tombstone encodings (they own no arena slot).
    #[inline]
    pub(crate) fn arena_pairs_at(&self, idx: usize) -> usize {
        if self.nodes[idx].is_multi() { self.multi_len_at(idx) } else { 0 }
    }

    /// Pairs-arena compaction trigger — the one knob. A sweep runs only when the
    /// dead-slot count exceeds this floor AND over half the arena is dead.
    ///
    /// The "over half" half is what makes it amortized: a sweep zeroes
    /// `dead_pairs`, so the next one cannot fire until the level has minted
    /// another live-arena's worth of garbage — O(1) sweep work per dead slot.
    /// The floor keeps levels whose whole arena is a few KiB out of the memmove
    /// path entirely.
    pub(crate) const PAIRS_COMPACT_MIN_DEAD: usize = 4096; // × 8 B/pair = 32 KiB

    /// Sweep unreferenced slots out of the `pairs` arena in place, if the
    /// garbage has grown past [`PAIRS_COMPACT_MIN_DEAD`](Self::PAIRS_COMPACT_MIN_DEAD).
    /// Returns whether the sweep ran.
    ///
    /// Twin contraction appends each merged union at the arena tail and abandons
    /// the source ranges, so a contraction-heavy level would otherwise hold
    /// unboundedly more dead arena than live. One memmove pass reclaims it:
    /// surviving ranges keep their pair CONTENT and its relative order
    /// byte-for-byte and only slide down, so every reader observes exactly what
    /// it did before.
    ///
    /// Callers must hold no pair-arena offset across the call. That is a
    /// one-level obligation: a range's start is stored only in the owning node's
    /// packed word or its `ext` entry (both rewritten here), and every other
    /// reader resolves a node to its slice through `multi_range` at use time.
    pub(crate) fn compact_pairs_if_stale(&mut self) -> bool {
        let dead = self.dead_pairs as usize;
        if dead <= Self::PAIRS_COMPACT_MIN_DEAD
            || dead * 2 <= self.pairs.len()
            // The survivor index packs each range start into the high 32 bits
            // of its sort key, so starts must fit `u32`. An arena past 4 G pairs
            // (32 GiB in one level) is out of reach of every configuration we
            // run — skip the sweep rather than pack a truncated start.
            || self.pairs.len() > u32::MAX as usize
        {
            return false;
        }

        // Index the arena's owners. Node order is NOT start order — a merged
        // survivor's union sits at the tail while unmerged nodes keep their low
        // starts — so the moves must be driven by a start-sorted index; walking
        // in node order would move a range down onto one not yet copied out.
        // Each entry is `(start << 32) | node_idx`, so the sort is a plain u64
        // sort: no key closure re-decoding the ext table on every comparison.
        // Live ranges are disjoint and non-empty, so starts are distinct and the
        // low half never decides the order.
        //
        // A plain local Vec, not a pooled buffer: the sweep is amortized-rare
        // (it zeroes `dead_pairs`, so the level must re-mint a live arena's
        // worth of garbage before the next one) and immediately runs a
        // whole-arena `copy_within` + `shrink_arrays`, so one allocation is
        // noise — whereas a pool would hold its peak-sized buffer resident for
        // the life of the thread.
        let mut order: Vec<u64> = Vec::new();
        debug_assert!(
            self.nodes.len() <= u32::MAX as usize,
            "compact_pairs_if_stale: node index must fit the packed key's low half"
        );
        for i in 0..self.nodes.len() {
            // Leaves and tombstones (`is_multi() == false` for both) and inline
            // nodes own no arena slot.
            let node = self.nodes[i];
            if !node.is_multi() {
                continue;
            }
            let range = self.multi_range(&node);
            if range.is_empty() {
                // An empty range owns nothing, but `&pairs[s..s]` still panics
                // once `s` is past the truncated length — park it at 0.
                self.set_multi_start(i, 0);
            } else {
                order.push(((range.start as u64) << 32) | (i as u64));
            }
        }
        order.sort_unstable();

        // Pairwise-disjoint live ranges are what make the slide safe: each
        // source range then starts at or after the write cursor, so a downward
        // `copy_within` can never clobber a range still to be copied. Every
        // arena writer allocates a fresh tail range and only ever re-points a
        // node at its own slots, so disjointness holds by construction — verify
        // it before touching a byte rather than corrupt the arena if some future
        // writer breaks it.
        let mut prev_end = 0usize;
        let mut ok = true;
        for &key in order.iter() {
            let start = (key >> 32) as usize;
            if start < prev_end {
                ok = false;
                break;
            }
            prev_end = start + self.multi_len_at(key as u32 as usize);
        }
        debug_assert!(ok, "compact_pairs_if_stale: overlapping live pair ranges");

        if ok {
            let mut write = 0usize;
            for &key in order.iter() {
                let node_idx = key as u32 as usize;
                let start = (key >> 32) as usize;
                let len = self.multi_len_at(node_idx);
                if start > write {
                    self.pairs.copy_within(start..start + len, write);
                }
                self.set_multi_start(node_idx, write);
                write += len;
            }
            self.pairs.truncate(write);
        }
        // The index is dead once the slide has rewritten every start, and
        // `shrink_arrays` below reallocates the arena it copies into — free the
        // index first so the two are never resident together at the peak.
        drop(order);
        // Zero the counter on both exits: after a sweep there is no garbage
        // left, and a level that trips the disjointness bail must re-accumulate
        // before trying again instead of re-scanning on every later merge.
        self.dead_pairs = 0;
        if ok {
            // Whether the reclaimed slack goes back to the allocator is the
            // level's one shrink policy's call, not a second threshold here.
            self.shrink_arrays();
        }
        ok
    }

    /// Push an internal node, appending its pairs to the arena.
    /// Single-pair nodes are stored inline (no arena entry); multi-pair nodes use the arena.
    /// Returns the new node's local index.
    #[inline]
    pub fn push_internal_node(&mut self, input_pairs: &[InputPair]) -> LocalNodeIdx {
        let idx = LocalNodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            self.nodes.push(TddNodeData::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            // Single pair that can't be inlined (right has LEAF_BIT or left has MULTI_BIT).
            // Use extended encoding — the only form that supports pair_len=1 without
            // aliasing either the leaf or multi_extended encoding.
            let pair_start = self.pairs.len();
            self.pairs.push(input_pairs[0]);
            let ext_idx = self.ext.len();
            self.ext.push(ExtMulti { start: pair_start as u64, len: 1 });
            self.nodes.push(TddNodeData::multi_extended(ext_idx as u32));
        } else {
            let pair_start = self.pairs.len();
            let pair_len = input_pairs.len();
            self.pairs.extend_from_slice(input_pairs);
            let data = self.encode_multi(pair_start, pair_len);
            self.nodes.push(data);
        }
        idx
    }

    /// Fallible `push_internal_node` — returns `Err(())` on alloc failure
    /// (caller maps to `ApplyError::OverBudget`). Mirrors the four-branch
    /// dispatch in `push_internal_node` but every push/extend is guarded.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the budget-gated buffer reservation failed; callers
    /// map this to `ApplyError::OverBudget`.
    #[inline]
    pub fn try_push_internal_node(
        &mut self,
        input_pairs: &[InputPair],
    ) -> Result<LocalNodeIdx, ()> {
        let idx = LocalNodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            self.nodes.try_reserve(1).map_err(|_| ())?;
            self.nodes.push(TddNodeData::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            let pair_start = self.pairs.len();
            self.pairs.try_reserve(1).map_err(|_| ())?;
            self.pairs.push(input_pairs[0]);
            let ext_idx = self.ext.len();
            self.ext.try_reserve(1).map_err(|_| ())?;
            self.ext.push(ExtMulti { start: pair_start as u64, len: 1 });
            self.nodes.try_reserve(1).map_err(|_| ())?;
            self.nodes.push(TddNodeData::multi_extended(ext_idx as u32));
        } else {
            let pair_start = self.pairs.len();
            let pair_len = input_pairs.len();
            self.pairs.try_reserve(pair_len).map_err(|_| ())?;
            self.pairs.extend_from_slice(input_pairs);
            // try_encode_multi mirrors encode_multi but guards the ext.push.
            let data = self.try_encode_multi(pair_start, pair_len)?;
            self.nodes.try_reserve(1).map_err(|_| ())?;
            self.nodes.push(data);
        }
        Ok(idx)
    }

    /// Push a multi-pair node (fallible). Pairs are assumed already in `self.pairs`.
    ///
    /// The new node's index is `self.nodes.len()` *before* the call; it is not
    /// returned, because every caller already reads `nodes.len()` itself
    /// immediately beforehand (it has to — the index goes into the caller's
    /// grid/result map before the push can fail).
    ///
    /// Shape: the same fast/cold split as `conjoin::budget::try_push_pair_into`
    /// — an inlinable "room available, operands fit 31 bits" store here, with the
    /// ENTIRE growth / extended-encoding body exiled to
    /// `push_multi_by_range_slow`. The two are observationally
    /// identical because under those two conditions the old monolithic body was
    /// already inert: `try_encode_multi` took its `fits_u31` branch (pure — no
    /// `ext` push, no allocation) and `nodes.try_reserve(1)` found
    /// `needs_to_grow == false`. Splitting them is a codegen fix, not a semantic
    /// one: the four cold call sites (two `RawVec::grow_one`, two `finish_grow`,
    /// the encode panic) forced a six-register frame push/pop onto every one of
    /// the ~1 G calls this takes per apply-heavy compile. `#[inline(never)]` on
    /// the cold arm is load-bearing — it is what removes the join.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the budget-gated buffer reservation failed; callers
    /// map this to `ApplyError::OverBudget`.
    ///
    /// # Panics (release-mode relaxation)
    ///
    /// `pair_len == 1` is forbidden — it aliases the `multi_extended` encoding
    /// (`EXT_SENTINEL == 1`). The cold arm still hard-`assert!`s it via
    /// `try_encode_multi`, but the fast path only `debug_assert!`s, so a
    /// violating caller corrupts silently in release instead of panicking. Both
    /// call sites dispatch the single-pair case to the inline/extended path
    /// before calling.
    #[inline(always)]
    pub fn try_push_multi_by_range(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<(), ()> {
        debug_assert!(
            pair_len >= 2,
            "try_push_multi_by_range: pair_len=1 aliases multi_extended encoding"
        );
        if self.nodes.len() < self.nodes.capacity()
            && pair_start < (1usize << 31)
            && pair_len < (1usize << 31)
        {
            // Spare capacity: `Vec::push` cannot reallocate, so there is nothing
            // to reserve or refuse; both operands fit, so the encoding is the
            // pure normal-multi word.
            self.nodes
                .push(TddNodeData::multi_pair(pair_start as u32, pair_len as u32));
            return Ok(());
        }
        self.push_multi_by_range_slow(pair_start, pair_len)
    }

    /// Growth / extended-encoding arm of `try_push_multi_by_range` — the
    /// original body, verbatim minus the dead index. Reached only when `nodes`
    /// is full (once per doubling event) or when `pair_start`/`pair_len` overflow
    /// 31 bits (pathological product grids), so the out-of-line call is
    /// amortized to nothing.
    #[cold]
    #[inline(never)]
    fn push_multi_by_range_slow(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<(), ()> {
        let data = self.try_encode_multi(pair_start, pair_len)?;
        self.nodes.try_reserve(1).map_err(|_| ())?;
        self.nodes.push(data);
        Ok(())
    }

    /// Fallible `encode_multi` — only the extended branch allocates.
    /// `pub(crate)` for the direct-emission at-slot finalize in `apply_clause`
    /// (pairs already in the arena; node word written to a tombstoned slot).
    #[inline]
    pub(crate) fn try_encode_multi(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<TddNodeData, ()> {
        assert!(pair_len != 1, "try_encode_multi: pair_len=1 aliases multi_extended encoding");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            Ok(TddNodeData::multi_pair(pair_start as u32, pair_len as u32))
        } else {
            let ext_idx = self.ext.len();
            self.ext.try_reserve(1).map_err(|_| ())?;
            self.ext.push(ExtMulti { start: pair_start as u64, len: pair_len as u64 });
            Ok(TddNodeData::multi_extended(ext_idx as u32))
        }
    }
}

// No add_leaf / add_leaf_implied methods — leaf levels are marginal (Pos=0,
// Neg=1, One=2) and never have stored nodes. Use push_internal_node for
// internal levels.
