//! The pair arena: node encoding, in-place resizing, compaction, and node pushes.

use crate::engine::Engine;
use crate::diagram::primitives::{MultiPairRange, InputPair, NodeIdx, TddNodeData, MULTI_BIT};
// `types/marginal.rs` already depends on the apply-side error/fallible-push
// primitives (`resolve_swapped_marginal_side`) — this is the same established
// cross-dependency, not a new one, needed for `reencode_shrunk_multi`'s
// `multi_pairs` push.
use crate::error::ApplyError;
use super::TddLevel;

/// The encoding a node lands on when its pair list shrinks — see
/// [`TddLevel::shrunk_encoding`].
enum ShrunkEncoding {
    /// Two or more survivors: the node keeps its encoding, only the length moves.
    Truncate,
    /// A sole survivor that fits in the node word.
    Inline(InputPair),
    /// A sole survivor that does not fit inline, over the range entry the node
    /// already owns.
    ReuseRangeEntry(usize),
    /// A sole survivor that does not fit inline, on a node holding no range
    /// entry yet: one has to be appended.
    NewRangeEntry,
}

impl TddLevel {
    /// Build a multi-pair node data from `(pair_start, pair_len)`, promoting to the
    /// extended encoding when either value doesn't fit in 31 bits and allocates an
    /// `multi_pairs` entry as needed.
    /// `pair_len == 1` should use inline instead; `pair_len == 0` is allowed,
    /// for an empty placeholder node.
    ///
    /// # Panics
    ///
    /// Panics if `pair_len == 1` (that value aliases the `multi_ranged` encoding;
    /// callers must use the inline path via `push_internal_node` instead).
    #[inline]
    pub(crate) fn encode_multi(&mut self, pair_start: usize, pair_len: usize) -> TddNodeData {
        assert!(pair_len != 1, "encode_multi: pair_len=1 aliases multi_ranged encoding; use push_internal_node");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            TddNodeData::multi_pair(pair_start as u32, pair_len as u32)
        } else {
            let multi_pairs_idx = self.multi_pairs.len();
            debug_assert!(multi_pairs_idx < (1usize << 31), "too many extended nodes in a single level");
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: pair_len as u64 });
            TddNodeData::multi_ranged(multi_pairs_idx as u32)
        }
    }

    /// Update `pair_len` for a multi-pair node (used after in-place dedup shrinks
    /// the pair count). Caller must ensure `new_len` >= 2; use inline conversion
    /// for 1-pair results. Correctly handles extended nodes by updating the side table.
    ///
    /// # Panics
    ///
    /// Panics if `new_len < 2` (`new_len == 1` aliases the `multi_ranged`
    /// encoding; convert to the inline or extended form instead).
    #[inline]
    pub(crate) fn set_pair_len(&mut self, node_idx: usize, new_len: u32) {
        assert!(new_len >= 2, "set_pair_len: new_len=1 aliases multi_ranged; convert to inline or extended");
        let node = &mut self.nodes[node_idx];
        if node.is_multi_ranged() {
            let multi_pairs_idx = (node.a & !MULTI_BIT) as usize;
            // Shrinking stays extended even if new_len now fits in u31 — the
            // multi_pairs slot is already allocated, and callers don't rely on form.
            self.multi_pairs[multi_pairs_idx].len = new_len as u64;
        } else {
            node.set_pair_len(new_len);
        }
    }

    /// Re-encode a multi-pair node at `node_idx` after an in-place rewrite has
    /// compacted its arena range `[start, start+old_len)` down to `new_len`
    /// live survivors sitting at the prefix `[start, start+new_len)`. Shared
    /// epilogue for every pass that compacts a node's own arena range with a
    /// write cursor behind a read cursor — `contract_leaf::rewrite_level`,
    /// `pair_fusion::rebuild_parent_level` and the twin-merge parent rewrite —
    /// each of which then needs the same re-encode: shrink in place
    /// (`new_len >= 2`), inline the sole survivor (`new_len == 1` and it fits),
    /// or fall back to a length-1 extended multi pointing at that one slot.
    ///
    /// Precondition: `new_len < old_len` (a strict shrink — an unchanged list
    /// is the caller's own early-out, not this helper's job) and `new_len >=
    /// 1` (emptying a node entirely goes through a different path).
    /// Debug-asserted; every caller establishes the shrink itself before
    /// calling.
    ///
    /// Returns the number of pair-arena slots this abandons — the caller's
    /// `dead_pairs` contribution, via whichever accounting style it already
    /// uses (`note_dead_pairs` immediately, or accumulated and applied once).
    ///
    /// ## Ext-slot reuse
    ///
    /// In the `new_len == 1`, can't-inline arm, reuse the node's OWN `multi_pairs`
    /// entry when it is already `is_multi_ranged()` (no allocation) and
    /// allocate a fresh `MultiPairRange` only when the node started life as a
    /// normal (packed) multi. Skipping this reuse would leak the node's prior
    /// `multi_pairs` entry: `multi_pairs` is append-only (never compacted), so an
    /// unconditionally-fresh push abandons the old slot as permanent garbage
    /// for the life of the level.
    ///
    /// # Errors
    ///
    /// `Err(ApplyError::OverBudget)` if the fresh `multi_pairs` entry (the one
    /// allocating arm) cannot be reserved.
    #[inline]
    pub(crate) fn reencode_shrunk_multi(
        &mut self, eng: &Engine, node_idx: usize,
        start: usize,
        old_len: usize,
        new_len: usize,
    ) -> Result<usize, ApplyError> {
        if matches!(self.shrunk_encoding(node_idx, start, new_len), ShrunkEncoding::NewRangeEntry) {
            eng.limits().reserve(&mut self.multi_pairs, 1)?;
        }
        Ok(self.reencode_shrunk_multi_reserved(node_idx, start, old_len, new_len))
    }

    /// [`reencode_shrunk_multi`](Self::reencode_shrunk_multi) for a caller that
    /// has already reserved the one `multi_pairs` entry the allocating arm can
    /// need, so the re-encode is infallible.
    ///
    /// Same preconditions and return value.
    #[inline]
    pub(crate) fn reencode_shrunk_multi_reserved(
        &mut self, node_idx: usize,
        start: usize,
        old_len: usize,
        new_len: usize,
    ) -> usize {
        debug_assert!(new_len < old_len, "reencode_shrunk_multi: not a shrink");
        debug_assert!(new_len >= 1, "reencode_shrunk_multi: emptying a node is a different path");
        let entry = MultiPairRange { start: start as u64, len: 1 };
        match self.shrunk_encoding(node_idx, start, new_len) {
            ShrunkEncoding::Truncate => {
                self.set_pair_len(node_idx, new_len as u32);
                return old_len - new_len;
            }
            ShrunkEncoding::Inline(surviving) => {
                self.nodes[node_idx] = TddNodeData::inline(surviving);
                return old_len; // an inline node owns no arena slot
            }
            ShrunkEncoding::ReuseRangeEntry(e) => self.multi_pairs[e] = entry,
            ShrunkEncoding::NewRangeEntry => {
                let e = self.multi_pairs.len();
                debug_assert!(
                    self.multi_pairs.capacity() > e,
                    "reencode_shrunk_multi_reserved: caller must reserve the range entry",
                );
                self.multi_pairs.push(entry);
                self.nodes[node_idx] = TddNodeData::multi_ranged(e as u32);
            }
        }
        old_len - 1
    }

    /// Which encoding a shrink to `new_len` lands the node on. The one place
    /// the arms are decided, so the fallible entry point can tell whether it
    /// has to reserve without restating the tests.
    #[inline]
    fn shrunk_encoding(&self, node_idx: usize, start: usize, new_len: usize) -> ShrunkEncoding {
        if new_len >= 2 {
            return ShrunkEncoding::Truncate;
        }
        let surviving = self.pairs[start];
        if surviving.can_inline() {
            return ShrunkEncoding::Inline(surviving);
        }
        let node = &self.nodes[node_idx];
        if node.is_multi_ranged() {
            ShrunkEncoding::ReuseRangeEntry(node.multi_pairs_idx() as usize)
        } else {
            ShrunkEncoding::NewRangeEntry
        }
    }

    /// Rewrite a multi-pair node's arena start, handling both encodings.
    /// Companion of [`set_pair_len`](Self::set_pair_len); only the pairs-arena
    /// sweep needs it, and only ever to *lower* a start.
    ///
    /// A normal-multi node keeps its packed form: the new start is ≤ the old one,
    /// which already fit 31 bits, so the encoding cannot overflow. An extended
    /// node stays extended (its `multi_pairs` slot is already allocated).
    #[inline]
    fn set_multi_start(&mut self, node_idx: usize, new_start: usize) {
        let node = &mut self.nodes[node_idx];
        debug_assert!(node.is_multi());
        if node.is_multi_ranged() {
            let multi_pairs_idx = (node.a & !MULTI_BIT) as usize;
            self.multi_pairs[multi_pairs_idx].start = new_start as u64;
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

    /// Pair-arena slots owned by the node at `idx` — the one definition of a
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
    /// packed word or its `multi_pairs` entry (both rewritten here), and every other
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

        // Index the arena's owners. Node order is not start order — a merged
        // survivor's union sits at the tail while unmerged nodes keep their low
        // starts — so the moves must be driven by a start-sorted index; walking
        // in node order would move a range down onto one not yet copied out.
        // Each entry is `(start << 32) | node_idx`, so the sort is a plain u64
        // sort: no key closure re-decoding the multi-pair range table on every comparison.
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

    /// Give the node at `at` a new pair list, keeping its index.
    ///
    /// Every other node keeps its index too, so no parent reference has to be
    /// rewritten. The node's old pair range is abandoned in the arena and
    /// reclaimed by the next compaction.
    pub fn replace_node_pairs(&mut self, at: NodeIdx, input_pairs: &[InputPair]) {
        let fresh = self.push_internal_node(input_pairs);
        self.nodes[at.idx()] = self.nodes[fresh.idx()];
        self.nodes.pop();
    }

    /// Append a node holding `pairs` in canonical form — sorted, with
    /// duplicates removed — and return its index.
    ///
    /// Canonical order is what lets two nodes be compared slice against slice;
    /// the dedup keeps the no-duplicate-pairs invariant independent of the
    /// caller's own reasoning about why its pairs are distinct (O(n) once
    /// sorted). `pairs` must be non-empty.
    pub(crate) fn push_internal_node_canonical(&mut self, pairs: &mut Vec<InputPair>) -> NodeIdx {
        super::sort_pairs(pairs);
        pairs.dedup();
        self.push_internal_node(pairs)
    }

    /// Append a node with the given pairs and return its index. Chooses the
    /// storage encoding itself; the only way to add a node when building a
    /// diagram by hand. `input_pairs` must be non-empty.
    #[inline]
    pub fn push_internal_node(&mut self, input_pairs: &[InputPair]) -> NodeIdx {
        let idx = NodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            self.nodes.push(TddNodeData::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            // Single pair that can't be inlined (right has LEAF_BIT or left has MULTI_BIT).
            // Use extended encoding — the only form that supports pair_len=1 without
            // aliasing either the leaf or multi_ranged encoding.
            let pair_start = self.pairs.len();
            self.pairs.push(input_pairs[0]);
            let multi_pairs_idx = self.multi_pairs.len();
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: 1 });
            self.nodes.push(TddNodeData::multi_ranged(multi_pairs_idx as u32));
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
    pub(crate) fn try_push_internal_node(
        &mut self,
        input_pairs: &[InputPair],
    ) -> Result<NodeIdx, ()> {
        let idx = NodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            self.nodes.try_reserve(1).map_err(|_| ())?;
            self.nodes.push(TddNodeData::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            let pair_start = self.pairs.len();
            self.pairs.try_reserve(1).map_err(|_| ())?;
            self.pairs.push(input_pairs[0]);
            let multi_pairs_idx = self.multi_pairs.len();
            self.multi_pairs.try_reserve(1).map_err(|_| ())?;
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: 1 });
            self.nodes.try_reserve(1).map_err(|_| ())?;
            self.nodes.push(TddNodeData::multi_ranged(multi_pairs_idx as u32));
        } else {
            let pair_start = self.pairs.len();
            let pair_len = input_pairs.len();
            self.pairs.try_reserve(pair_len).map_err(|_| ())?;
            self.pairs.extend_from_slice(input_pairs);
            // try_encode_multi mirrors encode_multi but guards the multi_pairs.push.
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
    /// `multi_pairs` push, no allocation) and `nodes.try_reserve(1)` found
    /// `needs_to_grow == false`. Splitting them is a codegen fix, not a semantic
    /// one: the cold call sites (the `Vec` growth paths and the encode panic)
    /// forced a six-register frame push/pop onto every one of
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
    /// `pair_len == 1` is forbidden — it aliases the `multi_ranged` encoding
    /// (`RANGE_SENTINEL == 1`). The cold arm still hard-`assert!`s it via
    /// `try_encode_multi`, but the fast path only `debug_assert!`s, so a
    /// violating caller corrupts silently in release instead of panicking. Both
    /// call sites dispatch the single-pair case to the inline/extended path
    /// before calling.
    #[inline(always)]
    pub(crate) fn try_push_multi_by_range(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<(), ()> {
        debug_assert!(
            pair_len >= 2,
            "try_push_multi_by_range: pair_len=1 aliases multi_ranged encoding"
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
    /// `pub(crate)` for the direct-emission at-slot finalize in `conjoin_clause`
    /// (pairs already in the arena; node word written to a tombstoned slot).
    #[inline]
    pub(crate) fn try_encode_multi(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<TddNodeData, ()> {
        assert!(pair_len != 1, "try_encode_multi: pair_len=1 aliases multi_ranged encoding");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            Ok(TddNodeData::multi_pair(pair_start as u32, pair_len as u32))
        } else {
            let multi_pairs_idx = self.multi_pairs.len();
            self.multi_pairs.try_reserve(1).map_err(|_| ())?;
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: pair_len as u64 });
            Ok(TddNodeData::multi_ranged(multi_pairs_idx as u32))
        }
    }
}
