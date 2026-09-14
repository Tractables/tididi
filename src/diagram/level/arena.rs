//! The pair arena: node encoding, in-place resizing, compaction, and node pushes.

use crate::engine::Engine;
use crate::diagram::primitives::{MultiPairRange, ChildPair, NodeIdx, EncodedNode, MULTI_BIT};
use crate::limits::{OperationError};
use super::TddLevel;

/// The encoding a node lands on when its pair list shrinks — see
/// [`TddLevel::shrunk_encoding`].
enum ShrunkEncoding {
    /// Two or more survivors: the node keeps its encoding, only the length moves.
    Truncate,
    /// A sole survivor that fits in the node word.
    Inline(ChildPair),
    /// A sole survivor that does not fit inline, over the range entry the node
    /// already owns.
    ReuseRangeEntry(usize),
    /// A sole survivor that does not fit inline, on a node holding no range
    /// entry yet: one has to be appended.
    NewRangeEntry,
}

/// How a node push grows the level's buffers.
///
/// The construction and reduction paths push through `Vec`'s own growth;
/// the apply emitters reserve first and report a refused allocation to the
/// caller, which maps it to `OperationError::OverBudget`.
pub(crate) trait Growth {
    type Err;
    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
}

/// [`Growth`] through `Vec`'s own reallocation.
pub(crate) struct Grow;

impl Growth for Grow {
    type Err = std::convert::Infallible;
    #[inline(always)]
    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        v.reserve(additional);
        Ok(())
    }
}

/// [`Growth`] that refuses instead of aborting: `try_reserve`, with a failed
/// reservation reported as `Err(())`.
pub(crate) struct TryGrow;

impl Growth for TryGrow {
    type Err = ();
    #[inline(always)]
    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        v.try_reserve(additional).map_err(|_| ())
    }
}

impl TddLevel {
    /// Build a multi-pair node data from `(pair_start, pair_len)`, promoting to the
    /// extended encoding when either value doesn't fit in 31 bits and allocates an
    /// `multi_pairs` entry as needed.
    /// `pair_len == 1` is [`encode_single`](Self::encode_single)'s; `pair_len == 0`
    /// is allowed, for an empty placeholder node.
    ///
    /// # Panics
    ///
    /// Panics if `pair_len == 1` (that value aliases the `multi_ranged` encoding).
    #[inline]
    pub(crate) fn encode_multi(&mut self, pair_start: usize, pair_len: usize) -> EncodedNode {
        self.encode_multi_in::<Grow>(pair_start, pair_len).unwrap_or_else(|never| match never {})
    }

    /// [`encode_multi`](Self::encode_multi) growing through `G`; only the
    /// extended branch allocates.
    ///
    /// # Errors
    ///
    /// The `multi_pairs` reservation `G` refused.
    #[inline]
    fn encode_multi_in<G: Growth>(&mut self, pair_start: usize, pair_len: usize) -> Result<EncodedNode, G::Err> {
        assert!(pair_len != 1, "encode_multi: pair_len=1 aliases multi_ranged encoding; use encode_single");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            Ok(EncodedNode::multi_pair(pair_start as u32, pair_len as u32))
        } else {
            let multi_pairs_idx = self.multi_pairs.len();
            debug_assert!(multi_pairs_idx < (1usize << 31), "too many extended nodes in a single level");
            G::reserve(&mut self.multi_pairs, 1)?;
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: pair_len as u64 });
            Ok(EncodedNode::multi_ranged(multi_pairs_idx as u32))
        }
    }

    /// Encode a node holding exactly `pair`, which sits at arena index `start`:
    /// inline when the pair fits in the node's own words (the arena slot is
    /// then unused), else a one-pair `multi_pairs` range, since a `pair_len`
    /// of 1 aliases the `multi_ranged` encoding.
    #[inline]
    pub(crate) fn encode_single(&mut self, start: usize, pair: ChildPair) -> EncodedNode {
        if pair.can_inline() {
            return EncodedNode::inline(pair);
        }
        let multi_pairs_idx = self.multi_pairs.len();
        self.multi_pairs.push(MultiPairRange { start: start as u64, len: 1 });
        EncodedNode::multi_ranged(multi_pairs_idx as u32)
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
    /// live survivors sitting at the prefix `[start, start+new_len)`: shrink
    /// in place (`new_len >= 2`), inline the sole survivor (`new_len == 1` and
    /// it fits), or a length-1 extended multi pointing at that one slot. The
    /// epilogue of every pass that compacts a node's own range in place.
    ///
    /// Precondition (debug-asserted): `1 <= new_len < old_len`.
    ///
    /// Returns the number of pair-arena slots this abandons, the caller's
    /// `dead_pairs` contribution.
    ///
    /// A node already `is_multi_ranged()` reuses its own `multi_pairs` entry;
    /// `multi_pairs` is never compacted, so a fresh push would leak the old one.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` if the fresh `multi_pairs` entry (the one
    /// allocating arm) cannot be reserved.
    #[inline]
    pub(crate) fn reencode_shrunk_multi(
        &mut self, eng: &Engine, node_idx: usize,
        start: usize,
        old_len: usize,
        new_len: usize,
    ) -> Result<usize, OperationError> {
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
                self.nodes[node_idx] = EncodedNode::inline(surviving);
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
                self.nodes[node_idx] = EncodedNode::multi_ranged(e as u32);
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

    /// Pairs-arena compaction trigger. A sweep runs only when the dead-slot
    /// count exceeds this floor and over half the arena is dead; the latter
    /// makes the sweep O(1) per dead slot, since a sweep zeroes `dead_pairs`.
    pub(crate) const PAIRS_COMPACT_MIN_DEAD: usize = 4096; // × 8 B/pair = 32 KiB

    /// Sweep unreferenced slots out of the `pairs` arena in place, if the
    /// garbage has grown past [`Self::PAIRS_COMPACT_MIN_DEAD`].
    /// Returns whether the sweep ran.
    ///
    /// Three phases: index the live ranges, verify they are disjoint, slide
    /// them down. Every node's pairs read the same afterwards.
    ///
    /// Callers must hold no pair-arena offset across the call; a range's
    /// start lives only in the owning node's word or its `multi_pairs` entry,
    /// both rewritten here.
    pub(crate) fn compact_pairs_if_stale(&mut self) -> bool {
        let dead = self.dead_pairs as usize;
        if dead <= Self::PAIRS_COMPACT_MIN_DEAD
            || dead * 2 <= self.pairs.len()
            // The survivor index packs each range start into the high 32 bits
            // of its sort key, so starts must fit `u32`. An arena past 4 G pairs
            // (32 GiB in one level) is not a size the sweep needs to handle:
            // skip it rather than pack a truncated start.
            || self.pairs.len() > u32::MAX as usize
        {
            return false;
        }

        let order = self.index_live_ranges();
        let ok = self.live_ranges_are_disjoint(&order);
        debug_assert!(ok, "compact_pairs_if_stale: overlapping live pair ranges");
        if ok {
            self.slide_ranges_down(&order);
        }
        // Free the index before `shrink_arrays` reallocates the arena.
        drop(order);
        // Zero on both exits: a level that failed the disjointness check must
        // re-accumulate before trying again.
        self.dead_pairs = 0;
        if ok {
            self.shrink_arrays();
        }
        ok
    }

    /// Compaction phase 1: a start-sorted index of the arena's live ranges,
    /// each entry `(start << 32) | node_idx`.
    ///
    /// Node order is not start order (a merged union sits at the tail), and
    /// sliding in node order would move a range onto one not yet copied out.
    /// Packing start and node index into one `u64` makes the sort a plain
    /// integer sort.
    fn index_live_ranges(&mut self) -> Vec<u64> {
        let mut order: Vec<u64> = Vec::new();
        debug_assert!(
            self.nodes.len() <= u32::MAX as usize,
            "index_live_ranges: node index must fit the packed key's low half"
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
        order
    }

    /// Compaction phase 2: whether the indexed ranges are pairwise disjoint.
    ///
    /// Disjointness is what makes the slide safe: each source range then starts
    /// at or after the write cursor, so a downward `copy_within` never clobbers
    /// a range still to be copied.
    fn live_ranges_are_disjoint(&self, order: &[u64]) -> bool {
        let mut prev_end = 0usize;
        for &key in order {
            let start = (key >> 32) as usize;
            if start < prev_end {
                return false;
            }
            prev_end = start + self.multi_len_at(key as u32 as usize);
        }
        true
    }

    /// Compaction phase 3: slide every indexed range down onto the write cursor
    /// and repoint its owner, then truncate the arena to what survived.
    ///
    /// Surviving ranges keep their pair content and its relative order
    /// byte-for-byte, so every reader observes exactly what it did before.
    fn slide_ranges_down(&mut self, order: &[u64]) {
        let mut write = 0usize;
        for &key in order {
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

    /// Append a node with the given pairs and return its index. Chooses the
    /// storage encoding itself; the only way to add a node when building a
    /// diagram by hand. `input_pairs` must be non-empty.
    #[inline]
    pub(crate) fn push_internal_node(&mut self, input_pairs: &[ChildPair]) -> NodeIdx {
        self.push_internal_node_in::<Grow>(input_pairs).unwrap_or_else(|never| match never {})
    }

    /// [`push_internal_node`](Self::push_internal_node) for the apply
    /// emitters: every push reserves first, and a refused reservation comes
    /// back as `Err(())`, which the caller maps to `OperationError::OverBudget`.
    ///
    /// # Errors
    ///
    /// A buffer reservation was refused.
    #[inline]
    pub(crate) fn try_push_internal_node(
        &mut self,
        input_pairs: &[ChildPair],
    ) -> Result<NodeIdx, ()> {
        self.push_internal_node_in::<TryGrow>(input_pairs)
    }

    /// Append a node through the fallible encoder and charge its arena growth to the engine.
    pub(crate) fn push_node_on(&mut self, eng: &Engine, pairs: &[ChildPair]) -> Result<NodeIdx, OperationError> {
        let lim = eng.limits();
        #[cfg(test)]
        if lim.refuses_reserve() { return Err(OperationError::OverBudget); }
        if pairs.len() == 1 && pairs[0].can_inline() && self.nodes.len() < self.nodes.capacity() {
            return self.try_push_internal_node(pairs).map_err(|_| OperationError::OverBudget);
        }
        let before = self.arena_capacity_bytes();
        lim.preflight_alloc(std::mem::size_of_val(pairs) as u64);
        let index = self.try_push_internal_node(pairs).map_err(|_| OperationError::OverBudget)?;
        lim.charge_bytes(self.arena_capacity_bytes().saturating_sub(before))?;
        Ok(index)
    }

    /// The allocated bytes of the three structural arenas.
    fn arena_capacity_bytes(&self) -> u64 {
        (self.nodes.capacity() * std::mem::size_of::<EncodedNode>()
            + self.pairs.capacity() * std::mem::size_of::<ChildPair>()
            + self.multi_pairs.capacity() * std::mem::size_of::<MultiPairRange>()) as u64
    }

    /// The node push, growing through `G`.
    ///
    /// # Errors
    ///
    /// A buffer reservation `G` refused.
    #[inline]
    fn push_internal_node_in<G: Growth>(
        &mut self,
        input_pairs: &[ChildPair],
    ) -> Result<NodeIdx, G::Err> {
        let idx = NodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            G::reserve(&mut self.nodes, 1)?;
            self.nodes.push(EncodedNode::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            // Single pair that can't be inlined (right has `LEAF_BIT` or left has `MULTI_BIT`).
            // Use extended encoding — the only form that supports pair_len=1 without
            // aliasing either the leaf or multi_ranged encoding.
            let pair_start = self.pairs.len();
            G::reserve(&mut self.pairs, 1)?;
            self.pairs.push(input_pairs[0]);
            let multi_pairs_idx = self.multi_pairs.len();
            G::reserve(&mut self.multi_pairs, 1)?;
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: 1 });
            G::reserve(&mut self.nodes, 1)?;
            self.nodes.push(EncodedNode::multi_ranged(multi_pairs_idx as u32));
        } else {
            let pair_start = self.pairs.len();
            let pair_len = input_pairs.len();
            G::reserve(&mut self.pairs, pair_len)?;
            self.pairs.extend_from_slice(input_pairs);
            let data = self.encode_multi_in::<G>(pair_start, pair_len)?;
            G::reserve(&mut self.nodes, 1)?;
            self.nodes.push(data);
        }
        Ok(idx)
    }

    /// Push a multi-pair node (fallible). Pairs are assumed already in `self.pairs`.
    ///
    /// The new node's index is `self.nodes.len()` before the call; it is not
    /// returned, since the caller records it before the push can fail.
    ///
    /// The store with spare capacity and operands under 31 bits is inlined;
    /// growth and the extended encoding are out of line in
    /// `push_multi_by_range_slow`, which keeps the per-node path's frame small.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the buffer reservation failed; callers map this to
    /// `OperationError::OverBudget`.
    ///
    /// # Panics
    ///
    /// `pair_len == 1` aliases the `multi_ranged` encoding. The cold arm
    /// `assert!`s it; the fast path only `debug_assert!`s, so a violating
    /// caller corrupts silently in release.
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
                .push(EncodedNode::multi_pair(pair_start as u32, pair_len as u32));
            return Ok(());
        }
        self.push_multi_by_range_slow(pair_start, pair_len)
    }

    /// Growth / extended-encoding arm of `try_push_multi_by_range`, reached
    /// only when `nodes` is full or when `pair_start`/`pair_len` overflow
    /// 31 bits.
    #[cold]
    #[inline(never)]
    fn push_multi_by_range_slow(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<(), ()> {
        let data = self.encode_multi_in::<TryGrow>(pair_start, pair_len)?;
        self.nodes.try_reserve(1).map_err(|_| ())?;
        self.nodes.push(data);
        Ok(())
    }
}
