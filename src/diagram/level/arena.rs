//! The pair arena: node encoding, in-place resizing, compaction, and node pushes.

use crate::Engine;
use crate::diagram::primitives::{MultiPairRange, ChildPair, NodeIdx, EncodedNode, NodeKind, MULTI_BIT};
use crate::limits::{Charged, OperationError};
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

/// Reserve room for `additional` more elements, refusing rather than
/// aborting. The allocator's own error carries nothing the caller can use —
/// the request size is known at the site that reports it — so it is dropped.
fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), ()> {
    v.try_reserve(additional).map_err(|_| ())
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
    /// Panics if `pair_len == 1` (that value aliases the `multi_ranged`
    /// encoding), or if the allocator refuses the `multi_pairs` entry — use
    /// [`try_encode_multi`](Self::try_encode_multi) where a refusal is an
    /// answer.
    #[inline]
    pub(crate) fn encode_multi(&mut self, pair_start: usize, pair_len: usize) -> EncodedNode {
        self.try_encode_multi(pair_start, pair_len).expect("out of memory encoding a node")
    }

    /// [`encode_multi`](Self::encode_multi) refusing instead of aborting; only
    /// the extended branch allocates.
    ///
    /// # Errors
    ///
    /// The `multi_pairs` reservation was refused.
    #[inline]
    fn try_encode_multi(&mut self, pair_start: usize, pair_len: usize) -> Result<EncodedNode, ()> {
        assert!(pair_len != 1, "encode_multi: pair_len=1 aliases multi_ranged encoding; use encode_single");
        let fits_u31 = pair_start < (1usize << 31) && pair_len < (1usize << 31);
        if fits_u31 {
            Ok(EncodedNode::multi_pair(pair_start as u32, pair_len as u32))
        } else {
            let multi_pairs_idx = self.multi_pairs.len();
            debug_assert!(multi_pairs_idx < (1usize << 31), "too many extended nodes in a single level");
            reserve(&mut self.multi_pairs, 1)?;
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
        if let NodeKind::MultiRanged(idx) = self.nodes[node_idx].kind() {
            // Shrinking stays extended even if new_len now fits in u31 — the
            // multi_pairs slot is already allocated, and callers don't rely on form.
            self.multi_pairs[idx as usize].len = new_len as u64;
        } else {
            self.nodes[node_idx].set_pair_len(new_len);
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
    /// Reserve the one `multi_pairs` entry a shrink to `new_len` can need, so
    /// the caller can rewrite its pair range and then finish with the
    /// infallible [`reencode_shrunk_multi_reserved`](Self::reencode_shrunk_multi_reserved).
    ///
    /// Call this *before* the rewrite. It therefore cannot read the surviving
    /// pair, which does not exist yet, so it reserves whenever the allocating
    /// arm is reachable and lets the entry go unused when the survivor turns
    /// out to be inlinable. Over-reserving costs one `MultiPairRange` of
    /// capacity and never a length, so the only effect is refusing marginally
    /// earlier.
    ///
    /// A node already `is_multi_ranged()` reuses its own `multi_pairs` entry
    /// and needs nothing; `multi_pairs` is never compacted, so a fresh push
    /// would leak the old one.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` if that entry cannot be reserved.
    #[inline]
    pub(crate) fn reserve_shrunk_multi(
        &mut self, eng: &Engine, node_idx: usize, new_len: usize,
    ) -> Result<(), OperationError> {
        // The pair-independent half of `shrunk_encoding`'s arm choice. The two
        // must move together: this is the only thing that decides whether the
        // reserve happens, and `shrunk_encoding` is the only thing that decides
        // whether the push happens.
        if new_len < 2 && !matches!(self.nodes[node_idx].kind(), NodeKind::MultiRanged(_)) {
            eng.limits().reserve(&mut self.multi_pairs, 1)?;
        }
        Ok(())
    }

    /// Re-encode a node whose pair range has just shrunk to `new_len`, for a
    /// caller that has already reserved through
    /// [`reserve_shrunk_multi`](Self::reserve_shrunk_multi), so the re-encode
    /// is infallible.
    ///
    /// Returns the number of pair-arena slots this abandons, the caller's
    /// `dead_pairs` contribution.
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
    /// the arms are decided; `reserve_shrunk_multi` is its conservative
    /// pre-rewrite shadow, deciding only whether the allocating arm is
    /// reachable.
    #[inline]
    fn shrunk_encoding(&self, node_idx: usize, start: usize, new_len: usize) -> ShrunkEncoding {
        if new_len >= 2 {
            return ShrunkEncoding::Truncate;
        }
        let surviving = self.pairs[start];
        if surviving.can_inline() {
            return ShrunkEncoding::Inline(surviving);
        }
        match self.nodes[node_idx].kind() {
            NodeKind::MultiRanged(idx) => ShrunkEncoding::ReuseRangeEntry(idx as usize),
            _ => ShrunkEncoding::NewRangeEntry,
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
        match self.nodes[node_idx].kind() {
            NodeKind::MultiRanged(idx) => self.multi_pairs[idx as usize].start = new_start as u64,
            NodeKind::Multi { .. } => {
                debug_assert!(new_start < (1usize << 31), "set_multi_start: start overflows the packed encoding");
                self.nodes[node_idx].a = (new_start as u32) | MULTI_BIT;
            }
            other => panic!("set_multi_start on {other:?}"),
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
        if self.nodes[idx].kind().pairs_in_arena() { self.multi_len_at(idx) } else { 0 }
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
            // Only a multi-pair node owns an arena slot; leaves, tombstones
            // and inline nodes own none.
            let node = self.nodes[i];
            if !node.kind().pairs_in_arena() {
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
    ///
    /// # Panics
    ///
    /// Panics if the allocator refuses a buffer — use
    /// [`try_push_internal_node`](Self::try_push_internal_node) where a
    /// refusal is an answer.
    #[inline]
    pub(crate) fn push_internal_node(&mut self, input_pairs: &[ChildPair]) -> NodeIdx {
        self.try_push_internal_node(input_pairs).expect("out of memory pushing a node")
    }

    /// Append a node through the fallible encoder and charge its arena growth to the engine.
    pub(crate) fn push_node_on(&mut self, eng: &Engine, pairs: &[ChildPair]) -> Result<NodeIdx, OperationError> {
        self.push_node_within(eng.limits(), pairs)
    }

    /// [`push_node_on`](Self::push_node_on) for a caller holding the limits
    /// rather than the engine, such as the rotation rebuild.
    pub(crate) fn push_node_within(&mut self, lim: &crate::limits::Limits, pairs: &[ChildPair]) -> Result<NodeIdx, OperationError> {
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

    /// Add one pair to the node at `idx`, in place.
    ///
    /// The node's pairs stay contiguous: a range already at the arena's tail
    /// simply grows, and any other range is copied to the tail and the slots
    /// it leaves behind are noted dead, the same move twin contraction makes
    /// when it concatenates two pair lists. An inline node's own pair moves to
    /// the arena with the new one.
    ///
    /// The caller owes the representation invariants: `pair` must be disjoint
    /// from the node's own pairs and owned by no other node of the level.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` if the arena growth is refused.
    pub(crate) fn push_pair_onto_node(
        &mut self, eng: &Engine, idx: usize, pair: ChildPair,
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        #[cfg(test)]
        if lim.refuses_reserve() { return Err(OperationError::OverBudget); }
        let before = self.arena_capacity_bytes();
        let node = self.nodes[idx].kind();
        match node {
            NodeKind::Inline(existing) => {
                let start = self.pairs.len();
                lim.reserve(&mut self.pairs, 2)?;
                self.pairs.push(existing);
                self.pairs.push(pair);
                self.nodes[idx] = self.try_encode_multi(start, 2).map_err(|()| OperationError::OverBudget)?;
            }
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => {
                let range = self.pair_range_at(idx);
                let len = range.len();
                if range.end == self.pairs.len() {
                    lim.reserve(&mut self.pairs, 1)?;
                    self.pairs.push(pair);
                    // A multi node holds two pairs or more, so the grown
                    // length never aliases the one-pair encoding.
                    self.set_pair_len(idx, (len + 1) as u32);
                } else {
                    let start = self.pairs.len();
                    lim.reserve(&mut self.pairs, len + 1)?;
                    self.pairs.extend_from_within(range);
                    self.pairs.push(pair);
                    self.nodes[idx] = self.try_encode_multi(start, len + 1).map_err(|()| OperationError::OverBudget)?;
                    self.note_dead_pairs(len);
                }
            }
            NodeKind::Leaf(_) | NodeKind::Tombstone => {
                panic!("push_pair_onto_node: node {idx} holds no pairs")
            }
        }
        lim.charge_bytes(self.arena_capacity_bytes().saturating_sub(before))?;
        Ok(())
    }

    /// Drop one pair from the node at `idx`, in place, and report whether the
    /// node still has pairs.
    ///
    /// Returns `Ok(false)` when `pair` was the node's only one, leaving the
    /// node untouched: a node with no pairs computes false, which invariant 2
    /// forbids, so emptying one is the caller's decision to make.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` if the re-encoding's range entry is
    /// refused.
    ///
    /// # Panics
    ///
    /// Panics if the node does not hold `pair`; a caller reaches this through
    /// the level's own pair list.
    pub(crate) fn remove_pair_from_node(
        &mut self, eng: &Engine, idx: usize, pair: ChildPair,
    ) -> Result<bool, OperationError> {
        let at = self.pairs_of_idx(idx).iter().position(|p| *p == pair)
            .unwrap_or_else(|| panic!("remove_pair_from_node: node {idx} does not hold {pair:?}"));
        let len = self.pairs_of_idx(idx).len();
        if len == 1 { return Ok(false); }
        self.reserve_shrunk_multi(eng, idx, len - 1)?;
        let range = self.pair_range_at(idx);
        self.pairs.copy_within(range.start + at + 1..range.end, range.start + at);
        let dead = self.reencode_shrunk_multi_reserved(idx, range.start, len, len - 1);
        self.note_dead_pairs(dead);
        Ok(true)
    }

    /// Size the arenas for `nodes` more nodes and `pairs` more pairs, charging
    /// the growth to `lim`.
    pub(crate) fn reserve_on(
        &mut self,
        lim: &crate::limits::Limits,
        nodes: usize,
        pairs: usize,
    ) -> Result<(), OperationError> {
        lim.reserve_exact(&mut self.nodes, nodes)?;
        lim.reserve_exact(&mut self.pairs, pairs)
    }

    /// The allocated bytes of the three structural arenas.
    pub(crate) fn arena_capacity_bytes(&self) -> u64 {
        self.nodes.charged_bytes() + self.pairs.charged_bytes() + self.multi_pairs.charged_bytes()
    }

    /// [`push_internal_node`](Self::push_internal_node) for the apply
    /// emitters: every buffer is reserved first, and a refused reservation
    /// comes back as `Err(())`, which the caller maps to
    /// `OperationError::OverBudget`.
    ///
    /// # Errors
    ///
    /// A buffer reservation was refused.
    #[inline]
    pub(crate) fn try_push_internal_node(
        &mut self,
        input_pairs: &[ChildPair],
    ) -> Result<NodeIdx, ()> {
        let idx = NodeIdx(self.nodes.len() as u32);
        if input_pairs.len() == 1 && input_pairs[0].can_inline() {
            reserve(&mut self.nodes, 1)?;
            self.nodes.push(EncodedNode::inline(input_pairs[0]));
        } else if input_pairs.len() == 1 {
            // Single pair that can't be inlined (right has `LEAF_BIT` or left has `MULTI_BIT`).
            // Use extended encoding — the only form that supports pair_len=1 without
            // aliasing either the leaf or `multi_ranged` encoding.
            let pair_start = self.pairs.len();
            reserve(&mut self.pairs, 1)?;
            self.pairs.push(input_pairs[0]);
            let multi_pairs_idx = self.multi_pairs.len();
            reserve(&mut self.multi_pairs, 1)?;
            self.multi_pairs.push(MultiPairRange { start: pair_start as u64, len: 1 });
            reserve(&mut self.nodes, 1)?;
            self.nodes.push(EncodedNode::multi_ranged(multi_pairs_idx as u32));
        } else {
            let pair_start = self.pairs.len();
            let pair_len = input_pairs.len();
            reserve(&mut self.pairs, pair_len)?;
            self.pairs.extend_from_slice(input_pairs);
            let data = self.try_encode_multi(pair_start, pair_len)?;
            reserve(&mut self.nodes, 1)?;
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

    /// Growth and extended-encoding arm of `try_push_multi_by_range`, reached
    /// only when `nodes` is full or when `pair_start`/`pair_len` overflow
    /// 31 bits.
    #[cold]
    #[inline(never)]
    fn push_multi_by_range_slow(
        &mut self,
        pair_start: usize,
        pair_len: usize,
    ) -> Result<(), ()> {
        let data = self.try_encode_multi(pair_start, pair_len)?;
        reserve(&mut self.nodes, 1)?;
        self.nodes.push(data);
        Ok(())
    }
}

impl Charged for TddLevel {
    /// The structural arenas: the meter charged their growth, so their
    /// capacity is what a dropped level hands back.
    #[inline]
    fn charged_bytes(&self) -> u64 {
        self.arena_capacity_bytes()
    }
}
