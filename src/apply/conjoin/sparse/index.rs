//! The reusable sparse workspace: reverse indices and buckets.

use super::*;
use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use crate::limits::{Charged, Limits};

/// A candidate of one f parent: a g parent and the two child products its
/// pair joins.
#[derive(Clone, Copy)]
pub(super) struct ParEntry {
    pub(super) g_parent: u32,   // g parent index
    pub(super) left_prod: u32,  // compacted left-child product index
    pub(super) right_prod: u32, // compacted right-child product index
}

/// Reusable workspace for sparse product construction.
///
/// Each checkout owns its buffers, including during nested operations. Capacity
/// is reused within the engine's shared retention ceiling; arrays are reset over
/// their live range before use.
#[derive(Default)]
pub(crate) struct SparseWorkspace {
    // ── Scatter: reverse indices (child → parent) ──
    pub(super) f_by_outer: Grouped<RevEntry>,
    pub(super) g_by_outer: Grouped<RevEntry>,
    // ── The leaf arm under a vtree leaf: g's pairs as a table of owners ──
    // Slot `OWNER_SLOTS * d + s`: the g node holding outer child `d` with the
    // label of slot `s` (`build_owner_table`). All `NO_PRODUCT` whenever
    // `owners_clean`.
    pub(super) owners: Vec<u32>,
    pub(super) owners_clean: bool,
    // The outer child's live products that meet an owner, each with its
    // slots (`gather_owned_products`), by f index through `outer_offsets`;
    // the first `owned_len` entries are this level's.
    pub(super) owned: Vec<OwnedProduct>,
    pub(super) owned_len: usize,
    // Whether the leaf arm found each of the outer child's live products
    // held for every label (`holds_every_label`): set by the arm for the
    // level being joined, false otherwise.
    pub(super) outer_held: bool,

    // ── The two children's product lists read by f index ──
    // A product list is emitted in ascending `f_idx` order by every
    // producer, so its bucket for one f index is a slice of it; these hold
    // the slice bounds (`bucket_offsets`) for the inner and the outer child.
    pub(super) inner_offsets: Vec<u32>,
    pub(super) outer_offsets: Vec<u32>,

    // ── The scatter (`scatter_join`) ──
    // Per-outer filtered g index: inner-g-child → [(`g_parent`, attached product)], rebuilt
    // each outer from the live set + the opposite-keyed g reverse index, so the
    // emit loop iterates only alive entries, with no dead probes.
    //   normal:  filtered[a2] = [(g_parent, right_prod)]   swapped: filtered[s2] = [(g_parent, left_prod)]
    pub(super) filtered: Rows<(u32, u32)>,
    pub(super) filtered_touched: Vec<u32>,          // indices of `filtered` written this outer, to clear
    pub(super) filtered_held: Vec<u64>,             // a bit per `filtered` bucket, set while it holds an entry

    // ── Output-sensitive join: the inner-g children the emit will read ──
    // The emit reads `filtered` only at the g children this outer's f parents
    // name through their live left products, so the build buckets only those
    // — a semi-join of the g side against the f side, one level up.
    pub(super) wanted: Vec<u32>,                    // inner-g children this outer's emit reads
    pub(super) wanted_epoch: u32,                   // the stamp that counts as marked
    pub(super) wanted_keys: Vec<u32>,               // the marked inner-g children, in marking order
    pub(super) inner_seen: Vec<u32>,                // inner f children already walked this outer
    pub(super) inner_seen_epoch: u32,               // likewise

    // ── Output-sensitive join: the second way to build `filtered` ──
    // g's reverse index keyed by the join's inner-g child, the opposite key
    // from `g_by_outer`. An outer whose g keys hold most of the level's g
    // pairs — a g operand free over the outer child has all of them under one
    // key — builds `filtered` from this index instead, walking the parents of
    // the wanted inner-g children and keeping those under one of the outer's
    // keys. `outer_keys` marks those keys for the walk and `outer_attached`
    // holds each one's product.
    pub(super) g_by_inner: Grouped<RevEntry>,
    pub(super) outer_keys: Vec<u32>,
    pub(super) outer_keys_epoch: u32,
    pub(super) outer_attached: Vec<u32>,

    // ── The general arm's walk of f's nodes: every outer's `filtered` kept ──
    pub(super) routes: Routes,
    // The outer keys whose route build marks the keys it reads, a bit each,
    // and the pairs `f_by_outer` keeps under each outer key on a routed
    // level (`build_kept_index`).
    pub(super) f_marked: Vec<u64>,
    pub(super) f_kept: Vec<u32>,

    // ── Candidates, and their dedup into products ──
    // The g parents of one f node's candidates, in candidate order, for the
    // walks of f's nodes that write the candidates' pairs straight into the
    // level (`emit_walked`): the scatter's and the probe's.
    pub(super) node_owners: Vec<u32>,

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // f-by-left, f-by-right, g-by-left, g-by-right.
    pub(super) est_counts: Vec<u32>,

    // ── The emit: one f parent's products into output nodes ──
    pub(super) emit: EmitScratch,

    // ── The probe join (`probe_level`) ──
    pub(super) probe: super::probe::ProbeBuffers,
}

/// What the emit (`emit_walked`) writes as
/// it turns one f parent's candidates into nodes, apart from the buffers
/// the candidates are read from, so that a walk can hold those while it
/// emits.
#[derive(Default)]
pub(crate) struct EmitScratch {
    /// `p2_map[g_parent]`: the product the parent's candidates under that
    /// `g` parent make, `NO_PRODUCT` for none yet. All `NO_PRODUCT` between
    /// parents: the emit restores the entries it wrote.
    pub(super) p2_map: Vec<u32>,
    /// Per product, its pair count, then its write cursor.
    pub(super) pair_counts: Vec<u32>,
    /// The pair of each one-pair product, stored inline.
    pub(super) single_pairs: Vec<ChildPair>,
    /// A walked parent's pairs, moved out of the level's arena while the
    /// emit writes them back by product.
    pub(super) moved_pairs: Vec<ChildPair>,
}

impl Buffers for EmitScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.p2_map);
        visit(&mut self.pair_counts);
        visit(&mut self.single_pairs);
        visit(&mut self.moved_pairs);
    }
}

/// Every outer key's `filtered` index of one level, kept for a walk of f's
/// nodes that reads them by outer key and inner-g child at random.
///
/// Key `k` of outer `o` (an inner-g child) holds the `(g_parent, attached)`
/// entries `filtered` would hold for it under `o`, as one run of `entries`,
/// written there as the outer's index is built. Each
/// outer's keys are a table of their own, its `segment`: an open-addressed
/// `slots[base..base + 2^bits]`, probed linearly from the key's hash and at
/// most half full, or, where the outer holds at least one key in
/// [`DENSE_SPAN`] of the inner-g children, the run bounds
/// `bounds[base..base + width + 1]` indexed by the key itself
/// ([`Segment::DENSE`]), the runs laid out in key order: key `k`'s run is
/// `entries[bounds[base + k]..bounds[base + k + 1]]`, so a key is found or
/// missed in one read of four bytes a key. An outer with no key has `bits`
/// 0 and no table.
#[derive(Default)]
pub(super) struct Routes {
    pub(super) segments: Vec<Segment>,
    pub(super) slots: Vec<RouteSlot>,
    /// The dense outers' run bounds, `width + 1` an outer.
    pub(super) bounds: Vec<u32>,
    pub(super) entries: Vec<(u32, u32)>,
    /// A bit per f node holding a pair under an outer with a route
    /// ([`Routes::mark_parents`]).
    pub(super) walked: Vec<u64>,
    /// The keys of the outer being built, each with its run, until
    /// [`Routes::place`] gives them a table.
    pub(super) pending: Vec<RouteSlot>,
    /// Per inner-g child, while an outer is built from entries that do not
    /// come grouped by key: its count, then its write cursor in `entries`.
    /// Zero between outers.
    pub(super) cursor: Vec<u32>,
}

/// One outer key's table in [`Routes`]: its first slot, and log2 of its
/// slot count, 0 for an outer with no key; or, with `bits`
/// [`Segment::DENSE`], its first run bound, the table indexed by key.
#[derive(Clone, Copy, Default)]
pub(super) struct Segment {
    pub(super) base: u32,
    pub(super) bits: u32,
}

impl Segment {
    /// `bits` of a table indexed by key: a run bound for every inner-g
    /// child, and one past the last.
    pub(super) const DENSE: u32 = 32;
}

/// An outer's keys take a table indexed by key where they are at least one
/// in this many of the inner-g children: at most 16 bytes a key, against
/// 24 to 48 in the open-addressed table, which holds two to four 12-byte
/// slots a key.
pub(super) const DENSE_SPAN: usize = 4;

/// One slot of a [`Routes`] table: an inner-g child and its run of
/// entries, or [`RouteSlot::EMPTY`].
#[derive(Clone, Copy)]
pub(super) struct RouteSlot {
    pub(super) key: u32,
    pub(super) start: u32,
    pub(super) len: u32,
}

impl RouteSlot {
    /// No key: a child index never reaches `u32::MAX`.
    pub(super) const EMPTY: RouteSlot = RouteSlot { key: u32::MAX, start: 0, len: 0 };
}

/// The first slot a table of `2^bits` slots, `bits >= 1`, probes for `key`:
/// the top bits of a multiplicative hash, which spreads runs of
/// consecutive keys.
#[inline(always)]
pub(super) fn route_slot(key: u32, bits: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B9) >> (32 - bits)) as usize
}

impl Routes {
    /// Empty, for a level of `outers` outer keys, each with no key yet, and
    /// `width` inner-g children.
    pub(super) fn reset(&mut self, lim: &Limits, outers: usize, width: usize) -> Result<(), OperationError> {
        self.segments.clear();
        lim.try_resize(&mut self.segments, outers, Segment::default())?;
        self.slots.clear();
        self.bounds.clear();
        self.entries.clear();
        self.pending.clear();
        self.cursor.clear();
        lim.try_resize(&mut self.cursor, width, 0)
    }

    /// Whether an outer of `keys` keys, of `width` inner-g children, takes
    /// a table indexed by key, its runs laid out in key order.
    pub(super) fn dense(keys: usize, width: usize) -> bool {
        width <= DENSE_SPAN * keys
    }

    /// Start outer `outer`'s table indexed by key, before its runs are
    /// written: the bounds that follow, [`Routes::bound`] at each of the
    /// `width` keys in order and [`Routes::close`] after the last, are its.
    pub(super) fn open(&mut self, lim: &Limits, outer: usize, width: usize) -> Result<(), OperationError> {
        let base = self.bounds.len();
        if base + width + 1 > u32::MAX as usize {
            return Err(OperationError::IndexOverflow);
        }
        // Grown by doubling, as `slots`.
        reserve_room(lim, &mut self.bounds, width + 1)?;
        self.segments[outer] = Segment { base: base as u32, bits: Segment::DENSE };
        Ok(())
    }

    /// The next key's run starts at `start` of `entries`. Within the room
    /// [`Routes::open`] reserved.
    #[inline(always)]
    pub(super) fn bound(&mut self, start: usize) {
        debug_assert!(self.bounds.len() < self.bounds.capacity(), "a bound past the room opened");
        self.bounds.push(start as u32);
    }

    /// End outer `outer`'s table indexed by key at `end` of `entries`; an
    /// outer whose runs are all empty has no table.
    pub(super) fn close(&mut self, outer: usize, end: usize) {
        let base = self.segments[outer].base as usize;
        if self.bounds[base] as usize == end {
            self.bounds.truncate(base);
            self.segments[outer] = Segment::default();
        } else {
            self.bound(end);
        }
    }

    /// Give outer `outer` an open-addressed table of the `pending` keys.
    pub(super) fn place(&mut self, lim: &Limits, outer: usize) -> Result<(), OperationError> {
        let keys = self.pending.len();
        if keys == 0 {
            return Ok(());
        }
        let bits = (2 * keys).next_power_of_two().trailing_zeros();
        let base = self.slots.len();
        let size = 1usize << bits;
        if base + size > u32::MAX as usize {
            return Err(OperationError::IndexOverflow);
        }
        // Grown by doubling: the level's tables are placed one outer at a
        // time, and an exact reserve would move them at nearly every one.
        reserve_room(lim, &mut self.slots, size)?;
        self.slots.resize(base + size, RouteSlot::EMPTY);
        let table = &mut self.slots[base..];
        for &slot in &self.pending {
            let mut at = route_slot(slot.key, bits);
            while table[at].key != u32::MAX {
                at = (at + 1) & (size - 1);
            }
            table[at] = slot;
        }
        self.segments[outer] = Segment { base: base as u32, bits };
        Ok(())
    }

    /// The tables and runs, for reading.
    pub(super) fn view(&self) -> RoutesView<'_> {
        RoutesView { segments: &self.segments, slots: &self.slots, bounds: &self.bounds, entries: &self.entries }
    }

    /// Set `walked`'s bit for each of the first `parents` nodes of `f_level`
    /// holding a pair under an outer with a route, the outer child the right
    /// one, or the left one where `SWAPPED`.
    pub(super) fn mark_parents<const SWAPPED: bool>(
        &mut self,
        lim: &Limits,
        f_level: &TddLevel,
        parents: usize,
    ) -> Result<(), OperationError> {
        let words = parents.div_ceil(64);
        self.walked.clear();
        lim.try_resize(&mut self.walked, words, 0u64)?;
        let Routes { segments, walked, .. } = self;
        f_level.try_for_each_node::<std::convert::Infallible>(0..parents, |p1, pairs| {
            let routed = pairs.iter().any(|pair| {
                let outer = if !SWAPPED { pair.right.0 } else { pair.left.0 };
                segments[outer as usize].bits != 0
            });
            walked[p1 >> 6] |= u64::from(routed) << (p1 & 63);
            Ok(())
        }).unwrap_or_else(|never| match never {});
        Ok(())
    }
}

impl Buffers for Routes {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.segments);
        visit(&mut self.slots);
        visit(&mut self.bounds);
        visit(&mut self.entries);
        visit(&mut self.walked);
        visit(&mut self.pending);
        visit(&mut self.cursor);
    }
}

/// [`Routes`] borrowed for reading.
#[derive(Clone, Copy)]
pub(super) struct RoutesView<'a> {
    pub(super) segments: &'a [Segment],
    pub(super) slots: &'a [RouteSlot],
    pub(super) bounds: &'a [u32],
    pub(super) entries: &'a [(u32, u32)],
}

impl<'a> RoutesView<'a> {
    /// The entries of inner-g child `key` under the outer of `segment`, empty
    /// where it has none.
    #[inline(always)]
    pub(super) fn find(self, segment: Segment, key: u32) -> &'a [(u32, u32)] {
        if segment.bits == Segment::DENSE {
            // Keys are the level's inner-g children, below the width the
            // table was opened with; an empty run has equal bounds.
            let at = segment.base as usize + key as usize;
            let bounds = &self.bounds[at..at + 2];
            return &self.entries[bounds[0] as usize..bounds[1] as usize];
        }
        let size = 1usize << segment.bits;
        let table = &self.slots[segment.base as usize..segment.base as usize + size];
        let mut at = route_slot(key, segment.bits);
        loop {
            let slot = table[at];
            if slot.key == key {
                return &self.entries[slot.start as usize..(slot.start + slot.len) as usize];
            }
            if slot.key == u32::MAX {
                return &[];
            }
            at = (at + 1) & (size - 1);
        }
    }

    /// Ask the cache for the place a lookup of `key` under the outer of
    /// `segment` reads first.
    #[inline(always)]
    pub(super) fn touch(self, segment: Segment, key: u32) {
        if segment.bits == Segment::DENSE {
            prefetch_at(self.bounds, segment.base as usize + key as usize);
        } else {
            prefetch_at(self.slots, segment.base as usize + route_slot(key, segment.bits));
        }
    }
}

/// One entry of a reverse index: a parent of the keyed child, and the child it
/// holds on the other side of the same pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RevEntry {
    pub(super) parent: u32,
    pub(super) other: u32,
}

/// Entries grouped by a key in `0..n`: key `k`'s run is
/// `entries[offsets[k]..offsets[k + 1]]`, so `offsets` holds `n + 1` bounds.
pub(super) struct Grouped<T> {
    pub(super) offsets: Vec<u32>,
    pub(super) entries: Vec<T>,
}

impl<T> Default for Grouped<T> {
    fn default() -> Self {
        Grouped { offsets: Vec::new(), entries: Vec::new() }
    }
}

impl<T> Grouped<T> {
    /// The grouping, borrowed for reading.
    #[inline]
    pub(super) fn view(&self) -> GroupedView<'_, T> {
        GroupedView { offsets: &self.offsets, entries: &self.entries }
    }

}

impl<T> Buffers for Grouped<T> {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.offsets);
        visit(&mut self.entries);
    }
}

/// A [`Grouped`] borrowed for reading, or any list already in key order
/// together with its bucket bounds.
#[derive(Clone, Copy)]
pub(super) struct GroupedView<'a, T> {
    pub(super) offsets: &'a [u32],
    pub(super) entries: &'a [T],
}

impl<'a, T> GroupedView<'a, T> {
    /// The entries under key `k`.
    #[inline]
    pub(super) fn bucket(self, k: usize) -> &'a [T] {
        &self.entries[self.offsets[k] as usize..self.offsets[k + 1] as usize]
    }

    /// Every entry of the first `keys` keys, in key order: the whole
    /// grouping when `keys` is its key count. The buffers are grow-only, so
    /// past that they may hold a wider grouping's entries.
    #[inline]
    pub(super) fn entries_of(self, keys: usize) -> &'a [T] {
        &self.entries[..self.offsets[keys] as usize]
    }

    /// How many entries key `k` holds.
    #[inline]
    pub(super) fn len(self, k: usize) -> usize {
        (self.offsets[k + 1] - self.offsets[k]) as usize
    }

    /// Ask the cache for the first two lines of key `k`'s entries, for a
    /// walk that reads them a few keys from now: its keys are random, so
    /// each run's start is a miss the walk would otherwise wait on. A hint
    /// only; no-op off `x86_64` and under Miri, which lacks the intrinsic.
    #[inline(always)]
    pub(super) fn prefetch_bucket(self, k: usize) {
        let at = self.entries.as_ptr().wrapping_add(self.offsets[k] as usize).cast::<u8>();
        prefetch_line(at);
        prefetch_line(at.wrapping_add(64));
    }

    /// Ask the cache for the lines of key `k`'s entries, up to
    /// [`BUCKET_LINES`] and no more than the run holds: the second stage of
    /// a walk that asks for a bucket some keys ahead. Reads key `k`'s
    /// bounds, which the first stage asked for ([`Self::prefetch_bounds`]).
    /// A hint only.
    #[inline(always)]
    pub(super) fn prefetch_run(self, k: usize) {
        let (start, end) = (self.offsets[k] as usize, self.offsets[k + 1] as usize);
        let at = self.entries.as_ptr().wrapping_add(start).cast::<u8>();
        let bytes = (end - start) * std::mem::size_of::<T>();
        let mut line = 0;
        while line < bytes && line < BUCKET_LINES * 64 {
            prefetch_line(at.wrapping_add(line));
            line += 64;
        }
    }

    /// Ask the cache for key `k`'s bounds, which [`Self::prefetch_run`]
    /// and [`Self::bucket`] read: the first stage of a walk that asks for a
    /// bucket some keys ahead, since forming a bucket's address is itself a
    /// read at a random place. A hint only.
    #[inline(always)]
    pub(super) fn prefetch_bounds(self, k: usize) {
        prefetch_line(self.offsets.as_ptr().wrapping_add(k).cast::<u8>());
    }
}

/// How many cache lines of a bucket [`GroupedView::prefetch_run`] asks
/// for: a longer run is read in order from there, which the hardware's own
/// prefetcher follows.
const BUCKET_LINES: usize = 4;

/// Ask the cache for the line holding `at`. A hint only: a prefetch reads
/// nothing the program sees and never faults, so any address is sound. A
/// no-op off `x86_64` and under Miri, which lacks the intrinsic.
#[inline(always)]
pub(super) fn prefetch_line(at: *const u8) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        // Sound whatever the address: a prefetch reads nothing the program
        // sees and never faults.
        unsafe { core::arch::x86_64::_mm_prefetch(at.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0) };
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = at;
}

/// How many f pairs ahead of the one it reads a walk through inner buckets
/// asks the cache for a bucket's bounds ([`GroupedView::prefetch_bounds`]),
/// and how many for the bucket's run ([`GroupedView::prefetch_run`]): the
/// run's address is read from the bounds, so the bounds are asked for
/// first, far enough ahead to have arrived when the run is.
pub(super) const BOUNDS_AHEAD: usize = 16;
pub(super) const RUN_AHEAD: usize = 8;

/// The prefetches of a walk that reads, for each f pair `under[j]` in turn,
/// the inner bucket its `other` child names: the bounds of the pair
/// [`BOUNDS_AHEAD`] on and the run of the pair [`RUN_AHEAD`] on. `under` is
/// a whole reverse index's entries ([`GroupedView::entries_of`]), so a walk
/// over one key's pairs asks for the next key's first ones too.
#[inline(always)]
pub(super) fn prefetch_walk<T: Copy>(under: &[RevEntry], j: usize, inner: GroupedView<'_, T>) {
    if let Some(r) = under.get(j + BOUNDS_AHEAD) {
        inner.prefetch_bounds(r.other as usize);
    }
    if let Some(r) = under.get(j + RUN_AHEAD) {
        inner.prefetch_run(r.other as usize);
    }
}

/// Ask the cache for `slice[k]`, which need not be in bounds; see
/// [`prefetch_line`].
#[inline(always)]
pub(super) fn prefetch_at<T>(slice: &[T], k: usize) {
    prefetch_line(slice.as_ptr().wrapping_add(k).cast::<u8>());
}

/// What a counting sort groups: items it reads twice, once to count their
/// keys and once to place them, in the same order both times.
pub(super) trait SortItems {
    type Item;
    /// Hand every item to `f`, in order.
    fn each(&self, f: impl FnMut(Self::Item));
}

/// An iterator is read by a clone of it for each pass.
impl<I: Iterator + Clone> SortItems for I {
    type Item = I::Item;
    #[inline(always)]
    fn each(&self, f: impl FnMut(I::Item)) {
        self.clone().for_each(f);
    }
}

/// A level's pairs, each with the index of the node holding it, read in
/// loops over its nodes ([`TddLevel::for_each_node_pair`]): a stored node's
/// pairs as a slice, an implicit level's generated from its description.
/// The fold of [`TddLevel::pairs_with_parent`] makes a call out of line for
/// each node, and these loops make none.
pub(super) struct NodePairs<'a>(pub(super) &'a TddLevel);

impl SortItems for NodePairs<'_> {
    type Item = (u32, ChildPair);
    #[inline(always)]
    fn each(&self, mut f: impl FnMut((u32, ChildPair))) {
        self.0.for_each_node_pair(|i, pair| f((i as u32, pair)));
    }
}

/// Group `items` by key into `out`: a counting sort in four passes. Count
/// each key — or take `counts`, the histogram when the caller already has
/// one — turn the counts into exclusive prefix sums, scatter every item
/// through its key's offset as the write cursor, then shift the offsets back
/// by one so each names the start of its run again. `items` is read twice
/// ([`SortItems`]); `item` gives each one's key and the value stored for it,
/// and `fill` is what the entries are sized with before the scatter. The
/// offsets are 32-bit, so more than `u32::MAX` items is
/// [`OperationError::IndexOverflow`] rather than a wrapped total.
pub(super) fn counting_sort<S, T: Copy>(
    lim: &Limits,
    n_keys: usize,
    items: impl SortItems<Item = S>,
    item: impl Fn(S) -> (usize, T),
    counts: Option<&[u32]>,
    fill: T,
    out: &mut Grouped<T>,
) -> Result<(), OperationError> {
    let Grouped { offsets, entries } = out;
    lim.try_resize(offsets, n_keys + 1, 0)?;
    offsets[n_keys] = 0;
    if let Some(counts) = counts {
        offsets[..n_keys].copy_from_slice(counts);
    } else {
        offsets[..n_keys].fill(0);
        // A key's count can only wrap once the items outnumber `u32::MAX`,
        // which the item count catches where the total would not.
        let mut seen = 0u64;
        items.each(|s| {
            let (key, _) = item(s);
            debug_assert!(key < n_keys, "key {key} outside the {n_keys} keys sorted");
            offsets[key] = offsets[key].wrapping_add(1);
            seen += 1;
        });
        if seen > u64::from(u32::MAX) {
            return Err(OperationError::IndexOverflow);
        }
    }
    let total = prefix_offsets(&mut offsets[..n_keys], false)?;
    offsets[n_keys] = total;
    lim.try_resize(entries, total as usize, fill)?;
    items.each(|s| {
        let (key, value) = item(s);
        let slot = offsets[key];
        entries[slot as usize] = value;
        // The cursor is stored, not stepped in place: stepped, it would be
        // read again after the entry's store, which for all the compiler
        // knows could have changed it.
        offsets[key] = slot + 1;
    });
    shift_offsets_right_by_one(&mut offsets[..=n_keys]);
    Ok(())
}

/// Replace counts with exclusive offsets. With inline singles, counts below
/// two use the reserved marker instead of arena space; no range may end there.
pub(super) fn prefix_offsets(counts: &mut [u32], inline_singles: bool) -> Result<u32, OperationError> {
    let limit = u32::MAX - u32::from(inline_singles);
    let mut total = 0u32;
    for slot in counts {
        let count = *slot;
        if inline_singles && count < 2 {
            *slot = u32::MAX;
        } else {
            *slot = total;
            total = total.checked_add(count).filter(|&n| n <= limit)
                .ok_or(OperationError::IndexOverflow)?;
        }
    }
    Ok(total)
}

/// Build a reverse index from a level's pairs, keyed by one child side:
///   `BY_RIGHT = false`: left_child_idx  → [(parent_idx, right_sibling_idx)]
///   `BY_RIGHT = true` : right_sibling_idx → [(parent_idx, left_child_idx)]
/// grouped by the key-side child index.
///
/// `BY_RIGHT` is a const generic so the `if BY_RIGHT` branches fold away.
/// `counts` is the histogram by this child when the direction estimate has
/// already counted this level's pairs.
pub(super) fn build_reverse_index<const BY_RIGHT: bool>(
    eng: &Engine,
    level: &TddLevel,
    key_width: usize,
    counts: Option<&[u32]>,
    index: &mut Grouped<RevEntry>,
) -> Result<(), OperationError> {
    counting_sort(
        eng.limits(), key_width, NodePairs(level),
        |(parent, pair)| {
            let (key, other) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
            (key as usize, RevEntry { parent, other })
        },
        counts, RevEntry { parent: 0, other: 0 }, index,
    )
}

/// The runs [`build_labelled_index`] cuts each outer child's entries into.
pub(super) const LABEL_RUNS: usize = 3;

/// Build a reverse index of a level whose other child is a vtree leaf, keyed
/// by the keyed child and then by the leaf label, Pos, One, Neg: the keyed
/// child `k`'s entries with label run `r` are bucket `LABEL_RUNS * k + r`,
/// in the order [`build_reverse_index`] would give them. A literal meets the
/// One run and its own, which are next to each other.
pub(super) fn build_labelled_index<const BY_RIGHT: bool>(
    eng: &Engine,
    level: &TddLevel,
    key_width: usize,
    index: &mut Grouped<RevEntry>,
) -> Result<(), OperationError> {
    // The run of each label, One, Pos, Neg in label order.
    const RUN: [usize; 3] = [1, 0, 2];
    counting_sort(
        eng.limits(), LABEL_RUNS * key_width, NodePairs(level),
        |(parent, pair)| {
            let (key, other) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
            (LABEL_RUNS * key as usize + RUN[other as usize], RevEntry { parent, other })
        },
        None, RevEntry { parent: 0, other: 0 }, index,
    )
}

/// The slots [`build_owner_table`] gives each keyed child, one per leaf
/// label: Pos, One, Neg.
pub(super) const OWNER_SLOTS: usize = 3;

/// The slot of each leaf label, One, Pos, Neg in label order, in a keyed
/// child's run of [`OWNER_SLOTS`].
const OWNER_SLOT: [usize; 3] = [1, 0, 2];

/// Fill `owners` with the owners of a level's pairs whose other child is a
/// vtree leaf: slot `OWNER_SLOTS * k + s` names the node holding the pair of
/// keyed child `k` and the label of slot `s`, `NO_PRODUCT` where no node
/// does. A deterministic level holds each such pair once; where a pair
/// repeats, in two nodes or twice in one as a marginal diagram's pair lists
/// may, the table cannot hold it and `None` is returned. Otherwise
/// `Some(ones)`, with `ones` whether some pair has the label One.
///
/// `owners` is all `NO_PRODUCT` whenever `clean`: a table left behind by a
/// level that did not clear it ([`clear_owner_table`]) is filled first.
pub(super) fn build_owner_table<const BY_RIGHT: bool>(
    lim: &Limits,
    level: &TddLevel,
    key_width: usize,
    owners: &mut Vec<u32>,
    clean: &mut bool,
) -> Result<Option<bool>, OperationError> {
    if !*clean {
        owners.fill(NO_PRODUCT);
        *clean = true;
    }
    lim.try_resize(owners, OWNER_SLOTS * key_width, NO_PRODUCT)?;
    *clean = false;
    let (mut repeated, mut ones) = (false, false);
    level.for_each_node_pair(|parent, pair| {
        let (key, label) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
        let slot = &mut owners[OWNER_SLOTS * key as usize + OWNER_SLOT[label as usize]];
        repeated |= *slot != NO_PRODUCT;
        *slot = parent as u32;
        ones |= label == crate::diagram::ONE_LEAF_IDX.0;
    });
    Ok((!repeated).then_some(ones))
}

/// Whether g's pairs hold a keyed child for every label an f pair can have,
/// from its slots in an owner table ([`build_owner_table`]): a g node holds
/// it with One, which meets every label, or g nodes hold it with both
/// literals, one of which meets each. A node index is below 2^31, so the
/// or of the two literals' slots is all ones only where one is missing,
/// and its and with the One slot only where that is missing too.
#[inline(always)]
pub(super) fn holds_every_label(slots: [u32; OWNER_SLOTS]) -> bool {
    (slots[0] | slots[2]) & slots[1] != NO_PRODUCT
}

/// A live product of the outer child, `prod`, with the slots of its g node
/// in an owner table ([`build_owner_table`]): the g nodes holding it with
/// the labels Pos, One, Neg.
#[derive(Clone, Copy)]
pub(super) struct OwnedProduct {
    pub(super) prod: u32,
    pub(super) slots: [u32; OWNER_SLOTS],
}

/// Fill `owned` with the products of `list`, in its order, whose g node has
/// an owner in `owners`, each with its slots, and `offsets` with their
/// bucket bounds over `width` f indices, as [`bucket_offsets`] gives
/// `list`'s; return how many there are, and whether each of them is held
/// for every label ([`holds_every_label`]), on a level with no g pair
/// labeled One by both literals. One pass over `list`, which every producer
/// emits in ascending f order, reading each product's slots once: a walk
/// then reads a product and its slots together, instead of once per f pair
/// that meets the product, each read waiting on the one before.
pub(super) fn gather_owned_products(
    lim: &Limits,
    list: &[ProductEntry],
    width: usize,
    owners: &[u32],
    offsets: &mut Vec<u32>,
    owned: &mut Vec<OwnedProduct>,
) -> Result<(usize, bool), OperationError> {
    lim.try_resize(offsets, width + 1, 0u32)?;
    lim.try_resize(owned, list.len(), OwnedProduct { prod: 0, slots: [NO_PRODUCT; OWNER_SLOTS] })?;
    let (owners, _) = owners.as_chunks::<OWNER_SLOTS>();
    let bounds = &mut offsets[..=width];
    bounds.fill(0);
    // Each product writes its f index's end, so the last of the index
    // leaves it; no branch waits on where one f index's products stop,
    // nor on the slots' read.
    let (mut kept, mut held, mut last_f, mut descents) = (0usize, 0usize, 0usize, 0usize);
    let owned = &mut owned[..list.len()];
    for e in list {
        let f = e.f_idx.idx();
        let slots = owners[e.g_idx.idx()];
        // Written whether kept or not: the next product overwrites one not
        // kept.
        debug_assert!(kept < owned.len(), "kept counts the entries before this one");
        // Safety: `kept` counts kept entries before this one, so it is below
        // this entry's index in `list`, whose length `owned` has.
        unsafe { *owned.get_unchecked_mut(kept) = OwnedProduct { prod: e.prod_idx.0, slots } };
        // Some slot is filled: a node index is below 2^31, so the slots'
        // and is all ones only where all three are empty.
        kept += usize::from(slots[0] & slots[1] & slots[2] != NO_PRODUCT);
        held += usize::from(holds_every_label(slots));
        bounds[f + 1] = kept as u32;
        descents += usize::from(f < last_f);
        last_f = f;
    }
    cheap_assert!(descents == 0, "a product list is not in ascending f order ({descents} descents in {} entries)", list.len());
    // An f index without products ends where the one before it does.
    let mut end = 0;
    for bound in &mut bounds[1..] {
        end = end.max(*bound);
        *bound = end;
    }
    Ok((kept, held == kept))
}

/// Fill `owned` with each product of `list`, which holds one per f index in
/// order, with its slots in `owners`, at its f index: one pass over `list`,
/// each slots' read independent of the others. Return whether each product
/// is numbered as its f index and held for every label, and whether each is
/// held for every label ([`holds_every_label`]).
pub(super) fn gather_listed_products(
    lim: &Limits,
    list: &[ProductEntry],
    owners: &[u32],
    owned: &mut Vec<OwnedProduct>,
) -> Result<(bool, bool), OperationError> {
    if owned.len() < list.len() {
        lim.try_resize(owned, list.len(), OwnedProduct { prod: 0, slots: [NO_PRODUCT; OWNER_SLOTS] })?;
    }
    let (owners, _) = owners.as_chunks::<OWNER_SLOTS>();
    let (mut plain, mut held) = (0usize, 0usize);
    for (i, (to, e)) in owned.iter_mut().zip(list).enumerate() {
        let slots = owners[e.g_idx.idx()];
        *to = OwnedProduct { prod: e.prod_idx.0, slots };
        let holds = usize::from(holds_every_label(slots));
        held += holds;
        plain += holds & usize::from(e.prod_idx.0 as usize == i);
    }
    Ok((plain == list.len(), held == list.len()))
}

/// Slots per node of a level below which [`clear_owner_table`] empties the
/// table by writing it whole rather than slot by slot: a line of slots
/// written in order costs about what one write at random does.
const FILL_SLOTS_PER_NODE: usize = 16;

/// Empty the slots [`build_owner_table`] filled from `level` over
/// `key_width` keyed children: the table's first `OWNER_SLOTS * key_width`
/// slots written in order where they are few against the level's nodes,
/// otherwise each slot a pair filled.
pub(super) fn clear_owner_table<const BY_RIGHT: bool>(
    level: &TddLevel,
    key_width: usize,
    owners: &mut [u32],
    clean: &mut bool,
) {
    let slots = OWNER_SLOTS * key_width;
    if slots <= FILL_SLOTS_PER_NODE * level.node_count() {
        owners[..slots].fill(NO_PRODUCT);
    } else {
        level.for_each_node_pair(|_, pair| {
            let (key, label) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
            owners[OWNER_SLOTS * key as usize + OWNER_SLOT[label as usize]] = NO_PRODUCT;
        });
    }
    *clean = true;
}

/// After a counting-sort fill pass leaves each `offsets[i]` pointing one-past
/// the end of bucket `i`, shift the slice right by one so every `offsets[i]`
/// is restored to the start of its bucket (and `offsets[0] = 0`).
#[inline]
fn shift_offsets_right_by_one(offsets: &mut [u32]) {
    let mut prev = 0u32;
    for slot in offsets.iter_mut() {
        std::mem::swap(&mut *slot, &mut prev);
    }
}

/// Rows indexed by a level's child or parent, reset to each level's width.
///
/// Rows past the width keep their allocations for a wider level later, which
/// is how the sparse workspace amortizes them across levels and applies. Only
/// the open rows, those below the width, can change while it is open, so the
/// bytes the others hold are kept as one sum and counting the allocations
/// walks the open rows, not every row the widest level so far grew.
pub(super) struct Rows<T> {
    rows: Vec<Vec<T>>,
    open: usize,
    /// Bytes the rows at and past `open` hold.
    closed_bytes: u64,
}

impl<T> Default for Rows<T> {
    fn default() -> Self {
        Rows { rows: Vec::new(), open: 0, closed_bytes: 0 }
    }
}

fn row_bytes<T>(rows: &[Vec<T>]) -> u64 {
    rows.iter().map(Charged::charged_bytes).sum()
}

impl<T> Rows<T> {
    /// Open the first `n` rows, empty.
    pub(super) fn reset(&mut self, lim: &Limits, n: usize) -> Result<(), OperationError> {
        if self.rows.len() < n {
            let additional = n - self.rows.len();
            lim.reserve_exact(&mut self.rows, additional)?;
            self.rows.resize_with(n, Vec::new);
        }
        if n < self.open {
            self.closed_bytes += row_bytes(&self.rows[n..self.open]);
        } else {
            self.closed_bytes -= row_bytes(&self.rows[self.open..n]);
        }
        self.open = n;
        for row in &mut self.rows[..n] {
            row.clear();
        }
        Ok(())
    }
}

impl<T> std::ops::Deref for Rows<T> {
    type Target = [Vec<T>];
    #[inline]
    fn deref(&self) -> &[Vec<T>] {
        &self.rows[..self.open]
    }
}

impl<T> std::ops::DerefMut for Rows<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Vec<T>] {
        &mut self.rows[..self.open]
    }
}

impl<T> Charged for Rows<T> {
    fn charged_bytes(&self) -> u64 {
        self.rows.charged_bytes() + row_bytes(&self.rows[..self.open]) + self.closed_bytes
    }
}

impl<T> Scratch for Rows<T> {
    fn release(&mut self) {
        *self = Rows::default();
    }
}

impl Buffers for SparseWorkspace {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        self.f_by_outer.buffers(visit);
        self.g_by_outer.buffers(visit);
        visit(&mut self.owners);
        visit(&mut self.owned);
        visit(&mut self.inner_offsets);
        visit(&mut self.outer_offsets);
        visit(&mut self.filtered);
        visit(&mut self.filtered_touched);
        visit(&mut self.filtered_held);
        visit(&mut self.wanted);
        visit(&mut self.wanted_keys);
        visit(&mut self.inner_seen);
        self.g_by_inner.buffers(visit);
        visit(&mut self.outer_keys);
        visit(&mut self.outer_attached);
        self.routes.buffers(visit);
        visit(&mut self.f_marked);
        visit(&mut self.f_kept);
        visit(&mut self.node_owners);
        visit(&mut self.est_counts);
        self.emit.buffers(visit);
        self.probe.buffers(visit);
    }
}

impl PooledScratch for SparseWorkspace {
    fn prepare(&mut self) {}
}

/// [`Limits::reserve`](crate::limits::Limits::reserve) behind its own test
/// for room, inline: a buffer filled a node or a key at a time pays one
/// comparison where it has room, and the reserve's charging and growth
/// only where it has none.
#[inline(always)]
pub(super) fn reserve_room<T>(lim: &crate::limits::Limits, v: &mut Vec<T>, additional: usize) -> Result<(), OperationError> {
    if additional <= v.capacity() - v.len() {
        return Ok(());
    }
    reserve_cold(lim, v, additional)
}

/// The arm of [`reserve_room`] that grows.
#[cold]
#[inline(never)]
fn reserve_cold<T>(lim: &crate::limits::Limits, v: &mut Vec<T>, additional: usize) -> Result<(), OperationError> {
    lim.reserve(v, additional)
}

#[cfg(test)]
#[path = "tests/rows.rs"]
mod tests;
