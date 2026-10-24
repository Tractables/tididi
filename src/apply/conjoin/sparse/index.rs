//! The reusable sparse workspace: reverse indices and buckets.

use super::*;
use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use crate::limits::{Charged, Limits};

/// Candidate that survived the sibling liveness filter, grouped by f-parent.
#[derive(Clone, Copy)]
pub(super) struct ParEntry {
    pub(super) g_parent: u32,   // g parent index
    pub(super) left_prod: u32,  // compacted left-child product index
    pub(super) right_prod: u32, // compacted right-child product index
}

/// A candidate with the f parent it belongs to, as the flat list holds it
/// before the sort by parent.
#[derive(Clone, Copy)]
pub(super) struct Candidate {
    pub(super) parent: u32,
    pub(super) entry: ParEntry,
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

    // ── Candidates, and their dedup into products ──
    pub(super) par_buckets: Rows<ParEntry>,          // surviving candidates bucketed by f-parent
    // The flat alternative a level with many more parents than candidates
    // takes (`flat_candidates_win`): the scatter appends every candidate
    // with its parent to `par_flat`, and `sort_candidates` groups them by
    // parent into `par_sorted`.
    pub(super) par_flat: Vec<Candidate>,
    pub(super) par_sorted: Grouped<ParEntry>,
    // The candidates of one f node, for the walks that find a node's
    // candidates all at once and emit it before the next (the leaf arm and
    // the probe).
    pub(super) node_candidates: Vec<ParEntry>,

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

/// What the emit ([`emit_parent`](super::scatter::emit_parent)) writes as
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
}

impl Buffers for EmitScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.p2_map);
        visit(&mut self.pair_counts);
        visit(&mut self.single_pairs);
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

/// Empty the slots [`build_owner_table`] filled from `level`.
pub(super) fn clear_owner_table<const BY_RIGHT: bool>(level: &TddLevel, owners: &mut [u32], clean: &mut bool) {
    level.for_each_node_pair(|_, pair| {
        let (key, label) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
        owners[OWNER_SLOTS * key as usize + OWNER_SLOT[label as usize]] = NO_PRODUCT;
    });
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
        visit(&mut self.par_buckets);
        visit(&mut self.par_flat);
        self.par_sorted.buffers(visit);
        visit(&mut self.node_candidates);
        visit(&mut self.est_counts);
        self.emit.buffers(visit);
        self.probe.buffers(visit);
    }
}

impl PooledScratch for SparseWorkspace {
    fn prepare(&mut self) {}
}

#[cfg(test)]
#[path = "tests/rows.rs"]
mod tests;
