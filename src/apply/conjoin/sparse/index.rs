//! The reusable sparse workspace: reverse indices and buckets.

use super::*;
use crate::execution::pool::{Buffers, Nested, PooledScratch, Scratch};
use crate::limits::Limits;

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
    pub(super) filtered: Vec<Vec<(u32, u32)>>,
    pub(super) filtered_touched: Vec<u32>,          // indices of `filtered` written this outer, to clear

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
    pub(super) par_buckets: Vec<Vec<ParEntry>>,     // surviving candidates bucketed by f-parent
    // The flat alternative a level with many more parents than candidates
    // takes (`flat_candidates_win`): the scatter appends every candidate
    // with its parent to `par_flat`, and `sort_candidates` groups them by
    // parent into `par_sorted`.
    pub(super) par_flat: Vec<Candidate>,
    pub(super) par_sorted: Grouped<ParEntry>,
    pub(super) p2_map: Vec<u32>,                    // flat lookup: p2_map[g_parent] → product idx, NO_PRODUCT if new

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // f-by-left, f-by-right, g-by-left, g-by-right.
    pub(super) est_counts: Vec<u32>,

    // ── The emit: one f parent's products into output nodes ──
    pub(super) pair_counts: Vec<u32>,               // per-product pair count, then write cursors
    pub(super) single_pairs: Vec<ChildPair>,        // the pair of each one-pair product, stored inline
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

    /// How many entries key `k` holds.
    #[inline]
    pub(super) fn len(self, k: usize) -> usize {
        (self.offsets[k + 1] - self.offsets[k]) as usize
    }
}

/// Group `items` by key into `out`: a counting sort in four passes. Count
/// each key — or take `counts`, the histogram when the caller already has
/// one — turn the counts into exclusive prefix sums, scatter every item
/// through its key's offset as the write cursor, then shift the offsets back
/// by one so each names the start of its run again. `items` is walked twice,
/// so it is an iterator that can be cloned; `item` gives each one's key and
/// the value stored for it, and `fill` is what the entries are sized with
/// before the scatter. The offsets are 32-bit, so more than `u32::MAX` items
/// is [`OperationError::IndexOverflow`] rather than a wrapped total.
pub(super) fn counting_sort<S, T: Copy, I>(
    lim: &Limits,
    n_keys: usize,
    items: I,
    item: impl Fn(S) -> (usize, T),
    counts: Option<&[u32]>,
    fill: T,
    out: &mut Grouped<T>,
) -> Result<(), OperationError>
where
    I: Iterator<Item = S> + Clone,
{
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
        for s in items.clone() {
            let (key, _) = item(s);
            debug_assert!(key < n_keys, "key {key} outside the {n_keys} keys sorted");
            offsets[key] = offsets[key].wrapping_add(1);
            seen += 1;
        }
        if seen > u64::from(u32::MAX) {
            return Err(OperationError::IndexOverflow);
        }
    }
    let mut total = 0u32;
    for slot in offsets.iter_mut().take(n_keys) {
        let count = *slot;
        *slot = total;
        total = total.checked_add(count).ok_or(OperationError::IndexOverflow)?;
    }
    offsets[n_keys] = total;
    lim.try_resize(entries, total as usize, fill)?;
    for s in items {
        let (key, value) = item(s);
        let slot = offsets[key] as usize;
        entries[slot] = value;
        offsets[key] += 1;
    }
    shift_offsets_right_by_one(&mut offsets[..=n_keys]);
    Ok(())
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
    let pairs = level.nodes.iter().enumerate()
        .flat_map(|(parent, node)| level.pairs_of(node).iter().map(move |&pair| (parent as u32, pair)));
    counting_sort(
        eng.limits(), key_width, pairs,
        |(parent, pair)| {
            let (key, other) = if BY_RIGHT { (pair.right.0, pair.left.0) } else { (pair.left.0, pair.right.0) };
            (key as usize, RevEntry { parent, other })
        },
        counts, RevEntry { parent: 0, other: 0 }, index,
    )
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

/// Ensure `buckets` has ≥ `n` inner Vecs (growing via `resize_with`), then clear
/// the first `n`. Buckets that already existed keep their reserved capacity —
/// this is how the sparse workspace amortizes allocations across calls.
pub(super) fn ensure_buckets_cleared<T>(eng: &Engine, buckets: &mut Vec<Vec<T>>, n: usize) -> Result<(), OperationError> {
    let lim = eng.limits();
    if buckets.len() < n {
        let additional = n - buckets.len();
        lim.reserve_exact(buckets, additional)?;
        buckets.resize_with(n, Vec::new);
    }
    for b in &mut buckets[..n] {
        b.clear();
    }
    Ok(())
}

impl Buffers for SparseWorkspace {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        self.f_by_outer.buffers(visit);
        self.g_by_outer.buffers(visit);
        visit(&mut self.inner_offsets);
        visit(&mut self.outer_offsets);
        visit(&mut Nested(&mut self.filtered));
        visit(&mut self.filtered_touched);
        visit(&mut self.wanted);
        visit(&mut self.wanted_keys);
        visit(&mut self.inner_seen);
        self.g_by_inner.buffers(visit);
        visit(&mut self.outer_keys);
        visit(&mut self.outer_attached);
        visit(&mut Nested(&mut self.par_buckets));
        visit(&mut self.par_flat);
        self.par_sorted.buffers(visit);
        visit(&mut self.p2_map);
        visit(&mut self.est_counts);
        visit(&mut self.pair_counts);
        visit(&mut self.single_pairs);
    }
}

impl PooledScratch for SparseWorkspace {
    fn prepare(&mut self) {}
}
