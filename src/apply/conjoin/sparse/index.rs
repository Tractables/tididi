//! The reusable sparse workspace: reverse indices and buckets.

use super::*;
use crate::limits::pool::SCRATCH_RETAIN_BYTES;
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

    // ── The scatter (`scatter_outsens`) ──
    // Per-outer filtered g index: inner-g-child → [(p2, attached_prod)], rebuilt
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

    // ── Dedup: parent dedup ──
    pub(super) par_buckets: Vec<Vec<ParEntry>>,     // surviving candidates bucketed by f-parent
    // The flat alternative a level with many more parents than candidates
    // takes (`flat_candidates_win`): the scatter appends every candidate
    // with its parent to `par_flat`, and `sort_candidates` groups them by
    // parent into `par_sorted`.
    pub(super) par_flat: Vec<Candidate>,
    pub(super) par_sorted: Grouped<ParEntry>,
    pub(super) p2_map: Vec<u32>,                    // flat lookup: p2_map[right_parent] → compacted idx, NO_PRODUCT if new
    pub(super) p2_map_touched: Vec<u32>,            // p2 values written into p2_map this p1's emit pass, to clear

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // f-by-left, f-by-right, g-by-left, g-by-right.
    pub(super) est_counts: Vec<u32>,

    // ── Node build: counting-sort pairs into output nodes ──
    pub(super) emit_pairs: Vec<(u32, ChildPair)>,   // (parent_prod_idx, pair) for all surviving pairs
    pub(super) pairs_by_parent: Grouped<ChildPair>, // the same pairs grouped by chunk-local parent
}

impl SparseWorkspace {
    /// Release inner Vec memory from bucket arrays whose retained capacity grew
    /// past [`SCRATCH_RETAIN_BYTES`]. Called after a large sparse level to avoid
    /// retaining peak allocations. Covers every `Vec<Vec<_>>` bucket array.
    fn release_if_large(&mut self, lim: &Limits) {
        crate::limits::pool::release_if_oversized(lim, &mut self.inner_offsets);
        crate::limits::pool::release_if_oversized(lim, &mut self.outer_offsets);
        drop_if_large(lim, &mut self.par_buckets);
        crate::limits::pool::release_if_oversized(lim, &mut self.par_flat);
        self.par_sorted.release_if_oversized(lim);
        drop_if_large(lim, &mut self.filtered);
        crate::limits::pool::release_if_oversized(lim, &mut self.wanted);
        crate::limits::pool::release_if_oversized(lim, &mut self.wanted_keys);
        crate::limits::pool::release_if_oversized(lim, &mut self.inner_seen);
        self.g_by_inner.release_if_oversized(lim);
        crate::limits::pool::release_if_oversized(lim, &mut self.outer_keys);
        crate::limits::pool::release_if_oversized(lim, &mut self.outer_attached);
    }
}

/// Drop and replace `v` with an empty Vec if its retained capacity — outer spine
/// (`capacity·size_of::<Vec<E>>`) plus Σ inner `capacity·size_of::<E>` — exceeds
/// [`SCRATCH_RETAIN_BYTES`], the same rule
/// [`release_if_oversized`](crate::limits::pool::release_if_oversized) applies to the flat buffers. Frees both the inner elements and the outer
/// allocation. Early-exits the summation as soon as the threshold is crossed, so
/// the common under-cap case pays at most one pass and the over-cap case stops
/// early. `size_of::<E>()` is a compile-time constant. The over-cap branch then
/// finishes the summation, because the amount handed back to `lim` has to be
/// what is actually freed and not just the threshold that tripped.
#[inline]
pub(super) fn drop_if_large<E>(lim: &Limits, v: &mut Vec<Vec<E>>) {
    let elem = std::mem::size_of::<E>();
    let spine = v.capacity().saturating_mul(std::mem::size_of::<Vec<E>>());
    let mut bytes = spine;
    let mut over = bytes > SCRATCH_RETAIN_BYTES;
    if !over {
        for inner in v.iter() {
            bytes = bytes.saturating_add(inner.capacity().saturating_mul(elem));
            if bytes > SCRATCH_RETAIN_BYTES {
                over = true;
                break;
            }
        }
    }
    if over {
        let freed = v.iter().fold(spine, |acc, inner| {
            acc.saturating_add(inner.capacity().saturating_mul(elem))
        });
        *v = Vec::new();
        lim.release_bytes(freed as u64);
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

    /// The bytes both buffers hold.
    pub(super) fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        capacity_bytes(&self.offsets).saturating_add(capacity_bytes(&self.entries))
    }

    /// Hand back either buffer whose capacity grew past the retention cap.
    pub(super) fn release_if_oversized(&mut self, lim: &Limits) {
        crate::limits::pool::release_if_oversized(lim, &mut self.offsets);
        crate::limits::pool::release_if_oversized(lim, &mut self.entries);
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
/// before the scatter.
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
        for s in items.clone() {
            let (key, _) = item(s);
            debug_assert!(key < n_keys, "key {key} outside the {n_keys} keys sorted");
            offsets[key] += 1;
        }
    }
    let mut total = 0u32;
    for slot in offsets.iter_mut().take(n_keys) {
        let count = *slot;
        *slot = total;
        total += count;
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
        .filter(|(_, node)| node.is_internal())
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

impl crate::limits::pool::PooledScratch for SparseWorkspace {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::{capacity_bytes, nested_bytes};
        [
            self.f_by_outer.retained_bytes(),
            self.g_by_outer.retained_bytes(),
            capacity_bytes(&self.inner_offsets),
            capacity_bytes(&self.outer_offsets),
            nested_bytes(&self.filtered),
            capacity_bytes(&self.filtered_touched),
            capacity_bytes(&self.wanted),
            capacity_bytes(&self.wanted_keys),
            capacity_bytes(&self.inner_seen),
            self.g_by_inner.retained_bytes(),
            capacity_bytes(&self.outer_keys),
            capacity_bytes(&self.outer_attached),
            nested_bytes(&self.par_buckets),
            capacity_bytes(&self.par_flat),
            self.par_sorted.retained_bytes(),
            capacity_bytes(&self.p2_map),
            capacity_bytes(&self.p2_map_touched),
            capacity_bytes(&self.est_counts),
            capacity_bytes(&self.emit_pairs),
            self.pairs_by_parent.retained_bytes(),
        ].into_iter().sum()
    }

    fn prepare(&mut self) {}
    fn retain(&mut self, lim: &Limits) { self.release_if_large(lim); }
}
