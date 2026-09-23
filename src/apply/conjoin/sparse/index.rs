//! The reusable sparse workspace: reverse indices, buckets and node-index newtypes.

use super::*;
use crate::limits::pool::SCRATCH_RETAIN_BYTES;
use crate::limits::Limits;

/// Candidate that survived the sibling liveness filter, grouped by f-parent.
#[derive(Clone, Copy)]
pub(crate) struct ParEntry {
    pub(crate) p2: u32,      // g parent index
    pub(crate) a_prod: u32,  // compacted left-child product index
    pub(crate) sib_idx: u32, // compacted right-child product index
}

/// A candidate with the f parent it belongs to, as the flat list holds it
/// before the sort by parent.
#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) parent: u32,
    pub(crate) entry: ParEntry,
}

/// Index of a node in `f.levels[t].nodes`. Distinct from `RightNodeIdx` and
/// `ProductNodeIdx` so that construction-site swaps are caught at compile time.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct LeftNodeIdx(pub(crate) u32);

impl LeftNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in `g.levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct RightNodeIdx(pub(crate) u32);

impl RightNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in the output `levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct ProductNodeIdx(pub(crate) u32);

/// A live product node: the conjunction `f[left_idx] ∧ g[right_idx]` produced
/// the output node at `prod_idx` in the output level.
#[derive(Clone, Copy)]
pub(crate) struct ProductEntry {
    pub(crate) left_idx: LeftNodeIdx,
    pub(crate) right_idx: RightNodeIdx,
    pub(crate) prod_idx: ProductNodeIdx,
}

/// Reusable workspace for sparse product construction.
///
/// Each checkout owns its buffers, including during nested operations. Capacity
/// is reused within the engine's shared retention ceiling; arrays are reset over
/// their live range before use.
#[derive(Default)]
pub(crate) struct SparseWorkspace {
    // ── Phase A: reverse indices (child → parent) for scatter ──
    pub(crate) rev_entries_c1: Vec<RevEntry>,
    pub(crate) rev_offsets_c1: Vec<u32>,         // prefix-sum offsets, length = child_width + 1
    pub(crate) rev_entries_c2: Vec<RevEntry>,
    pub(crate) rev_offsets_c2: Vec<u32>,

    // ── The two children's product lists read by f index ──
    // A product list is emitted in ascending `left_idx` order by every
    // producer, so its bucket for one f index is a slice of it; these hold
    // the slice bounds (`bucket_offsets`) for the inner and the outer child.
    pub(crate) inner_offsets: Vec<u32>,
    pub(crate) outer_offsets: Vec<u32>,

    // ── Output-sensitive join (`scatter_outsens`) ──
    // Per-outer filtered g index: inner-g-child → [(p2, attached_prod)], rebuilt
    // each outer from the live set + the opposite-keyed g reverse index, so the
    // emit loop iterates only alive entries, with no dead probes.
    //   normal:  filtered[a2] = [(p2, sib_idx)]   swapped: filtered[s2] = [(p2, a_prod)]
    pub(crate) filtered: Vec<Vec<(u32, u32)>>,
    pub(crate) filtered_touched: Vec<u32>,          // indices of `filtered` written this outer, to clear

    // ── Output-sensitive join: the inner-g children the emit will read ──
    // The emit reads `filtered` only at the g children this outer's f parents
    // name through their live left products, so the build buckets only those
    // — a semi-join of the g side against the f side, one level up.
    pub(crate) wanted: Vec<u32>,                    // inner-g children this outer's emit reads
    pub(crate) wanted_epoch: u32,                   // the stamp that counts as marked
    pub(crate) wanted_keys: Vec<u32>,               // the marked inner-g children, in marking order
    pub(crate) inner_seen: Vec<u32>,                // inner f children already walked this outer
    pub(crate) inner_seen_epoch: u32,               // likewise

    // ── Output-sensitive join: the second way to build `filtered` ──
    // g's reverse index keyed by the join's inner-g child, the opposite key
    // from `rev_entries_c2`. An outer whose g keys hold most of the level's g
    // pairs — a g operand free over the outer child has all of them under one
    // key — builds `filtered` from this index instead, walking the parents of
    // the wanted inner-g children and keeping those under one of the outer's
    // keys. `outer_keys` marks those keys for the walk and `outer_attached`
    // holds each one's product.
    pub(crate) rev_entries_c3: Vec<RevEntry>,
    pub(crate) rev_offsets_c3: Vec<u32>,
    pub(crate) outer_keys: Vec<u32>,
    pub(crate) outer_keys_epoch: u32,
    pub(crate) outer_attached: Vec<u32>,

    // ── Phase E: parent dedup ──
    pub(crate) par_buckets: Vec<Vec<ParEntry>>,     // surviving candidates bucketed by f-parent
    // The flat alternative a level with many more parents than candidates
    // takes (`flat_candidates_win`): the scatter appends every candidate
    // with its parent to `par_flat`, and `sort_candidates` counting-sorts
    // them into `par_sorted`, parent `p1`'s run being
    // `par_sorted[par_offsets[p1]..par_offsets[p1 + 1]]`. `flat_candidates`
    // says which representation the current level filled.
    pub(crate) flat_candidates: bool,
    pub(crate) par_flat: Vec<Candidate>,
    pub(crate) par_sorted: Vec<ParEntry>,
    pub(crate) par_offsets: Vec<u32>,
    pub(crate) p2_map: Vec<u32>,                    // flat lookup: p2_map[right_parent] → compacted idx, NO_PRODUCT if new
    pub(crate) p2_map_touched: Vec<u32>,            // p2 values written into p2_map this p1's emit pass, to clear

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // f-by-left, f-by-right, g-by-left, g-by-right.
    pub(crate) est_counts: Vec<u32>,

    // ── Phase F: counting-sort pairs into output nodes ──
    pub(crate) emit_pairs: Vec<(u32, ChildPair)>,   // (parent_prod_idx, pair) for all surviving pairs
    pub(crate) pair_counts: Vec<u32>,               // per-parent pair count, then prefix-sum offsets
    pub(crate) sorted_pairs: Vec<ChildPair>,        // output buffer for counting sort

    /// True when some level of an operand (or of the output built so far) is
    /// marginal, which makes a node's pair list a legal *multiset* rather than a
    /// set (see `content_twin.rs`). Set on entry to
    /// `apply_sparse_level`; read only by the debug-only duplicate-pair check in
    /// Phase F, and always `false` in release (the scan is `cfg!`-gated so it
    /// compiles out).
    pub(crate) duplicates_legal: bool,
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
        crate::limits::pool::release_if_oversized(lim, &mut self.par_sorted);
        crate::limits::pool::release_if_oversized(lim, &mut self.par_offsets);
        drop_if_large(lim, &mut self.filtered);
        crate::limits::pool::release_if_oversized(lim, &mut self.wanted);
        crate::limits::pool::release_if_oversized(lim, &mut self.wanted_keys);
        crate::limits::pool::release_if_oversized(lim, &mut self.inner_seen);
        crate::limits::pool::release_if_oversized(lim, &mut self.rev_entries_c3);
        crate::limits::pool::release_if_oversized(lim, &mut self.rev_offsets_c3);
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
pub(crate) fn drop_if_large<E>(lim: &Limits, v: &mut Vec<Vec<E>>) {
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
pub(crate) struct RevEntry {
    pub(crate) parent: u32,
    pub(crate) other: u32,
}

/// Build a reverse index from a level's pairs, keyed by one child side:
///   `BY_RIGHT = false`: left_child_idx  → [(parent_idx, right_sibling_idx)]
///   `BY_RIGHT = true` : right_sibling_idx → [(parent_idx, left_child_idx)]
/// stored counting-sort style as a flat `entries` buffer plus prefix-sum `offsets`.
///
/// After this call: `entries[offsets[key] .. offsets[key + 1]]` is the slice of
/// `(parent_idx, other_side_idx)` pairs for each key-side child index.
///
/// `BY_RIGHT` is a const generic so the `if BY_RIGHT` branches fold away.
/// Four-pass counting sort:
///   1. Count: `offsets[key] = number of pairs with that key-side child` —
///      or copy `counts`, the same numbers when the direction estimate has
///      already counted this level's pairs by this child
///   2. Exclusive prefix sum: `offsets[i]` becomes the start-of-bucket for `i`
///   3. Fill: scatter `(parent_idx, other_side)` using `offsets` as write cursors,
///      leaving each `offsets[i]` one-past-the-end of bucket `i`
///   4. Restore: shift right by one so `offsets[i]` is back at start-of-bucket
pub(crate) fn build_reverse_index<const BY_RIGHT: bool>(
    eng: &Engine,
    level: &TddLevel,
    key_width: usize,
    counts: Option<&[u32]>,
    offsets: &mut Vec<u32>,
    entries: &mut Vec<RevEntry>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    // Pass 1: count
    lim.try_resize(offsets, key_width + 1, 0)?;
    offsets[key_width] = 0;
    if let Some(counts) = counts {
        offsets[..key_width].copy_from_slice(counts);
    } else {
        offsets[..key_width].fill(0);
        // Unpacked slice iterator (vectorizable).
        for node in level.nodes.iter() {
            if !node.is_internal() { continue; }
            for pair in level.pairs_of(node) {
                let key = if BY_RIGHT { pair.right.0 } else { pair.left.0 } as usize;
                offsets[key] += 1;
            }
        }
    }
    // Pass 2: exclusive prefix sum
    let mut total = 0u32;
    for slot in offsets.iter_mut().take(key_width) {
        let count = *slot;
        *slot = total;
        total += count;
    }
    offsets[key_width] = total;
    // Pass 3: fill, bumping offsets[key] as a write cursor
    lim.try_resize(entries, total as usize, RevEntry { parent: 0, other: 0 })?;
    for (parent_idx, node) in level.nodes.iter().enumerate() {
        if !node.is_internal() { continue; }
        for pair in level.pairs_of(node) {
            let key = if BY_RIGHT { pair.right.0 } else { pair.left.0 } as usize;
            let other = if BY_RIGHT { pair.left.0 } else { pair.right.0 };
            let slot = offsets[key] as usize;
            entries[slot] = RevEntry { parent: parent_idx as u32, other };
            offsets[key] += 1;
        }
    }
    // Pass 4: shift right by one so offsets[i] is back at start-of-bucket i
    shift_offsets_right_by_one(&mut offsets[..=key_width]);
    Ok(())
}

/// After a counting-sort fill pass leaves each `offsets[i]` pointing one-past
/// the end of bucket `i`, shift the slice right by one so every `offsets[i]`
/// is restored to the start of its bucket (and `offsets[0] = 0`).
#[inline]
pub(crate) fn shift_offsets_right_by_one(offsets: &mut [u32]) {
    let mut prev = 0u32;
    for slot in offsets.iter_mut() {
        std::mem::swap(&mut *slot, &mut prev);
    }
}

/// Ensure `buckets` has ≥ `n` inner Vecs (growing via `resize_with`), then clear
/// the first `n`. Buckets that already existed keep their reserved capacity —
/// this is how the sparse workspace amortizes allocations across calls.
pub(crate) fn ensure_buckets_cleared<T>(eng: &Engine, buckets: &mut Vec<Vec<T>>, n: usize) -> Result<(), OperationError> {
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
            capacity_bytes(&self.rev_entries_c1),
            capacity_bytes(&self.rev_offsets_c1),
            capacity_bytes(&self.rev_entries_c2),
            capacity_bytes(&self.rev_offsets_c2),
            capacity_bytes(&self.inner_offsets),
            capacity_bytes(&self.outer_offsets),
            nested_bytes(&self.filtered),
            capacity_bytes(&self.filtered_touched),
            capacity_bytes(&self.wanted),
            capacity_bytes(&self.wanted_keys),
            capacity_bytes(&self.inner_seen),
            capacity_bytes(&self.rev_entries_c3),
            capacity_bytes(&self.rev_offsets_c3),
            capacity_bytes(&self.outer_keys),
            capacity_bytes(&self.outer_attached),
            nested_bytes(&self.par_buckets),
            capacity_bytes(&self.par_flat),
            capacity_bytes(&self.par_sorted),
            capacity_bytes(&self.par_offsets),
            capacity_bytes(&self.p2_map),
            capacity_bytes(&self.p2_map_touched),
            capacity_bytes(&self.est_counts),
            capacity_bytes(&self.emit_pairs),
            capacity_bytes(&self.pair_counts),
            capacity_bytes(&self.sorted_pairs),
        ].into_iter().sum()
    }

    fn prepare(&mut self) {}
    fn retain(&mut self, lim: &Limits) { self.release_if_large(lim); }
}
