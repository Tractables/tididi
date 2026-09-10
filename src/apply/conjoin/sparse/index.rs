//! The reusable sparse workspace: reverse indices, buckets and node-index newtypes.

use super::*;
use crate::engine::pool::SCRATCH_RETAIN_BYTES;

/// Candidate that survived the sibling liveness filter, grouped by f-parent.
#[derive(Clone, Copy)]
pub(crate) struct ParEntry {
    pub(crate) p2: u32,      // g parent index
    pub(crate) a_prod: u32,  // compacted left-child product index
    pub(crate) sib_idx: u32, // compacted right-child product index
}

/// Index of a node in `f.levels[t].nodes`. Distinct from `RightNodeIdx` and
/// `ProductNodeIdx` so that construction-site swaps are caught at compile time.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct LeftNodeIdx(pub(crate) u32);

impl LeftNodeIdx {
    #[inline(always)]
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in `g.levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct RightNodeIdx(pub(crate) u32);

impl RightNodeIdx {
    #[inline(always)]
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in the output `levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct ProductNodeIdx(pub(crate) u32);

impl ProductNodeIdx {
    // Only referenced from a `debug_assert_eq!` below, so it is unused in
    // release builds — silence the dead-code lint there rather than dropping it.
    #[inline(always)]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// A live product node: the conjunction f[left_idx] ∧ g[right_idx] produced
/// the output node at `prod_idx` in the output level.
#[derive(Clone, Copy)]
pub(crate) struct ProductEntry {
    pub(crate) left_idx: LeftNodeIdx,
    pub(crate) right_idx: RightNodeIdx,
    pub(crate) prod_idx: ProductNodeIdx,
}

/// Reusable workspace for sparse product construction.
///
/// Thread-local via `RefCell` (apply_and is never re-entrant). All Vecs grow
/// monotonically and are never shrunk — capacity is retained across calls to
/// amortize allocation cost. Cleared/resized at the start of each use.
///
/// Exception: after a large call whose bucket arrays exceed
/// [`SCRATCH_RETAIN_BYTES`] of retained capacity, they are dropped. This caps
/// the memory retained from rare large calls without hurting performance on
/// typical calls.
#[derive(Default)]
pub(crate) struct SparseWorkspace {
    // ── Phase A: reverse indices (child → parent) for scatter ──
    pub(crate) rev_entries_c1: Vec<RevEntry>,
    pub(crate) rev_offsets_c1: Vec<u32>,         // prefix-sum offsets, length = child_width + 1
    pub(crate) rev_entries_c2: Vec<RevEntry>,
    pub(crate) rev_offsets_c2: Vec<u32>,

    // ── Fused scatter-filter: product lookup ──
    pub(crate) prod_by_a1: Vec<Vec<(u32, u32)>>,    // a1 → [(a2, a_prod)] from alive left products
    pub(crate) prod_by_s1: Vec<Vec<(u32, u32)>>,    // s1 → [(s2, sib_prod)] from alive right products (swapped dir)

    // ── Phase C: sibling/child liveness filter ──
    pub(crate) right_buckets: Vec<Vec<(u32, u32)>>, // right products bucketed by f-index: (right_idx, prod_idx)
    pub(crate) left_buckets: Vec<Vec<(u32, u32)>>,  // left products bucketed by f-index (swapped direction)

    // ── Output-sensitive join (`scatter_outsens`) ──
    // Per-outer filtered g index: inner-g-child → [(p2, attached_prod)], rebuilt
    // each outer from the live set + the opposite-keyed g reverse index, so the
    // emit loop iterates only alive entries, with no dead probes.
    //   normal:  filtered[a2] = [(p2, sib_idx)]   swapped: filtered[s2] = [(p2, a_prod)]
    pub(crate) filtered: Vec<Vec<(u32, u32)>>,
    pub(crate) filtered_touched: Vec<u32>,          // indices of `filtered` written this outer, to clear

    // ── Phase E: parent dedup ──
    pub(crate) par_buckets: Vec<Vec<ParEntry>>,     // surviving candidates bucketed by f-parent
    pub(crate) p2_map: Vec<u32>,                    // flat lookup: p2_map[right_parent] → compacted idx, NO_PRODUCT if new
    pub(crate) p2_map_touched: Vec<u32>,            // p2 values written into p2_map this p1's emit pass, to clear

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // f-by-left, f-by-right, g-by-left, g-by-right.
    pub(crate) est_counts: Vec<u32>,

    // ── Phase F: counting-sort pairs into output nodes ──
    pub(crate) emit_pairs: Vec<(u32, InputPair)>,   // (parent_prod_idx, pair) for all surviving pairs
    pub(crate) pair_counts: Vec<u32>,               // per-parent pair count, then prefix-sum offsets
    pub(crate) sorted_pairs: Vec<InputPair>,        // output buffer for counting sort

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
    fn release_if_large(&mut self) {
        drop_if_large(&mut self.prod_by_a1);
        drop_if_large(&mut self.prod_by_s1);
        drop_if_large(&mut self.right_buckets);
        drop_if_large(&mut self.left_buckets);
        drop_if_large(&mut self.par_buckets);
        drop_if_large(&mut self.filtered);
    }
}

/// Drop and replace `v` with an empty Vec if its retained capacity — outer spine
/// (`capacity·size_of::<Vec<E>>`) plus Σ inner `capacity·size_of::<E>` — exceeds
/// [`SCRATCH_RETAIN_BYTES`], the same rule
/// [`release_if_oversized`](crate::engine::pool::release_if_oversized) applies to the flat buffers. Frees both the inner elements and the outer
/// allocation. Early-exits the summation as soon as the threshold is crossed, so
/// the common under-cap case pays at most one pass and the over-cap case stops
/// early. `size_of::<E>()` is a compile-time constant.
#[inline]
pub(crate) fn drop_if_large<E>(v: &mut Vec<Vec<E>>) {
    let elem = std::mem::size_of::<E>();
    let mut bytes = v.capacity().saturating_mul(std::mem::size_of::<Vec<E>>());
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
        *v = Vec::new();
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
/// The const generic selects the key side at monomorphization, so each
/// instantiation (`::<false>` / `::<true>`) is codegen-identical to a
/// hand-written keyed variant — the `if BY_RIGHT` branches fold away. Classic
/// four-pass counting sort:
///   1. Count: `offsets[key] = number of pairs with that key-side child`
///   2. Exclusive prefix sum: `offsets[i]` becomes the start-of-bucket for `i`
///   3. Fill: scatter `(parent_idx, other_side)` using `offsets` as write cursors,
///      leaving each `offsets[i]` one-past-the-end of bucket `i`
///   4. Restore: shift right by one so `offsets[i]` is back at start-of-bucket
pub(crate) fn build_reverse_index<const BY_RIGHT: bool>(
    eng: &Engine,
    level: &TddLevel,
    key_width: usize,
    offsets: &mut Vec<u32>,
    entries: &mut Vec<RevEntry>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    // Pass 1: count
    lim.try_resize(offsets, key_width + 1, 0)?;
    offsets[..key_width + 1].fill(0);
    // Unpacked slice iterator (vectorizable).
    for node in level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in level.pairs_of(node) {
            let key = if BY_RIGHT { pair.right.0 } else { pair.left.0 } as usize;
            offsets[key] += 1;
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
pub(crate) fn ensure_buckets_cleared<T>(eng: &Engine, buckets: &mut Vec<Vec<T>>, n: usize) -> Result<(), ApplyError> {
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

/// Release the engine's sparse workspace bucket memory if it grew too large.
///
/// Called from `apply_and_fallible` after each sparse level to cap retained peak.
pub(crate) fn release_sparse_ws_if_large(eng: &Engine) {
    eng.sparse().borrow_mut().release_if_large();
}

/// Fully drop the engine's sparse workspace, replacing it with a fresh
/// `SparseWorkspace::default()` — every bucket array, reverse index, and emit
/// buffer released to the allocator. Unlike `release_sparse_ws_if_large` (the
/// conditional per-array trim on the normal apply exit), this frees all retained
/// capacity unconditionally.
///
/// Safe only at an inter-compile boundary — no apply in flight on this engine.
/// The panic that unwinds a failed sub-compile drops the workspace's `RefCell`
/// borrow guard, but the workspace itself is owned by the engine, so its
/// bucket arrays survive the unwind at full capacity. This reset is the
/// reclaim for that pin; calling it while `apply_sparse_level` holds the borrow
/// would panic on the double borrow.
pub(crate) fn reset_sparse_ws(eng: &Engine) {
    *eng.sparse().borrow_mut() = SparseWorkspace::default();
}

#[cfg(test)]
#[path = "../sparse_scatter_direction_pool_tests.rs"]
mod scatter_direction_pool_tests;

#[cfg(test)]
#[path = "../sparse_retention_tests.rs"]
mod retention_tests;

#[cfg(test)]
#[path = "../sparse_a4_self_conjunction_tests.rs"]
mod a4_self_conjunction_tests;

#[cfg(test)]
#[path = "../sparse_reset_ws_tests.rs"]
mod reset_ws_tests;

#[cfg(test)]
#[path = "../sparse_regression_tests.rs"]
mod regression_tests;
