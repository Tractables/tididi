//! Sparse product construction for the apply algorithm.
//!
//! For levels where k1 * k2 > SPARSE_THRESHOLD, the dense grid iteration is
//! replaced by a scatter-filter-dedup pipeline. This module also contains the
//! leaf-level processing, identity product lists, and output index computation.

use std::cell::{Cell, RefCell};

use smallvec::SmallVec;

use crate::vtree::VtreeIdx;
use super::{ApplyError, DEAD, Tdd, TddLevel, InputPair, ZERO,
    LocalNodeIdx, LevelGrid,
    try_push, try_resize, try_resize_dead, budget_reserve_exact, budget_reserve,
    CONJOIN_GRID, bump_live_count,
};

#[derive(Clone, Copy)]
pub(super) struct SparseConfig {
    pub(super) min_grid: usize,
    pub(super) sparsity_factor: u128,
}

/// Default sparse-path config. Production uses these fixed values.
const SPARSE_CONFIG_DEFAULT: SparseConfig = SparseConfig { min_grid: 4096, sparsity_factor: 64 };

thread_local! {
    /// Test-only override for `sparse_config`. None = use the fixed default.
    /// Scoped by `with_sparse_config`. Per-thread so parallel tests don't race.
    static SPARSE_CONFIG_OVERRIDE: Cell<Option<SparseConfig>> = const { Cell::new(None) };
}

pub(super) fn sparse_config() -> SparseConfig {
    if let Some(cfg) = SPARSE_CONFIG_OVERRIDE.with(|c| c.get()) {
        return cfg;
    }
    SPARSE_CONFIG_DEFAULT
}
/// Estimate which scatter direction (normal vs swapped) does fewer inner probes,
/// for the general (both-non-leaf) path. The probe count factorizes per pair:
///   normal = Σ_{(a1,a2)∈pl_left}  cnt_C1_left[a1]·deg_C2_left[a2]
///   swap   = Σ_{(s1,s2)∈pl_right} cnt_C1_right[s1]·deg_C2_right[s2]
/// where cnt_C1_* counts c1 PAIRS by left/right child and deg_C2_* counts c2 PAIRS
/// by left/right child. Cost is O(|c1 pairs|+|c2 pairs|+|pl_left|+|pl_right|) — tiny
/// next to the billions of probes the choice governs. Returns `true` when swapping
/// is cheaper, i.e. `est_swap < est_normal`. The grid-size heuristic (which this
/// replaces) ignores selectivity, mispicking on wide×wide segment conjoins.
///
/// The four counter arrays are carved out of `SparseWorkspace::est_counts` — one
/// pooled buffer, grown once and re-zeroed per level — rather than four fresh
/// `vec![0u32; k]`s. The estimator runs on the widest levels in the compile, so
/// those four allocations landed exactly where headroom is tightest; sizing them
/// through `try_resize` also makes the estimator's own memory OverBudget-catchable
/// instead of an abort.
fn estimate_scatter_direction(
    est_counts: &mut Vec<u32>,
    c1_level: &TddLevel,
    c2_level: &TddLevel,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    k1_left: usize, k2_left: usize,
    k1_right: usize, k2_right: usize,
) -> Result<bool, ApplyError> {
    let total = k1_left + k1_right + k2_left + k2_right;
    try_resize(est_counts, total, 0u32)?;
    // The buffer is pooled and grow-only, so the prefix in use must be re-zeroed
    // per level — a wider level's residue would otherwise be counted again here.
    let buf = &mut est_counts[..total];
    buf.fill(0);
    // One buffer, four back-to-back index spaces (c1-by-left, c1-by-right,
    // c2-by-left, c2-by-right) — the counting loops below need two of them live
    // at once, so they must be disjoint slices.
    let (cnt_c1_left, rest) = buf.split_at_mut(k1_left);
    let (cnt_c1_right, rest) = rest.split_at_mut(k1_right);
    let (deg_c2_left, deg_c2_right) = rest.split_at_mut(k2_left);
    for node in c1_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in c1_level.pairs_of(node) {
            cnt_c1_left[pair.left.0 as usize] += 1;
            cnt_c1_right[pair.right.0 as usize] += 1;
        }
    }
    for node in c2_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in c2_level.pairs_of(node) {
            deg_c2_left[pair.left.0 as usize] += 1;
            deg_c2_right[pair.right.0 as usize] += 1;
        }
    }
    let mut est_normal: u128 = 0;
    for e in pl_left {
        est_normal += cnt_c1_left[e.c1_idx.0 as usize] as u128
            * deg_c2_left[e.c2_idx.0 as usize] as u128;
    }
    let mut est_swap: u128 = 0;
    for e in pl_right {
        est_swap += cnt_c1_right[e.c1_idx.0 as usize] as u128
            * deg_c2_right[e.c2_idx.0 as usize] as u128;
    }
    Ok(est_swap < est_normal)
}

/// Test helper: run `f` with `sparse_config()` returning the given values on
/// this thread. The previous override is restored on exit. Production code
/// should not use this.

#[cfg(test)]
pub(crate) fn with_sparse_config<F: FnOnce() -> R, R>(min_grid: usize, sparsity_factor: u128, f: F) -> R {
    let cfg = SparseConfig { min_grid, sparsity_factor };
    let prev = SPARSE_CONFIG_OVERRIDE.with(|c| c.replace(Some(cfg)));
    let result = f();
    SPARSE_CONFIG_OVERRIDE.with(|c| c.set(prev));
    result
}

/// Soft byte budget for the sparse Phase E+F transient buffers
/// (`emit_pairs` + `sorted_pairs` + consumed `par_buckets` rows).
/// When `Σ par_buckets[p].len() * BYTES_PER_PAR_ENTRY` exceeds the budget,
/// Phase E+F is emitted in chunks of c1-parent ranges, dropping each chunk's
/// `par_buckets` allocations before the next chunk's `emit_pairs` grows.
///
/// Default = 256 MiB. Levels whose total projected transient fits in one
/// chunk (typical MCC instances) produce `boundaries = [0, k1]` from
/// `plan_e_f_chunks` and run a single `flush_chunk` with `drop_consumed=false`
/// — preserving the cross-apply `par_buckets` capacity reuse. Wide levels
/// (canary OOMs) get split into multiple chunks with `drop_consumed=true`,
/// capping within-call peak. Fixed at 256 MiB — not tunable at runtime.
const SPARSE_CHUNK_BYTES_DEFAULT: usize = 256 * 1024 * 1024;

thread_local! {
    /// Test-only override for `sparse_chunk_bytes`. None = use the fixed default.
    /// Scoped by `with_sparse_chunk_bytes`. Per-thread so parallel tests don't race.
    static SPARSE_CHUNK_BYTES_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn sparse_chunk_bytes() -> usize {
    if let Some(v) = SPARSE_CHUNK_BYTES_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    SPARSE_CHUNK_BYTES_DEFAULT
}

/// Test helper: run `f` with `sparse_chunk_bytes()` returning `v` on this thread.
/// The previous value is restored on exit. Production code should not use this.

#[cfg(test)]
pub(crate) fn with_sparse_chunk_bytes<F: FnOnce() -> R, R>(v: usize, f: F) -> R {
    let prev = SPARSE_CHUNK_BYTES_OVERRIDE.with(|c| c.replace(Some(v)));
    let result = f();
    SPARSE_CHUNK_BYTES_OVERRIDE.with(|c| c.set(prev));
    result
}

/// Projected transient cost per surviving `ParEntry`:
///   sizeof(ParEntry)             = 12   (Phase C/E input)
/// + sizeof((u32, InputPair))     = 12   (Phase E output → emit_pairs)
/// + sizeof(InputPair)            = 8    (Phase F output → sorted_pairs)
/// Used by `plan_e_f_chunks` to size chunks under the byte budget.
pub(super) const BYTES_PER_PAR_ENTRY: usize = 32;

// ── Sparse product construction ──────────────────────────────────────────────
//
// For levels where k1 * k2 > SPARSE_THRESHOLD, the dense grid iteration is
// replaced by a scatter-filter-dedup pipeline inspired by the upward branch.
// Instead of iterating all (i, j) cells, we:
//   1. Build reverse indices: child_idx → [(parent_idx, sibling_idx)]
//   2. Scatter from live child products upward to candidate parents
//   3. Filter candidates by sibling liveness (lazy-cleared flat lookup)
//   4. Dedup parent products (lazy-cleared flat p2_map)
//   5. Emit output pairs and nodes
//
// This is O(n * degree²) where n = live products, vs O(k1 * k2) for dense.

/// Candidate that survived the sibling liveness filter, grouped by c1-parent.
#[derive(Clone, Copy)]
struct ParEntry {
    p2: u32,      // c2 parent index
    a_prod: u32,  // compacted left-child product index
    sib_idx: u32, // compacted right-child product index (from sib_lookup)
}

/// Index of a node in `c1.levels[t].nodes`. Distinct from `C2NodeIdx` and
/// `ProdNodeIdx` so that construction-site swaps are caught at compile time.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) struct C1NodeIdx(pub(super) u32);

impl C1NodeIdx {
    #[inline(always)]
    pub(super) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in `c2.levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) struct C2NodeIdx(pub(super) u32);

impl C2NodeIdx {
    #[inline(always)]
    pub(super) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in the output `levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) struct ProdNodeIdx(pub(super) u32);

impl ProdNodeIdx {
    // Only referenced from a `debug_assert_eq!` below, so it is unused in
    // release builds — silence the dead-code lint there rather than dropping it.
    #[inline(always)]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(super) fn idx(self) -> usize { self.0 as usize }
}

/// A live product node: the conjunction c1[c1_idx] ∧ c2[c2_idx] produced
/// the output node at `prod_idx` in the output level.
#[derive(Clone, Copy)]
pub(super) struct ProductEntry {
    pub(super) c1_idx: C1NodeIdx,
    pub(super) c2_idx: C2NodeIdx,
    pub(super) prod_idx: ProdNodeIdx,
}

/// Reusable workspace for sparse product construction.
///
/// Thread-local via `RefCell` (apply_and is never re-entrant). All Vecs grow
/// monotonically and are never shrunk — capacity is retained across calls to
/// amortize allocation cost. Cleared/resized at the start of each use.
///
/// Exception: after a large call whose bucket arrays exceed
/// `SPARSE_BUCKET_BYTE_LIMIT` of retained capacity, they are dropped. This caps
/// the memory retained from rare large calls without hurting performance on
/// typical calls.
#[derive(Default)]
struct SparseWorkspace {
    // ── Phase A: reverse indices (child → parent) for scatter ──
    rev_entries_c1: Vec<(u32, u32)>,  // (parent_idx, sibling_idx)
    rev_offsets_c1: Vec<u32>,         // prefix-sum offsets, length = child_width + 1
    rev_entries_c2: Vec<(u32, u32)>,
    rev_offsets_c2: Vec<u32>,

    // ── Fused scatter-filter: product lookup ──
    prod_by_a1: Vec<Vec<(u32, u32)>>,    // a1 → [(a2, a_prod)] from alive left products
    prod_by_s1: Vec<Vec<(u32, u32)>>,    // s1 → [(s2, sib_prod)] from alive right products (swapped dir)

    // ── Phase C: sibling/child liveness filter ──
    right_buckets: Vec<Vec<(u32, u32)>>, // right products bucketed by c1-index: (c2_idx, prod_idx)
    left_buckets: Vec<Vec<(u32, u32)>>,  // left products bucketed by c1-index (swapped direction)
    sib_lookup: Vec<u32>,                // flat lookup: sib_lookup[c2_sibling] → prod_idx, DEAD if absent
    child_lookup: Vec<u32>,              // flat lookup: child_lookup[c2_left] → prod_idx (swapped dir)

    // ── Output-sensitive join (`scatter_outsens`) ──
    // Per-outer filtered c2 index: inner-c2-child → [(p2, attached_prod)], rebuilt
    // each outer from the live set + the opposite-keyed c2 reverse index, so the
    // emit loop iterates ONLY alive entries (no dead `sib_lookup` probes).
    //   normal:  filtered[a2] = [(p2, sib_idx)]   swapped: filtered[s2] = [(p2, a_prod)]
    filtered: Vec<Vec<(u32, u32)>>,
    filtered_touched: Vec<u32>,          // indices of `filtered` written this outer, to clear

    // ── Phase E: parent dedup ──
    par_buckets: Vec<Vec<ParEntry>>,     // surviving candidates bucketed by c1-parent
    p2_map: Vec<u32>,                    // flat lookup: p2_map[c2_parent] → compacted idx, DEAD if new
    p2_map_touched: Vec<u32>,            // p2 values written into p2_map this p1's emit pass, to clear

    // ── Scatter-direction estimator ──
    // Per-child-index pair counters for the four (operand × side) index spaces
    // `estimate_scatter_direction` sums over, packed back-to-back in one buffer:
    // c1-by-left, c1-by-right, c2-by-left, c2-by-right.
    est_counts: Vec<u32>,

    // ── Phase F: counting-sort pairs into output nodes ──
    emit_pairs: Vec<(u32, InputPair)>,   // (parent_prod_idx, pair) for all surviving pairs
    pair_counts: Vec<u32>,               // per-parent pair count, then prefix-sum offsets
    sorted_pairs: Vec<InputPair>,        // output buffer for counting sort

    /// Set true on entry to `apply_sparse_level`, cleared on successful exit.
    /// If true at next entry, the previous call bailed mid-iteration
    /// (`OverBudget` from try_push/try_resize) and the lazy-cleared lookup
    /// tables (`sib_lookup`, `child_lookup`, `p2_map`) may hold stale
    /// non-DEAD entries that the scatter-clean cleanup never restored. When
    /// dirty, the next call must full-fill these tables with DEAD before use
    /// — `try_resize` alone is a no-op on entries already in range.
    /// Observed as the td-fc-pri × --mc undercount on mc2022_track1_048
    /// (bug entry 2026-05-25).
    dirty: bool,

    /// True when some level of an operand (or of the output built so far) is
    /// marginal, which makes a node's pair list a legal *multiset* rather than a
    /// set (maintainer ruling 2026-07-27; see `content_twin.rs`). Set on entry to
    /// `apply_sparse_level`; read only by the debug-only duplicate-pair check in
    /// Phase F, and always `false` in release (the scan is `cfg!`-gated so it
    /// compiles out).
    dups_legal: bool,
}

/// Byte cap on the *retained* capacity of a single bucket array. A bucket array
/// whose footprint — outer spine + Σ inner capacities — exceeds this is dropped
/// after the level so a rare fat level doesn't park its peak in the thread-local
/// for the rest of the compile. Mirrors the flat-arena policy `pool_put_bounded`
/// uses on `SCRATCH_*` (same 32 MiB `MAX_LEVEL_ARENA_BYTES`). The old
/// outer-*length* trigger missed few-but-fat-row levels: a bucket array with a
/// handful of outer rows, each holding a product-list-sized inner Vec (the
/// `prod_by_*` / bucket rows are NOT bounded by the chunker), stayed under the
/// length cap while parking large memory.
const SPARSE_BUCKET_BYTE_LIMIT: usize = crate::tdd::types::MAX_LEVEL_ARENA_BYTES;

impl SparseWorkspace {
    /// Release inner Vec memory from bucket arrays whose retained capacity grew
    /// past `SPARSE_BUCKET_BYTE_LIMIT`. Called after a large sparse level to
    /// avoid retaining peak allocations. Covers every `Vec<Vec<_>>` bucket array
    /// — including `prod_by_a1`/`prod_by_s1`, whose product-list-sized rows the
    /// length-based predecessor never released.
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
/// `SPARSE_BUCKET_BYTE_LIMIT`. Frees both the inner elements and the outer
/// allocation. Early-exits the summation as soon as the threshold is crossed, so
/// the common under-cap case pays at most one pass and the over-cap case stops
/// early. `size_of::<E>()` is a compile-time constant.
#[inline]
fn drop_if_large<E>(v: &mut Vec<Vec<E>>) {
    let elem = std::mem::size_of::<E>();
    let mut bytes = v.capacity().saturating_mul(std::mem::size_of::<Vec<E>>());
    let mut over = bytes > SPARSE_BUCKET_BYTE_LIMIT;
    if !over {
        for inner in v.iter() {
            bytes = bytes.saturating_add(inner.capacity().saturating_mul(elem));
            if bytes > SPARSE_BUCKET_BYTE_LIMIT {
                over = true;
                break;
            }
        }
    }
    if over {
        *v = Vec::new();
    }
}

thread_local! {
    static SPARSE_WS: RefCell<SparseWorkspace> = RefCell::new(SparseWorkspace::default());
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
fn build_reverse_index<const BY_RIGHT: bool>(
    level: &TddLevel,
    key_width: usize,
    offsets: &mut Vec<u32>,
    entries: &mut Vec<(u32, u32)>,
) -> Result<(), ApplyError> {
    // Pass 1: count
    try_resize(offsets, key_width + 1, 0)?;
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
    for i in 0..key_width {
        let count = offsets[i];
        offsets[i] = total;
        total += count;
    }
    offsets[key_width] = total;
    // Pass 3: fill, bumping offsets[key] as a write cursor
    try_resize(entries, total as usize, (0, 0))?;
    for (parent_idx, node) in level.nodes.iter().enumerate() {
        if !node.is_internal() { continue; }
        for pair in level.pairs_of(node) {
            let key = if BY_RIGHT { pair.right.0 } else { pair.left.0 } as usize;
            let other = if BY_RIGHT { pair.left.0 } else { pair.right.0 };
            let slot = offsets[key] as usize;
            entries[slot] = (parent_idx as u32, other);
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
fn shift_offsets_right_by_one(offsets: &mut [u32]) {
    let mut prev = 0u32;
    for slot in offsets.iter_mut() {
        let cur = *slot;
        *slot = prev;
        prev = cur;
    }
}

/// Ensure `buckets` has ≥ `n` inner Vecs (growing via `resize_with`), then clear
/// the first `n`. Buckets that already existed keep their reserved capacity —
/// this is how the sparse workspace amortizes allocations across calls.
fn ensure_buckets_cleared<T>(buckets: &mut Vec<Vec<T>>, n: usize) -> Result<(), ApplyError> {
    if buckets.len() < n {
        let additional = n - buckets.len();
        budget_reserve_exact(buckets, additional)?;
        buckets.resize_with(n, Vec::new);
    }
    for b in &mut buckets[..n] {
        b.clear();
    }
    Ok(())
}

/// True when `c1` and `c2` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, ext)` on every level is
/// sufficient — but ONLY when no level is marginal (a marginal level hides its
/// content outside `nodes`/`pairs`, so the structural test can't see it; see A4).
pub(super) fn is_self_conjunction(c1: &Tdd, c2: &Tdd) -> bool {
    // The shortcut lets `c1 ∧ c2` return `c1.clone()` when the operands are the
    // same function. It is a pure perf optimization, never needed for
    // correctness. A marginal level clears `nodes`/`pairs` (integer-marginal) or
    // `pairs` (weight-marginal) and moves its real content into
    // `marginal_counts`/the external weight store — which this structural test
    // does NOT compare. Two operands agreeing on every explicit level but
    // differing in marginal mass (or holding a marginal×marginal unsound
    // schedule the callers debug-assert against) would compare equal and
    // silently drop one side's content. Bail whenever either operand carries any
    // marginal level (A4).
    if c1.levels.iter().any(|l| l.is_marginal()) || c2.levels.iter().any(|l| l.is_marginal()) {
        return false;
    }
    c1.output == c2.output
        && c1.levels.iter().zip(c2.levels.iter()).all(|(l1, l2)| {
            // `ext` too: equal nodes+pairs with a differently-arranged `ext` table
            // is a different function (A4).
            l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ext == l2.ext
        })
}

/// Fill `pl` with the identity product mapping for a level where one TDD operand
/// is constant-true. Returns `true` if the level was identity (c2-identity or
/// c1-identity), `false` otherwise. On the identity path also sets
/// `*has_pl_slot = true` itself (co-located with the fill); on the non-identity
/// path `has_pl_slot` is left untouched for the caller to set once it fills `pl`
/// some other way.
///
/// Identity means x ∧ 1 = x — the constant-true operand contributes a single
/// fixed index. The One label is at local index 0 on every level (leaf and
/// internal alike, since `LeafLabel::One = 0` and identity levels are width-1).
pub(super) fn fill_identity_product_list(
    k1: usize,
    k2: usize,
    c2_id: bool,
    c1_id: bool,
    pl: &mut Vec<ProductEntry>,
    has_pl_slot: &mut bool,
) -> Result<bool, ApplyError> {
    // The constant-true operand's One node is at index 0 regardless of leaf-ness
    // (`ONE_LEAF_IDX.0 == LeafLabel::One as u32 == 0`), so no leaf/internal split.
    const ID_IDX: u32 = 0;
    if c2_id {
        budget_reserve(pl, k1)?;
        for i in 0..k1 as u32 {
            // x ∧ 1 = x: output index equals c1 index (identity mapping).
            pl.push(ProductEntry { c1_idx: C1NodeIdx(i), c2_idx: C2NodeIdx(ID_IDX), prod_idx: ProdNodeIdx(i) });
        }
        *has_pl_slot = true;
        Ok(true)
    } else if c1_id {
        budget_reserve(pl, k2)?;
        for j in 0..k2 as u32 {
            // 1 ∧ x = x: output index equals c2 index (identity mapping).
            pl.push(ProductEntry { c1_idx: C1NodeIdx(ID_IDX), c2_idx: C2NodeIdx(j), prod_idx: ProdNodeIdx(j) });
        }
        *has_pl_slot = true;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Output-sensitive scatter: THE scatter engine — the four-way join
/// of c1/c2 parent and child/sibling product lists. `SWAPPED = false` outer-loops
/// by right sibling s1; `SWAPPED = true` by left child a1 (every difference is a
/// pure left↔right role rename; the `if SWAPPED` branches fold at compile time).
/// A per-outer FILTERED c2 index makes the emit walk only alive `(p2, prod)`
/// entries. The emitted ParEntry *set* into `par_buckets` is order-free — sound
/// because pair lists are order-independent.
///
/// Two arms behind a shared front-end (the two reverse-index builds):
///
/// **Leaf arm** (`leaf_side_is_leaf`): iterate the non-leaf product list,
/// `CONJOIN_GRID` computes the leaf-side product. Do NOT rewrite this arm into
/// the filtered-index form: on a 3-label alphabet the grid has at most 2/9 dead
/// entries, so output-sensitivity buys nothing there, and the
/// per-product-entry loop is already selective (c2 grouped by the
/// non-leaf child). The rev_c2 keying below is identical to what the leaf
/// arm needs (normal → by right, swapped → by left; entries carry the
/// leaf-side child = leaf label), so the front-end is shared unchanged.
///
/// **General arm** (both sides non-leaf), per outer key:
///   1. Build `filtered`: for each live `(inner_live, attached)` in the outer's
///      liveness bucket, walk the opposite-keyed c2 index and bucket its parents
///      by the join's inner-c2 child, attaching the live product.
///   2. Emit: for each c1-parent sharing the outer, for each alive inner product,
///      push the precomputed alive `(p2, prod)` entries — zero dead probes.
///   3. Clear only the `filtered` buckets touched this outer.
fn scatter_outsens<const SWAPPED: bool>(
    ws: &mut SparseWorkspace,
    c1_level: &TddLevel,
    c2_level: &TddLevel,
    k1_left: usize, k2_left: usize,
    k1_right: usize, k2_right: usize,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    // `left_is_leaf` when `!SWAPPED`; `right_is_leaf` when `SWAPPED`.
    leaf_side_is_leaf: bool,
) -> Result<(), ApplyError> {
    // c1 reverse index keyed by the outer-loop dimension:
    //   normal → by right sibling s1; swapped → by left child a1.
    if !SWAPPED {
        build_reverse_index::<true>(c1_level, k1_right, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
    } else {
        build_reverse_index::<false>(c1_level, k1_left, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
    }
    // c2 reverse index keyed by the FILTER-INNER dimension (the general arm's
    // filtering axis — and exactly the keying the leaf arm needs, which groups
    // c2 by the non-leaf outer child for selectivity; one build serves both arms):
    //   normal → by right s2 → entries (p2, a2); swapped → by left a2 → entries (p2, s2).
    if !SWAPPED {
        build_reverse_index::<true>(c2_level, k2_right, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    } else {
        build_reverse_index::<false>(c2_level, k2_left, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    }

    if leaf_side_is_leaf {
        // ── Leaf arm ──
        // Iterate the non-leaf product list; the leaf-side product comes from
        // CONJOIN_GRID. rev_entries_c2's inner child IS the leaf label here
        // (normal: a2 with left the leaf; swapped: s2 with right the leaf).
        //
        // A3: amortized cancellation/deadline poll — same rationale/soundness
        // as the general arm below; bail lands where `try_push` recovers.
        let mut ticker = super::budget::PollTicker::new(super::budget::APPLY_POLL_STRIDE);
        let pl_outer = if !SWAPPED { pl_right } else { pl_left };
        for &ProductEntry { c1_idx: C1NodeIdx(outer1), c2_idx: C2NodeIdx(outer2), prod_idx: ProdNodeIdx(outer_prod) } in pl_outer {
            let off_c1 = ws.rev_offsets_c1[outer1 as usize] as usize;
            let end_c1 = ws.rev_offsets_c1[outer1 as usize + 1] as usize;
            if off_c1 == end_c1 { continue; }
            let off_c2 = ws.rev_offsets_c2[outer2 as usize] as usize;
            let end_c2 = ws.rev_offsets_c2[outer2 as usize + 1] as usize;
            for &(p2, inner2) in &ws.rev_entries_c2[off_c2..end_c2] {
                for &(p1, inner1) in &ws.rev_entries_c1[off_c1..end_c1] {
                    // normal:  outer_prod = sib_idx (right pl), grid computes a_prod.
                    // swapped: outer_prod = a_prod  (left pl),  grid computes sib_idx.
                    let grid_prod = CONJOIN_GRID[inner1 as usize][inner2 as usize];
                    if grid_prod != DEAD {
                        let (a_prod, sib_idx) = if !SWAPPED {
                            (grid_prod, outer_prod)
                        } else {
                            (outer_prod, grid_prod)
                        };
                        try_push(&mut ws.par_buckets[p1 as usize], ParEntry {
                            p2, a_prod, sib_idx,
                        })?;
                    }
                }
                ticker.tick_by((end_c1 - off_c1) as u64)?;
            }
        }
        return Ok(());
    }

    // ── General arm (both sides non-leaf) ──
    // Inner product table for the non-iterated product list:
    //   normal: prod_by_a1[a1] = [(a2, a_prod)] from pl_left
    //   swapped: prod_by_s1[s1] = [(s2, sib_prod)] from pl_right
    if !SWAPPED {
        ensure_buckets_cleared(&mut ws.prod_by_a1, k1_left)?;
        for &ProductEntry { c1_idx: C1NodeIdx(a1), c2_idx: C2NodeIdx(a2), prod_idx: ProdNodeIdx(a_prod) } in pl_left {
            try_push(&mut ws.prod_by_a1[a1 as usize], (a2, a_prod))?;
        }
    } else {
        ensure_buckets_cleared(&mut ws.prod_by_s1, k1_right)?;
        for &ProductEntry { c1_idx: C1NodeIdx(s1), c2_idx: C2NodeIdx(s2), prod_idx: ProdNodeIdx(sib_prod) } in pl_right {
            try_push(&mut ws.prod_by_s1[s1 as usize], (s2, sib_prod))?;
        }
    }
    // Outer-loop liveness buckets:
    //   normal: right_buckets[r1] = [(r2=s2, sib_idx)] from pl_right
    //   swapped: left_buckets[a1] = [(a2, a_prod)] from pl_left
    if !SWAPPED {
        ensure_buckets_cleared(&mut ws.right_buckets, k1_right)?;
        for &ProductEntry { c1_idx: C1NodeIdx(r1), c2_idx: C2NodeIdx(r2), prod_idx: ProdNodeIdx(sib_idx) } in pl_right {
            try_push(&mut ws.right_buckets[r1 as usize], (r2, sib_idx))?;
        }
    } else {
        ensure_buckets_cleared(&mut ws.left_buckets, k1_left)?;
        for &ProductEntry { c1_idx: C1NodeIdx(a1), c2_idx: C2NodeIdx(a2), prod_idx: ProdNodeIdx(a_prod) } in pl_left {
            try_push(&mut ws.left_buckets[a1 as usize], (a2, a_prod))?;
        }
    }

    // Per-outer filtered index: keyed by the join's inner-c2 child.
    //   normal: filtered[a2] = [(p2, sib_idx)]  (sized k2_left)
    //   swapped: filtered[s2] = [(p2, a_prod)]  (sized k2_right)
    let filtered_dim = if !SWAPPED { k2_left } else { k2_right };
    ensure_buckets_cleared(&mut ws.filtered, filtered_dim)?;
    ws.filtered_touched.clear();

    // A3: amortized cancellation/deadline poll. The sparse join had no mid-level
    // break, so a wide level could wait out a cancelled race lane / expired
    // deadline / due preempt slice. Accumulate emitted-candidate work and poll
    // every ~1M units (see `budget::PollTicker`); the bail lands at a loop level
    // already covered by `try_push`'s recovery, so the workspace stays reusable.
    let mut ticker = super::budget::PollTicker::new(super::budget::APPLY_POLL_STRIDE);
    let outer_k1 = if !SWAPPED { k1_right } else { k1_left };
    for outer in 0..outer_k1 {
        let outer_empty = if !SWAPPED {
            ws.right_buckets[outer].is_empty()
        } else {
            ws.left_buckets[outer].is_empty()
        };
        if outer_empty { continue; }

        // ── Build the per-outer filtered c2 index ──
        // For each live (inner_live, attached), walk the opposite-keyed c2 index
        // and bucket each c2 parent by its inner child, carrying `attached`.
        let live_len = if !SWAPPED { ws.right_buckets[outer].len() } else { ws.left_buckets[outer].len() };
        for li in 0..live_len {
            let (c2_key, attached) = if !SWAPPED {
                ws.right_buckets[outer][li]   // (r2=s2, sib_idx)
            } else {
                ws.left_buckets[outer][li]    // (a2, a_prod)
            };
            let off = ws.rev_offsets_c2[c2_key as usize] as usize;
            let end = ws.rev_offsets_c2[c2_key as usize + 1] as usize;
            for ei in off..end {
                let (p2, inner_c2) = ws.rev_entries_c2[ei]; // normal: (p2, a2); swapped: (p2, s2)
                let bucket = &mut ws.filtered[inner_c2 as usize];
                if bucket.is_empty() {
                    ws.filtered_touched.push(inner_c2);
                }
                // re-borrow after the touched push (push borrows a different field)
                try_push(&mut ws.filtered[inner_c2 as usize], (p2, attached))?;
            }
        }

        // ── Emit: walk c1-parents sharing this outer; for each alive inner
        //    product, replay the precomputed alive filtered entries ──
        let c1_off = ws.rev_offsets_c1[outer] as usize;
        let c1_end = ws.rev_offsets_c1[outer + 1] as usize;
        for ci in c1_off..c1_end {
            let (p1, inner1) = ws.rev_entries_c1[ci];
            // Defensive bounds check (deep-vsplit invariant break).
            if !SWAPPED {
                if (inner1 as usize) >= ws.prod_by_a1.len() { return Err(ApplyError::OverBudget); }
            } else {
                if (inner1 as usize) >= ws.prod_by_s1.len() { return Err(ApplyError::OverBudget); }
            }
            let bucket = &mut ws.par_buckets[p1 as usize];
            if !SWAPPED {
                for &(a2, a_prod) in &ws.prod_by_a1[inner1 as usize] {
                    let fb = &ws.filtered[a2 as usize];
                    for &(p2, sib_idx) in fb {
                        try_push(bucket, ParEntry { p2, a_prod, sib_idx })?;
                    }
                    ticker.tick_by(fb.len() as u64)?;
                }
            } else {
                for &(s2, sib_prod) in &ws.prod_by_s1[inner1 as usize] {
                    let fb = &ws.filtered[s2 as usize];
                    for &(p2, a_prod) in fb {
                        try_push(bucket, ParEntry { p2, a_prod, sib_idx: sib_prod })?;
                    }
                    ticker.tick_by(fb.len() as u64)?;
                }
            }
        }

        // ── Clear only the filtered buckets touched this outer ──
        for ti in 0..ws.filtered_touched.len() {
            let idx = ws.filtered_touched[ti] as usize;
            ws.filtered[idx].clear();
        }
        ws.filtered_touched.clear();
    }
    Ok(())
}

/// Greedy bin-pack of c1-parent indices into chunks whose projected Phase E+F
/// transient byte cost stays under `bytes_budget`. Returns boundary indices
/// `[0, p1_a, p1_b, ..., k1]`; each chunk processes `par_buckets[boundaries[i] .. boundaries[i+1]]`.
///
/// A single p1's bucket is never split. Returns the single-chunk degenerate
/// list `[0, k1]` when `bytes_budget` is `0` or `usize::MAX`, or when the
/// whole level fits in one chunk — in which case the call site's loop runs
/// exactly once and the path is byte-for-byte equivalent to the unchunked code.
#[inline]
fn plan_e_f_chunks(
    par_buckets: &[Vec<ParEntry>],
    k1: usize,
    bytes_budget: usize,
) -> SmallVec<[u32; 8]> {
    let mut out: SmallVec<[u32; 8]> = SmallVec::new();
    out.push(0);
    if bytes_budget == 0 || bytes_budget == usize::MAX {
        out.push(k1 as u32);
        return out;
    }
    let entries_budget = bytes_budget / BYTES_PER_PAR_ENTRY;
    let mut acc = 0usize;
    for p1 in 0..k1 {
        let n = par_buckets[p1].len();
        if acc != 0 && acc.saturating_add(n) > entries_budget {
            out.push(p1 as u32);
            acc = 0;
        }
        acc += n;
    }
    out.push(k1 as u32);
    out
}

/// Emit output nodes for c1-parents in `[p1_start..p1_end)` (Phase E + Phase F
/// applied to one chunk). Reuses `ws.emit_pairs`, `ws.pair_counts`,
/// `ws.sorted_pairs` from scratch (cleared/resized at chunk entry) so peak
/// transient bytes stay bounded by the chunk size. After emit, drops the
/// inner allocations of `ws.par_buckets[p1_start..p1_end]` so the next
/// chunk's `emit_pairs` grows in already-released address space.
///
/// Pre: `ws.par_buckets[p1_start..p1_end]` is populated by Phase C/D.
/// `pl_output.len()` at entry == number of parents emitted by previous chunks
/// (so `prod_idx` stays sequential globally).
///
/// `emit_pairs` stores the *chunk-local* parent index (`prod_idx - chunk_parent_start`),
/// allowing `pair_counts` to be sized to `num_new_parents` rather than the
/// running total.
#[inline]
fn flush_chunk(
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), ApplyError> {
    let chunk_parent_start = pl_output.len() as u32;
    ws.emit_pairs.clear();

    flush_chunk_phase_e(ws, pl_output, chunk_parent_start, p1_start, p1_end, drop_consumed)?;
    flush_chunk_phase_f(ws, level, pl_output, chunk_parent_start)?;
    Ok(())
}

/// Phase E (chunk-local): dedup parent products via `p2_map`, emit `InputPair`s
/// into `ws.emit_pairs`, and optionally drop consumed `par_buckets` rows.
///
/// Called exclusively from `flush_chunk`.
#[inline(always)]
fn flush_chunk_phase_e(
    ws: &mut SparseWorkspace,
    pl_output: &mut Vec<ProductEntry>,
    chunk_parent_start: u32,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), ApplyError> {
    // Defensive: guard against a prior call bailing mid-loop and leaving
    // stale touched entries (mirrors `scatter_outsens`'s own defensive
    // `ws.filtered_touched.clear()`).
    ws.p2_map_touched.clear();

    // Index-based iteration so the mutable accesses to `ws.p2_map` and
    // `ws.emit_pairs` inside the loop don't conflict with the immutable
    // borrow of `ws.par_buckets[p1]`.
    for p1 in p1_start..p1_end {
        let bucket_len = ws.par_buckets[p1].len();
        if bucket_len == 0 { continue; }

        for ei in 0..bucket_len {
            let entry = ws.par_buckets[p1][ei];
            let global_idx = {
                let slot_val = ws.p2_map[entry.p2 as usize];
                if slot_val == DEAD {
                    let idx = pl_output.len() as u32;
                    ws.p2_map[entry.p2 as usize] = idx;
                    ws.p2_map_touched.push(entry.p2);
                    try_push(pl_output, ProductEntry {
                        c1_idx: C1NodeIdx(p1 as u32),
                        c2_idx: C2NodeIdx(entry.p2),
                        prod_idx: ProdNodeIdx(idx),
                    })?;
                    idx
                } else {
                    slot_val
                }
            };
            let local = global_idx - chunk_parent_start;
            // Marginal children never reach the sparse path (guarded at
            // apply_sparse_level entry), so child refs are plain structural
            // indices — no bit-30 slot tagging here.
            let left_raw = entry.a_prod;
            let right_raw = entry.sib_idx;
            try_push(&mut ws.emit_pairs, (local, InputPair {
                left: LocalNodeIdx(left_raw),
                right: LocalNodeIdx(right_raw),
            }))?;
        }

        // Lazy-clear p2_map (only entries actually written this p1, via the
        // touched list — avoids rescanning `par_buckets[p1]` a second time).
        for ti in 0..ws.p2_map_touched.len() {
            let p2 = ws.p2_map_touched[ti];
            ws.p2_map[p2 as usize] = DEAD;
        }
        ws.p2_map_touched.clear();
    }

    // Multi-chunk mode: drop consumed par_buckets allocations (replace with
    // Vec::new()) so the backing memory is freed before the next chunk's
    // emit_pairs / sorted_pairs grow — this is the whole point of chunking.
    //
    // Single-chunk mode: leave buckets alone. The next apply's
    // ensure_buckets_cleared will `.clear()` (length=0, retain capacity),
    // preserving the cross-apply capacity reuse that the old non-chunked code
    // relied on for cheap scatter pushes.
    if drop_consumed {
        for p1 in p1_start..p1_end {
            ws.par_buckets[p1] = Vec::new();
        }
    }
    Ok(())
}

/// Phase F (chunk-local): counting-sort `ws.emit_pairs` by chunk-local parent
/// index and create output nodes in `level`.
///
/// Called exclusively from `flush_chunk`. No-ops when `ws.emit_pairs` produced
/// zero new parents for this chunk.
#[inline(always)]
fn flush_chunk_phase_f(
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &[ProductEntry],
    chunk_parent_start: u32,
) -> Result<(), ApplyError> {
    let num_new_parents = pl_output.len() - chunk_parent_start as usize;
    if num_new_parents == 0 { return Ok(()); }

    // Copied out before `ws.sorted_pairs` is mutably borrowed below.
    let dups_legal = ws.dups_legal;

    let pc = &mut ws.pair_counts;
    try_resize(pc, num_new_parents + 1, 0)?;
    pc[..num_new_parents + 1].fill(0);
    for &(local_parent, _) in &ws.emit_pairs {
        debug_assert!((local_parent as usize) < num_new_parents);
        pc[local_parent as usize] += 1;
    }
    let mut total = 0u32;
    for i in 0..num_new_parents {
        let c = pc[i];
        pc[i] = total;
        total += c;
    }
    pc[num_new_parents] = total;

    let n = total as usize;
    let sp = &mut ws.sorted_pairs;
    try_resize(sp, n, InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) })?;
    for &(local_parent, pair) in &ws.emit_pairs {
        let pos = pc[local_parent as usize] as usize;
        sp[pos] = pair;
        pc[local_parent as usize] += 1;
    }
    // After the fill pass each pc[i] points one-past-end of its bucket;
    // shift right so pc[i] is back at start-of-bucket (same pass-4 restore
    // as build_reverse_index) for the node-creation loop below.
    shift_offsets_right_by_one(&mut pc[..=num_new_parents]);

    budget_reserve(&mut level.nodes, num_new_parents)?;
    for i in 0..num_new_parents {
        let start = pc[i] as usize;
        let end = pc[i + 1] as usize;
        let pair_slice = &sp[start..end];
        // No sort, no dedup. Pair lists are unordered sets and twin contraction
        // is order-independent, so the emit
        // order is free — no canonicalizing sort is required.
        //
        // The scatter cannot produce duplicate pairs when child levels are
        // canonical (no duplicate nodes ⇒ grid lookups are injective; the
        // pair-level corollary of the no-compress proof).
        // A defensive dedup here would be dead weight — one measured over the
        // whole suite removed zero pairs. In a purely Boolean diagram the pair
        // list is never a legitimate multiset, so a duplicate signals an
        // upstream canonicity violation to fix at the source. Once *any* level
        // is marginal, duplicates are legal (`ws.dups_legal`; maintainer ruling
        // 2026-07-27 — pair lists are then multisets feeding a sum) and are
        // inherited from an operand parent whose own list holds the pair twice.
        debug_assert!(
            dups_legal || {
                let mut seen = std::collections::HashSet::new();
                pair_slice.iter().all(|p| seen.insert(*p))
            },
            "sparse apply: duplicate pair emitted — canonicity violated"
        );
        // Output-pair meter, charged around the sparse builder's own emit: it
        // grows `level.pairs` with a raw `try_reserve`, so the dense walk's
        // choke point never sees these. See `ApplyLimits::pairs_in_flight`.
        let pre_pairs_cap = level.pairs.capacity();
        level.try_push_internal_node(pair_slice)
            .map_err(|_| ApplyError::OverBudget)?;
        super::budget::account_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    }
    Ok(())
}

/// Process a single internal level using the sparse scatter-filter-dedup pipeline.
///
/// Instead of iterating all k1*k2 cells, builds reverse indices from parent pairs
/// and scatters from live child products upward. Only alive products are touched.
///
/// The scatter and sibling-liveness filter are fused: we iterate by
/// right-sibling s1, populate sib_lookup once per s1, then scatter with
/// inline s2 filtering — avoiding an intermediate candidate buffer.
///
/// Phases:
///   A+C: Fused scatter-filter by right sibling via sib_lookup[s2]
///   E:   Dedup parent products via p2_map[p2]; emit InputPairs
///   F:   Counting-sort pairs by parent product, create output nodes
///
/// Phases E+F are chunked by c1-parent index range when the projected transient
/// cost exceeds `TIDIDI_SPARSE_CHUNK_BYTES` (default 256 MiB) — each chunk's
/// `par_buckets` rows are dropped before the next chunk's `emit_pairs` grows,
/// capping within-call peak on wide levels (e.g. MCC 2025 canaries).
pub(super) fn apply_sparse_level(
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    c1: &Tdd,
    c2: &Tdd,
    levels: &mut [TddLevel],
    c1_widths: &[usize],
    c2_widths: &[usize],
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    pl_output: &mut Vec<ProductEntry>,
    left_is_leaf: bool,
    right_is_leaf: bool,
    // Whether this level's vars are marginalized after (sparse-STREAM lever
    // eligibility). Used only by the `instrument` build's transient accounting.
    #[allow(unused_variables)] is_marg_target: bool,
) -> Result<(), ApplyError> {
    let t_idx = t.idx();
    let k1 = c1_widths[t_idx];
    let k2 = c2_widths[t_idx];
    let k1_left = c1_widths[left.idx()];
    let k2_left = c2_widths[left.idx()];
    let k1_right = c1_widths[right.idx()];
    let k2_right = c2_widths[right.idx()];

    // No-marginal-leakage guard (tier-0, every build incl. release). The sparse
    // path is never routed for a marginal child level — every marginal-parent
    // level goes to the dedicated marginal-parent dispatch in apply_inner. This
    // matters because the reverse-index below buckets parents by the decoded
    // child coordinate `decode_marg_coord(pair.left.0, …)`; under inline encoding
    // a marginal ref decodes to the COUNT, not a per-node index, collapsing
    // equal-count children into one bucket → dropped multiplicity (the mc007 ×4).
    // The operand-child (c1/c2) checks are load-bearing — an inline marginal ref
    // can only exist on a marginal child level. Always-on so a routing regression
    // aborts loudly instead of silently miscounting.
    cheap_assert!(
        !c1.levels[left.idx()].is_marginal() && !c1.levels[right.idx()].is_marginal()
            && !c2.levels[left.idx()].is_marginal() && !c2.levels[right.idx()].is_marginal()
            && !levels[left.idx()].is_marginal() && !levels[right.idx()].is_marginal(),
        "apply_sparse_level reached with a marginal child (t={t_idx} l={} r={}): \
         the dedicated marginal-parent dispatch was bypassed",
        left.idx(), right.idx()
    );

    SPARSE_WS.with_borrow_mut(|ws| -> Result<(), ApplyError> {

        // Dirty-flag recovery: if the previous sparse apply bailed mid-iteration
        // (e.g. OverBudget from try_push inside the scatter loop), the lazy-cleared
        // lookup tables `sib_lookup`/`child_lookup`/`p2_map` may still hold
        // non-DEAD entries that the scatter-clean cleanup never restored.
        // `try_resize` below is a no-op when the table is already large enough,
        // so without this reset the new apply would read stale prod indices and
        // emit spurious pairs (silent undercount; td-fc-pri × --mc on
        // mc2022_track1_048 reproduced this against the unbudgeted retry).
        if ws.dirty {
            ws.sib_lookup.fill(DEAD);
            ws.child_lookup.fill(DEAD);
            ws.p2_map.fill(DEAD);
        }
        ws.dirty = true;

        // Duplicate pairs in one node's list are legal once any level of the
        // diagram is marginal — pair lists are then multisets feeding a sum
        // (maintainer ruling 2026-07-27). A duplicate here is *inherited*: an
        // operand parent whose own list holds the same pair twice produces the
        // same product pair twice, which is exactly the multiplicity the count
        // recurrence needs. Only the pure-Boolean case still guarantees
        // set-ness, so that is where the Phase F check stays armed. `cfg!` is a
        // compile-time constant, so the level scan is dead code in release.
        ws.dups_legal = cfg!(debug_assertions)
            && (c1.levels.iter().any(|l| l.is_marginal())
                || c2.levels.iter().any(|l| l.is_marginal())
                || levels.iter().any(|l| l.is_marginal()));

        // ── Fused scatter-filter ──────────────────────────────────────
        //
        // Four-way join: parent(p1,p2) <- c1(p1,a1,s1) /\ c2(p2,a2,s2)
        //                                /\ left_alive(a1,a2) /\ right_alive(s1,s2)
        //
        // Direction chosen by child grid size:
        //   left_grid <= right_grid: outer=s1, probe=sib_lookup (normal)
        //   left_grid >  right_grid: outer=a1, probe=child_lookup (swapped)
        //
        // When the iterated child is a leaf, the reverse index for the
        // opposite operand is keyed by the non-leaf child for selectivity,
        // and CONJOIN_GRID replaces the lookup table for the leaf product.

        let left_grid = k1_left * k2_left;
        let right_grid = k1_right * k2_right;
        // Direction: selectivity estimator (general path) picks the side with fewer
        // dead probes. Do not substitute a plain grid-size proxy — it ignores
        // selectivity and mispicks on wide×wide segment conjoins.
        let both_non_leaf = !left_is_leaf && !right_is_leaf;
        let swap_direction = if both_non_leaf {
            estimate_scatter_direction(
                &mut ws.est_counts,
                &c1.levels[t_idx], &c2.levels[t_idx], pl_left, pl_right,
                k1_left, k2_left, k1_right, k2_right,
            )?
        } else {
            left_grid > right_grid
        };

        ensure_buckets_cleared(&mut ws.par_buckets, k1)?;
        try_resize(&mut ws.p2_map, k2, DEAD)?;

        // Output-sensitive join: THE scatter engine, for both leaf and general
        // levels. The general arm carries no dead-probe inner loop (that probe
        // ran 91-98% dead on dense segment conjoins); the leaf arm keeps the
        // leaf fast-path shape. There is no alternative engine to select.
        if !swap_direction {
            scatter_outsens::<false>(ws, &c1.levels[t_idx], &c2.levels[t_idx],
                k1_left, k2_left, k1_right, k2_right,
                pl_left, pl_right, left_is_leaf)?;
        } else {
            scatter_outsens::<true>(ws, &c1.levels[t_idx], &c2.levels[t_idx],
                k1_left, k2_left, k1_right, k2_right,
                pl_left, pl_right, right_is_leaf)?;
        }

        // `plan_e_f_chunks` greedy-packs c1-parent indices into Phase E+F chunks
        // under `sparse_chunk_bytes()` (default 256 MiB; `usize::MAX` disables).
        // Typical MCC instances fit in one chunk — single `flush_chunk` call with
        // `drop_consumed=false`, preserving cross-apply par_buckets capacity reuse.
        // Wide levels split into multiple chunks with `drop_consumed=true`,
        // releasing each consumed range's `par_buckets[p1]` before the next
        // chunk's `emit_pairs` grows.
        let level = &mut levels[t_idx];
        let boundaries = plan_e_f_chunks(&ws.par_buckets, k1, sparse_chunk_bytes());
        let is_chunked = boundaries.len() > 2;
        for window in boundaries.windows(2) {
            flush_chunk(ws, level, pl_output,
                window[0] as usize, window[1] as usize, is_chunked)?;
        }

        #[cfg(debug_assertions)]
        {
            // par_buckets contents are still present in single-chunk mode (we
            // iterated by reference) and replaced with Vec::new() in multi-chunk
            // mode. Either way they're "logically consumed" — the next apply's
            // ensure_buckets_cleared will reset length. No structural assertion
            // here; pl_output / level.nodes invariants below catch real bugs.
            //
            // pl_output grew monotonically and prod_idx[i] == i.
            for (i, e) in pl_output.iter().enumerate() {
                debug_assert_eq!(e.prod_idx.idx(), i,
                    "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
            }
            debug_assert!(levels[t_idx].nodes.len() == pl_output.len(),
                "level.nodes.len() {} != pl_output.len() {}",
                levels[t_idx].nodes.len(), pl_output.len());
        }

        // Scatter-clean cleanup completed; lookup tables are all DEAD again.
        // The dirty-flag recovery at entry is unnecessary on the next call.
        ws.dirty = false;
        Ok(())
    })
}

/// Compute the conjunction (AND) of two TDDs via compacting product construction.
///
/// Given TDDs of width k and k' over the same vtree, produces a TDD for their
/// conjunction with width ≤ k·k'. Dead nodes (zero conjunctions) are omitted
/// from the output (compaction), so the "no false nodes" invariant is preserved.
///
/// Short-circuits to ZERO if either input is UNSAT.
///
/// ## Structure (for navigating this 650+ line function)
///
/// 1. **Early exit**: ZERO inputs → return ZERO immediately
/// 2. **Product grid setup**: flat `node_idx` array mapping (level, i, j) → output index
/// 3. **Identity tracking**: detect constant-true subtrees to skip product computation
/// 4. **Leaf processing**: 4×4 truth table conjunctions, identity leaf fast-paths
/// 5. **Internal processing**: cross-product of input pairs with four specializations:
///    - 1×1 (single pair each): direct lookup, no allocation
///    - N×1 / 1×N: linear scan of one operand's pairs
///    - N×M: full cross-product with dead-pair pre-filtering (bitmask or coarse)
/// 6. **Output**: look up the conjunction of the two output nodes
/// Fill grid entries at leaf vtree levels from the static `CONJOIN_GRID` table.
///
/// At leaf levels the conjunction is a constant 3×3 truth table (Pos, Neg, One),
/// so we just copy from `CONJOIN_GRID` into `node_idx`. When `might_use_sparse`,
/// grid space is bump-allocated as we go and live counts are recorded for
/// parent density checks; otherwise the grid offsets are pre-computed.
pub(super) fn apply_leaf_levels(
    vtree: &crate::vtree::Vtree,
    c1_widths: &[usize],
    c2_widths: &[usize],
    grids: &mut [LevelGrid],
    node_idx: &mut Vec<u32>,
    grid_end: &mut usize,
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    might_use_sparse: bool,
    // Spine-bounded apply: the leaves that are children of a rebuilt level.
    // Every other leaf's grid is unreachable — its parent rides through
    // untouched — so building it would be pure waste. `None` = every leaf.
    only: Option<&[crate::vtree::VtreeIdx]>,
) -> Result<(), ApplyError> {
    let mut one_leaf = |t: crate::vtree::VtreeIdx| -> Result<(), ApplyError> {
        let t_idx = t.idx();
        let k1 = c1_widths[t_idx];
        let k2 = c2_widths[t_idx];
        let t_base = if might_use_sparse {
            let base = *grid_end;
            *grid_end += k1 * k2;
            try_resize_dead(node_idx, *grid_end)?;
            base
        } else {
            grids[t_idx].base_unchecked()
        };
        grids[t_idx] = LevelGrid::Leaf { base: t_base };
        let mut count = 0usize;
        for i in 0..k1 {
            for j in 0..k2 {
                let val = CONJOIN_GRID[i][j];
                node_idx[t_base + i * k2 + j] = val;
                if val != DEAD { count += 1; }
            }
        }
        if might_use_sparse { bump_live_count(live_counts, out_nodes_so_far, t_idx, count); }
        Ok(())
    };
    match only {
        Some(l) => for &t in l { one_leaf(t)?; },
        None => for (t, _leaf_var) in vtree.leaf_bottomup() { one_leaf(t)?; },
    }
    Ok(())
}

/// Compute the output local index for the conjunction TDD.
///
/// Three cases by how the root level was processed:
/// - Dense grid: O(1) lookup at `node_idx[out_flat]`.
/// - Sparse with product list: scan list for the (c1_out, c2_out) entry.
/// - Sparse identity: pass through the non-identity operand's output.
///
/// Returns ZERO if the root conjunction is unsatisfiable.
pub(super) fn compute_apply_output(
    c1: &Tdd,
    c2: &Tdd,
    grids: &[LevelGrid],
    node_idx: &[u32],
    c2_widths: &[usize],
    c1_identity: &[bool],
    c2_identity: &[bool],
    has_pl: &[bool],
    product_lists: &[Vec<ProductEntry>],
) -> LocalNodeIdx {
    let out_ti = c1.output.vtree.idx();
    if let Some(out_base) = grids[out_ti].base() {
        let out_flat = out_base
            + c1.output.local.idx() * c2_widths[out_ti]
            + c2.output.local.idx();
        let val = node_idx[out_flat];
        if val != DEAD { LocalNodeIdx(val) } else { ZERO }
    } else {
        let c1_out = c1.output.local.0;
        let c2_out = c2.output.local.0;
        if !has_pl[out_ti] {
            // Root is an identity level: pass through the non-identity operand's output.
            if c2_identity[out_ti] {
                LocalNodeIdx(c1_out)
            } else if c1_identity[out_ti] {
                LocalNodeIdx(c2_out)
            } else {
                // Invariant violation, not an UNSAT result: fabricating ZERO here
                // would silently miscount. Abort loudly in every build (A7).
                cheap_assert!(
                    false,
                    "compute_apply_output: root level t={out_ti} has no grid, no \
                     product list, and neither identity flag — apply-routing invariant \
                     violated (c1_out={c1_out} c2_out={c2_out})"
                );
                ZERO
            }
        } else {
            product_lists[out_ti]
                .iter()
                .find(|e| e.c1_idx == C1NodeIdx(c1_out) && e.c2_idx == C2NodeIdx(c2_out))
                .map(|e| LocalNodeIdx(e.prod_idx.0))
                .unwrap_or(ZERO)
        }
    }
}


/// Release the thread-local sparse workspace bucket memory if it grew too large.
///
/// Called from `apply_and_fallible` after each sparse level to cap retained peak.
pub(super) fn release_sparse_ws_if_large() {
    SPARSE_WS.with_borrow_mut(|ws| ws.release_if_large());
}

/// Fully drop the thread-local sparse workspace, replacing it with a fresh
/// `SparseWorkspace::default()` — every bucket array, reverse index, and emit
/// buffer released to the allocator. Unlike `release_sparse_ws_if_large` (the
/// conditional per-array trim on the normal apply exit), this frees ALL retained
/// capacity unconditionally.
///
/// Safe ONLY at an inter-compile boundary — no apply in flight on this thread.
/// The panic that unwinds a failed sub-compile drops the `SPARSE_WS` `RefCell`
/// borrow guard, but the workspace itself is OWNED by the thread-local, so its
/// bucket arrays (`par_buckets` alone measured ~1.8 GiB live at a depth-1
/// recovery split) survive the unwind at full capacity. This reset is the
/// reclaim for that pin; calling it while `apply_sparse_level` holds the borrow
/// would panic on the double borrow.
pub(crate) fn reset_sparse_ws() {
    SPARSE_WS.with_borrow_mut(|ws| *ws = SparseWorkspace::default());
}

#[cfg(test)]
#[path = "sparse_scatter_direction_pool_tests.rs"]
mod scatter_direction_pool_tests;

#[cfg(test)]
#[path = "sparse_p5_retention_tests.rs"]
mod p5_retention_tests;

#[cfg(test)]
#[path = "sparse_a4_self_conjunction_tests.rs"]
mod a4_self_conjunction_tests;

#[cfg(test)]
#[path = "sparse_reset_ws_tests.rs"]
mod reset_ws_tests;

#[cfg(test)]
#[path = "sparse_regression_tests.rs"]
mod regression_tests;
