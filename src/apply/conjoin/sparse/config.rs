//! Thresholds that decide when a level takes the sparse route.

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct SparseConfig {
    pub(crate) min_grid: usize,
    pub(crate) sparsity_factor: u128,
}

/// Default sparse-path config. Production uses these fixed values.
pub(crate) const SPARSE_CONFIG_DEFAULT: SparseConfig = SparseConfig { min_grid: 4096, sparsity_factor: 64 };

#[cfg(test)]
thread_local! {
    /// Test-only override for `sparse_config`, scoped by `with_sparse_config`.
    pub(crate) static SPARSE_CONFIG_OVERRIDE: Cell<Option<SparseConfig>> = const { Cell::new(None) };
}

pub(crate) fn sparse_config() -> SparseConfig {
    #[cfg(test)]
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
pub(crate) fn estimate_scatter_direction(
    eng: &Engine,
    est_counts: &mut Vec<u32>,
    c1_level: &TddLevel,
    c2_level: &TddLevel,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    k1_left: usize, k2_left: usize,
    k1_right: usize, k2_right: usize,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    let total = k1_left + k1_right + k2_left + k2_right;
    lim.try_resize(est_counts, total, 0u32)?;
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

/// Run `f` with `sparse_config()` returning the given values on this thread.
#[cfg(test)]
pub(crate) fn with_sparse_config<F: FnOnce() -> R, R>(min_grid: usize, sparsity_factor: u128, f: F) -> R {
    let cfg = SparseConfig { min_grid, sparsity_factor };
    crate::scoped::Scoped::run(&SPARSE_CONFIG_OVERRIDE, Some(cfg), f)
}

/// Soft byte budget for the sparse Phase E+F transient buffers
/// (`emit_pairs` + `sorted_pairs` + consumed `par_buckets` rows).
/// When `Σ par_buckets[p].len() * BYTES_PER_PAR_ENTRY` exceeds the budget,
/// Phase E+F is emitted in chunks of c1-parent ranges, dropping each chunk's
/// `par_buckets` allocations before the next chunk's `emit_pairs` grows.
///
/// A policy value of 256 MiB, not tunable at runtime. A level whose whole
/// projected transient fits in one chunk produces `boundaries = [0, k1]` from
/// `plan_e_f_chunks` and runs a single `flush_chunk` with `drop_consumed=false`,
/// which preserves the cross-apply `par_buckets` capacity reuse; that is the
/// common case, and the cap exists for the wide levels that are not, which split
/// into several chunks with `drop_consumed=true`.
pub(crate) const SPARSE_CHUNK_BYTES_DEFAULT: usize = 256 * 1024 * 1024;

#[cfg(test)]
thread_local! {
    /// Test-only override for `sparse_chunk_bytes`, scoped by `with_sparse_chunk_bytes`.
    pub(crate) static SPARSE_CHUNK_BYTES_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(crate) fn sparse_chunk_bytes() -> usize {
    #[cfg(test)]
    if let Some(v) = SPARSE_CHUNK_BYTES_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    SPARSE_CHUNK_BYTES_DEFAULT
}

/// Run `f` with `sparse_chunk_bytes()` returning `v` on this thread.
#[cfg(test)]
pub(crate) fn with_sparse_chunk_bytes<F: FnOnce() -> R, R>(v: usize, f: F) -> R {
    crate::scoped::Scoped::run(&SPARSE_CHUNK_BYTES_OVERRIDE, Some(v), f)
}

/// Projected transient cost per surviving `ParEntry`:
///   sizeof(ParEntry)             = 12   (Phase C/E input)
/// + sizeof((u32, InputPair))     = 12   (Phase E output → emit_pairs)
/// + sizeof(InputPair)            = 8    (Phase F output → sorted_pairs)
/// Used by `plan_e_f_chunks` to size chunks under the byte budget.
pub(crate) const BYTES_PER_PAR_ENTRY: usize = 32;

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
