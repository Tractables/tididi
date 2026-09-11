//! Thresholds that decide when a level takes the sparse route.
//!
//! The values themselves are engine fields; this module is the estimator that
//! spends them.

use super::*;

/// Estimate which scatter direction (normal vs swapped) does fewer inner probes,
/// for the general (both-non-leaf) path. The probe count factorizes per pair:
///   `normal = Σ_{(a1,a2)∈pl_left}  cnt_C1_left[a1]·deg_C2_left[a2]`
///   `swap   = Σ_{(s1,s2)∈pl_right} cnt_C1_right[s1]·deg_C2_right[s2]`
/// where cnt_C1_* counts f pairs by left/right child and deg_C2_* counts g pairs
/// by left/right child. Cost is O(|f pairs|+|g pairs|+|pl_left|+|pl_right|) — tiny
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_scatter_direction(
    eng: &Engine,
    est_counts: &mut Vec<u32>,
    left_level: &TddLevel,
    right_level: &TddLevel,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    shape: crate::apply::conjoin::setup::LevelShape,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    let crate::apply::conjoin::setup::LevelShape { f, g, .. } = shape;
    let total = f.left + f.right + g.left + g.right;
    lim.try_resize(est_counts, total, 0u32)?;
    // The buffer is pooled and grow-only, so the prefix in use must be re-zeroed
    // per level — a wider level's residue would otherwise be counted again here.
    let buf = &mut est_counts[..total];
    buf.fill(0);
    // One buffer, four back-to-back index spaces (f-by-left, f-by-right,
    // g-by-left, g-by-right) — the counting loops below need two of them live
    // at once, so they must be disjoint slices.
    let (cnt_c1_left, rest) = buf.split_at_mut(f.left);
    let (cnt_c1_right, rest) = rest.split_at_mut(f.right);
    let (deg_c2_left, deg_c2_right) = rest.split_at_mut(g.left);
    for node in left_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in left_level.pairs_of(node) {
            cnt_c1_left[pair.left.0 as usize] += 1;
            cnt_c1_right[pair.right.0 as usize] += 1;
        }
    }
    for node in right_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in right_level.pairs_of(node) {
            deg_c2_left[pair.left.0 as usize] += 1;
            deg_c2_right[pair.right.0 as usize] += 1;
        }
    }
    let mut est_normal: u128 = 0;
    for e in pl_left {
        est_normal += cnt_c1_left[e.left_idx.0 as usize] as u128
            * deg_c2_left[e.right_idx.0 as usize] as u128;
    }
    let mut est_swap: u128 = 0;
    for e in pl_right {
        est_swap += cnt_c1_right[e.left_idx.0 as usize] as u128
            * deg_c2_right[e.right_idx.0 as usize] as u128;
    }
    Ok(est_swap < est_normal)
}

/// Projected transient cost per surviving `ParEntry`:
///
/// ```text
///   sizeof(ParEntry)             = 12   (Phase C/E input)
/// + sizeof((u32, InputPair))     = 12   (Phase E output → emit_pairs)
/// + sizeof(InputPair)            = 8    (Phase F output → sorted_pairs)
/// ```
///
/// Used by `plan_e_f_chunks` to size chunks under the byte budget.
pub(crate) const BYTES_PER_PAR_ENTRY: usize = 32;

// ── Sparse product construction ──────────────────────────────────────────────
//
// For levels where left_width * right_width exceeds `sparse_min_grid`, the dense grid iteration is
// replaced by a scatter-filter-dedup pipeline inspired by the upward branch.
// Instead of iterating all (i, j) cells, we:
//   1. Build reverse indices: child_idx → [(parent_idx, sibling_idx)]
//   2. Scatter from live child products upward to candidate parents
//   3. Filter candidates by sibling liveness (lazy-cleared flat lookup)
//   4. Dedup parent products (lazy-cleared flat p2_map)
//   5. Emit output pairs and nodes
//
// This is O(n * degree²) where n = live products, vs O(left_width * right_width) for dense.
