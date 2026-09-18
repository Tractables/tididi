//! The thresholds that decide when a level takes the sparse route, and the
//! estimator that spends them.

use super::*;

/// The thresholds one apply decides its sparse routing by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseThresholds {
    /// Grid cells above which a level takes the sparse route.
    pub(crate) min_grid: usize,
    /// How much sparser than its grid a level must be to take the sparse route.
    pub(crate) sparsity_factor: u128,
    /// Soft byte budget for the sparse path's transient emission buffers.
    ///
    /// A level whose whole projected transient fits inside the budget is
    /// emitted in one chunk, which preserves the cross-apply bucket capacity
    /// reuse; the budget exists for the wide levels that do not fit, which
    /// split into several chunks and release each consumed range before the
    /// next one grows. `usize::MAX` never splits.
    pub(crate) chunk_bytes: usize,
}

impl SparseThresholds {
    /// The thresholds every apply decides by.
    pub(crate) const PRODUCTION: SparseThresholds = SparseThresholds {
        min_grid: 4096,
        sparsity_factor: 64,
        chunk_bytes: 256 * 1024 * 1024,
    };
}

/// The thresholds in force: [`SparseThresholds::PRODUCTION`], unless a test
/// has installed others on this thread.
pub(crate) fn sparse_thresholds() -> SparseThresholds {
    forced().unwrap_or(SparseThresholds::PRODUCTION)
}

/// Estimate which scatter direction (normal vs swapped) does fewer inner probes,
/// for the general (both-non-leaf) path. The probe count factorizes per pair:
///   `normal = Σ_{(a1,a2)∈pl_left}  cnt_C1_left[a1]·deg_C2_left[a2]`
///   `swap   = Σ_{(s1,s2)∈pl_right} cnt_C1_right[s1]·deg_C2_right[s2]`
/// where cnt_C1_* counts f pairs by left/right child and deg_C2_* counts g pairs
/// by left/right child. Cost is O(|f pairs|+|g pairs|+|pl_left|+|pl_right|) — tiny
/// next to the probes the choice governs. Returns `true` when swapping is
/// cheaper, i.e. `est_swap < est_normal`.
///
/// The four counter arrays are carved out of the pooled `est_counts` buffer,
/// sized through `try_resize` so a refusal is `OverBudget` rather than an abort.
pub(crate) fn estimate_scatter_direction(
    eng: &Engine,
    est_counts: &mut Vec<u32>,
    left_level: &TddLevel,
    right_level: &TddLevel,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    shape: crate::apply::conjoin::setup::LevelShape,
) -> Result<bool, OperationError> {
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
/// + sizeof((u32, ChildPair))     = 12   (Phase E output → emit_pairs)
/// + sizeof(ChildPair)            = 8    (Phase F output → sorted_pairs)
/// ```
///
/// Used by `plan_e_f_chunks` to size chunks under the byte budget.
pub(crate) const BYTES_PER_PAR_ENTRY: usize = 32;

// Tests override the routing thresholds; production always uses the defaults.
#[cfg(test)]
use super::tests::forced_thresholds as forced;

#[cfg(not(test))]
fn forced() -> Option<SparseThresholds> {
    None
}
