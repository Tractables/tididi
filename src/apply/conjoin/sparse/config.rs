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

/// Choose the scatter's direction on a level whose children are both
/// internal: which child of f the general arm keys its outer loop by, the
/// right one (`false`) or the left one (`true`).
///
/// Both directions emit the same candidates, so the emit's output is no
/// basis for the choice. What differs is the work around it, and each part
/// has a bound one pass over the level's pairs and product lists can sum,
/// with `O` the outer child and `I` the inner one:
///
/// - the outer loop, one round per live product of the outer child;
/// - the walk from every f pair under an outer key through the inner
///   child's products, which the marking and the emit each make once:
///   `Σ_{(i, i2) ∈ PL_I} cnt_f_I[i]`;
/// - the filtered index build, the cheaper of its two walks: by the outer's
///   g keys, `Σ_{(o, o2) ∈ PL_O} deg_g_O[o2]`, or by the wanted inner-g
///   children, at most every g pair an f pair reaches through the inner
///   products, `Σ_{(i, i2) ∈ PL_I} cnt_f_I[i] · deg_g_I[i2]`.
///
/// `cnt_f_*` counts f's pairs by one child and `deg_g_*` counts g's. The
/// build's second walk is bounded rather than counted because the wanted
/// set is only known once the outer's f pairs are walked; where the outer's
/// g keys hold most of the level (a g free over the outer child keeps every
/// pair under one key) the bound is the one that decides, and it is exact
/// when each inner child reaches one g pair. When the counts tie, the
/// direction whose inner child is the narrower wins: the inner dimension is
/// the one the walk and the build index at random.
///
/// Cost is O(|f pairs| + |g pairs| + |pl_left| + |pl_right|), small next to
/// the work the choice governs. The four counter arrays are carved out of
/// the pooled `est_counts` buffer, sized through `try_resize` so a refusal
/// is `OverBudget` rather than an abort.
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
    let (cnt_f_left, rest) = buf.split_at_mut(f.left);
    let (cnt_f_right, rest) = rest.split_at_mut(f.right);
    let (deg_g_left, deg_g_right) = rest.split_at_mut(g.left);
    for node in left_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in left_level.pairs_of(node) {
            cnt_f_left[pair.left.0 as usize] += 1;
            cnt_f_right[pair.right.0 as usize] += 1;
        }
    }
    for node in right_level.nodes.iter() {
        if !node.is_internal() { continue; }
        for pair in right_level.pairs_of(node) {
            deg_g_left[pair.left.0 as usize] += 1;
            deg_g_right[pair.right.0 as usize] += 1;
        }
    }
    // Each product list serves both directions: as the inner child's it
    // gives the walk and the bound on the build by inner-g child, as the
    // outer child's it gives the build by g key.
    let (walk_by_left, reach_by_left, keys_by_left) = walk_and_keys(pl_left, cnt_f_left, deg_g_left);
    let (walk_by_right, reach_by_right, keys_by_right) = walk_and_keys(pl_right, cnt_f_right, deg_g_right);
    let cost_normal = pl_right.len() as u128 + walk_by_left + keys_by_right.min(reach_by_left);
    let cost_swapped = pl_left.len() as u128 + walk_by_right + keys_by_left.min(reach_by_right);
    Ok(cost_swapped < cost_normal
        || (cost_swapped == cost_normal && f.right + g.right < f.left + g.left))
}

/// The three sums one child's product list contributes to the direction
/// estimate: the walk of f's pairs through the child's products,
/// `Σ cnt_f[i]`; the g pairs that walk reaches, `Σ cnt_f[i] · deg_g[i2]`;
/// and the g pairs under the child's live g keys, `Σ deg_g[i2]`.
fn walk_and_keys(pl: &[ProductEntry], cnt_f: &[u32], deg_g: &[u32]) -> (u128, u128, u128) {
    let (mut walk, mut reach, mut keys) = (0u128, 0u128, 0u128);
    for e in pl {
        let cnt = cnt_f[e.left_idx.idx()] as u128;
        let deg = deg_g[e.right_idx.idx()] as u128;
        walk += cnt;
        reach += cnt * deg;
        keys += deg;
    }
    (walk, reach, keys)
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
