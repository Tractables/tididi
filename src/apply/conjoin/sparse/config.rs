//! The thresholds that decide when a level takes the sparse route, and the
//! estimator that spends them.

use super::*;
use crate::limits::{Limits, SparseRoute};

/// The thresholds one apply decides its sparse routing by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseThresholds {
    /// Grid cells above which a level takes the sparse route.
    pub(crate) min_grid: usize,
    /// How much sparser than its grid a level must be to take the sparse route.
    pub(crate) sparsity_factor: u128,
    /// Pairs at or above which an f node walked in order hands its
    /// candidates to [`emit_parent`] as one list, rather than writing them
    /// to the level's pair arena first: such a node has many candidates of
    /// several g parents, which the emit would copy back off the arena.
    pub(crate) collected_pairs: usize,
    /// Bytes of route tables at or above which the routed walk asks the
    /// cache for a group of lookups' slots before it probes any, and for
    /// the hits' runs before it reads any.
    pub(crate) pipelined_bytes: usize,
}

impl SparseThresholds {
    /// The thresholds an engine starts with.
    pub(crate) const PRODUCTION: SparseThresholds = SparseThresholds {
        min_grid: SparseRoute::DEFAULT.min_grid,
        sparsity_factor: SparseRoute::DEFAULT.sparsity as u128,
        collected_pairs: 64,
        pipelined_bytes: 16 << 20,
    };
}

/// The thresholds in force: [`SparseThresholds::PRODUCTION`] with the
/// engine's [`SparseRoute`], unless a test has installed others on this
/// thread.
pub(crate) fn sparse_thresholds(lim: &Limits) -> SparseThresholds {
    forced().unwrap_or_else(|| {
        let route = lim.sparse_route();
        SparseThresholds {
            min_grid: route.min_grid,
            sparsity_factor: u128::from(route.sparsity),
            ..SparseThresholds::PRODUCTION
        }
    })
}

/// The direction estimate's verdict for one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScatterChoice {
    /// Key the outer loop by f's left child rather than its right one.
    pub(super) swapped: bool,
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
/// is `OverBudget` rather than an abort, and are left there for the
/// reverse-index builds, which start from the same counts: see
/// [`EstCounts`].
pub(super) fn estimate_scatter_direction(
    eng: &Engine,
    est_counts: &mut Vec<u32>,
    f_level: &TddLevel,
    g_level: &TddLevel,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    shape: crate::apply::conjoin::setup::LevelShape,
) -> Result<ScatterChoice, OperationError> {
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
    f_level.for_each_node_pair(|_, pair| {
        cnt_f_left[pair.left.0 as usize] += 1;
        cnt_f_right[pair.right.0 as usize] += 1;
    });
    g_level.for_each_node_pair(|_, pair| {
        deg_g_left[pair.left.0 as usize] += 1;
        deg_g_right[pair.right.0 as usize] += 1;
    });
    // Each product list serves both directions: as the inner child's it
    // gives the walk and the bound on the build by inner-g child, as the
    // outer child's it gives the build by g key.
    let (walk_by_left, reach_by_left, keys_by_left) = walk_and_keys(pl_left, cnt_f_left, deg_g_left);
    let (walk_by_right, reach_by_right, keys_by_right) = walk_and_keys(pl_right, cnt_f_right, deg_g_right);
    let cost_normal = pl_right.len() as u128 + walk_by_left + keys_by_right.min(reach_by_left);
    let cost_swapped = pl_left.len() as u128 + walk_by_right + keys_by_left.min(reach_by_right);
    let swapped = cost_swapped < cost_normal
        || (cost_swapped == cost_normal && f.right + g.right < f.left + g.left);
    Ok(ScatterChoice { swapped })
}

/// The four pair counts [`estimate_scatter_direction`] leaves in
/// `est_counts`, one per child of each operand, packed back to back.
///
/// A reverse index keyed by one of those children starts from the same
/// count, so a build after the estimate takes it from here rather than
/// counting the pairs again.
pub(super) struct EstCounts<'a> {
    pub(super) f_left: &'a [u32],
    pub(super) f_right: &'a [u32],
    pub(super) g_left: &'a [u32],
    pub(super) g_right: &'a [u32],
}

impl<'a> EstCounts<'a> {
    /// Split the buffer the estimate filled for `shape`.
    pub(super) fn of(est_counts: &'a [u32], shape: crate::apply::conjoin::setup::LevelShape) -> EstCounts<'a> {
        let crate::apply::conjoin::setup::LevelShape { f, g, .. } = shape;
        let (f_left, rest) = est_counts.split_at(f.left);
        let (f_right, rest) = rest.split_at(f.right);
        let (g_left, rest) = rest.split_at(g.left);
        EstCounts { f_left, f_right, g_left, g_right: &rest[..g.right] }
    }
}

/// The three sums one child's product list contributes to the direction
/// estimate: the walk of f's pairs through the child's products,
/// `Σ cnt_f[i]`; the g pairs that walk reaches, `Σ cnt_f[i] · deg_g[i2]`;
/// and the g pairs under the child's live g keys, `Σ deg_g[i2]`.
fn walk_and_keys(pl: &[ProductEntry], cnt_f: &[u32], deg_g: &[u32]) -> (u128, u128, u128) {
    let (mut walk, mut reach, mut keys) = (0u128, 0u128, 0u128);
    for e in pl {
        let cnt = cnt_f[e.f_idx.idx()] as u128;
        let deg = deg_g[e.g_idx.idx()] as u128;
        walk += cnt;
        reach += cnt * deg;
        keys += deg;
    }
    (walk, reach, keys)
}

// Tests override the routing thresholds; production always uses the defaults.
#[cfg(test)]
use super::tests::forced_thresholds as forced;

#[cfg(not(test))]
fn forced() -> Option<SparseThresholds> {
    None
}
