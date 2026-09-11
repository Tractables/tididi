//! Apply setup: phases 1-3 of `apply_and_fallible_inner` (width/marginal-entry
//! snapshot, sparse/budget pre-scan, grid/product-list allocation), bundled into
//! `ApplyRun` and produced by `apply_and_setup`. The driver in `conjoin/mod.rs`
//! destructures `ApplyRun` back into its locals. Scratch pools,
//! `MARGINAL_ENTRY_*`, `APPLY_BYTES_PER_CELL`, and `APPLY_LIMITS` belong to
//! `mod.rs`/`budget` and are reached via `super::`.

use crate::engine::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::{self, *};
use super::{liveness, ApplyError, LevelGrid, APPLY_BYTES_PER_CELL};
use super::grid_arena::GridArena;
use crate::value::StreamCache;
use super::output::LiveCounts;
use super::marginal_plan::EntryMarginality;
use super::sparse::ProductEntry;
use super::route::{LevelMarg, SparseGate};
use super::targets::MarginalTargets;

/// Bundled result of `apply_and_setup` — the per-apply working state produced
/// before the bottom-up level sweep.
pub(super) struct ApplyRun {
    pub(super) levels: Vec<TddLevel>,
    pub(super) left_widths: Vec<usize>,
    pub(super) right_widths: Vec<usize>,
    pub(super) min_grid: usize,
    pub(super) sparsity_factor: u128,
    /// Lazily computed child columns for the streaming-marginal path. See
    /// [`StreamCache`].
    pub(super) stream_cache: StreamCache,
    /// The flat product-grid slab and every level's claim on it. See
    /// [`GridArena`].
    pub(super) arena: GridArena,
    pub(super) product_lists: Vec<Vec<ProductEntry>>,
    pub(super) live_counts: LiveCounts,
    pub(super) has_pl: Vec<bool>,
    /// Which levels of each operand were marginal at apply entry. See
    /// [`EntryMarginality`].
    pub(super) entry_marginality: EntryMarginality,
    /// `right_identity[t]` — g computes constant-true over subtree `t`, so f's
    /// nodes pass through unchanged. Lazily accreted, so a false reading only
    /// costs a fallback to the dense grid.
    pub(super) right_identity: Vec<bool>,
    /// The symmetric flag for f.
    pub(super) left_identity: Vec<bool>,
    /// Decode buffers for one cell's pairs, one per operand.
    pub(super) inputs1_scratch: Vec<InputPair>,
    pub(super) inputs2_scratch: Vec<InputPair>,
    /// The four dead-pair pre-filter masks, reused across internal levels.
    pub(super) prefilter_masks: liveness::PrefilterMaskScratch,
}

/// One internal vtree level's identity: the node, its two children, and both
/// operands' widths at each of the three.
///
/// The widths come from the entry snapshot, not from the levels themselves —
/// an identity fast path steals an operand's level mid-sweep, which zeroes the
/// width the level would report.
#[derive(Clone, Copy)]
pub(super) struct LevelShape {
    pub(super) t: VtreeIdx,
    pub(super) left: VtreeIdx,
    pub(super) right: VtreeIdx,
    /// f's widths at the three nodes.
    pub(super) f: OperandWidths,
    /// g's, likewise.
    pub(super) g: OperandWidths,
}

/// One operand's widths across a level and its two children.
///
/// The three names are the vtree axis — `here` is the level itself, `left` and
/// `right` its children — so which operand a width belongs to is said once, by
/// the field of [`LevelShape`] this sits in.
#[derive(Clone, Copy)]
pub(super) struct OperandWidths {
    pub(super) here: usize,
    pub(super) left: usize,
    pub(super) right: usize,
}

impl ApplyRun {
    /// The shape of the level at `t`, read off the entry width snapshot.
    pub(super) fn shape(&self, t: VtreeIdx, left: VtreeIdx, right: VtreeIdx) -> LevelShape {
        let (t_idx, left_idx, right_idx) = (t.idx(), left.idx(), right.idx());
        LevelShape {
            t, left, right,
            f: OperandWidths {
                here: self.left_widths[t_idx],
                left: self.left_widths[left_idx],
                right: self.left_widths[right_idx],
            },
            g: OperandWidths {
                here: self.right_widths[t_idx],
                left: self.right_widths[left_idx],
                right: self.right_widths[right_idx],
            },
        }
    }

    /// This level's marginality, in the two senses [`route_level`](super::route::route_level) needs.
    pub(super) fn level_marginal(
        &self,
        f: &Tdd,
        g: &Tdd,
        shape: LevelShape,
        marginalize_targets: MarginalTargets<'_>,
    ) -> LevelMarg {
        let (t_idx, left_idx, right_idx) = (shape.t.idx(), shape.left.idx(), shape.right.idx());
        let now = |i: usize| self.levels[i].is_marginal();
        let any = |i: usize| {
            self.levels[i].is_marginal()
                || f.levels[i].is_marginal()
                || g.levels[i].is_marginal()
        };
        LevelMarg {
            left_now: now(left_idx),
            right_now: now(right_idx),
            left_any: any(left_idx),
            right_any: any(right_idx),
            is_target: marginalize_targets.is_target(t_idx),
        }
    }

    /// Whether the sparse routes are open at this level, and whether the
    /// children's live products are sparse enough for the scatter walk.
    ///
    /// The density test is exact arithmetic on `u128`: the two maxima are
    /// products of widths and overflow `u64` on wide levels.
    pub(super) fn sparse_gate(&self, shape: LevelShape) -> SparseGate {
        let LevelShape { left, right, f, g, .. } = shape;
        let max_left = (f.left * g.left) as u128;
        let max_right = (f.right * g.right) as u128;
        let live_l = self.live_counts.at(left.idx()) as u128;
        let live_r = self.live_counts.at(right.idx()) as u128;
        SparseGate {
            available: self.arena.is_bump(),
            density_wins: max_left > 0
                && max_right > 0
                && self.sparsity_factor * live_l * live_r < max_left * max_right,
            min_grid: self.min_grid,
        }
    }

    /// Hand every pooled buffer back to the engine and return the built levels.
    ///
    /// Heavy buffers are capped at the scratch-retention cap on the way out, so a
    /// single wide conjunction cannot park GiB-scale allocations in the pools.
    pub(super) fn finish(mut self, eng: &Engine) -> Vec<TddLevel> {
        let pool = eng.apply();
        let (slab, grids) = self.arena.into_parts();
        pool.node_idx.put_bounded(slab);
        pool.grids.put(grids);
        pool.right_identity.put(self.right_identity);
        pool.left_identity.put(self.left_identity);
        for pl in &mut self.product_lists {
            crate::limits::pool::release_if_oversized(pl);
        }
        pool.product_lists.put(self.product_lists);
        self.live_counts.into_pool(&pool.live_counts);
        pool.has_pl.put(self.has_pl);
        pool.left_widths.put(self.left_widths);
        pool.right_widths.put(self.right_widths);
        pool.inputs1.put_bounded(self.inputs1_scratch);
        pool.inputs2.put_bounded(self.inputs2_scratch);
        // Same retention rule, applied to the bundle's four fields.
        self.prefilter_masks.release_oversized();
        pool.prefilter_masks.put(self.prefilter_masks);
        self.stream_cache.put(&pool.stream_cache);
        self.levels
    }
}

/// Phases 1–3 of `apply_and_fallible_inner`: width/marginal-entry snapshot,
/// sparse/budget pre-scan, grid/product-list allocation.
///
/// Returns owned scratch vectors so the caller can destructure them into the
/// same local names, leaving phases 4–6 untouched.
#[inline(always)]
#[allow(clippy::type_complexity)]
/// Snapshot both operands' per-level widths, note whether either carries a
/// marginal level at entry, and accumulate the dense-route cell count the
/// predictive budget check below reads.
///
/// All three come out of one pass. The cell sum reads exactly the two widths
/// the loop already has in registers over exactly the same range, so folding it
/// in costs nothing and saves a pass. The sparse pre-scan is deliberately not
/// fused: it ranges over the cached topo order — reachable internal nodes only
/// — which is a different set.
///
/// The widths must be read before the bottom-up sweep's identity swaps steal
/// levels, which zero `effective_width` and clear `is_marginal`.
fn snapshot_widths(
    f: &Tdd,
    g: &Tdd,
    num_nodes: usize,
    min_grid: usize,
    left_widths: &mut [usize],
    right_widths: &mut [usize],
) -> (u64, bool) {
    let mut any_entry_marginal = false;
    let mut total_cells: u64 = 0;
    for i in 0..num_nodes {
        let w1 = f.effective_width(VtreeIdx(i as u32));
        let w2 = g.effective_width(VtreeIdx(i as u32));
        left_widths[i] = w1;
        right_widths[i] = w2;
        any_entry_marginal |= f.levels[i].is_marginal() | g.levels[i].is_marginal();
        let cells = (w1 as u64).saturating_mul(w2 as u64);
        if cells <= min_grid as u64 {
            total_cells = total_cells.saturating_add(cells);
        }
    }
    (total_cells, any_entry_marginal)
}

/// Clear the per-level sparse bookkeeping of every level.
fn reset_level_tracking(
    num_nodes: usize,
    product_lists: &mut [Vec<ProductEntry>],
    has_pl: &mut [bool],
) {
    for i in 0..num_nodes {
        product_lists[i].clear();
        has_pl[i] = false;
    }
}

/// Build the product-grid arena in the shape this apply needs.
///
/// With any sparse level possible the arena bumps: every level starts
/// ungridded and claims space when it is reached. Otherwise every level's base
/// is computed up front and the slab is sized once.
fn layout_grids(
    eng: &Engine,
    might_use_sparse: bool,
    num_nodes: usize,
    left_widths: &[usize],
    right_widths: &[usize],
    grids: Vec<LevelGrid>,
) -> Result<GridArena, ApplyError> {
    let cells = eng.apply().node_idx.take();
    if might_use_sparse {
        Ok(GridArena::bump(cells, grids, 0..=num_nodes))
    } else {
        GridArena::preplanned(
            eng, cells, grids,
            (0..num_nodes)
                .map(|i| (i, left_widths[i] * right_widths[i]))
                .chain(std::iter::once((num_nodes, 0))),
        )
    }
}

/// Refuse before allocating anything if the cells this apply is *guaranteed* to
/// materialize already exceed the remaining soft budget.
///
/// `total_cells` counts dense-path levels only. A level above the sparse
/// threshold takes a conjoin that never materializes its grid, so its dense
/// width product is a worst-case fiction — it is quadratic in the level's
/// width, so one wide level alone can name more cells than any machine has
/// memory for — and counting it here would refuse over memory that is never
/// allocated. A sparse level's real cost is its surviving
/// pair count, which the soft budget still sees, just per-push at each call
/// site rather than through this predictor.
///
/// The per-cell byte factor is deliberately conservative: pairs (8B) + nodes
/// (8B) + scratch (4–8B) ≈ 24B.
///
/// The arenas themselves are deliberately not bulk-reserved anywhere near
/// here. Under `ulimit -v` that consumes address space the apply never uses —
/// Linux's lazy commit bounds RSS, but the limit measures address space — and every
/// `Vec` growth in the apply body is fallible at its own call site anyway.
///
/// # Errors
///
/// [`ApplyError::OverBudget`] when the prediction does not fit.
fn preflight_dense_budget(lim: &crate::limits::Limits, total_cells: u64) -> Result<(), ApplyError> {
    if let Some(rem) = lim.budget()
        && total_cells.saturating_mul(APPLY_BYTES_PER_CELL) > rem {
            return Err(ApplyError::OverBudget);
        }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_and_setup(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    num_nodes: usize,
    marginalize_targets: MarginalTargets<'_>,
    weighted: bool,
) -> Result<ApplyRun, ApplyError> {
    let lim = eng.limits();
    let levels: Vec<TddLevel> = diagram::take_levels(eng, num_nodes);

    let mut grids: Vec<LevelGrid> = eng.apply().grids.take();
    if grids.len() < num_nodes + 1 {
        grids.resize(num_nodes + 1, LevelGrid::Sparse);
    }

    let mut left_widths = eng.apply().left_widths.take();
    let mut right_widths = eng.apply().right_widths.take();
    if left_widths.len() < num_nodes { left_widths.resize(num_nodes, 0); }
    if right_widths.len() < num_nodes { right_widths.resize(num_nodes, 0); }
    let min_grid = eng.tuning().sparse_min_grid;
    let sparsity_factor = eng.tuning().sparse_sparsity_factor;
    let (total_cells, any_entry_marginal) = snapshot_widths(
        f, g, num_nodes, min_grid, &mut left_widths, &mut right_widths,
    );

    let entry_marginality = EntryMarginality::snapshot(f, g, num_nodes, any_entry_marginal);

    preflight_dense_budget(lim, total_cells)?;

    // With no level over the threshold, all the sparse infrastructure — product
    // lists, live counts, bump allocator — is skipped outright.
    let might_use_sparse = vtree.internal_bottomup().any(|(t, _, _)| {
        left_widths[t.idx()].saturating_mul(right_widths[t.idx()]) > min_grid
    });

    // Streaming-marginal scratch: lazily computed child columns for
    // streaming-target levels whose children are still explicit.
    let stream_cache = StreamCache::take(
        &eng.apply().stream_cache,
        num_nodes,
        marginalize_targets.any().then_some(weighted),
    );

    // Product lists, live counts, and has_pl are only used when might_use_sparse.
    let mut product_lists = eng.apply().product_lists.take();
    let live_counts = LiveCounts::take(&eng.apply().live_counts, num_nodes);
    let mut has_pl = eng.apply().has_pl.take();

    if product_lists.len() < num_nodes { product_lists.resize_with(num_nodes, Vec::new); }
    has_pl.resize(num_nodes, false);

    reset_level_tracking(num_nodes, &mut product_lists, &mut has_pl);

    let arena = layout_grids(
        eng,
        might_use_sparse, num_nodes, &left_widths, &right_widths, grids,
    )?;

    let mut inputs1_scratch: Vec<InputPair> = eng.apply().inputs1.take();
    let mut inputs2_scratch: Vec<InputPair> = eng.apply().inputs2.take();
    inputs1_scratch.clear();
    inputs2_scratch.clear();

    Ok(ApplyRun {
        levels, left_widths, right_widths,
        min_grid, sparsity_factor,
        stream_cache,
        arena,
        product_lists, live_counts, has_pl,
        entry_marginality,
        right_identity: eng.apply().right_identity.take(),
        left_identity: eng.apply().left_identity.take(),
        inputs1_scratch,
        inputs2_scratch,
        prefilter_masks: eng.apply().prefilter_masks.take(),
    })
}
