//! Apply setup: the width and marginal-entry snapshot, the sparse and budget
//! pre-scan, and the grid and product-list allocation, bundled into `ApplyRun`
//! by `apply_and_setup` for the driver to sweep with.

use crate::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::*;
use super::{liveness, OperationError, APPLY_BYTES_PER_CELL};
use super::products::Products;
use super::scratch::ApplyWorkspace;
use crate::value::StreamCache;
use super::marginal_plan::EntryMarginality;
use super::sparse::{sparse_thresholds, SparseThresholds};
use super::route::{LevelMarg, SparseGate};

/// The vtree nodes whose levels the bottom-up sweep marginalizes.
#[derive(Clone, Copy)]
pub(crate) enum MarginalTargets<'a> {
    /// This apply marginalizes nothing.
    None,
    /// `true` at every vtree node whose level is summed out.
    At(&'a [bool]),
}

impl<'a> MarginalTargets<'a> {
    /// The set given, or `None` when none was.
    pub(crate) fn new(targets: Option<&'a [bool]>) -> Self {
        match targets {
            Some(t) => MarginalTargets::At(t),
            None => MarginalTargets::None,
        }
    }

    /// Whether anything is summed out at all — what the streaming scratch and
    /// its pooling are conditioned on.
    #[inline]
    pub(crate) fn any(self) -> bool {
        matches!(self, MarginalTargets::At(_))
    }

    /// Whether level `t_idx` is summed out. A target streams its marginal:
    /// the emit-growth mode decision and `build_stream_state`'s setup read
    /// this per level, and the commit then keys off `stream_state` being
    /// `Some` rather than re-reading the predicate.
    #[inline]
    pub(crate) fn is_target(self, t_idx: usize) -> bool {
        match self {
            MarginalTargets::None => false,
            MarginalTargets::At(t) => t[t_idx],
        }
    }
}

/// Bundled result of `apply_and_setup` — the per-apply working state produced
/// before the bottom-up level sweep.
pub(super) struct ApplyRun<'a> {
    pub(super) levels: &'a mut [TddLevel],
    pub(super) left_widths: &'a mut Vec<usize>,
    pub(super) right_widths: &'a mut Vec<usize>,
    /// The sparse-route thresholds this apply decides by.
    pub(super) thresholds: SparseThresholds,
    /// Lazily computed child columns for the streaming-marginal path. See
    /// [`StreamCache`].
    pub(super) stream_cache: &'a mut StreamCache,
    pub(super) products: &'a mut Products,
    /// Which levels of each operand were marginal at apply entry. See
    /// [`EntryMarginality`].
    pub(super) entry_marginality: EntryMarginality,
    /// `right_identity[t]` — g computes constant-true over subtree `t`, so f's
    /// nodes pass through unchanged. Lazily accreted, so a false reading only
    /// costs a fallback to the dense grid.
    pub(super) right_identity: &'a mut Vec<bool>,
    /// The symmetric flag for f.
    pub(super) left_identity: &'a mut Vec<bool>,
    /// Decode buffers for one cell's pairs, one per operand.
    pub(super) inputs1_scratch: &'a mut Vec<ChildPair>,
    pub(super) inputs2_scratch: &'a mut Vec<ChildPair>,
    /// The four dead-pair pre-filter masks, reused across internal levels.
    pub(super) prefilter_masks: &'a mut liveness::PrefilterMaskScratch,
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

impl ApplyRun<'_> {
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
    ///
    /// The two routes also differ in what they do to the children first. A
    /// child the sparse pipeline built has a product list and no grid, and a
    /// dense route here fills that grid before its own, at the product of the
    /// operands' widths there; a child the dense pipeline built has a grid and
    /// no product list, and the sparse route scans every cell of it for one.
    /// `child_grid_wins` says whether the grids the dense route would fill
    /// outweigh the grids the sparse route would scan — and at least one of
    /// the former is over the size floor and sparse against its live products
    /// — so that the dense route's setup alone loses to the scatter walk,
    /// however small this level's own grid is.
    pub(super) fn sparse_gate(&self, shape: LevelShape) -> SparseGate {
        let LevelShape { left, right, f, g, .. } = shape;
        let max_left = (f.left * g.left) as u128;
        let max_right = (f.right * g.right) as u128;
        let live_l = self.products.live(left.idx()) as u128;
        let live_r = self.products.live(right.idx()) as u128;
        let factor = self.thresholds.sparsity_factor;
        let min_grid = self.thresholds.min_grid;
        let ungridded = |child: VtreeIdx| self.products.arena.is_sparse(child.idx());
        let wide_and_sparse = |child: VtreeIdx, max: u128, live: u128| {
            ungridded(child) && max > min_grid as u128 && factor * live < max
        };
        let split = |child: VtreeIdx, max: u128| if ungridded(child) { (max, 0) } else { (0, max) };
        let (fill_l, scan_l) = split(left, max_left);
        let (fill_r, scan_r) = split(right, max_right);
        SparseGate {
            available: self.products.arena.is_bump(),
            density_wins: max_left > 0
                && max_right > 0
                && factor * live_l * live_r < max_left * max_right,
            child_grid_wins: (wide_and_sparse(left, max_left, live_l)
                || wide_and_sparse(right, max_right, live_r))
                && fill_l + fill_r > scan_l + scan_r,
            min_grid,
        }
    }

    pub(super) fn reclaim_child_grids(&mut self, left: usize, right: usize) {
        self.products.reclaim_children([
            (left, self.left_widths[left] * self.right_widths[left]),
            (right, self.left_widths[right] * self.right_widths[right]),
        ]);
    }

    pub(super) fn ensure_product_list_for_child(&mut self, eng: &Engine, t: usize, left: usize, right: usize) -> Result<(), OperationError> {
        self.products.ensure_product_list_for_child(eng, t, left, right, self.right_identity[t], self.left_identity[t])
    }

    pub(super) fn materialize_dense_child(&mut self, eng: &Engine, t: usize, left: usize, right: usize) -> Result<(), OperationError> {
        self.products.materialize_dense_child(eng, t, left, right, self.right_identity[t], self.left_identity[t])
    }


}

/// Snapshot both operands' per-level widths, note whether either carries a
/// marginal level at entry, and sum the dense-route cell count
/// `preflight_dense_budget` reads; all three in one pass over the levels.
///
/// Must run before the sweep's identity swaps steal levels, which zeroes
/// `reference_slot_count` and clears `is_marginal`.
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
        let w1 = f.reference_slot_count(VtreeIdx(i as u32));
        let w2 = g.reference_slot_count(VtreeIdx(i as u32));
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

/// Refuse before allocating anything if the cells this apply is *guaranteed* to
/// materialize already exceed the remaining soft budget.
///
/// `total_cells` counts levels at or under the sparse threshold only: a level
/// above it never materializes its grid, and its cost, the surviving pairs, is
/// charged per push at each growth site. `APPLY_BYTES_PER_CELL` is the pair,
/// node and scratch bytes one dense cell costs.
///
/// Nothing is reserved here; every arena growth in the apply is fallible at
/// its own site.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the prediction does not fit.
fn preflight_dense_budget(lim: &crate::limits::Limits, total_cells: u64) -> Result<(), OperationError> {
    if let Some(rem) = lim.budget()
        && total_cells.saturating_mul(APPLY_BYTES_PER_CELL) > rem {
            return Err(OperationError::OverBudget);
        }
    Ok(())
}

/// Build the [`ApplyRun`] for one conjunction: snapshot the operands, refuse
/// if the dense cells alone exceed the budget, and take every pooled buffer
/// the sweep needs.
///
/// # Errors
///
/// [`OperationError::OverBudget`] from the dense preflight or a buffer reservation.
pub(super) fn apply_and_setup<'a>(
    eng: &Engine,
    f: &Tdd,
    g: &Tdd,
    marginalize_targets: MarginalTargets<'_>,
    weighted: bool,
    levels: &'a mut [TddLevel],
    scratch: &'a mut ApplyWorkspace,
) -> Result<ApplyRun<'a>, OperationError> {
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    let lim = eng.limits();
    let thresholds = sparse_thresholds();
    let min_grid = thresholds.min_grid;

    let ApplyWorkspace { left_widths, right_widths, left_identity, right_identity,
        inputs1_scratch, inputs2_scratch, products, stream_cache, prefilter_masks } = scratch;
    if left_widths.len() < num_nodes { left_widths.resize(num_nodes, 0); }
    if right_widths.len() < num_nodes { right_widths.resize(num_nodes, 0); }
    let (total_cells, any_entry_marginal) = snapshot_widths(
        f, g, num_nodes, min_grid, left_widths, right_widths,
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
    stream_cache.reset(num_nodes, marginalize_targets.any().then_some(weighted));
    products.reset(eng, might_use_sparse, num_nodes, left_widths, right_widths)?;

    Ok(ApplyRun {
        levels, left_widths, right_widths,
        thresholds,
        stream_cache,
        products,
        entry_marginality,
        right_identity,
        left_identity,
        inputs1_scratch,
        inputs2_scratch,
        prefilter_masks,
    })
}
