//! Apply setup: the width and marginal-entry snapshot, the sparse and budget
//! pre-scan, and the grid and product-list allocation, bundled into `ApplyRun`
//! by `apply_and_setup` for the driver to sweep with.

use crate::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::*;
use super::{liveness, OperationError};
use super::products::Products;
use super::scratch::ApplyWorkspace;
use crate::value::StreamCache;
use super::marginal_plan::EntryMarginality;
use super::sparse::{sparse_thresholds, SparseThresholds};
use super::route::{LevelMarg, SparseGate};

/// A set of vtree nodes handed to the sweep: the levels it sums out, or the
/// subtrees it collapses. A membership array, or nothing.
#[derive(Clone, Copy, Default)]
pub(crate) struct VtreeMask<'a>(Option<&'a [bool]>);

impl<'a> VtreeMask<'a> {
    /// The set `members` describes; `None` is the empty set.
    pub(crate) fn new(members: Option<&'a [bool]>) -> Self {
        VtreeMask(members)
    }

    /// Whether the set was given at all — what the streaming scratch and its
    /// pooling are conditioned on.
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self.0.is_none()
    }

    /// Whether vtree node `t_idx` is in the set.
    #[inline]
    pub(crate) fn contains(self, t_idx: usize) -> bool {
        self.0.is_some_and(|members| members[t_idx])
    }
}

/// Where the free levels ([`ApplyRun::free`]) of a conjunction lie, one byte
/// per vtree node, read where the sweep and its setup ask: which operand is
/// free at an internal node, and whether a node lies under a free level.
/// Empty when neither operand has a free level.
#[derive(Clone, Copy, Default)]
pub(super) struct Regions<'a>(&'a [u8]);

impl Regions<'_> {
    /// `f` is free at this internal node.
    const F_FREE: u8 = 1;
    /// `g` is free at this internal node.
    const G_FREE: u8 = 2;
    /// The parent of this node is free in either operand.
    const UNDER: u8 = 4;

    #[inline]
    fn role(self, t: usize) -> u8 {
        self.0.get(t).copied().unwrap_or(0)
    }

    /// Whether either operand is free at `t`: the sweep does not build it.
    #[inline]
    pub(super) fn free_at(self, t: usize) -> bool {
        self.role(t) & (Self::F_FREE | Self::G_FREE) != 0
    }

    /// Whether `f` is free at `t`.
    #[inline]
    pub(super) fn free_in_f(self, t: usize) -> bool {
        self.role(t) & Self::F_FREE != 0
    }

    /// Whether `g` is free at `t`.
    #[inline]
    pub(super) fn free_in_g(self, t: usize) -> bool {
        self.role(t) & Self::G_FREE != 0
    }

    /// Whether `t` lies under a free level, inside a region below its top:
    /// no level reads its products.
    #[inline]
    pub(super) fn under_free(self, t: usize) -> bool {
        self.role(t) & Self::UNDER != 0
    }
}

/// Bundled result of `apply_and_setup` — the per-apply working state produced
/// before the bottom-up level sweep.
pub(super) struct ApplyRun<'a, 'r> {
    pub(super) levels: &'a mut [TddLevel],
    pub(super) f_widths: &'a mut Vec<usize>,
    pub(super) g_widths: &'a mut Vec<usize>,
    /// The sparse-route thresholds this apply decides by.
    pub(super) thresholds: SparseThresholds,
    /// Lazily computed child columns for the streaming-marginal path. See
    /// [`StreamCache`].
    pub(super) stream_cache: &'a mut StreamCache,
    pub(super) products: &'a mut Products,
    /// Which levels of each operand were marginal at apply entry. See
    /// [`EntryMarginality`].
    pub(super) entry_marginality: EntryMarginality,
    /// `g_identity[t]` — g computes constant-true over subtree `t`, so f's
    /// nodes pass through unchanged. Lazily accreted, so a false reading only
    /// costs a fallback to the dense grid.
    pub(super) g_identity: &'a mut Vec<bool>,
    /// The symmetric flag for f.
    pub(super) f_identity: &'a mut Vec<bool>,
    /// Decode buffers for one cell's pairs, one per operand.
    pub(super) f_pairs_scratch: &'a mut Vec<ChildPair>,
    pub(super) g_pairs_scratch: &'a mut Vec<ChildPair>,
    /// The four dead-pair pre-filter masks, reused across internal levels.
    pub(super) prefilter_masks: &'a mut liveness::PrefilterMaskScratch,
    /// The levels the identity fast paths moved into the output, as
    /// `(level, from f)`, in the order they moved.
    pub(super) carried: Vec<(usize, bool)>,
    /// Whether the sweep must give its operands back when refused: it drops
    /// no operand level, and `carried` goes back to the operands.
    pub(super) restoring: bool,
    /// Each operand's free levels: internal nodes no variable it was placed
    /// with is under, whose levels are left empty and read as the constant
    /// true level, one node true on both sides, they stand for. The sweep
    /// takes the other operand's level at each, and builds one only where
    /// both are free or the other is the constant true level too.
    pub(super) free: Regions<'r>,
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

/// One value per operand, named by the operand rather than by a child side.
#[derive(Clone, Copy, Default)]
pub(super) struct Operands<T> {
    pub(super) f: T,
    pub(super) g: T,
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

impl ApplyRun<'_, '_> {
    /// The shape of the level at `t`, read off the entry width snapshot.
    pub(super) fn shape(&self, t: VtreeIdx, left: VtreeIdx, right: VtreeIdx) -> LevelShape {
        let (t_idx, left_idx, right_idx) = (t.idx(), left.idx(), right.idx());
        LevelShape {
            t, left, right,
            f: OperandWidths {
                here: self.f_widths[t_idx],
                left: self.f_widths[left_idx],
                right: self.f_widths[right_idx],
            },
            g: OperandWidths {
                here: self.g_widths[t_idx],
                left: self.g_widths[left_idx],
                right: self.g_widths[right_idx],
            },
        }
    }

    /// This level's marginality, in the two senses [`route_level`](super::route::route_level) needs.
    pub(super) fn level_marginal(
        &self,
        f: &Tdd,
        g: &Tdd,
        shape: LevelShape,
        targets: VtreeMask<'_>,
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
            is_target: targets.contains(t_idx),
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
            (left, self.f_widths[left] * self.g_widths[left]),
            (right, self.f_widths[right] * self.g_widths[right]),
        ]);
    }

    pub(super) fn ensure_product_list_for_child(&mut self, eng: &Engine, child: usize, f_width: usize, g_width: usize) -> Result<(), OperationError> {
        self.products.ensure_product_list_for_child(eng, child, f_width, g_width, self.g_identity[child], self.f_identity[child])
    }

    pub(super) fn materialize_dense_child(&mut self, eng: &Engine, child: usize, f_width: usize, g_width: usize) -> Result<(), OperationError> {
        self.products.materialize_dense_child(eng, child, f_width, g_width, self.g_identity[child], self.f_identity[child])
    }


}

/// What [`snapshot_widths`] reads off the operands besides their widths.
struct Snapshot {
    /// The dense-route cell count `preflight_dense_budget` reads.
    total_cells: u64,
    /// Whether either operand has a marginal level at entry.
    any_entry_marginal: bool,
    /// Whether some internal level's grid is over the sparse gate's
    /// `min_grid`.
    might_use_sparse: bool,
}

/// Snapshot both operands' per-level widths, note whether either carries a
/// marginal level at entry, sum the dense-route cell count
/// `preflight_dense_budget` reads, and say whether any internal level's grid
/// is over `min_grid`; all in one pass over the levels, which also fills
/// `roles` for [`Regions`] when an operand has free levels. A free level
/// ([`ApplyRun::free`]) has the width of the level it stands for.
///
/// Must run before the sweep's identity swaps steal levels, which zeroes
/// `reference_slot_count` and clears `is_marginal`.
fn snapshot_widths(
    f: &Tdd,
    g: &Tdd,
    free: Operands<VtreeMask<'_>>,
    min_grid: usize,
    f_widths: &mut [usize],
    g_widths: &mut [usize],
    roles: &mut Vec<u8>,
) -> Snapshot {
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    // Nodes are stored leaves first.
    let leaves = vtree.num_leaves() as usize;
    roles.clear();
    let with_regions = !(free.f.is_empty() && free.g.is_empty());
    if with_regions {
        roles.resize(num_nodes, 0);
    }
    let (f_levels, g_levels) = (&f.levels[..num_nodes], &g.levels[..num_nodes]);
    // The leaves, a constant width each.
    f_widths[..leaves].fill(LEAF_WIDTH);
    g_widths[..leaves].fill(LEAF_WIDTH);
    let mut any_entry_marginal = f_levels[..leaves].iter().zip(&g_levels[..leaves])
        .fold(false, |any, (fl, gl)| any | fl.is_marginal() | gl.is_marginal());
    let leaf_cells = (LEAF_WIDTH * LEAF_WIDTH) as u64;
    // The cells are summed saturating, which no order of the terms changes.
    let mut total_cells = match leaf_cells <= min_grid as u64 {
        true => (leaves as u64).saturating_mul(leaf_cells),
        false => 0,
    };
    // The internal nodes.
    let mut might_use_sparse = false;
    let internal = f_widths[leaves..num_nodes].iter_mut().zip(&mut g_widths[leaves..num_nodes])
        .zip(f_levels[leaves..].iter().zip(&g_levels[leaves..]));
    for (i, ((fw, gw), (fl, gl))) in (leaves..).zip(internal) {
        let (free_f, free_g) = (free.f.contains(i), free.g.contains(i));
        if with_regions && (free_f || free_g) {
            roles[i] |= (u8::from(free_f) * Regions::F_FREE) | (u8::from(free_g) * Regions::G_FREE);
            let (left, right) = vtree.children(VtreeIdx(i as u32));
            roles[left.idx()] |= Regions::UNDER;
            roles[right.idx()] |= Regions::UNDER;
        }
        let w1 = if free_f { 1 } else { fl.slot_count() };
        let w2 = if free_g { 1 } else { gl.slot_count() };
        *fw = w1;
        *gw = w2;
        any_entry_marginal |= fl.is_marginal() | gl.is_marginal();
        let cells = (w1 as u64).saturating_mul(w2 as u64);
        might_use_sparse |= cells > min_grid as u64;
        if cells <= min_grid as u64 {
            total_cells = total_cells.saturating_add(cells);
        }
    }
    Snapshot { total_cells, any_entry_marginal, might_use_sparse }
}

/// Conservative per-cell byte factor for the apply's product grid:
/// pairs (8B) + nodes (8B) + scratch (4–8B) ≈ 24B.
const APPLY_BYTES_PER_CELL: u64 = 24;

/// Refuse before allocating anything if a lower bound on the cells this apply
/// materializes already exceeds the soft budget less what the operation has
/// charged so far.
///
/// `total_cells` counts the levels at or under the sparse gate's `min_grid`,
/// which take a dense route whatever their density. It is a lower bound: a
/// level above the gate goes dense too when its products are not sparse, and
/// a level under it goes sparse when a child grid wins. A sparse level's
/// cost, the surviving pairs, is charged per push at each growth site.
/// `APPLY_BYTES_PER_CELL` is the pair, node and scratch bytes one dense cell
/// costs.
///
/// Nothing is reserved here; every arena growth in the apply is fallible at
/// its own site.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the prediction does not fit.
fn preflight_dense_budget(lim: &crate::limits::Limits, total_cells: u64) -> Result<(), OperationError> {
    if let Some(rem) = lim.budget_headroom()
        && total_cells.saturating_mul(APPLY_BYTES_PER_CELL) > rem {
            return Err(OperationError::OverBudget);
        }
    Ok(())
}

/// Build the [`ApplyRun`] for one conjunction: snapshot the operands, refuse
/// if the dense cells alone exceed the budget, and take every pooled buffer
/// the sweep needs. The [`Regions`] of the operands' free levels in `free`
/// are written to `roles`.
///
/// # Errors
///
/// [`OperationError::OverBudget`] from the dense preflight or a buffer reservation.
#[expect(clippy::too_many_arguments)]
pub(super) fn apply_and_setup<'a, 'r>(
    eng: &Engine,
    f: &Tdd,
    g: &Tdd,
    targets: VtreeMask<'_>,
    free: Operands<VtreeMask<'_>>,
    weighted: bool,
    levels: &'a mut [TddLevel],
    scratch: &'a mut ApplyWorkspace,
    roles: &'r mut Vec<u8>,
) -> Result<ApplyRun<'a, 'r>, OperationError> {
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    let lim = eng.limits();
    let thresholds = sparse_thresholds(lim);
    let min_grid = thresholds.min_grid;

    let ApplyWorkspace { f_widths, g_widths, f_identity, g_identity,
        f_pairs_scratch, g_pairs_scratch, products, stream_cache, prefilter_masks } = scratch;
    if f_widths.len() < num_nodes { f_widths.resize(num_nodes, 0); }
    if g_widths.len() < num_nodes { g_widths.resize(num_nodes, 0); }
    let Snapshot { total_cells, any_entry_marginal, might_use_sparse } = snapshot_widths(
        f, g, free, min_grid, f_widths, g_widths, roles,
    );
    let free = Regions(roles);

    let entry_marginality = EntryMarginality::snapshot(f, g, num_nodes, any_entry_marginal);

    preflight_dense_budget(lim, total_cells)?;

    // Streaming-marginal scratch: lazily computed child columns for
    // streaming-target levels whose children are still explicit.
    stream_cache.reset(num_nodes, (!targets.is_empty()).then_some(weighted));
    // With no level over the threshold, all the sparse infrastructure — product
    // lists, live counts, bump allocator — is skipped outright. A level under
    // a free level is taken whole with its region, and no level reads its
    // products.
    products.reset(eng, might_use_sparse, num_nodes, f_widths, g_widths, free)?;

    Ok(ApplyRun {
        levels, f_widths, g_widths,
        thresholds,
        stream_cache,
        products,
        entry_marginality,
        g_identity,
        f_identity,
        f_pairs_scratch,
        g_pairs_scratch,
        prefilter_masks,
        carried: Vec::new(),
        restoring: false,
        free,
    })
}

#[cfg(test)]
#[path = "tests/setup.rs"]
mod tests;
