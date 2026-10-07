//! Apply setup: the levels a conjunction visits, their width and
//! marginal-entry snapshot, the sparse and budget pre-scan, and the grid and
//! product-list allocation, bundled into `ApplyRun` by `apply_and_setup` for
//! the driver to sweep with.

use crate::Engine;
use crate::vtree::{Vtree, VtreeIdx};
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

/// The levels a conjunction visits: every level but those under a free
/// level ([`ApplyRun::cone`]), which the result takes whole with its region
/// and no other level reads. With no free level that is every level, read
/// off the vtree; otherwise the setup lists them in one pass over the
/// internal levels, which also counts the nodes of the free levels.
#[derive(Clone, Copy)]
pub(super) struct Cone<'a> {
    /// The internal levels no operand is free at, bottom-up: the levels the
    /// sweep builds or carries as identity levels.
    pub(super) built: &'a [VtreeIdx],
    /// For each level of `built`, the node count of the free levels before it
    /// bottom-up; empty when no level is free.
    free_before: &'a [u64],
    /// The free levels whose parent is not free, the tops of the regions.
    pub(super) tops: &'a [VtreeIdx],
    /// Every free internal level, its region's top included, bottom-up.
    pub(super) free: &'a [VtreeIdx],
    /// The leaves under no free level.
    pub(super) leaves: &'a [VtreeIdx],
    /// The node count of the free levels, as the result holds them.
    pub(super) free_nodes: u64,
    /// Each operand's free levels.
    masks: Operands<VtreeMask<'a>>,
}

impl Cone<'_> {
    /// Whether either operand is free at `t`: the sweep does not build it.
    #[inline]
    pub(super) fn free_at(self, t: usize) -> bool {
        self.masks.f.contains(t) || self.masks.g.contains(t)
    }

    /// Whether `f` is free at `t`.
    #[inline]
    pub(super) fn free_in_f(self, t: usize) -> bool {
        self.masks.f.contains(t)
    }

    /// Whether `g` is free at `t`.
    #[inline]
    pub(super) fn free_in_g(self, t: usize) -> bool {
        self.masks.g.contains(t)
    }

    /// The node count of the free levels before the `k`-th level of `built`
    /// bottom-up.
    #[inline]
    pub(super) fn free_before(self, k: usize) -> u64 {
        self.free_before.get(k).copied().unwrap_or(0)
    }

    /// Every level the conjunction visits: the leaves, the built levels and
    /// the tops.
    pub(super) fn nodes(self) -> impl Iterator<Item = usize> {
        self.leaves.iter().chain(self.built).chain(self.tops).map(|t| t.idx())
    }
}

/// The lists a [`Cone`] reads when some level is free, kept between
/// conjunctions in the [`ApplyWorkspace`].
#[derive(Default)]
pub(super) struct ConeLists {
    built: Vec<VtreeIdx>,
    free_before: Vec<u64>,
    tops: Vec<VtreeIdx>,
    free: Vec<VtreeIdx>,
    leaves: Vec<VtreeIdx>,
}

impl ConeLists {
    pub(super) fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.built);
        visit(&mut self.free_before);
        visit(&mut self.tops);
        visit(&mut self.free);
        visit(&mut self.leaves);
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
    /// The levels the identity fast paths and the relabelling route moved
    /// into the output, as `(level, from f)`, in the order they moved.
    pub(super) carried: Vec<(usize, bool)>,
    /// Those of `carried` the relabelling route moved: the level is its
    /// carrier's, but a subtree under it is not, so the output owes it as a
    /// level it built.
    pub(super) relabel_moved: Vec<usize>,
    /// Whether the sweep must give its operands back when refused: it drops
    /// no operand level, and `carried` goes back to the operands.
    pub(super) restoring: bool,
    /// Each operand's free levels: internal nodes no variable it was placed
    /// with is under, whose levels are left empty and read as the constant
    /// true level, one node true on both sides, they stand for. The result
    /// takes the other operand's level at each, a free level where both are
    /// free, before the sweep; the conjunction visits only the levels of the
    /// cone, which no free level is over.
    pub(super) cone: Cone<'r>,
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
    ///
    /// Each test runs only where `route_level` reads its answer, and is false
    /// elsewhere: the child grids' where the routes are available and this
    /// level's own grid is at most `min_grid`, and only past a child built
    /// sparse; the density test where the grid is big. Most levels are small
    /// over dense children and run neither.
    pub(super) fn sparse_gate(&self, shape: LevelShape) -> SparseGate {
        let LevelShape { left, right, f, g, .. } = shape;
        let (max_left, max_right) = (f.left * g.left, f.right * g.right);
        let live = |child: VtreeIdx| self.products.live(child.idx()) as u128;
        let factor = self.thresholds.sparsity_factor;
        let min_grid = self.thresholds.min_grid;
        let ungridded = |child: VtreeIdx| self.products.arena.is_sparse(child.idx());
        let child_grids = || {
            let wide_and_sparse = |child: VtreeIdx, max: usize| {
                ungridded(child) && max > min_grid && factor * live(child) < max as u128
            };
            if !(wide_and_sparse(left, max_left) || wide_and_sparse(right, max_right)) {
                return false;
            }
            let split = |child: VtreeIdx, max: usize| {
                if ungridded(child) { (max as u128, 0) } else { (0, max as u128) }
            };
            let ((fill_l, scan_l), (fill_r, scan_r)) = (split(left, max_left), split(right, max_right));
            fill_l + fill_r > scan_l + scan_r
        };
        let available = self.products.arena.is_bump();
        let own_grid = f.here * g.here > min_grid;
        let child_grid_wins = available && !own_grid && child_grids();
        SparseGate {
            available,
            density_wins: available
                && (own_grid || child_grid_wins)
                && max_left > 0
                && max_right > 0
                && factor * live(left) * live(right) < max_left as u128 * max_right as u128,
            child_grid_wins,
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

/// What [`survey`] reads off the operands besides their widths.
struct Snapshot {
    /// The dense-route cell count `preflight_dense_budget` reads.
    total_cells: u64,
    /// Whether either operand has a marginal level at entry.
    any_entry_marginal: bool,
    /// Whether some internal level's grid is over the sparse gate's
    /// `min_grid`.
    might_use_sparse: bool,
}

/// Find the levels a conjunction visits ([`Cone`]), listing them in
/// `lists` when some level is free, and snapshot both operands' widths
/// there; note whether either operand carries a marginal level at entry,
/// sum the dense-route cell count `preflight_dense_budget` reads, and say
/// whether any internal level's grid is over `min_grid`: all in one pass
/// over the internal levels.
///
/// A free level has width 1 in its operand, so its cells are the nodes of
/// the level the result takes there: the other operand's, or one node where
/// both are free. Only an operand with no marginal level is given free
/// levels ([`apply_and_kept`](super::drive::apply_and_kept)), and a free
/// level is empty, so no leaf under a free level is read for marginality.
///
/// Must run before the sweep's identity swaps steal levels, which zeroes
/// `reference_slot_count` and clears `is_marginal`.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a list cannot grow.
#[expect(clippy::too_many_arguments)]
fn survey<'r>(
    eng: &Engine,
    vtree: &'r Vtree,
    f: &Tdd,
    g: &Tdd,
    masks: Operands<VtreeMask<'r>>,
    min_grid: usize,
    f_widths: &mut [usize],
    g_widths: &mut [usize],
    lists: &'r mut ConeLists,
) -> Result<(Snapshot, Cone<'r>), OperationError> {
    let internal = vtree.internal_bottomup_slice();
    let leaf_cells = (LEAF_WIDTH * LEAF_WIDTH) as u64;
    // The cells are summed saturating, which no order of the terms changes.
    let mut total_cells = match leaf_cells <= min_grid as u64 {
        true => u64::from(vtree.num_leaves()).saturating_mul(leaf_cells),
        false => 0,
    };
    let mut might_use_sparse = false;
    let mut any_entry_marginal = false;
    let mut cells = |w1: usize, w2: usize| {
        let cells = (w1 as u64).saturating_mul(w2 as u64);
        might_use_sparse |= cells > min_grid as u64;
        if cells <= min_grid as u64 {
            total_cells = total_cells.saturating_add(cells);
        }
        cells
    };
    // A level no operand is free at, and a leaf, read off both operands.
    let mut level = |t: VtreeIdx, f_widths: &mut [usize], g_widths: &mut [usize]| {
        let (fl, gl) = (&f.levels[t.idx()], &g.levels[t.idx()]);
        any_entry_marginal |= fl.is_marginal() | gl.is_marginal();
        let (w1, w2) = match vtree.node(t).is_leaf() {
            true => (LEAF_WIDTH, LEAF_WIDTH),
            false => (fl.slot_count(), gl.slot_count()),
        };
        f_widths[t.idx()] = w1;
        g_widths[t.idx()] = w2;
        (w1, w2)
    };
    if masks.f.is_empty() && masks.g.is_empty() {
        let leaves = vtree.leaf_bottomup_slice();
        for &t in leaves {
            level(t, f_widths, g_widths);
        }
        for &t in internal {
            let (w1, w2) = level(t, f_widths, g_widths);
            cells(w1, w2);
        }
        let cone = Cone { built: internal, free_before: &[], tops: &[], free: &[], leaves, free_nodes: 0, masks };
        let snapshot = Snapshot { total_cells, any_entry_marginal, might_use_sparse };
        return Ok((snapshot, cone));
    }
    let lim = eng.limits();
    let ConeLists { built, free_before, tops, free, leaves } = lists;
    built.clear();
    free_before.clear();
    tops.clear();
    free.clear();
    leaves.clear();
    lim.reserve(built, internal.len())?;
    lim.reserve(free_before, internal.len())?;
    lim.reserve(tops, internal.len())?;
    lim.reserve(free, internal.len())?;
    lim.reserve(leaves, vtree.num_leaves() as usize)?;
    let free_at = |t: VtreeIdx| masks.f.contains(t.idx()) || masks.g.contains(t.idx());
    let mut free_nodes = 0u64;
    for &t in internal {
        let at = t.idx();
        let (free_f, free_g) = (masks.f.contains(at), masks.g.contains(at));
        if free_f || free_g {
            debug_assert!(
                !f.levels[at].is_marginal() && !g.levels[at].is_marginal(),
                "an operand with a free level has a marginal level at {at}",
            );
            let w1 = if free_f { 1 } else { f.levels[at].slot_count() };
            let w2 = if free_g { 1 } else { g.levels[at].slot_count() };
            f_widths[at] = w1;
            g_widths[at] = w2;
            free_nodes = free_nodes.saturating_add(cells(w1, w2));
            free.push(t);
            continue;
        }
        let (w1, w2) = level(t, f_widths, g_widths);
        cells(w1, w2);
        built.push(t);
        free_before.push(free_nodes);
        let (left, right) = vtree.children(t);
        for child in [left, right] {
            if vtree.node(child).is_leaf() {
                level(child, f_widths, g_widths);
                leaves.push(child);
            } else if free_at(child) {
                tops.push(child);
            }
        }
    }
    // A free root is the top of its region, under no built level; a leaf
    // root is the one level.
    let root = vtree.root();
    if vtree.node(root).is_leaf() {
        level(root, f_widths, g_widths);
        leaves.push(root);
    } else if free_at(root) {
        tops.push(root);
    }
    let cone = Cone { built, free_before, tops, free, leaves, free_nodes, masks };
    let snapshot = Snapshot { total_cells, any_entry_marginal, might_use_sparse };
    Ok((snapshot, cone))
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

/// Build the [`ApplyRun`] for one conjunction: find the levels it visits
/// and snapshot the operands there, refuse if the dense cells alone exceed
/// the budget, and take every pooled buffer the sweep needs. Each operand's
/// free levels are in `free`.
///
/// # Errors
///
/// [`OperationError::OverBudget`] from the dense preflight or a buffer reservation.
#[expect(clippy::too_many_arguments)]
pub(super) fn apply_and_setup<'a, 'r: 'a>(
    eng: &Engine,
    vtree: &'r Vtree,
    f: &Tdd,
    g: &Tdd,
    targets: VtreeMask<'_>,
    free: Operands<VtreeMask<'r>>,
    weighted: bool,
    levels: &'a mut [TddLevel],
    scratch: &'r mut ApplyWorkspace,
) -> Result<ApplyRun<'a, 'r>, OperationError> {
    let num_nodes = vtree.num_nodes();
    let lim = eng.limits();
    let thresholds = sparse_thresholds(lim);
    let min_grid = thresholds.min_grid;

    let ApplyWorkspace { f_widths, g_widths, f_identity, g_identity,
        f_pairs_scratch, g_pairs_scratch, products, stream_cache, prefilter_masks, cone } = scratch;
    if f_widths.len() < num_nodes { f_widths.resize(num_nodes, 0); }
    if g_widths.len() < num_nodes { g_widths.resize(num_nodes, 0); }
    let (Snapshot { total_cells, any_entry_marginal, might_use_sparse }, cone) = survey(
        eng, vtree, f, g, free, min_grid, f_widths, g_widths, cone,
    )?;

    let entry_marginality = EntryMarginality::snapshot(f, g, num_nodes, any_entry_marginal);

    preflight_dense_budget(lim, total_cells)?;

    // Streaming-marginal scratch: lazily computed child columns for
    // streaming-target levels whose children are still explicit.
    stream_cache.reset(num_nodes, (!targets.is_empty()).then_some(weighted));
    // With no level over the threshold, all the sparse infrastructure — product
    // lists, live counts, bump allocator — is skipped outright. A level under
    // a free level is taken whole with its region, and no level reads its
    // products.
    products.reset(eng, might_use_sparse, num_nodes, f_widths, g_widths, cone)?;

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
        relabel_moved: Vec::new(),
        restoring: false,
        cone,
    })
}

#[cfg(test)]
#[path = "tests/setup.rs"]
mod tests;
