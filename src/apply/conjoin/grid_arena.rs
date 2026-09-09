//! The flat product-grid arena: where each level's grid lives, and how space
//! for it is claimed and reclaimed.
//!
//! Every level's product grid is a slice of one flat `Vec<u32>`: cell `(i, j)`
//! of level `t` sits at `base(t) + i * right_width[t] + j` and holds the output index
//! for `f[i] ∧ g[j]`, or `DEAD` where that product was zero. `u32` rather
//! than `u16` because widths pass 65k on hard instances.
//!
//! The arena has two shapes, and they differ in every operation, so they are
//! two variants of one type rather than one code path steered by a flag:
//!
//! * [`GridArena::Preplanned`] — no level can go sparse, so every level's base
//!   is computed up front and the slab is sized once. Nothing is ever
//!   reclaimed: the whole layout is live until the apply ends.
//! * [`GridArena::Bump`] — some level may take the sparse route and skip its
//!   grid, so space is claimed level by level and released again as soon as a
//!   level's single parent has consumed it. This caps the slab at the live
//!   frontier instead of the sum of every densified level.
//!
//! Correctness of the reclaim rests on the single-consumer invariant: each
//! vtree node's grid is read by exactly its one parent and is dead afterwards,
//! so a region is freed exactly once and never while still read. The root grid
//! is never freed (the root has no parent) and is the only grid the output
//! computation reads. Reused regions are always DEAD-filled before use.
//!
//! VALIDITY. A cell holds a meaningful value only where its producer wrote
//! one. The dense routes fill a level's whole grid, `DEAD` included; the sparse
//! route writes only the cells its scatter produced and leaves the rest as
//! whatever the region's previous tenant left. So a read is sound only for a
//! level whose [`LevelGrid`] says the grid was materialized — which is what
//! [`GridArena::materialized`] returns and what every consumer goes through.

use crate::engine::Engine;
use super::{ApplyError, LevelGrid, DEAD};
use super::budget::try_resize_dead;
use super::setup::ApplyRun;
use super::sparse::{ProductEntry, LeftNodeIdx, RightNodeIdx, ProductNodeIdx, fill_identity_product_list};

/// Offset of a level's grid within the arena's flat slab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct GridBase(usize);

impl GridBase {
    /// The offset as an index into the slab.
    #[inline(always)]
    pub(super) fn idx(self) -> usize { self.0 }
}

/// A stretch of the slab that is not currently any level's grid.
#[derive(Clone, Copy)]
pub(in crate::apply::conjoin) struct Region {
    base: GridBase,
    len: usize,
}

/// The flat product-grid slab and every level's claim on it.
pub(super) enum GridArena {
    /// Whole layout computed up front; no level goes sparse and nothing is
    /// reclaimed.
    Preplanned { cells: Vec<u32>, grids: Vec<LevelGrid> },
    /// Space claimed and released as the sweep proceeds.
    Bump { cells: Vec<u32>, end: usize, free: Vec<Region>, grids: Vec<LevelGrid> },
}

impl GridArena {
    /// The pre-planned arena: every level gets the base its width dictates and
    /// the slab is sized once to hold them all.
    pub(super) fn preplanned(
        eng: &Engine,
        mut cells: Vec<u32>,
        mut grids: Vec<LevelGrid>,
        layout: impl Iterator<Item = (usize, usize)>,
    ) -> Result<Self, ApplyError> {
        let mut cursor = 0usize;
        for (i, level_cells) in layout {
            grids[i] = LevelGrid::Dense { base: cursor };
            cursor += level_cells;
        }
        try_resize_dead(eng, &mut cells, cursor)?;
        Ok(GridArena::Preplanned { cells, grids })
    }

    /// The bump arena: every touched level starts ungridded.
    pub(super) fn bump(
        cells: Vec<u32>,
        mut grids: Vec<LevelGrid>,
        touched: impl Iterator<Item = usize>,
    ) -> Self {
        for i in touched {
            grids[i] = LevelGrid::Sparse;
        }
        GridArena::Bump { cells, end: 0, free: Vec::new(), grids }
    }

    /// True when levels claim space as they are reached — the one behavioural
    /// difference the sweep still has to ask about, because a level's live
    /// count is only maintained (and only read) in that shape.
    #[inline(always)]
    pub(super) fn is_bump(&self) -> bool {
        matches!(self, GridArena::Bump { .. })
    }

    /// The flat slab every grid is a slice of.
    #[inline(always)]
    pub(super) fn slab(&self) -> &[u32] {
        match self {
            GridArena::Preplanned { cells, .. } | GridArena::Bump { cells, .. } => cells,
        }
    }

    /// The flat slab, for a producer writing its level's cells.
    #[inline(always)]
    pub(super) fn slab_mut(&mut self) -> &mut [u32] {
        match self {
            GridArena::Preplanned { cells, .. } | GridArena::Bump { cells, .. } => cells,
        }
    }

    /// Give the slab and the per-level descriptors back to the engine's pools.
    pub(super) fn into_parts(self) -> (Vec<u32>, Vec<LevelGrid>) {
        match self {
            GridArena::Preplanned { cells, grids }
            | GridArena::Bump { cells, grids, .. } => (cells, grids),
        }
    }

    #[inline(always)]
    fn grids(&self) -> &[LevelGrid] {
        match self {
            GridArena::Preplanned { grids, .. } | GridArena::Bump { grids, .. } => grids,
        }
    }

    #[inline(always)]
    fn grids_mut(&mut self) -> &mut [LevelGrid] {
        match self {
            GridArena::Preplanned { grids, .. } | GridArena::Bump { grids, .. } => grids,
        }
    }

    /// Level `t`'s grid base, or `None` if it has none — the one way to learn
    /// whether a level's cells may be read.
    #[inline(always)]
    pub(super) fn materialized(&self, t: usize) -> Option<GridBase> {
        self.grids()[t].base().map(GridBase)
    }

    /// True when level `t` has no grid, so consumers must go through its
    /// product list.
    #[inline(always)]
    pub(super) fn is_sparse(&self, t: usize) -> bool {
        self.grids()[t].is_sparse()
    }

    /// Record that level `t`'s grid has been materialized by a producer.
    #[inline(always)]
    pub(super) fn set_dense(&mut self, t: usize, base: GridBase) {
        self.grids_mut()[t] = LevelGrid::Dense { base: base.0 };
    }

    /// Record that level `t`'s grid holds the fixed leaf conjunction table.
    #[inline(always)]
    pub(super) fn set_leaf(&mut self, t: usize, base: GridBase) {
        self.grids_mut()[t] = LevelGrid::Leaf { base: base.0 };
    }

    /// Record that level `t` has no grid.
    #[inline(always)]
    pub(super) fn set_sparse(&mut self, t: usize) {
        self.grids_mut()[t] = LevelGrid::Sparse;
    }

    /// Claim `cells` of slab for level `t`.
    ///
    /// Pre-planned, that is a lookup: the base was decided at setup. Bumping,
    /// a freed region is reused when one fits (best-fit, to curb
    /// fragmentation) and the frontier grows only on a miss.
    pub(super) fn alloc(
        &mut self,
        eng: &Engine,
        t: usize,
        cells: usize,
    ) -> Result<GridBase, ApplyError> {
        match self {
            GridArena::Preplanned { grids, .. } => Ok(GridBase(grids[t].base_unchecked())),
            GridArena::Bump { cells: slab, end, free, .. } => {
                if cells == 0 {
                    // Zero-width grid: never read or freed meaningfully; hand
                    // out the cursor without growing.
                    return Ok(GridBase(*end));
                }
                // Best-fit: smallest free region that still fits. <500 regions,
                // so linear.
                let mut best: Option<usize> = None;
                for (i, r) in free.iter().enumerate() {
                    if r.len >= cells && best.is_none_or(|b| r.len < free[b].len) {
                        best = Some(i);
                    }
                }
                if let Some(i) = best {
                    let Region { base, len } = free[i];
                    if len == cells {
                        free.swap_remove(i);
                    } else {
                        free[i] = Region { base: GridBase(base.0 + cells), len: len - cells };
                    }
                    return Ok(base);
                }
                let base = *end;
                *end += cells;
                try_resize_dead(eng, slab, *end)?;
                Ok(GridBase(base))
            }
        }
    }

    /// Return a consumed region, coalescing with physically adjacent free
    /// regions to limit fragmentation. A no-op on a pre-planned layout, which
    /// owns one contiguous block and must not be freed piecemeal.
    pub(super) fn free(&mut self, base: GridBase, cells: usize) {
        let GridArena::Bump { free, .. } = self else { return };
        if cells == 0 { return; }
        let mut region = Region { base, len: cells };
        // Repeatedly absorb a neighbor on either side (at most a left and a right).
        let mut merged = true;
        while merged {
            merged = false;
            for i in 0..free.len() {
                let n = free[i];
                if n.base.0 + n.len == region.base.0 {
                    region = Region { base: n.base, len: region.len + n.len };
                } else if region.base.0 + region.len == n.base.0 {
                    region.len += n.len;
                } else {
                    continue;
                }
                free.swap_remove(i);
                merged = true;
                break;
            }
        }
        free.push(region);
    }

    /// Release child level `v`'s grid (if it has one) and mark it ungridded, so
    /// its region can be reused and no dangling base remains.
    pub(super) fn free_child(&mut self, v: usize, cells: usize) {
        if !self.is_bump() { return; }
        if let Some(base) = self.materialized(v) {
            self.free(base, cells);
            self.set_sparse(v);
        }
    }

    /// Ensure level `ti` has a grid: claim space, DEAD-fill it, and populate it
    /// from `product_list` (which the caller has already built).
    pub(super) fn ensure_grid(
        &mut self,
        eng: &Engine,
        ti: usize, k1: usize, right_width: usize,
        product_list: &[ProductEntry],
    ) -> Result<(), ApplyError> {
        if !self.is_sparse(ti) { return Ok(()); }
        let cells = k1 * right_width;
        let base = self.alloc(eng, ti, cells)?.idx();
        self.set_dense(ti, GridBase(base));
        let slab = self.slab_mut();
        slab[base..base + cells].fill(DEAD);
        for &ProductEntry { c1_idx, c2_idx, prod_idx } in product_list {
            slab[base + c1_idx.idx() * right_width + c2_idx.idx()] = prod_idx.0;
        }
        Ok(())
    }

    /// Ensure level `ti` has a product list, scanning its grid if not.
    pub(super) fn ensure_product_list(
        &self,
        eng: &Engine,
        ti: usize, k1: usize, right_width: usize,
        product_list: &mut Vec<ProductEntry>, has_pl: &mut [bool],
    ) -> Result<(), ApplyError> {
        let lim = eng.limits();
        if has_pl[ti] { return Ok(()); }
        has_pl[ti] = true;
        let base = self.materialized(ti).expect("expected allocated grid, found Sparse").idx();
        let slab = self.slab();
        for i in 0..k1 {
            for j in 0..right_width {
                let idx = slab[base + i * right_width + j];
                if idx != DEAD {
                    lim.try_push(product_list, ProductEntry {
                        c1_idx: LeftNodeIdx(i as u32),
                        c2_idx: RightNodeIdx(j as u32),
                        prod_idx: ProductNodeIdx(idx),
                    })?;
                }
            }
        }
        Ok(())
    }
}

impl ApplyRun {
    /// Reclaim the two consumed child grids — dead once this level is built
    /// (each node has exactly one parent). Invoked at every level-finishing
    /// exit.
    #[inline(always)]
    pub(super) fn reclaim_child_grids(&mut self, left_idx: usize, right_idx: usize) {
        for v in [left_idx, right_idx] {
            self.arena.free_child(v, self.c1_widths[v] * self.c2_widths[v]);
        }
    }

    /// Ensure `product_lists[ci]` is populated. Tries the cheap identity fast
    /// path first (constant-true operand → the product list is just the
    /// non-identity operand's nodes); falls back to scanning the dense grid.
    /// Used on both the sparse and dense paths of the level loop.
    pub(super) fn ensure_product_list_for_child(
        &mut self,
        eng: &Engine,
        ci: usize, k1: usize, right_width: usize,
    ) -> Result<(), ApplyError> {
        if self.has_pl[ci] { return Ok(()); }
        if !fill_identity_product_list(
            eng,
            k1, right_width,
            self.c2_identity[ci], self.c1_identity[ci],
            &mut self.product_lists[ci],
            &mut self.has_pl[ci],
        )? {
            self.arena.ensure_product_list(
                eng, ci, k1, right_width, &mut self.product_lists[ci], &mut self.has_pl,
            )?;
        }
        Ok(())
    }

    /// Materialize one child on the dense-parent path: build its product list
    /// if not already built, then grid it.
    ///
    /// This is the dense path, so the child is known to be ungridded — the
    /// identity fast path is the only way to build its product list.
    pub(super) fn materialize_dense_child(
        &mut self,
        eng: &Engine,
        idx: usize,
        k1c: usize,
        k2c: usize,
    ) -> Result<(), ApplyError> {
        if !self.has_pl[idx] {
            fill_identity_product_list(
                eng, k1c, k2c,
                self.c2_identity[idx], self.c1_identity[idx],
                &mut self.product_lists[idx], &mut self.has_pl[idx],
            )?;
            self.has_pl[idx] = true;
        }
        self.arena.ensure_grid(eng, idx, k1c, k2c, &self.product_lists[idx])
    }
}
