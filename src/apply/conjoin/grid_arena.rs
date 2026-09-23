//! The flat product-grid arena: where each level's grid lives, and how space
//! for it is claimed and reclaimed.
//!
//! Every level's product grid is a slice of one flat `Vec<u32>`: cell `(i, j)`
//! of level `t` sits at `base(t) + i * right_width[t] + j` and holds the output index
//! for `f[i] ∧ g[j]`, or `NO_PRODUCT` where that product was zero. `u32` rather
//! than `u16` because widths pass 65 535.
//!
//! The arena has two shapes:
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
//! computation reads. Reused regions are always `NO_PRODUCT`-filled before use.
//!
//! A cell holds a meaningful value only where its producer wrote
//! one. The dense routes fill a level's whole grid, `NO_PRODUCT` included; the sparse
//! route writes only the cells its scatter produced and leaves the rest as
//! whatever the region's previous tenant left. So a read is sound only for a
//! level whose [`LevelGrid`] says the grid was materialized — which is what
//! [`GridArena::materialized`] returns and what every consumer goes through.

use crate::Engine;
use super::{OperationError, NO_PRODUCT};
use super::budget::try_resize_dead;
use super::sparse::{ProductEntry, LeftNodeIdx, RightNodeIdx, ProductNodeIdx};

/// Per-level descriptor for the product grid's slice of the flat arena.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LevelGrid {
    /// No grid allocated; consumers must go through `product_lists[t]`.
    Sparse,
    /// Grid allocated at `base` and filled — by the leaf `CONJOIN_GRID` lookup
    /// or by one of the internal-level producers (the scatter from a product
    /// list, the sequential dense emit, an identity shortcut).
    Materialized { base: usize },
}

impl LevelGrid {
    /// Base offset into the flat `node_idx` arena, or `None` if unallocated.
    #[inline]
    pub(crate) fn base(&self) -> Option<usize> {
        match self {
            LevelGrid::Sparse => None,
            LevelGrid::Materialized { base } => Some(*base),
        }
    }

    /// Base offset; panics if the grid is not allocated.
    #[inline]
    pub(crate) fn base_unchecked(&self) -> usize {
        self.base().expect("expected allocated grid, found Sparse")
    }

    /// Whether no grid is allocated for this level.
    #[inline]
    pub(crate) fn is_sparse(&self) -> bool { matches!(self, LevelGrid::Sparse) }
}

/// Offset of a level's grid within the arena's flat slab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct GridBase(usize);

impl GridBase {
    /// The offset as an index into the slab.
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

impl Default for GridArena {
    fn default() -> Self { Self::Preplanned { cells: Vec::new(), grids: Vec::new() } }
}

impl GridArena {
    pub(super) fn reset(&mut self, eng: &Engine, sparse: bool, n: usize, left: &[usize], right: &[usize]) -> Result<(), OperationError> {
        let (cells, mut grids) = match std::mem::take(self) {
            Self::Preplanned { cells, grids } | Self::Bump { cells, grids, .. } => (cells, grids),
        };
        grids.resize(n + 1, LevelGrid::Sparse);
        if sparse {
            grids.fill(LevelGrid::Sparse);
            *self = Self::Bump { cells, grids, end: 0, free: Vec::new() };
        } else {
            let mut cursor = 0;
            for i in 0..n {
                grids[i] = LevelGrid::Materialized { base: cursor };
                cursor += left[i] * right[i];
            }
            grids[n] = LevelGrid::Materialized { base: cursor };
            *self = Self::Preplanned { cells, grids };
            if let Self::Preplanned { cells, .. } = self { try_resize_dead(eng, cells, cursor)?; }
        }
        Ok(())
    }

    pub(super) fn retain(&mut self, lim: &crate::limits::Limits) {
        match self {
            Self::Preplanned { cells, .. } | Self::Bump { cells, .. } => {
                crate::limits::pool::release_if_oversized(lim, cells);
            }
        }
    }

    /// True when levels claim space as they are reached — the one behavioural
    /// difference the sweep still has to ask about, because a level's live
    /// count is only maintained (and only read) in that shape.
    pub(super) fn is_bump(&self) -> bool {
        matches!(self, GridArena::Bump { .. })
    }

    /// The flat slab every grid is a slice of.
    pub(super) fn slab(&self) -> &[u32] {
        match self {
            GridArena::Preplanned { cells, .. } | GridArena::Bump { cells, .. } => cells,
        }
    }

    /// The flat slab, for a producer writing its level's cells.
    pub(super) fn slab_mut(&mut self) -> &mut [u32] {
        match self {
            GridArena::Preplanned { cells, .. } | GridArena::Bump { cells, .. } => cells,
        }
    }

    fn grids(&self) -> &[LevelGrid] {
        match self {
            GridArena::Preplanned { grids, .. } | GridArena::Bump { grids, .. } => grids,
        }
    }

    fn grids_mut(&mut self) -> &mut [LevelGrid] {
        match self {
            GridArena::Preplanned { grids, .. } | GridArena::Bump { grids, .. } => grids,
        }
    }

    /// Level `t`'s grid base, or `None` if it has none — the one way to learn
    /// whether a level's cells may be read.
    pub(super) fn materialized(&self, t: usize) -> Option<GridBase> {
        self.grids()[t].base().map(GridBase)
    }

    /// True when level `t` has no grid, so consumers must go through its
    /// product list.
    pub(super) fn is_sparse(&self, t: usize) -> bool {
        self.grids()[t].is_sparse()
    }

    /// Record that level `t`'s grid has been materialized by a producer.
    pub(super) fn set_dense(&mut self, t: usize, base: GridBase) {
        self.grids_mut()[t] = LevelGrid::Materialized { base: base.0 };
    }

    /// Record that level `t` has no grid.
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
    ) -> Result<GridBase, OperationError> {
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

    /// Ensure level `ti` has a grid: claim space, `NO_PRODUCT`-fill it, and populate it
    /// from `product_list` (which the caller has already built).
    pub(super) fn ensure_grid(
        &mut self,
        eng: &Engine,
        ti: usize, left_width: usize, right_width: usize,
        product_list: &[ProductEntry],
    ) -> Result<(), OperationError> {
        if !self.is_sparse(ti) { return Ok(()); }
        let cells = left_width * right_width;
        let base = self.alloc(eng, ti, cells)?.idx();
        self.set_dense(ti, GridBase(base));
        let slab = self.slab_mut();
        slab[base..base + cells].fill(NO_PRODUCT);
        for &ProductEntry { left_idx, right_idx, prod_idx } in product_list {
            slab[base + left_idx.idx() * right_width + right_idx.idx()] = prod_idx.0;
        }
        Ok(())
    }

    /// Ensure level `ti` has a product list, scanning its grid if not.
    pub(super) fn ensure_product_list(
        &self,
        eng: &Engine,
        ti: usize, left_width: usize, right_width: usize,
        product_list: &mut Vec<ProductEntry>, has_pl: &mut [bool],
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        if has_pl[ti] { return Ok(()); }
        has_pl[ti] = true;
        let base = self.materialized(ti).expect("expected allocated grid, found Sparse").idx();
        let slab = self.slab();
        for i in 0..left_width {
            for j in 0..right_width {
                let idx = slab[base + i * right_width + j];
                if idx != NO_PRODUCT {
                    lim.try_push(product_list, ProductEntry {
                        left_idx: LeftNodeIdx(i as u32),
                        right_idx: RightNodeIdx(j as u32),
                        prod_idx: ProductNodeIdx(idx),
                    })?;
                }
            }
        }
        Ok(())
    }
}
