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
//! * Pre-planned — no level can go sparse, so every level's base is computed
//!   up front and the slab is sized once. Nothing is ever reclaimed: the whole
//!   layout is live until the apply ends.
//! * Bumping — some level may take the sparse route and skip its grid, so
//!   space is claimed level by level and released again as soon as a level's
//!   single parent has consumed it. This caps the slab at the live frontier
//!   instead of the sum of every densified level.
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
//! level whose grid was materialized — which is what
//! [`GridArena::materialized`] returns and what every consumer goes through.

use crate::Engine;
use super::{OperationError, NO_PRODUCT};
use super::budget::try_resize_dead;
use super::products::{ProductEntry, LeftNodeIdx, RightNodeIdx, ProductNodeIdx};

/// Offset of a level's grid within the arena's flat slab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct GridBase(usize);

impl GridBase {
    /// The offset as an index into the slab.
    pub(super) fn idx(self) -> usize { self.0 }
}

/// A stretch of the slab that is not currently any level's grid.
#[derive(Clone, Copy)]
struct Region {
    base: GridBase,
    len: usize,
}

/// The state of a layout that claims and releases space as the sweep proceeds.
#[derive(Default)]
struct Bump {
    /// The frontier: no cell past it has been claimed.
    end: usize,
    /// Released regions below the frontier, reused best-fit.
    free: Vec<Region>,
}

/// The flat product-grid slab and every level's claim on it.
#[derive(Default)]
pub(super) struct GridArena {
    cells: Vec<u32>,
    /// Each level's grid base, `None` while the level has no grid and its
    /// consumers must go through its product list.
    grids: Vec<Option<GridBase>>,
    /// The bump allocator's state; `None` on a pre-planned layout.
    bump: Option<Bump>,
}

impl GridArena {
    pub(super) fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        let free = self.bump.as_ref().map_or(0, |bump| capacity_bytes(&bump.free));
        capacity_bytes(&self.cells).saturating_add(capacity_bytes(&self.grids)).saturating_add(free)
    }

    pub(super) fn reset(&mut self, eng: &Engine, sparse: bool, n: usize, left: &[usize], right: &[usize]) -> Result<(), OperationError> {
        self.grids.clear();
        self.grids.resize(n, None);
        let mut bump = self.bump.take().unwrap_or_default();
        bump.end = 0;
        bump.free.clear();
        if sparse {
            self.bump = Some(bump);
        } else {
            let mut cursor = 0;
            for i in 0..n {
                self.grids[i] = Some(GridBase(cursor));
                cursor += left[i] * right[i];
            }
            try_resize_dead(eng, &mut self.cells, cursor)?;
        }
        Ok(())
    }

    pub(super) fn retain(&mut self, lim: &crate::limits::Limits) {
        crate::limits::pool::release_if_oversized(lim, &mut self.cells);
    }

    /// True when levels claim space as they are reached — the one behavioural
    /// difference the sweep still has to ask about, because a level's live
    /// count is only maintained (and only read) in that shape.
    pub(super) fn is_bump(&self) -> bool {
        self.bump.is_some()
    }

    /// The flat slab every grid is a slice of.
    pub(super) fn slab(&self) -> &[u32] {
        &self.cells
    }

    /// The flat slab, for a producer writing its level's cells.
    pub(super) fn slab_mut(&mut self) -> &mut [u32] {
        &mut self.cells
    }

    /// Level `t`'s grid base, or `None` if it has none — the one way to learn
    /// whether a level's cells may be read.
    pub(super) fn materialized(&self, t: usize) -> Option<GridBase> {
        self.grids[t]
    }

    /// True when level `t` has no grid, so consumers must go through its
    /// product list.
    pub(super) fn is_sparse(&self, t: usize) -> bool {
        self.grids[t].is_none()
    }

    /// Record that level `t`'s grid has been materialized by a producer.
    pub(super) fn set_dense(&mut self, t: usize, base: GridBase) {
        self.grids[t] = Some(base);
    }

    /// Record that level `t` has no grid.
    pub(super) fn set_sparse(&mut self, t: usize) {
        self.grids[t] = None;
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
        let Some(bump) = &mut self.bump else {
            return Ok(self.grids[t].expect("a pre-planned layout grids every level"));
        };
        if cells == 0 {
            // Zero-width grid: never read or freed meaningfully; hand
            // out the cursor without growing.
            return Ok(GridBase(bump.end));
        }
        // Best-fit: smallest free region that still fits. <500 regions,
        // so linear.
        let mut best: Option<usize> = None;
        for (i, r) in bump.free.iter().enumerate() {
            if r.len >= cells && best.is_none_or(|b| r.len < bump.free[b].len) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let Region { base, len } = bump.free[i];
            if len == cells {
                bump.free.swap_remove(i);
            } else {
                bump.free[i] = Region { base: GridBase(base.0 + cells), len: len - cells };
            }
            return Ok(base);
        }
        let base = bump.end;
        bump.end += cells;
        try_resize_dead(eng, &mut self.cells, bump.end)?;
        Ok(GridBase(base))
    }

    /// Return a consumed region, coalescing with physically adjacent free
    /// regions to limit fragmentation. A no-op on a pre-planned layout, which
    /// owns one contiguous block and must not be freed piecemeal.
    pub(super) fn free(&mut self, base: GridBase, cells: usize) {
        let Some(bump) = &mut self.bump else { return };
        if cells == 0 { return; }
        let free = &mut bump.free;
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

    /// Append level `ti`'s live cells to `product_list`, scanning its grid.
    pub(super) fn scan_product_list(
        &self,
        eng: &Engine,
        ti: usize, left_width: usize, right_width: usize,
        product_list: &mut Vec<ProductEntry>,
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        let base = self.materialized(ti).expect("a product list is scanned from a materialized grid").idx();
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
