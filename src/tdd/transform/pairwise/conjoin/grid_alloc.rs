//! Grid-reclaim bump allocator over the apply's flat `node_idx` arena, plus the
//! grid/product-list materialization helpers. Pure code motion out of
//! `conjoin/mod.rs`; the driver calls the `pub(super)` entries. `ensure_grid`
//! stays private (only `materialize_dense_child` calls it).

use super::{ApplyError, LevelGrid, DEAD, try_push};
use super::budget::try_resize_dead;
use super::sparse::{ProductEntry, C1NodeIdx, C2NodeIdx, ProdNodeIdx, fill_identity_product_list};

/// Ensure level `ti` has a grid in node_idx. If not yet allocated,
/// bump-allocates space, DEAD-fills, and populates from the product list.
/// Requires the product list to be already built (has_pl[ti] == true).
/// Marks the level as `DenseWeak` — values come from scatter, not a
/// sequential emit, so strict monotonicity does not hold.
#[inline(always)]
fn ensure_grid(
    ti: usize, k1: usize, k2: usize,
    grids: &mut [LevelGrid], node_idx: &mut Vec<u32>, grid_end: &mut usize,
    free_regions: &mut Vec<(usize, usize)>,
    product_list: &[ProductEntry],
) -> Result<(), ApplyError> {
    if !grids[ti].is_sparse() { return Ok(()); }
    let cells = k1 * k2;
    let base = grid_alloc(node_idx, grid_end, free_regions, cells)?;
    grids[ti] = LevelGrid::DenseWeak { base };
    node_idx[base..base + cells].fill(DEAD);
    for &ProductEntry { c1_idx, c2_idx, prod_idx } in product_list {
        node_idx[base + c1_idx.idx() * k2 + c2_idx.idx()] = prod_idx.0;
    }
    Ok(())
}

/// Instrumented wrapper around [`ensure_grid`] for the `sparse-conjunction`
/// Materialize one child on the dense-parent path (B6.2): build its product
/// list if not already built, then `ensure_grid` it into `node_idx`.
/// Left/right call sites were near-identical twins; unified here. Takes
/// `&mut Vec<ProductEntry>` for just the one child (not the whole
/// `product_lists`) so the borrow stays scoped to that element and doesn't
/// conflict with the other child's.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_dense_child(
    idx: usize,
    k1c: usize,
    k2c: usize,
    c2_ident: bool,
    c1_ident: bool,
    has_pl_slot: &mut bool,
    pl: &mut Vec<ProductEntry>,
    grids: &mut [LevelGrid],
    node_idx: &mut Vec<u32>,
    grid_end: &mut usize,
    free_regions: &mut Vec<(usize, usize)>,
) -> Result<(), ApplyError> {
    if !*has_pl_slot {
        fill_identity_product_list(k1c, k2c, c2_ident, c1_ident, pl, has_pl_slot)?;
        *has_pl_slot = true;
    }
    ensure_grid(idx, k1c, k2c, grids, node_idx, grid_end, free_regions, &*pl)
}

// ── Grid-reclaim allocator ──────────────────────────────────────────────────
//
// Bump-or-reuse allocator over `node_idx`. `grid_alloc` first tries to satisfy a
// request from the free-list (best-fit, to curb fragmentation); only on a miss
// does it grow `grid_end`. `grid_free` returns a consumed region and coalesces it
// with any physically adjacent free region. Together they cap `node_idx` at the
// live-grid frontier instead of the cumulative sum of every densified level.
//
// Correctness rests on the single-consumer invariant: each vtree node's grid is
// read by exactly its one parent and is dead afterward, so a region is freed
// exactly once (at the parent's `reclaim_child_grids`) and never while still read. The
// root grid is never freed (root has no parent) and is the only grid
// `compute_apply_output` reads. Reused regions are always DEAD-filled before use.

/// Allocate a `cells`-long region in `node_idx`, reusing a freed region when one
/// fits (best-fit) or bumping `grid_end` otherwise. Returns the region base.
pub(super) fn grid_alloc(
    node_idx: &mut Vec<u32>,
    grid_end: &mut usize,
    free_regions: &mut Vec<(usize, usize)>,
    cells: usize,
) -> Result<usize, ApplyError> {
    if cells == 0 {
        // Zero-width grid: never read or freed meaningfully; hand out the cursor
        // without growing.
        return Ok(*grid_end);
    }
    // Best-fit: smallest free region that still fits. <500 regions, so linear.
    let mut best: Option<usize> = None;
    for (i, &(_, len)) in free_regions.iter().enumerate() {
        if len >= cells && best.is_none_or(|b| len < free_regions[b].1) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let (base, len) = free_regions[i];
        if len == cells {
            free_regions.swap_remove(i);
        } else {
            free_regions[i] = (base + cells, len - cells); // keep the remainder
        }
        return Ok(base);
    }
    // No fit: grow the arena.
    let base = *grid_end;
    *grid_end += cells;
    try_resize_dead(node_idx, *grid_end)?;
    Ok(base)
}

/// Return a consumed region to the free-list, coalescing with physically adjacent
/// free regions to limit fragmentation.
pub(super) fn grid_free(free_regions: &mut Vec<(usize, usize)>, base: usize, cells: usize) {
    if cells == 0 { return; }
    let mut new_base = base;
    let mut new_len = cells;
    // Repeatedly absorb a neighbor on either side (at most a left and a right).
    let mut merged = true;
    while merged {
        merged = false;
        for i in 0..free_regions.len() {
            let (b, l) = free_regions[i];
            if b + l == new_base {
                new_base = b;
                new_len += l;
                free_regions.swap_remove(i);
                merged = true;
                break;
            }
            if new_base + new_len == b {
                new_len += l;
                free_regions.swap_remove(i);
                merged = true;
                break;
            }
        }
    }
    free_regions.push((new_base, new_len));
}

/// Free child node `v`'s grid (if it has one) back to the reclaim free-list and
/// mark it sparse, so its region can be reused and no dangling base remains.
/// Sparse-mode only — dense mode owns one upfront contiguous block and never
/// reclaims piecemeal.
#[inline]
pub(super) fn grid_free_child(
    grids: &mut [LevelGrid],
    free_regions: &mut Vec<(usize, usize)>,
    c1_widths: &[usize],
    c2_widths: &[usize],
    v: usize,
) {
    if let Some(base) = grids[v].base() {
        grid_free(free_regions, base, c1_widths[v] * c2_widths[v]);
        grids[v] = LevelGrid::Sparse;
    }
}

/// Ensure level `ti` has a product list. If not built yet, scans the grid.
#[inline(always)]
pub(super) fn ensure_product_list(
    ti: usize, k1: usize, k2: usize,
    grids: &[LevelGrid], node_idx: &[u32],
    product_list: &mut Vec<ProductEntry>, has_pl: &mut [bool],
) -> Result<(), ApplyError> {
    if has_pl[ti] { return Ok(()); }
    has_pl[ti] = true;
    let base = grids[ti].base_unchecked();
    for i in 0..k1 {
        for j in 0..k2 {
            let idx = node_idx[base + i * k2 + j];
            if idx != DEAD {
                try_push(product_list, ProductEntry { c1_idx: C1NodeIdx(i as u32), c2_idx: C2NodeIdx(j as u32), prod_idx: ProdNodeIdx(idx) })?;
            }
        }
    }
    Ok(())
}
