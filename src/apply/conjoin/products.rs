//! Dense and sparse views of the conjunction's intermediate products.
//!
//! This owner keeps list validity, density counts, conversions and pooled
//! storage together. Kernels borrow the arenas they write; completion publishes
//! the corresponding representation before a parent reads it.

use crate::{Engine, OperationError};
use crate::diagram::TddLevel;
use super::grid_arena::GridArena;

/// Index of a node in `f.levels[t].nodes`. Distinct from `GNodeIdx` and
/// `ProductNodeIdx` so that construction-site swaps are caught at compile time.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct FNodeIdx(pub(crate) u32);

impl FNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in `g.levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct GNodeIdx(pub(crate) u32);

impl GNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in the output `levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct ProductNodeIdx(pub(crate) u32);

/// A live product node: the conjunction `f[f_idx] ∧ g[g_idx]` produced
/// the output node at `prod_idx` in the output level.
#[derive(Clone, Copy)]
pub(crate) struct ProductEntry {
    pub(crate) f_idx: FNodeIdx,
    pub(crate) g_idx: GNodeIdx,
    pub(crate) prod_idx: ProductNodeIdx,
}

/// The three product lists one sparse level reads and writes.
pub(crate) struct ProductLists<'a> {
    pub(crate) left: &'a [ProductEntry],
    pub(crate) right: &'a [ProductEntry],
    pub(crate) out: &'a mut Vec<ProductEntry>,
}

/// Fill `pl` with the identity product mapping for a level where one operand
/// is constant-true, and say whether it did: `x ∧ 1 = x`, so the product list
/// maps each node of the other operand to itself. The constant-true operand's
/// One node is at index 0 on every level, leaf or internal
/// (`ONE_LEAF_IDX.0 == LeafLabel::One as u32 == 0`), so there is no
/// leaf/internal split. With neither operand constant-true `pl` is left
/// untouched for the caller to fill some other way.
fn fill_identity_product_list(
    eng: &Engine,
    f_width: usize,
    g_width: usize,
    right_id: bool,
    left_id: bool,
    pl: &mut Vec<ProductEntry>,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    const ID_IDX: u32 = 0;
    if right_id {
        lim.reserve(pl, f_width)?;
        for i in 0..f_width as u32 {
            pl.push(ProductEntry { f_idx: FNodeIdx(i), g_idx: GNodeIdx(ID_IDX), prod_idx: ProductNodeIdx(i) });
        }
        Ok(true)
    } else if left_id {
        lim.reserve(pl, g_width)?;
        for j in 0..g_width as u32 {
            pl.push(ProductEntry { f_idx: FNodeIdx(ID_IDX), g_idx: GNodeIdx(j), prod_idx: ProductNodeIdx(j) });
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

#[derive(Default)]
pub(super) struct Products {
    pub(super) arena: GridArena,
    product_lists: Vec<Vec<ProductEntry>>,
    live_counts: Vec<usize>,
    has_pl: Vec<bool>,
}

impl Products {
    pub(super) fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        self.arena.buffers(visit);
        visit(&mut crate::execution::pool::Nested(&mut self.product_lists));
        visit(&mut self.live_counts);
        visit(&mut self.has_pl);
    }

    pub(super) fn reset(&mut self, eng: &Engine, sparse: bool, n: usize, f_widths: &[usize], g_widths: &[usize]) -> Result<(), OperationError> {
        if self.product_lists.len() < n { self.product_lists.resize_with(n, Vec::new); }
        self.has_pl.resize(n, false);
        for i in 0..n { self.product_lists[i].clear(); self.has_pl[i] = false; }
        self.live_counts.clear();
        self.live_counts.resize(n, 0);
        self.arena.reset(eng, sparse, n, f_widths, g_widths)
    }

    pub(super) fn filter_level(
        &mut self, eng: &Engine, shape: super::LevelShape,
        keep: &mut dyn FnMut(crate::vtree::VtreeIdx, super::NodeIdx, super::NodeIdx) -> bool,
    ) -> Result<(), OperationError> {
        let t = shape.t.idx();
        self.ensure_product_list_for_child(eng, t, shape.f.here, shape.g.here, false, false)?;
        let base = self.arena.materialized(t);
        let grid = self.arena.slab_mut();
        let list = &mut self.product_lists[t];
        let mut write = 0;
        let mut poll = eng.limits().gate();
        for read in 0..list.len() {
            poll.poll(1)?;
            let entry = list[read];
            if keep(shape.t, super::NodeIdx(entry.f_idx.0), super::NodeIdx(entry.g_idx.0)) {
                list[write] = entry; write += 1;
            } else if let Some(base) = base {
                grid[base.idx() + entry.f_idx.0 as usize * shape.g.here + entry.g_idx.0 as usize] = super::NO_PRODUCT;
            }
        }
        poll.flush()?;
        list.truncate(write);
        self.live_counts[t] = write;
        Ok(())
    }

    pub(super) fn live(&self, level: usize) -> usize { self.live_counts[level] }
    pub(super) fn record_live(&mut self, level: usize, count: usize) { self.live_counts[level] = count; }

    pub(super) fn finish_sparse(&mut self, level: &mut TddLevel, t: usize) {
        self.live_counts[t] = level.nodes.len();
        self.has_pl[t] = true;
        level.shrink_arrays();
    }

    /// Level `t`'s product list, as the last build or
    /// [`Self::ensure_product_list_for_child`] left it.
    pub(super) fn list(&self, t: usize) -> &[ProductEntry] {
        &self.product_lists[t]
    }

    pub(super) fn lists(&mut self, left: usize, right: usize, out: usize) -> ProductLists<'_> {
        let [left, right, out] = self.product_lists.get_disjoint_mut([left, right, out])
            .expect("a level and its children are distinct");
        ProductLists { left, right, out }
    }

    pub(super) fn row_buffers(&mut self, t: usize) -> (&mut [u32], &mut Vec<ProductEntry>) {
        (self.arena.slab_mut(), &mut self.product_lists[t])
    }

    /// Resolve a completed product without exposing its representation to the caller.
    pub(super) fn lookup(&self, t: usize, f: u32, g: u32, g_width: usize, g_identity: bool, f_identity: bool) -> Option<super::NodeIdx> {
        let value = if let Some(base) = self.arena.materialized(t) {
            self.arena.slab()[base.idx() + f as usize * g_width + g as usize]
        } else if self.has_pl[t] {
            self.product_lists[t].iter().find(|entry| entry.f_idx.0 == f && entry.g_idx.0 == g)?.prod_idx.0
        } else if g_identity { f }
        else if f_identity { g }
        else { panic!("product level {t} has neither stored products nor an identity operand") };
        (value != super::NO_PRODUCT).then_some(super::NodeIdx(value))
    }

    pub(super) fn reclaim_children(&mut self, children: [(usize, usize); 2]) {
        for (level, cells) in children { self.arena.free_child(level, cells); }
    }

    /// Build level `ci`'s product list if it is not built yet: from the
    /// identity mapping when an operand is constant-true, by scanning the
    /// level's grid otherwise. Used on both the sparse and dense paths of the
    /// level loop.
    pub(super) fn ensure_product_list_for_child(
        &mut self,
        eng: &Engine,
        ci: usize, f_width: usize, g_width: usize,
        g_identity: bool, f_identity: bool,
    ) -> Result<(), OperationError> {
        if self.has_pl[ci] { return Ok(()); }
        let list = &mut self.product_lists[ci];
        if !fill_identity_product_list(eng, f_width, g_width, g_identity, f_identity, list)? {
            self.arena.scan_product_list(eng, ci, f_width, g_width, list)?;
        }
        self.has_pl[ci] = true;
        Ok(())
    }

    /// Materialize one child on the dense-parent path: build its product list
    /// if not already built, then grid it.
    ///
    /// This is the dense path, so the child is known to be ungridded — the
    /// identity mapping is the only way to build its product list.
    pub(super) fn materialize_dense_child(
        &mut self,
        eng: &Engine,
        idx: usize,
        f_width: usize,
        g_width: usize,
        g_identity: bool, f_identity: bool,
    ) -> Result<(), OperationError> {
        cheap_assert!(self.has_pl[idx] || g_identity || f_identity,
            "an ungridded child on the dense path has an identity operand");
        self.ensure_product_list_for_child(eng, idx, f_width, g_width, g_identity, f_identity)?;
        self.arena.ensure_grid(eng, idx, f_width, g_width, &self.product_lists[idx])
    }
}
