//! Dense and sparse views of the conjunction's intermediate products.
//!
//! This owner keeps list validity, density counts, conversions and pooled
//! storage together. Kernels borrow the arenas they write; completion publishes
//! the corresponding representation before a parent reads it.

use crate::{Engine, OperationError};
use crate::diagram::TddLevel;
use super::grid_arena::GridArena;

/// Index of a node in `f.levels[t].nodes`. Distinct from `RightNodeIdx` and
/// `ProductNodeIdx` so that construction-site swaps are caught at compile time.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct LeftNodeIdx(pub(crate) u32);

impl LeftNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in `g.levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct RightNodeIdx(pub(crate) u32);

impl RightNodeIdx {
    pub(crate) fn idx(self) -> usize { self.0 as usize }
}

/// Index of a node in the output `levels[t].nodes`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct ProductNodeIdx(pub(crate) u32);

/// A live product node: the conjunction `f[left_idx] ∧ g[right_idx]` produced
/// the output node at `prod_idx` in the output level.
#[derive(Clone, Copy)]
pub(crate) struct ProductEntry {
    pub(crate) left_idx: LeftNodeIdx,
    pub(crate) right_idx: RightNodeIdx,
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
    left_width: usize,
    right_width: usize,
    right_id: bool,
    left_id: bool,
    pl: &mut Vec<ProductEntry>,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    const ID_IDX: u32 = 0;
    if right_id {
        lim.reserve(pl, left_width)?;
        for i in 0..left_width as u32 {
            pl.push(ProductEntry { left_idx: LeftNodeIdx(i), right_idx: RightNodeIdx(ID_IDX), prod_idx: ProductNodeIdx(i) });
        }
        Ok(true)
    } else if left_id {
        lim.reserve(pl, right_width)?;
        for j in 0..right_width as u32 {
            pl.push(ProductEntry { left_idx: LeftNodeIdx(ID_IDX), right_idx: RightNodeIdx(j), prod_idx: ProductNodeIdx(j) });
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
    pub(super) fn retained_bytes(&self) -> usize {
        use crate::limits::pool::{capacity_bytes, nested_bytes};
        [
            self.arena.retained_bytes(),
            nested_bytes(&self.product_lists),
            capacity_bytes(&self.live_counts),
            capacity_bytes(&self.has_pl),
        ].into_iter().sum()
    }

    pub(super) fn reset(&mut self, eng: &Engine, sparse: bool, n: usize, left: &[usize], right: &[usize]) -> Result<(), OperationError> {
        if self.product_lists.len() < n { self.product_lists.resize_with(n, Vec::new); }
        self.has_pl.resize(n, false);
        for i in 0..n { self.product_lists[i].clear(); self.has_pl[i] = false; }
        self.live_counts.clear();
        self.live_counts.resize(n, 0);
        self.arena.reset(eng, sparse, n, left, right)
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
            if keep(shape.t, super::NodeIdx(entry.left_idx.0), super::NodeIdx(entry.right_idx.0)) {
                list[write] = entry; write += 1;
            } else if let Some(base) = base {
                grid[base.idx() + entry.left_idx.0 as usize * shape.g.here + entry.right_idx.0 as usize] = super::NO_PRODUCT;
            }
        }
        poll.flush()?;
        list.truncate(write);
        self.live_counts[t] = write;
        Ok(())
    }

    pub(super) fn retain(&mut self, lim: &crate::limits::Limits) {
        self.arena.retain(lim);
        for list in &mut self.product_lists {
            crate::limits::pool::release_if_oversized(lim, list);
        }
    }

    pub(super) fn live(&self, level: usize) -> usize { self.live_counts[level] }
    pub(super) fn record_live(&mut self, level: usize, count: usize) { self.live_counts[level] = count; }

    pub(super) fn finish_sparse(&mut self, level: &mut TddLevel, t: usize) {
        self.live_counts[t] = level.nodes.len();
        self.has_pl[t] = true;
        level.shrink_arrays();
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
    pub(super) fn lookup(&self, t: usize, left: u32, right: u32, right_width: usize, right_identity: bool, left_identity: bool) -> Option<super::NodeIdx> {
        let value = if let Some(base) = self.arena.materialized(t) {
            self.arena.slab()[base.idx() + left as usize * right_width + right as usize]
        } else if self.has_pl[t] {
            self.product_lists[t].iter().find(|entry| entry.left_idx.0 == left && entry.right_idx.0 == right)?.prod_idx.0
        } else if right_identity { left }
        else if left_identity { right }
        else { panic!("product level {t} has neither stored products nor an identity operand") };
        (value != super::NO_PRODUCT).then_some(super::NodeIdx(value))
    }

    pub(super) fn reclaim_children(&mut self, children: [(usize, usize); 2]) {
        for (level, cells) in children { self.arena.free_child(level, cells); }
    }

    /// Ensure `product_lists[ci]` is populated. Tries the cheap identity fast
    /// path first (constant-true operand → the product list is just the
    /// non-identity operand's nodes); falls back to scanning the dense grid.
    /// Used on both the sparse and dense paths of the level loop.
    pub(super) fn ensure_product_list_for_child(
        &mut self,
        eng: &Engine,
        ci: usize, left_width: usize, right_width: usize,
        right_identity: bool, left_identity: bool,
    ) -> Result<(), OperationError> {
        if self.has_pl[ci] { return Ok(()); }
        if !fill_identity_product_list(
            eng,
            left_width, right_width,
            right_identity, left_identity,
            &mut self.product_lists[ci],
        )? {
            self.arena.ensure_product_list(
                eng, ci, left_width, right_width, &mut self.product_lists[ci], &mut self.has_pl,
            )?;
        }
        self.has_pl[ci] = true;
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
        left_width_c: usize,
        right_width_c: usize,
        right_identity: bool, left_identity: bool,
    ) -> Result<(), OperationError> {
        if !self.has_pl[idx] {
            let filled = fill_identity_product_list(
                eng, left_width_c, right_width_c,
                right_identity, left_identity,
                &mut self.product_lists[idx],
            )?;
            cheap_assert!(filled, "an ungridded child on the dense path has an identity operand");
            self.has_pl[idx] = true;
        }
        self.arena.ensure_grid(eng, idx, left_width_c, right_width_c, &self.product_lists[idx])
    }
}
