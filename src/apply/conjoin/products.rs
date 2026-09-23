//! Dense and sparse views of the conjunction's intermediate products.
//!
//! This owner keeps list validity, density counts, conversions and pooled
//! storage together. Kernels borrow the arenas they write; completion publishes
//! the corresponding representation before a parent reads it.

use crate::{Engine, OperationError};
use crate::diagram::TddLevel;
use super::grid_arena::GridArena;
use super::sparse::{ProductEntry, ProductLists, fill_identity_product_list};

#[derive(Default)]
pub(super) struct Products {
    pub(super) arena: GridArena,
    product_lists: Vec<Vec<ProductEntry>>,
    live_counts: Vec<usize>,
    has_pl: Vec<bool>,
}

impl Products {
    pub(super) fn reset(&mut self, eng: &Engine, sparse: bool, n: usize, left: &[usize], right: &[usize]) -> Result<(), OperationError> {
        if self.product_lists.len() < n { self.product_lists.resize_with(n, Vec::new); }
        self.has_pl.resize(n, false);
        for i in 0..n { self.product_lists[i].clear(); self.has_pl[i] = false; }
        self.live_counts.clear();
        self.live_counts.resize(n, 0);
        self.arena.reset(eng, sparse, n, left, right)
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
            &mut self.has_pl[ci],
        )? {
            self.arena.ensure_product_list(
                eng, ci, left_width, right_width, &mut self.product_lists[ci], &mut self.has_pl,
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
        left_width_c: usize,
        right_width_c: usize,
        right_identity: bool, left_identity: bool,
    ) -> Result<(), OperationError> {
        if !self.has_pl[idx] {
            let filled = fill_identity_product_list(
                eng, left_width_c, right_width_c,
                right_identity, left_identity,
                &mut self.product_lists[idx], &mut self.has_pl[idx],
            )?;
            cheap_assert!(filled, "an ungridded child on the dense path has an identity operand");
        }
        self.arena.ensure_grid(eng, idx, left_width_c, right_width_c, &self.product_lists[idx])
    }
}
