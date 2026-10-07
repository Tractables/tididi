//! Dense and sparse views of the conjunction's intermediate products.
//!
//! This owner keeps list validity, density counts, conversions and pooled
//! storage together. Kernels borrow the arenas they write; completion publishes
//! the corresponding representation before a parent reads it.

use crate::{Engine, OperationError};
use crate::diagram::TddLevel;
use super::grid_arena::GridArena;
use super::setup::Operands;

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

/// Fill `pl` with the products of a complete level that has no grid: every
/// cell `(i, j)` of its `f_width × g_width` is node `i * g_width + j`.
fn fill_complete_product_list(
    eng: &Engine,
    f_width: usize,
    g_width: usize,
    pl: &mut Vec<ProductEntry>,
) -> Result<(), OperationError> {
    eng.limits().reserve(pl, f_width * g_width)?;
    for i in 0..f_width as u32 {
        for j in 0..g_width as u32 {
            let prod = i * g_width as u32 + j;
            pl.push(ProductEntry { f_idx: FNodeIdx(i), g_idx: GNodeIdx(j), prod_idx: ProductNodeIdx(prod) });
        }
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct Products {
    pub(super) arena: GridArena,
    product_lists: Vec<Vec<ProductEntry>>,
    live_counts: Vec<usize>,
    has_pl: Vec<bool>,
    /// Which levels are complete; see [`Self::is_complete`].
    complete: Vec<bool>,
}

impl Products {
    pub(super) fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        self.arena.buffers(visit);
        visit(&mut crate::execution::pool::Nested(&mut self.product_lists));
        visit(&mut self.live_counts);
        visit(&mut self.has_pl);
        visit(&mut self.complete);
    }

    /// Drop the product lists beyond the last operation's levels, so that
    /// parking the workspace costs that operation's own levels and not the
    /// most any operation ever had. A workspace that had seen a diagram of
    /// 86 000 levels made every later conjunction of a few levels walk all
    /// 86 000 lists twice on its way back to the pool. A later, larger
    /// operation grows the vector again with empty lists.
    pub(super) fn drop_unused_lists(&mut self, lim: &crate::limits::Limits) {
        let used = self.live_counts.len();
        if self.product_lists.len() > used {
            for list in self.product_lists.drain(used..) { lim.discard(list); }
        }
        self.has_pl.truncate(used);
        self.complete.truncate(used);
    }

    /// Clear the products of a conjunction over `n` levels at the levels of
    /// `cone`; a level under a free level is read by no other level, and its
    /// entries are left as they are and never read.
    pub(super) fn reset(
        &mut self, eng: &Engine, sparse: bool, n: usize, f_widths: &[usize], g_widths: &[usize],
        cone: super::setup::Cone<'_>,
    ) -> Result<(), OperationError> {
        if self.product_lists.len() < n { self.product_lists.resize_with(n, Vec::new); }
        self.has_pl.resize(n, false);
        self.complete.resize(n, false);
        self.live_counts.resize(n, 0);
        for i in cone.nodes() {
            self.product_lists[i].clear(); self.has_pl[i] = false; self.complete[i] = false; self.live_counts[i] = 0;
        }
        self.arena.reset(eng, sparse, n, f_widths, g_widths, cone)
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
        if write < list.len() {
            self.complete[t] = false;
        }
        list.truncate(write);
        self.live_counts[t] = write;
        Ok(())
    }

    /// Whether level `t` is complete: every product `f[i] ∧ g[j]` of its
    /// `f_width × g_width` cells is a node, numbered in cell order from 0, so
    /// its grid, materialized or not, holds `i * g_width + j` at cell
    /// `(i, j)`. A parent reads a complete child by arithmetic
    /// ([`CompleteLookup`](super::child_lookup::CompleteLookup)) and no
    /// candidate dies on it.
    ///
    /// Set by the builds that know it at no cost: a dense emit that
    /// produced a node in every cell ([`Self::note_built`]), and an identity
    /// fast path, whose carried level is the identity mapping. False until
    /// then, and for every other level; a false reading only costs the grid
    /// lookup.
    pub(super) fn is_complete(&self, t: usize) -> bool {
        self.complete[t]
    }

    /// Record that a build emitted `nodes` nodes, in cell order from node 0,
    /// over the `f_width × g_width` cells of level `t`'s product: the level
    /// is complete when every cell produced one.
    pub(super) fn note_built(&mut self, t: usize, f_width: usize, g_width: usize, nodes: usize) {
        self.complete[t] = nodes == f_width * g_width;
        #[cfg(debug_assertions)]
        if self.complete[t] {
            self.debug_assert_complete(t, f_width, g_width);
        }
    }

    /// Check a level marked complete against its stored products: every cell
    /// of its grid, or every entry of its product list, names the node its
    /// cell order says.
    #[cfg(debug_assertions)]
    fn debug_assert_complete(&self, t: usize, f_width: usize, g_width: usize) {
        let cells = f_width * g_width;
        if let Some(base) = self.arena.materialized(t) {
            let grid = &self.arena.slab()[base.idx()..base.idx() + cells];
            debug_assert!(grid.iter().enumerate().all(|(c, &n)| n as usize == c), "level {t} is not complete");
        } else if self.has_pl[t] {
            let list = &self.product_lists[t];
            debug_assert_eq!(list.len(), cells, "level {t} is not complete");
            debug_assert!(
                list.iter().enumerate().all(|(c, e)| e.prod_idx.0 as usize == c && e.f_idx.idx() * g_width + e.g_idx.idx() == c),
                "level {t} is not complete"
            );
        }
    }

    /// Record that level `t` is complete: an identity fast path carried it,
    /// so cell `i` names node `i`.
    pub(super) fn note_complete(&mut self, t: usize) {
        self.complete[t] = true;
    }

    pub(super) fn live(&self, level: usize) -> usize { self.live_counts[level] }
    pub(super) fn record_live(&mut self, level: usize, count: usize) { self.live_counts[level] = count; }

    pub(super) fn finish_sparse(&mut self, level: &mut TddLevel, t: usize) {
        self.live_counts[t] = level.nodes().len();
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
        else if self.complete[t] { f * g_width as u32 + g }
        else { panic!("product level {t} has neither stored products nor an identity operand") };
        (value != super::NO_PRODUCT).then_some(super::NodeIdx(value))
    }

    /// Child level `c`'s products along one node of the narrow operand, as a
    /// map from the carrier's node index to the product's node, `NO_PRODUCT`
    /// where the product is false: with `carrier_f`, cell `(i, fixed)` for
    /// each node `i` of `f`; without, cell `(fixed, j)` for each node `j` of
    /// `g`.
    ///
    /// Returns `false` without touching `out` when the map is the identity,
    /// which is what a complete level, or one an operand is constant-true
    /// over, gives when the narrow operand has one node there; `out` is
    /// filled otherwise.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the map's growth is refused.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn column(
        &self,
        eng: &Engine,
        c: usize, f_width: usize, g_width: usize,
        carrier_f: bool, fixed: u32,
        identity: Operands<bool>,
        out: &mut Vec<u32>,
    ) -> Result<bool, OperationError> {
        let (width, narrow) = if carrier_f { (f_width, g_width) } else { (g_width, f_width) };
        let cell = |k: usize| if carrier_f { (k, fixed as usize) } else { (fixed as usize, k) };
        let lim = eng.limits();
        out.clear();
        if self.complete[c] || identity.f || identity.g {
            // A complete level holds cell `(i, j)` at `i * g_width + j`, and so
            // does a level one operand is constant-true over: its one node is
            // at index 0 and the products are the other operand's nodes.
            if narrow == 1 { return Ok(false); }
            lim.reserve(out, width)?;
            out.extend((0..width).map(|k| { let (i, j) = cell(k); (i * g_width + j) as u32 }));
        } else if let Some(base) = self.arena.materialized(c) {
            let slab = &self.arena.slab()[base.idx()..base.idx() + f_width * g_width];
            lim.reserve(out, width)?;
            out.extend((0..width).map(|k| { let (i, j) = cell(k); slab[i * g_width + j] }));
        } else if self.has_pl[c] {
            lim.try_resize(out, width, super::NO_PRODUCT)?;
            for e in &self.product_lists[c] {
                let (k, other) = if carrier_f { (e.f_idx.idx(), e.g_idx.0) } else { (e.g_idx.idx(), e.f_idx.0) };
                if other == fixed { out[k] = e.prod_idx.0; }
            }
        } else {
            panic!("product level {c} has neither stored products nor an identity operand");
        }
        Ok(true)
    }

    /// Publish a level built by relabelling the carrier's nodes: `map` holds
    /// the output node of each carrier node, or `NO_PRODUCT` where the
    /// product is false, and the narrow operand has one node at the level.
    /// Writes the level's grid where it has one and its product list
    /// otherwise, and marks it complete when no product is false, since the
    /// nodes are then numbered in cell order.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the product list's growth is refused.
    pub(super) fn publish_relabelled(
        &mut self, eng: &Engine, t: usize, carrier_f: bool, map: &[u32], nodes: usize,
    ) -> Result<(), OperationError> {
        if let Some(base) = self.arena.materialized(t) {
            self.arena.slab_mut()[base.idx()..base.idx() + map.len()].copy_from_slice(map);
        } else {
            let list = &mut self.product_lists[t];
            list.clear();
            eng.limits().reserve(list, nodes)?;
            for (k, &prod) in map.iter().enumerate() {
                if prod == super::NO_PRODUCT { continue; }
                let (i, j) = if carrier_f { (k as u32, 0) } else { (0, k as u32) };
                list.push(ProductEntry { f_idx: FNodeIdx(i), g_idx: GNodeIdx(j), prod_idx: ProductNodeIdx(prod) });
            }
            self.has_pl[t] = true;
        }
        self.live_counts[t] = nodes;
        self.complete[t] = nodes == map.len();
        Ok(())
    }

    /// Mark a level the sparse route built complete when it is: a node in
    /// every cell, numbered in cell order. The scatter emits the products of
    /// each `f` node together, in node order, so a level whose `g` operand
    /// has one node there is complete whenever no product died.
    pub(super) fn note_sparse_built(&mut self, t: usize, f_width: usize, g_width: usize) {
        let list = &self.product_lists[t];
        self.complete[t] = self.has_pl[t]
            && list.len() == f_width * g_width
            && list.iter().enumerate().all(|(c, e)| {
                e.prod_idx.0 as usize == c && e.f_idx.idx() * g_width + e.g_idx.idx() == c
            });
    }

    pub(super) fn reclaim_children(&mut self, children: [(usize, usize); 2]) {
        for (level, cells) in children { self.arena.free_child(level, cells); }
    }

    /// Build level `ci`'s product list if it is not built yet: from the
    /// identity mapping when an operand is constant-true, by scanning the
    /// level's grid when it has one, and in cell order when it has neither
    /// but is complete, as a level the relabelling route carried is. Used on
    /// both the sparse and dense paths of the level loop.
    pub(super) fn ensure_product_list_for_child(
        &mut self,
        eng: &Engine,
        ci: usize, f_width: usize, g_width: usize,
        g_identity: bool, f_identity: bool,
    ) -> Result<(), OperationError> {
        if self.has_pl[ci] { return Ok(()); }
        let list = &mut self.product_lists[ci];
        if !fill_identity_product_list(eng, f_width, g_width, g_identity, f_identity, list)? {
            if self.arena.materialized(ci).is_none() && self.complete[ci] {
                fill_complete_product_list(eng, f_width, g_width, list)?;
            } else {
                self.arena.scan_product_list(eng, ci, f_width, g_width, list)?;
            }
        }
        self.has_pl[ci] = true;
        Ok(())
    }

    /// Materialize one child on the dense-parent path: build its product list
    /// if not already built, then grid it.
    ///
    /// This is the dense path, so the child is known to be ungridded — the
    /// identity mapping, or the cell order of a complete child, is the only
    /// way to build its product list.
    pub(super) fn materialize_dense_child(
        &mut self,
        eng: &Engine,
        idx: usize,
        f_width: usize,
        g_width: usize,
        g_identity: bool, f_identity: bool,
    ) -> Result<(), OperationError> {
        cheap_assert!(self.has_pl[idx] || g_identity || f_identity || self.complete[idx],
            "an ungridded child on the dense path has an identity operand or is complete");
        self.ensure_product_list_for_child(eng, idx, f_width, g_width, g_identity, f_identity)?;
        self.arena.ensure_grid(eng, idx, f_width, g_width, &self.product_lists[idx])
    }
}
