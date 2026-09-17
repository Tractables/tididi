//! The per-level table of resolved g column slices.

use super::*;
use crate::diagram::ChildDecoder;
use crate::limits::ByteCharge;

/// One g column's resolved pair slice, held as raw parts.
///
/// Raw rather than `&[ChildPair]` so the table can live in a `Cell` scratch
/// pool: [`Pool`](crate::limits::pool::Pool) requires a `'static` buffer type, which a lifetime-
/// carrying slice is not. Every construction site below writes the parts of a
/// live `&[ChildPair]`; [`RightColumns::get`] is the only reader.
#[derive(Clone, Copy)]
pub(crate) struct ColumnSlice {
    ptr: *const ChildPair,
    len: usize,
}

// Safety: descriptors never dereference or drop their pointees. Only
// RightColumns::get reads them, while borrowing the source level and engine;
// that live table cannot move across threads because Engine is not Sync.
// Pool entries are cleared before reuse and contain no borrowed values.
unsafe impl Send for ColumnSlice {}

/// Per-level g column table: column `j`'s pair slice resolved once per level
/// instead of once per (row, column) cell.
///
/// With identity masks (no marginal child) the descriptors borrow g's own
/// `nodes`/`pairs` and `flat` stays empty. With marginal masks g's pairs are
/// decoded once into `flat`, whose reservation is budget-charged and released
/// on drop; when that reservation is refused `build` returns `None` and the
/// walkers decode per cell instead.
///
/// `cols` is pooled scratch (`eng.apply().right_cols`), not budget-charged.
pub(crate) struct RightColumns<'a> {
    /// Decode arena, non-empty only on marginal-mask levels. Filled once in
    /// `build` and never touched again, so the heap block the descriptors
    /// point into stays put for the table's life; the field's job is to own
    /// that block, and it is never read through.
    #[allow(dead_code)]
    flat: Vec<ChildPair>,
    /// One descriptor per column `j ∈ 0..right_width`.
    cols: Vec<ColumnSlice>,
    /// The byte-budget charge for `flat`, held for the table's life and
    /// released on drop, including the level's early exits.
    #[allow(dead_code)]
    charge: ByteCharge<'a>,
    /// The engine whose pool the descriptor buffer goes back to.
    eng: &'a Engine,
}

impl<'a> RightColumns<'a> {
    /// Column `j`'s pairs — the per-level replacement for the per-cell
    /// `pairs_view_decoded` derivation.
    #[inline(always)]
    pub(crate) fn get(&self, j: usize) -> &[ChildPair] {
        let c = self.cols[j];
        // Safety: `c` was built by `build` from either a `&[ChildPair]`
        // borrowed from the g level, or a subrange of `self.flat`. `flat` is
        // owned by `self` and never mutated after `build`. The g level is
        // only read between the table's construction and its drop (the table
        // is a local of one iteration of the per-level loop, which takes no
        // `&mut g`), so its `pairs` cannot reallocate. An empty column carries
        // the aligned non-null pointer of the `&[]` it came from, which
        // `from_raw_parts` accepts at length 0.
        unsafe { std::slice::from_raw_parts(c.ptr, c.len) }
    }

    /// Resolve every column of `right_level` under `left_view`/`right_view`.
    ///
    /// Returns `None` (per-cell derivation fallback) when the table can't be
    /// built: a marginal-encoded g level (it stores count payloads, not pair
    /// structure — the marginal-at-t cases are consumed by fast paths before
    /// the cell build, so the walkers never read its pairs, and neither may
    /// we), an arena past `u32::MAX` pairs (one that large has no business
    /// existing), or a budget that rejects the arena / descriptor reservation.
    pub(crate) fn build(
        eng: &'a Engine,
        right_level: &'a TddLevel,
        right_width: usize,
        left_view: ChildDecoder,
        right_view: ChildDecoder,
    ) -> Option<RightColumns<'a>> {
        let lim = eng.limits();
        if right_level.is_marginal() {
            return None;
        }
        // `right_width` is the width cached before the sweep; resolving every
        // column reads `nodes[j]` for all of `0..right_width`, so decline if
        // the level holds fewer nodes than that.
        if right_width > right_level.nodes.len() {
            return None;
        }
        let identity = !left_view.is_marginal() && !right_view.is_marginal();

        // Identity masks borrow g's storage directly (the per-cell view was
        // already a zero-copy borrow — never materialize what was borrowed),
        // so the arena and its budget charge exist only for marginal masks.
        let mut flat: Vec<ChildPair> = Vec::new();
        let mut charge = ByteCharge::none(lim);
        if !identity {
            let mut total: usize = 0;
            for j in 0..right_width {
                if right_level.nodes[j].is_internal() {
                    total += right_level.pair_count_at(j);
                }
            }
            if total > u32::MAX as usize {
                return None;
            }
            // A reserve can fail after charging (try_reserve succeeds, the
            // soft-budget check trips), so the charge covers whatever capacity
            // the vec actually holds either way.
            let failed = lim.reserve_exact(&mut flat, total).is_err();
            charge.owe(Self::cap_bytes(&flat));
            if failed {
                return None;
            }
        }

        let mut cols: Vec<ColumnSlice> = eng.apply().right_cols.take();
        cols.clear();
        if cols.try_reserve(right_width).is_err() {
            eng.apply().right_cols.put_bounded(lim, cols);
            return None;
        }

        if identity {
            for j in 0..right_width {
                // The one per-column resolution — the same accessor the per-cell
                // identity fast path (`pairs_view_decoded` → `pairs_of_idx`)
                // calls, hoisted out of the row loop.
                let s = right_level.pairs_of_idx(j);
                cols.push(ColumnSlice { ptr: s.as_ptr(), len: s.len() });
            }
        } else {
            // Pass 1: decode every column into the arena, recording only
            // lengths — the arena's base is not final until it is full.
            for j in 0..right_width {
                let before = flat.len();
                right_level.decode_pairs_into(j, &mut flat, left_view, right_view);
                cols.push(ColumnSlice { ptr: std::ptr::null(), len: flat.len() - before });
            }
            // Pass 2: point each descriptor at its subrange of the finished
            // arena. The lengths sum to `flat.len()` by construction, so the
            // running offset never passes the end.
            let base = flat.as_ptr();
            let mut off = 0usize;
            for c in cols.iter_mut() {
                // Safety: `off <= flat.len() <= flat.capacity()` at every step,
                // so `base.add(off)` is inside the allocation (or exactly
                // one-past-the-end for a trailing empty column).
                c.ptr = unsafe { base.add(off) };
                off += c.len;
            }
        }

        Some(RightColumns { flat, cols, charge, eng })
    }

    fn cap_bytes(flat: &Vec<ChildPair>) -> u64 {
        (flat.capacity() as u64) * (std::mem::size_of::<ChildPair>() as u64)
    }
}

impl Drop for RightColumns<'_> {
    fn drop(&mut self) {
        // Hand the descriptor buffer back to the pool under the module-wide
        // retain cap, so one very wide level can't park its table there and
        // tax every later small apply.
        self.cols.clear();
        self.eng
            .apply()
            .right_cols
            .put_bounded(self.eng.limits(), std::mem::take(&mut self.cols));
    }
}
