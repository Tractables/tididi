//! The per-level table of resolved g column slices.

use super::*;
use crate::diagram::ChildDecoder;
use crate::limits::Transient;

/// Pairs a g column and an f row both need before the N×M walk groups their
/// cell by shared `.left`.
pub(crate) const GROUPED_MIN_PAIRS: usize = 64;

/// One g column's resolved pair slice, held as raw parts.
///
/// Raw rather than `&[ChildPair]` so the table can live in a `Cell` scratch
/// pool: [`Pool`](crate::execution::pool::Pool) requires a `'static` buffer type, which a lifetime-
/// carrying slice is not. Every construction site below writes the parts of a
/// live `&[ChildPair]`; [`ColumnSlice::pairs`] is the only reader.
#[derive(Clone, Copy)]
pub(crate) struct ColumnSlice {
    ptr: *const ChildPair,
    len: usize,
    /// The column's range in `RightColumns::run_ends`; empty when the column is
    /// not grouped.
    run_range: (u32, u32),
}

impl ColumnSlice {
    /// The pairs this descriptor was built from.
    ///
    /// # Safety
    ///
    /// The slice the descriptor was built from must be alive and unmoved for
    /// `'s`.
    #[inline(always)]
    unsafe fn pairs<'s>(self) -> &'s [ChildPair] {
        // Safety: the caller's contract. An empty column carries the aligned
        // non-null pointer of the `&[]` it came from, which `from_raw_parts`
        // accepts at length 0.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
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
/// On a level whose N×M cells can be grouped, every column of at least
/// [`GROUPED_MIN_PAIRS`] pairs also has its runs of pairs sharing `.left`
/// recorded here, once, rather than found again by every row that reaches it.
///
/// `cols` is pooled scratch (`eng.scratch.apply.right_cols`), not budget-charged.
pub(crate) struct RightColumns<'a> {
    /// Decode arena, non-empty only on marginal-mask levels. Filled once in
    /// `build` and never touched again, so the heap block the descriptors
    /// point into stays put for the table's life; the field's job is to own
    /// that block and its byte charge, and it is never read through.
    #[expect(dead_code)]
    flat: Transient<'a, Vec<ChildPair>>,
    /// One descriptor per column `j ∈ 0..right_width`.
    cols: Vec<ColumnSlice>,
    /// The grouped columns' run ends, column after column: each is the
    /// offset within its column one past a run of pairs sharing `.left`.
    /// Budget-charged, released on drop.
    run_ends: Transient<'a, Vec<u32>>,
    /// The engine whose pool the descriptor buffer goes back to.
    eng: &'a Engine,
}

impl<'a> RightColumns<'a> {
    /// Column `j`'s pairs — the per-level replacement for the per-cell
    /// `pairs_view_decoded` derivation.
    #[inline(always)]
    pub(crate) fn get(&self, j: usize) -> &[ChildPair] {
        // Safety: every descriptor was built by `build` from either a
        // `&[ChildPair]` borrowed from the g level, or a subrange of
        // `self.flat`. `flat` is owned by `self` and never mutated after
        // `build`. The g level is only read between the table's construction
        // and its drop (the table is a local of one iteration of the
        // per-level loop, which takes no `&mut g`), so its `pairs` cannot
        // reallocate.
        unsafe { self.cols[j].pairs() }
    }

    /// Column `j`'s run ends: each entry is the offset one past a run of
    /// consecutive pairs sharing `.left`, in order, the last one the column's
    /// length. `None` when the column is not grouped.
    #[inline(always)]
    pub(crate) fn runs(&self, j: usize) -> Option<&[u32]> {
        let (start, end) = self.cols[j].run_range;
        (start < end).then(|| &self.run_ends[start as usize..end as usize])
    }

    /// Resolve every column of `right_level` under `left_view`/`right_view`.
    ///
    /// Returns `None` (per-cell derivation fallback) when the table can't be
    /// built: a marginal-encoded g level (it stores count payloads, not pair
    /// structure — the marginal-at-t cases are consumed by fast paths before
    /// the cell build, so the walkers never read its pairs, and neither may
    /// we), an arena past `u32::MAX` pairs (one that large has no business
    /// existing), or a budget that rejects the arena / descriptor reservation.
    ///
    /// `grouped` says the level's N×M cells can take the grouped walk: both
    /// operand levels hold multi-pair nodes and neither child side is a
    /// pass-through. Only then are the long columns' runs recorded.
    pub(crate) fn build(
        eng: &'a Engine,
        right_level: &'a TddLevel,
        right_width: usize,
        left_view: ChildDecoder,
        right_view: ChildDecoder,
        grouped: bool,
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
        let mut flat = Transient::new(lim, Vec::<ChildPair>::new());
        if !identity {
            let mut total: usize = 0;
            for j in 0..right_width {
                total += right_level.pair_count_at(j);
            }
            if total > u32::MAX as usize {
                return None;
            }
            // A reserve can fail after charging (`try_reserve` succeeds, the
            // soft-budget check trips); dropping the transient hands back
            // whatever capacity the vec holds either way.
            lim.reserve_exact(&mut flat, total).ok()?;
        }

        let mut cols: Vec<ColumnSlice> = eng.scratch.apply.right_cols.take(eng);
        cols.clear();
        if cols.try_reserve(right_width).is_err() {
            eng.scratch.apply.right_cols.put(eng, cols);
            return None;
        }

        if identity {
            for j in 0..right_width {
                // The one per-column resolution — the same accessor the per-cell
                // identity fast path (`pairs_view_decoded` → `pairs_of_idx`)
                // calls, hoisted out of the row loop.
                let s = right_level.pairs_of_idx(j);
                cols.push(ColumnSlice { ptr: s.as_ptr(), len: s.len(), run_range: (0, 0) });
            }
        } else {
            // Pass 1: decode every column into the arena, recording only
            // lengths — the arena's base is not final until it is full.
            for j in 0..right_width {
                let before = flat.len();
                right_level.decode_pairs_into(j, &mut flat, left_view, right_view);
                cols.push(ColumnSlice { ptr: std::ptr::null(), len: flat.len() - before, run_range: (0, 0) });
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

        let mut table = RightColumns { flat, cols, run_ends: Transient::new(lim, Vec::new()), eng };
        if grouped {
            table.record_runs();
        }
        Some(table)
    }

    /// Record the runs of every column of at least [`GROUPED_MIN_PAIRS`]
    /// pairs.
    ///
    /// A push the budget refuses stops the recording there: that column and
    /// the ones after it stay ungrouped, and their cells take the ungrouped
    /// walk, which emits the same pairs in another order.
    fn record_runs(&mut self) {
        let lim = self.eng.limits();
        let long = |c: &ColumnSlice| c.len >= GROUPED_MIN_PAIRS;
        // No run end and no index into `run_ends` exceeds the long columns' total
        // pair count, so a level within `u32` stores both as `u32`.
        let total: usize = self.cols.iter().filter(|c| long(c)).map(|c| c.len).sum();
        if total > u32::MAX as usize {
            return;
        }
        for j in 0..self.cols.len() {
            let c = self.cols[j];
            if !long(&c) {
                continue;
            }
            // Safety: as in `get`. Pushing to `run_ends` moves neither `flat` nor
            // the g level.
            let pairs = unsafe { c.pairs() };
            let start = self.run_ends.len();
            let mut i = 0;
            while i < pairs.len() {
                let left = pairs[i].left;
                i += 1;
                while i < pairs.len() && pairs[i].left == left {
                    i += 1;
                }
                if lim.try_push(&mut self.run_ends, i as u32).is_err() {
                    self.run_ends.truncate(start);
                    return;
                }
            }
            self.cols[j].run_range = (start as u32, self.run_ends.len() as u32);
        }
    }
}

impl Drop for RightColumns<'_> {
    fn drop(&mut self) {
        // Hand the descriptor buffer back to the pool under the module-wide
        // retain cap, so one very wide level can't park its table there and
        // tax every later small apply.
        self.cols.clear();
        self.eng
            .scratch
            .apply
            .right_cols
            .put(self.eng, std::mem::take(&mut self.cols));
    }
}
