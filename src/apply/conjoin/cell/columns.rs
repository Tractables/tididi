//! The per-level table of resolved c2 column slices.

use super::*;
use crate::diagram::SideView;
use crate::engine::ByteCharge;

/// One c2 column's resolved pair slice, held as raw parts.
///
/// Raw rather than `&[InputPair]` so the table can live in a `Cell` scratch
/// pool: [`Pool`] requires a `'static` buffer type, which a lifetime-
/// carrying slice is not. Every construction site below writes the parts of a
/// live `&[InputPair]`; [`C2Columns::get`] is the only reader.
#[derive(Clone, Copy)]
pub(crate) struct ColSlice {
    ptr: *const InputPair,
    len: usize,
}

/// Per-level c2 column table: column `j`'s pair slice resolved ONCE per LEVEL
/// instead of once per (row, column) CELL.
///
/// Every `process_cell(i, j)` call used to re-derive the same column-`j` slice
/// from scratch — the mask-identity test, the `nodes[j]` bounds check, the
/// leaf/inline/multi encoding tests, the `ext`-sentinel range resolve, and (on
/// marg-mask levels) a full re-decode of the column's pairs into scratch. All
/// of that depends only on `j` and the level, never on the row, so with `k1`
/// rows it ran `k1` times per column. This resolves each column once, before
/// the row sweep; the cell prologue then indexes the table.
///
/// TWO storage regimes behind ONE table — the per-column resolution logic
/// lives here and nowhere else:
/// - **identity masks** (no marginal child): the descriptors are zero-copy
///   borrows of c2's own `nodes`/`pairs` storage, exactly what the per-cell
///   `pairs_view_decoded` fast path handed back. Nothing is copied and `flat`
///   stays empty.
/// - **marg masks**: c2's pairs need decoding, so they are decoded once into
///   `flat` and the descriptors point into it.
///   `flat` is O(Σ c2 pairs) — a real transient the budget must see, so it
///   reserves through `budget_reserve_exact` and un-charges the in-flight
///   accounting on drop (level end). On `OverBudget` the build returns `None`
///   and the walkers fall back to per-cell decode: strictly no worse than the
///   per-cell behavior on the OOM-critical path.
///
/// `cols` is pooled scratch (`eng.apply().c2_cols`), not diagram memory, so it is
/// not budget-charged; its retained capacity is capped on return to the pool
/// like every other apply scratch buffer.
pub(crate) struct C2Columns<'a> {
    /// Decode arena — non-empty ONLY on marg-mask levels. Filled once at
    /// build time and never touched again, so the heap block the descriptors
    /// point into is fixed for the table's whole life (moving the `Vec`, e.g.
    /// out of `build`, moves the 3-word header, never the block).
    ///
    /// Deliberately never read through this field — `build` resolves the
    /// descriptors against the arena's base before handing it over, so the
    /// field's whole job is to OWN the block and free it when the table
    /// drops. Removing it would dangle every marg-level descriptor.
    #[allow(dead_code)]
    flat: Vec<InputPair>,
    /// One descriptor per column `j ∈ 0..k2`.
    cols: Vec<ColSlice>,
    /// The byte-budget charge for `flat`, released wherever the table goes out
    /// of scope — including the level's early exits.
    ///
    /// Deliberately never read: the field's whole job is to hold the charge for
    /// the table's life and give it back on drop.
    #[allow(dead_code)]
    charge: ByteCharge<'a>,
    /// The engine whose pool the descriptor buffer goes back to.
    eng: &'a Engine,
}

impl<'a> C2Columns<'a> {
    /// Column `j`'s pairs — the per-level replacement for the per-cell
    /// `pairs_view_decoded` derivation.
    #[inline(always)]
    pub(crate) fn get(&self, j: usize) -> &[InputPair] {
        let c = self.cols[j];
        // SAFETY: `c` was built by `build` below out of either (a) a live
        // `&[InputPair]` borrowed from the c2 level, or (b) a subrange of
        // `self.flat`.
        //
        // (b) is owned by `self` and never mutated after `build`, so it is
        // alive and its heap block unmoved for as long as the returned borrow.
        //
        // (a) is alive by the sole caller's shape: the table is a local of ONE
        // iteration of the apply's per-level loop, and for the rest of that
        // iteration `c2` is only ever READ (`c2.level(t)`, `c2.levels[..]`) —
        // there is no `&mut c2` between the table's construction and its drop,
        // so c2's `nodes`/`pairs` cannot be pushed to and cannot reallocate.
        // The row sweep's own writes go to the OUTPUT level, a separate
        // allocation from either operand, and it holds `c2_level_t:
        // &TddLevel` across its full duration.
        //
        // An empty column carries the aligned-non-null pointer of the `&[]`
        // it came from, which `from_raw_parts` accepts at length 0.
        unsafe { std::slice::from_raw_parts(c.ptr, c.len) }
    }

    /// Resolve every column of `c2_level` under `left_view`/`right_view`.
    ///
    /// Returns `None` (per-cell derivation fallback) when the table can't be
    /// built: a marginal-encoded c2 level (it stores count payloads, not pair
    /// structure — the marginal-at-t cases are consumed by fast paths before
    /// the cell build, so the walkers never read its pairs, and neither may
    /// we), an arena past `u32::MAX` pairs (one that large has no business
    /// existing), or a budget that rejects the arena / descriptor reservation.
    pub(crate) fn build(
        eng: &'a Engine,
        c2_level: &TddLevel,
        k2: usize,
        left_view: SideView,
        right_view: SideView,
    ) -> Option<C2Columns<'a>> {
        let lim = eng.limits();
        if c2_level.is_marginal() {
            return None;
        }
        // `k2` is the level width cached before the sweep; resolving a column
        // reads `nodes[j]`, and the table resolves ALL of 0..k2 where the
        // per-cell path only reached the columns of a level with ≥1 live row.
        // If the two ever disagreed, hoisting would index past `nodes` on a
        // level the per-cell path never touched — decline instead, which is
        // exactly the pre-existing per-cell behavior.
        if k2 > c2_level.nodes.len() {
            return None;
        }
        let identity = !left_view.is_valued() && !right_view.is_valued();

        // Identity masks borrow c2's storage directly (the per-cell view was
        // already a zero-copy borrow — never materialize what was borrowed),
        // so the arena and its budget charge exist only for marg masks.
        let mut flat: Vec<InputPair> = Vec::new();
        let mut charge = ByteCharge::none(lim);
        if !identity {
            let mut total: usize = 0;
            for j in 0..k2 {
                if c2_level.nodes[j].is_internal() {
                    total += c2_level.pair_count_at(j);
                }
            }
            if total > u32::MAX as usize {
                return None;
            }
            // A reserve can fail AFTER charging (try_reserve succeeds, the
            // soft-budget check trips), so the charge covers whatever capacity
            // the vec actually holds either way.
            let failed = lim.reserve_exact(&mut flat, total).is_err();
            charge.owe(Self::cap_bytes(&flat));
            if failed {
                return None;
            }
        }

        let mut cols: Vec<ColSlice> = eng.apply().c2_cols.take();
        cols.clear();
        if cols.try_reserve(k2).is_err() {
            eng.apply().c2_cols.put_bounded(cols, MAX_LEVEL_ARENA_BYTES);
            return None;
        }

        if identity {
            for j in 0..k2 {
                // THE per-column resolution — the same accessor the per-cell
                // identity fast path (`pairs_view_decoded` → `pairs_view_into`)
                // calls, hoisted out of the row loop.
                let s = c2_level.pairs_of_idx(j);
                cols.push(ColSlice { ptr: s.as_ptr(), len: s.len() });
            }
        } else {
            // Pass 1: decode every column into the arena, recording only
            // lengths — the arena's base is not final until it is full.
            for j in 0..k2 {
                let before = flat.len();
                c2_level.decode_pairs_into(j, &mut flat, left_view, right_view);
                cols.push(ColSlice { ptr: std::ptr::null(), len: flat.len() - before });
            }
            // Pass 2: point each descriptor at its subrange of the finished
            // arena. The lengths sum to `flat.len()` by construction, so the
            // running offset never passes the end.
            let base = flat.as_ptr();
            let mut off = 0usize;
            for c in cols.iter_mut() {
                // SAFETY: `off <= flat.len() <= flat.capacity()` at every step,
                // so `base.add(off)` is inside the allocation (or exactly
                // one-past-the-end for a trailing empty column).
                c.ptr = unsafe { base.add(off) };
                off += c.len;
            }
        }

        Some(C2Columns { flat, cols, charge, eng })
    }

    fn cap_bytes(flat: &Vec<InputPair>) -> u64 {
        (flat.capacity() as u64) * (std::mem::size_of::<InputPair>() as u64)
    }
}

impl Drop for C2Columns<'_> {
    fn drop(&mut self) {
        // Hand the descriptor buffer back to the pool under the module-wide
        // retain cap, so one very wide level can't park its table there and
        // tax every later small apply.
        self.eng
            .apply()
            .c2_cols
            .put_bounded(std::mem::take(&mut self.cols), MAX_LEVEL_ARENA_BYTES);
    }
}
