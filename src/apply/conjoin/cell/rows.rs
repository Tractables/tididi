//! The row loop every build route shares, and the four routes themselves.

use super::*;
use super::super::liveness::ROW_LIVE;
use crate::diagram::{EncodedChildRef, NodeKind};
use std::mem::MaybeUninit;

// ============================= Row-loop driver =============================
//
// The four build routes — marginal-child emit (`run_level_rows_marginal`),
// sparse-marginal emit (`run_level_rows_marginal_sparse`), streaming collapse
// (`run_level_rows_stream_count`) and plain emit (`run_level_rows_plain`) — differ
// only in what they do per row and per cell. Everything around that (the
// per-row `NO_PRODUCT` reset, the f pair decode, the empty/dead-row skips, the alive
// masks, the right_width column sweep, the between-cell poll) is one loop body, living
// once in `run_level_rows` and parameterized by a [`CellAction`] — the same
// shape as `process_cell`, which is one product walk parameterized by a
// [`PairSink`].

/// The per-level references every build route reads, bundled and passed by
/// value.
///
/// Built once per level, after the output level and its two children have been
/// split apart, and threaded down the route chain unchanged.
#[derive(Clone, Copy)]
pub(crate) struct RowLoop<'a> {
    /// `f`'s level at this vtree node: one row per node.
    pub(crate) f_level: &'a TddLevel,
    /// `g`'s level at this vtree node: one column per node.
    pub(crate) g_level: &'a TddLevel,
    /// The output level's two children, read in place by the streaming fold.
    pub(crate) children: Sides<&'a TddLevel>,
    pub(crate) ctx: &'a CellCtx<'a>,
    /// Number of rows — `f`'s width at this vtree node.
    pub(crate) f_width: usize,
}

/// The three buffers the row loop writes through.
///
/// Kept apart from [`RowLoop`] so the shared half stays `Copy`: these are the
/// exclusive borrows, and a route that hands them on moves the bundle.
pub(crate) struct RowScratch<'a> {
    /// Decode buffer for the current row's `f` pairs.
    pub(crate) f_pairs: &'a mut Vec<ChildPair>,
    /// Decode buffer for the current cell's `g` pairs.
    pub(crate) g_pairs: &'a mut Vec<ChildPair>,
    /// The product-grid slab, which the row loop extends over the rows of a
    /// grid claimed past its end ([`GridArena::claim`]).
    ///
    /// [`GridArena::claim`]: super::super::grid_arena::GridArena::claim
    pub(crate) node_idx: &'a mut Vec<u32>,
}

/// Everything one cell of the row loop needs, bundled and passed by value.
///
/// A bundle rather than a dozen `cell` parameters, which every impl would
/// otherwise have to restate.
pub(super) struct CellArgs<'a, 'c, L, R> {
    /// Column index — the g node.
    pub(super) j: usize,
    /// The f row index itself — not necessarily the grid row: the sparse-marginal route
    /// builds every row at grid row 0 and needs this as the product entry's
    /// `f_idx`.
    pub(super) i: usize,
    /// Flat slab offset of the grid row this cell writes into —
    /// `ctx.output_grid_base + CellAction::grid_row(i) * ctx.right_width`, computed once per row by
    /// the driver for its `NO_PRODUCT` reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    pub(super) row_base: usize,
    /// Decoded pairs of f row `i` (never empty — empty rows are skipped).
    pub(super) f_pairs: &'a [ChildPair],
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) right_level_t: &'a TddLevel,
    pub(super) g_pairs_scratch: &'a mut Vec<ChildPair>,
    pub(super) node_idx: &'a mut [u32],
    pub(super) left: &'a L,
    pub(super) right: &'a R,
    /// The level's one work-clock gate. Every cell charges the pairs it walks
    /// into it, and the row loop adds one unit per cell; it is flushed when the
    /// level ends. Per-cell gates cannot do this — a gate narrower than its
    /// stride charges nothing at all, which is what a one-sided cell always is.
    pub(super) gate: &'a mut crate::limits::PollGate<'c>,
}

/// Per-row / per-cell action of the shared row loop ([`run_level_rows`]).
///
/// All hooks are `#[inline(always)]` in impls, so each instantiation
/// monomorphizes to a loop specialized to its route.
pub(super) trait CellAction<L: ChildLookup, R: ChildLookup> {
    /// Whether this action owns a dense `left_width × right_width` slab — one grid row per f row,
    /// so the per-row `NO_PRODUCT` resets tile the level's whole slab exactly once and
    /// can be replaced by a single fill (see [`run_level_rows`]). `false` for
    /// [`SparseMargEmit`], which reuses grid row 0 for every structural row and
    /// therefore must re-fill that one row between rows.
    const DENSE_SLAB: bool;

    /// Whether the level has one column, which the action was built for: the
    /// row loop then knows its width without reading it.
    const ONE_COLUMN: bool = false;

    /// Output grid row for f row `i`. Drives both the per-row `NO_PRODUCT` reset and
    /// the kernel's `grid_pos` (through `CellArgs::row_base`), so the two can
    /// never drift.
    fn grid_row(&self, i: usize) -> usize;

    /// One cell of the row.
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError>;

    /// Whether the row loop offers the action [`whole_row`](Self::whole_row).
    const WHOLE_ROWS: bool = false;

    /// The cells of a row whose every column holds one, in column order, as
    /// [`cell`](Self::cell) a column would write them, for an action with a
    /// loop of its own for them: how many columns it wrote, leaving the
    /// cells from there to [`cell`](Self::cell).
    #[inline(always)]
    fn whole_row(&mut self, _eng: &Engine, _row: WholeRow<'_, '_, L, R>) -> Result<usize, OperationError> {
        Ok(0)
    }
}

/// What [`CellAction::whole_row`] reads of a row: [`CellArgs`] without the
/// column and the f row's index.
pub(super) struct WholeRow<'a, 'c, L, R> {
    pub(super) row_base: usize,
    pub(super) f_pairs: &'a [ChildPair],
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) right_level_t: &'a TddLevel,
    pub(super) g_pairs_scratch: &'a mut Vec<ChildPair>,
    pub(super) node_idx: &'a mut [u32],
    pub(super) left: &'a L,
    pub(super) right: &'a R,
    pub(super) gate: &'a mut crate::limits::PollGate<'c>,
}

/// Fill small product grids once before traversal; reset larger grids row by
/// row to keep the current row cache-local. Both paths write the same values.
pub(super) const DEAD_SLAB_FILL_MAX_CELLS: usize = 1 << 16;

/// Extend `slab` to `end` cells, `NO_PRODUCT`-filled, and say whether all
/// of `start..end` was new: the cells of a grid claimed past the slab's end
/// are first written here, a row as the row loop reaches it, into capacity
/// the claim reserved. A range the slab already reached into holds whatever
/// an earlier grid left there.
#[inline(always)]
pub(super) fn extend_slab(slab: &mut Vec<u32>, start: usize, end: usize) -> bool {
    let len = slab.len();
    if len >= end {
        return false;
    }
    debug_assert!(slab.capacity() >= end, "a grid's claim reserves its cells");
    // A row of up to eight cells is written as `mark_dead` writes one, into
    // the spare capacity; a longer one is a `resize`.
    let n = end - len;
    if n <= 8 {
        const DEAD: [MaybeUninit<u32>; 4] = [MaybeUninit::new(NO_PRODUCT); 4];
        let cells = &mut slab.spare_capacity_mut()[..n];
        if n < 4 {
            (cells[0], cells[n / 2], cells[n - 1]) = (DEAD[0], DEAD[0], DEAD[0]);
        } else {
            cells[..4].copy_from_slice(&DEAD);
            cells[n - 4..].copy_from_slice(&DEAD);
        }
        // Safety: the `n` slots past the length were written above.
        unsafe { slab.set_len(end) };
    } else {
        slab.resize(end, NO_PRODUCT);
    }
    len <= start
}

/// The columns of `reach` (at most 64) whose reach meets `mask`, as the
/// bits of a word from the first.
#[inline(always)]
fn reaching(mask: u128, reach: &[u128]) -> u64 {
    debug_assert!(reach.len() <= 64);
    // From the last column down, a shift by one in place of a shift by the
    // column's index.
    reach.iter().rev().fold(0, |live, &r| live << 1 | u64::from(mask & r != 0))
}

/// Word `w` of the g nodes whose reach meets `mask`, read off the reach
/// columns `columns` of `words` words a bit (`liveness::build_reach_columns`):
/// the union of that word of each of the mask's bits that some reach mask has.
#[inline(always)]
fn reach_word(mask: u128, columns: &[u64], words: usize, w: usize) -> u64 {
    let bits = columns.len() / words;
    let mut m = if bits >= 128 { mask } else { mask & ((1u128 << bits) - 1) };
    let mut live = 0;
    while m != 0 {
        let b = m.trailing_zeros() as usize;
        m &= m - 1;
        live |= columns[b * words + w];
    }
    live
}

/// Reset `cells` to `NO_PRODUCT`.
///
/// Most product grids have a few cells, and a `fill` of a length the compiler
/// cannot see is a call to `memset`. Up to eight cells are written here as
/// two four-cell stores that overlap, or three single stores below four
/// cells; a longer run is a `fill`.
#[inline]
pub(super) fn mark_dead(cells: &mut [u32]) {
    const DEAD: [u32; 4] = [NO_PRODUCT; 4];
    let n = cells.len();
    match n {
        0 => {}
        1..=3 => {
            cells[0] = NO_PRODUCT;
            cells[n / 2] = NO_PRODUCT;
            cells[n - 1] = NO_PRODUCT;
        }
        4..=8 => {
            cells[..4].copy_from_slice(&DEAD);
            cells[n - 4..].copy_from_slice(&DEAD);
        }
        _ => cells.fill(NO_PRODUCT),
    }
}

/// [`mark_dead`] behind a call, for the row loop over a level with a
/// complete side. Inlined there, the reset left that loop short of registers
/// and its work counter went to the stack at every poll; the loop over two
/// grids keeps it inlined, where a call would cost more than the stores.
#[inline(never)]
fn mark_dead_out_of_line(cells: &mut [u32]) {
    mark_dead(cells);
}

/// The rows of an implicit level of `width` nodes, as a row loop reads
/// them, into `rows`: every pair the description generates for them, in
/// node order, `k` a node, decoded as the views `left` and `right` read
/// them; returns `k`. One pass over the description in place of a read a
/// row, into the loop's own decode buffer, which the rows of an implicit
/// level leave unused. `None` on a level that stores its pairs, whose rows
/// are slices of its arena, and where the budget refuses the buffer's
/// growth: each row is then generated as it is read.
#[inline(never)]
pub(super) fn described_rows(
    lim: &crate::limits::Limits,
    level: &TddLevel,
    width: usize,
    left: crate::diagram::ChildDecoder,
    right: crate::diagram::ChildDecoder,
    rows: &mut Vec<ChildPair>,
) -> Option<usize> {
    let d = level.implicit()?;
    let k = d.pairs_per_node();
    if width > d.nodes() {
        return None;
    }
    rows.clear();
    lim.reserve(rows, width.checked_mul(k)?).ok()?;
    d.pairs_of_first(width, rows);
    if left.is_marginal() || right.is_marginal() {
        for p in rows.iter_mut() {
            *p = crate::diagram::decoded(*p, left, right);
        }
    }
    Some(k)
}

/// The one row/cell loop of the dense product build.
///
/// `const DENSE` skips the per-row alive-mask fold on levels where it provably
/// cannot skip anything: `ctx.masked` false and neither side a pass-through, where
/// `row_alive_masks` always returns `(0, u128::MAX)`, or neither side able to
/// [kill](ChildLookup::kills) a candidate, where no mask can clear. Only the
/// plain routes reach those regimes; the other three always fold.
///
/// `right_width` and `output_grid_base` are read from `ctx` rather than passed alongside it, so
/// the row reset and the kernel's `grid_pos` derive from the same values by
/// construction.
#[inline(always)]
pub(super) fn run_level_rows<const DENSE: bool, L, R, A>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    action: &mut A,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    A: CellAction<L, R>,
{
    let RowLoop { f_level: left_level_t, g_level: right_level_t, ctx, f_width: left_width, .. } = rows;
    // Named once, ahead of the row loop, so the loop body indexes locals rather
    // than reaching back through the bundle at every cell.
    let RowScratch { f_pairs: f_pairs_scratch, g_pairs: g_pairs_scratch, node_idx: slab } = scratch;
    let lim = eng.limits();
    // Amortized wall-deadline/cancel poll: one read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    debug_assert!(!A::ONE_COLUMN || ctx.right_width == 1, "a one-column action runs on a level of one column");
    let right_width = if A::ONE_COLUMN { 1 } else { ctx.right_width };
    // Whether the per-column cull below can fire at all: the reach masks exist
    // only on a masked level, a pass-through side has no grid to be dead in,
    // and `DENSE` is the regime where the masks are not folded.
    let cull_left = !DENSE && ctx.masked && !left.passthrough() && left.kills();
    let cull_right = !DENSE && ctx.masked && !right.passthrough() && right.kills();
    // Whether a row may read its live columns off the reach columns: every
    // culling side built them.
    let (left_cols, right_cols) = (ctx.sides.left.reach_cols, ctx.sides.right.reach_cols);
    let by_columns = (cull_left || cull_right)
        && (!cull_left || !left_cols.is_empty())
        && (!cull_right || !right_cols.is_empty());
    let words = right_width.div_ceil(64);
    let culls = usize::from(cull_left) + usize::from(cull_right);
    // Whether a row reads its live columns off the row columns: every
    // culling side built them, and a side that culls nothing is read whole.
    let (left_rows, right_rows) = (ctx.sides.left.row_cols, ctx.sides.right.row_cols);
    let by_rows = (cull_left || cull_right) && ctx.sides.left.rows && ctx.sides.right.rows;
    debug_assert!(!by_rows || right_width <= super::super::liveness::ROW_COLUMNS_MAX_WIDTH);
    // A side read whole: every g node, and a live cell in the row.
    let whole = if by_rows { ROW_LIVE | ((1u64 << right_width) - 1) } else { 0 };

    // One slab fill instead of `left_width` row fills. On a dense-slab action the
    // per-row resets below tile `output_grid_base .. output_grid_base + left_width*right_width` exactly once each
    // (`grid_row(i) == i`), so hoisting them into a single `fill` writes the same
    // cells with the same byte pattern; only the order changes, and nothing reads
    // this level's slab between rows (the child lookups read the child levels'
    // disjoint slabs). Above the size gate the per-row form stays: interleaving
    // the reset with the row's cell work is what keeps the active row L1-resident
    // on a large grid.
    //
    // Where neither side can kill a candidate, a dense-slab action writes a
    // node at every cell of a live row and a live column, so only a dead row
    // is reset, and a dead column's cell is reset by the cell
    // ([`process_cell`]): no cell is written twice.
    let every_cell = A::DENSE_SLAB && DENSE && !left.kills() && !right.kills();
    let slab_fill = A::DENSE_SLAB && !every_cell && left_width.saturating_mul(right_width) <= DEAD_SLAB_FILL_MAX_CELLS;
    // A side that cannot kill is a complete one ([`mark_dead_out_of_line`]).
    let reset = |cells: &mut [u32]| {
        if left.kills() && right.kills() { mark_dead(cells) } else { mark_dead_out_of_line(cells) }
    };
    if slab_fill {
        let end = ctx.output_grid_base + left_width * right_width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            reset(&mut slab[ctx.output_grid_base..end]);
        }
    }

    // An implicit f level's rows generated before the loop reads them; a
    // stored level's rows are slices of its arena.
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, left_level_t, left_width, left_view, right_view, f_pairs_scratch);
    for i in 0..left_width {
        let row_base = ctx.output_grid_base + action.grid_row(i) * right_width;
        // A row the slab is extended over is `NO_PRODUCT` already.
        let fresh = extend_slab(slab, row_base, row_base + right_width);
        let node_idx = &mut slab[..];
        if !slab_fill && !every_cell && !fresh {
            reset(&mut node_idx[row_base..row_base + right_width]);
        }

        let f_pairs = match f_described {
            Some(k) => &f_pairs_scratch[i * k..(i + 1) * k],
            None => left_level_t.pairs_view_decoded(i, f_pairs_scratch, left_view, right_view),
        };
        // Empty pairs means dead (zero-containing) node — skip this row.
        if f_pairs.is_empty() {
            if every_cell && !fresh {
                reset(&mut node_idx[row_base..row_base + right_width]);
            }
            continue;
        }

        // The first column the cells below start at.
        let mut first = 0;
        if A::WHOLE_ROWS && !by_rows && !cull_left && !cull_right {
            // The row's every column is a cell, as the loop below would
            // find once the masks' fold kept the row.
            if !DENSE && row_alive_masks(ctx, f_pairs).is_none() {
                continue;
            }
            let row = WholeRow {
                row_base, f_pairs, ctx, right_level_t,
                g_pairs_scratch: &mut *g_pairs_scratch, node_idx: &mut *node_idx, left, right, gate: &mut poll,
            };
            first = action.whole_row(eng, row)?;
            if first == right_width {
                poll.poll(right_width as u64)?;
                continue;
            }
        }
        let mut cell = |j: usize| {
            action.cell(
                eng,
                CellArgs {
                    j,
                    i,
                    row_base,
                    f_pairs,
                    ctx,
                    right_level_t,
                    left,
                    right,
                    g_pairs_scratch: &mut *g_pairs_scratch,
                    node_idx: &mut *node_idx,
                    gate: &mut poll,
                },
            )
        };
        if by_rows {
            // The row's pairs' row columns folded on each culling side:
            // their intersection is the g nodes the reach scan below would find, and
            // a side without a live child row skips the row, as the alive
            // masks' fold would. An OR a pair a side, in place of the
            // masks' fold and a test a g node. A side that culls nothing
            // has a live cell in every child row, its child having columns.
            let l = if cull_left { f_pairs.iter().fold(0, |l, p| l | left_rows[p.left.raw() as usize]) } else { whole };
            let r = if cull_right { f_pairs.iter().fold(0, |r, p| r | right_rows[p.right.raw() as usize]) } else { whole };
            if l & r & ROW_LIVE == 0 {
                continue;
            }
            super::super::note_row_columns_row(!(cull_left && cull_right));
            let mut live = l & r & !ROW_LIVE;
            while live != 0 {
                let j = live.trailing_zeros() as usize;
                live &= live - 1;
                cell(j)?;
            }
            poll.poll(right_width as u64)?;
            continue;
        }
        let (left_alive_mask, right_alive_mask) = if DENSE {
            (0u128, u128::MAX)
        } else {
            match row_alive_masks(ctx, f_pairs) {
                Some(masks) => masks,
                // Row skip: if f[i]'s pairs all reference dead child rows,
                // no cell in this row can produce output.
                None => continue,
            }
        };
        if cull_left || cull_right {
            // The cell's own dead-cell test, taken before the cell is entered.
            // A column whose g node reaches no live child of this row cannot
            // produce anything, whatever arm the cell would have taken, so the
            // cheapest place to find that out is here — ahead of the column's
            // pair slice, the empty test and the arm dispatch. Both sides read
            // the row's mask, which is the union over the row's references, so
            // a clear intersection proves every cell of the column dead.
            //
            // The tests run 64 columns at a time into a word of live columns,
            // and the cells are entered in column order from its bits. Tested
            // ahead of each cell, most of a row's columns failing them, the
            // loop reloaded the context's reach slices at every column, the
            // cell's stores being able to change them as far as the compiler
            // knew, and branched on every test.
            //
            // A row whose masks have few bits reads its live columns off the
            // reach columns instead, a word a mask bit for 64 columns, where
            // the scan tests every column: about two thirds the instructions
            // a word as a column's test takes.
            let bits = if cull_left { left_alive_mask.count_ones() } else { 0 }
                + if cull_right { right_alive_mask.count_ones() } else { 0 };
            if by_columns && 2 * bits as usize * words < 3 * culls * right_width {
                super::super::note_reach_columns_row();
                for w in 0..words {
                    let n = (right_width - w * 64).min(64);
                    let mut live = u64::MAX >> (64 - n);
                    if cull_left {
                        live &= reach_word(left_alive_mask, left_cols, words, w);
                    }
                    if cull_right && live != 0 {
                        live &= reach_word(right_alive_mask, right_cols, words, w);
                    }
                    while live != 0 {
                        let j = w * 64 + live.trailing_zeros() as usize;
                        live &= live - 1;
                        cell(j)?;
                    }
                }
                poll.poll(right_width as u64)?;
                continue;
            }
            let mut j0 = 0;
            while j0 < right_width {
                let n = (right_width - j0).min(64);
                let mut live = u64::MAX >> (64 - n);
                if cull_left {
                    live &= reaching(left_alive_mask, &ctx.sides.left.reach[j0..j0 + n]);
                }
                if cull_right && live != 0 {
                    live &= reaching(right_alive_mask, &ctx.sides.right.reach[j0..j0 + n]);
                }
                while live != 0 {
                    let j = j0 + live.trailing_zeros() as usize;
                    live &= live - 1;
                    cell(j)?;
                }
                j0 += n;
            }
        } else {
            for j in first..right_width {
                cell(j)?;
            }
        }
        // One `poll(right_width)` per row instead of one per cell: the gate only
        // meters accumulated work, so the same total is booked either way. A row
        // wider than the stride now polls once rather than once per stride's worth
        // of cells — the poll is an idempotent deadline read, so firing once per
        // crossing is equivalent, and a row that skipped the `j` loop (empty or
        // dead) books nothing, exactly as before (`right_width == 0` books nothing either).
        poll.poll(right_width as u64)?;
    }
    // The level's residual: what the gate holds is under one stride by
    // construction, and on a level narrower than a stride it is everything.
    poll.flush()?;
    Ok(())
}

/// Materializing action: each surviving cell becomes a product node in the
/// dense `left_width × right_width` output slab, written through the sink
/// `S`. Both dense routes are this action, and the streaming route's with
/// a sink that folds each cell into the level's value column.
pub(super) struct Emit<S> {
    pub(super) sink: S,
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for Emit<S> {
    const DENSE_SLAB: bool = true;
    const WHOLE_ROWS: bool = true;

    #[inline(always)]
    fn whole_row(&mut self, _eng: &Engine, row: WholeRow<'_, '_, L, R>) -> Result<usize, OperationError> {
        emit_row(&mut self.sink, row)
    }

    /// Dense slab: one grid row per f row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        process_cell::<_, _, _>(
            eng,
            a.j,
            a.row_base,
            a.f_pairs,
            a.ctx,
            a.right_level_t,
            a.g_pairs_scratch,
            a.node_idx,
            a.left,
            a.right,
            &mut self.sink,
            a.gate,
        )
    }
}

/// [`Emit`] on a level whose g side is one node of one pair, `g`: every cell
/// is that column's, and its pairs are known before the row loop, which
/// then reads no column.
struct EmitBesideOne<S> {
    sink: S,
    g: [ChildPair; 1],
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for EmitBesideOne<S> {
    const DENSE_SLAB: bool = true;
    const ONE_COLUMN: bool = true;

    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        cell_pairs::<_, _, _>(
            eng,
            a.j,
            a.row_base + a.j,
            a.f_pairs,
            &self.g,
            a.ctx,
            a.node_idx,
            a.left,
            a.right,
            &mut self.sink,
            a.gate,
        )
    }
}

/// The `Route::MarginalChild` row loop: a level with at least one marginal
/// child. Iterates rows `0..left_width` in forward order, running the emit
/// kernel for each cell.
///
/// Never streams: streaming marginal-child levels fold their cells as they
/// find them ([`run_level_rows_stream_count`],
/// [`run_level_rows_stream_words`]) — there is no materialize-then-convert
/// fallback for them.
pub(crate) fn run_level_rows_marginal(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
) -> Result<(), OperationError> {
    let left = MarginalLookup::new(&rows.ctx.sides.left);
    let right = MarginalLookup::new(&rows.ctx.sides.right);
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut Emit { sink: EmitSink { level } },
    )
}

/// [`EmitSink`] on a level built a row at a time in one reused grid row
/// starting at `base`: each cell that gets a node is recorded as a
/// `ProductEntry` of the current [row](PairSink::row) instead of being left
/// in a dense slab.
pub(super) struct ListSink<'a> {
    pub(super) emit: EmitSink<'a>,
    pub(super) product_list: &'a mut Vec<ProductEntry>,
    pub(super) base: usize,
    pub(super) row: usize,
}

impl ListSink<'_> {
    /// Record the cell at `grid_pos` if it got a node.
    #[inline(always)]
    fn record(&mut self, eng: &Engine, node_idx: &[u32], grid_pos: usize) -> Result<(), OperationError> {
        let nid = node_idx[grid_pos];
        if nid == NO_PRODUCT {
            return Ok(());
        }
        let entry = ProductEntry {
            f_idx: FNodeIdx(self.row as u32),
            g_idx: GNodeIdx((grid_pos - self.base) as u32),
            prod_idx: ProductNodeIdx(nid),
        };
        eng.limits().try_push(self.product_list, entry)
    }
}

impl PairSink for ListSink<'_> {
    #[inline(always)]
    fn single(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.emit.single(eng, node_idx, grid_pos, lc, rc)?;
        self.record(eng, node_idx, grid_pos)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.emit.begin()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.emit.pair(eng, lc, rc)
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.emit.buf()
    }

    #[inline(always)]
    fn end(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, start: usize) -> Result<(), OperationError> {
        self.emit.end(eng, node_idx, grid_pos, start)?;
        self.record(eng, node_idx, grid_pos)
    }

    #[inline(always)]
    fn row(&mut self, i: usize) {
        self.row = i;
    }
}

/// Sparse-output action: emit into the reused row scratch through a
/// [`ListSink`], which records each surviving cell as a `ProductEntry`.
struct SparseMargEmit<'a> {
    sink: ListSink<'a>,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for SparseMargEmit<'_> {
    /// Not a dense slab — every structural row is rebuilt in the same `right_width`-wide
    /// scratch row, so its `NO_PRODUCT` reset must fire between rows and cannot be
    /// hoisted into a one-shot slab fill.
    const DENSE_SLAB: bool = false;

    /// One reused `right_width`-wide row scratch instead of a dense slab: every
    /// structural row builds at grid row 0, so `row_base == ctx.output_grid_base`.
    /// The true row index survives only in `CellArgs::i`, which the product
    /// entry's `f_idx` reads below.
    #[inline(always)]
    fn grid_row(&self, _i: usize) -> usize {
        0
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        self.sink.row(a.i);
        process_cell::<_, _, _>(
            eng,
            a.j,
            a.row_base,
            a.f_pairs,
            a.ctx,
            a.right_level_t,
            a.g_pairs_scratch,
            a.node_idx,
            a.left,
            a.right,
            &mut self.sink,
            a.gate,
        )
    }
}

/// The `Route::SparseMarg` row loop: an *exactly-one*-marginal-child level
/// whose output is structural (never a marginalization target).
///
/// Same emit kernel as [`run_level_rows_marginal`], but instead of a dense
/// `left_width*right_width` slab it reuses one `right_width`-wide row scratch
/// at `cell_ctx.output_grid_base` and records each surviving cell in
/// `product_list`. The marginal child is a pass-through carrier that kills no
/// pair, so the structural sibling alone decides which cells are alive and a
/// dense slab would be mostly `NO_PRODUCT`. The parent densifies the
/// `product_list` through `ensure_grid` when it needs the grid.
///
/// Every cell is built at grid row 0, so the kernel's
/// `grid_pos == cell_ctx.output_grid_base + j`, then read back from the scratch
/// and, if alive, pushed as
/// `ProductEntry { f_idx: row i, g_idx: col j, prod_idx: node }`.
///
/// Never streams: [`Route::SparseMarg`](crate::apply::conjoin::route::Route::SparseMarg)
/// is chosen only for a level that is not a marginalization target.
pub(crate) fn run_level_rows_marginal_sparse(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    product_list: &mut Vec<ProductEntry>,
) -> Result<(), OperationError> {
    let left = MarginalLookup::new(&rows.ctx.sides.left);
    let right = MarginalLookup::new(&rows.ctx.sides.right);
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut SparseMargEmit {
            sink: ListSink { emit: EmitSink { level }, product_list, base: rows.ctx.output_grid_base, row: 0 },
        },
    )
}

/// The row loop of `Route::PlainDense` and `Route::Dense`: a level with no
/// marginal child. Iterates rows in forward order.
/// Runs the emit kernel with positional grid lookups on both sides (no
/// pass-through, no mask decode — the caller guarantees no marginal child).
///
/// `DENSE = true` asserts two level-invariant facts, `cell_ctx.masked ==
/// false` and no pass-through side, so the inner loop carries no branch on
/// them; `DENSE = false` keeps the masked row-skip checks.
///
/// Never streams: streaming levels fold their cells as they find them
/// ([`run_level_rows_stream_count`], [`run_level_rows_stream_words`]) —
/// there is no post-cell snapshot conversion.
#[inline(always)]
pub(crate) fn run_level_rows_plain<const DENSE: bool>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
) -> Result<(), OperationError> {
    let left = DenseLookup { base: rows.ctx.sides.left.base, stride: rows.ctx.sides.left.stride };
    let right = DenseLookup { base: rows.ctx.sides.right.base, stride: rows.ctx.sides.right.stride };
    // When `DENSE`, the alive masks are constants (no masks, no pass-through):
    //   left_alive_mask  = 0u128      (the !masked branch of `row_alive_masks`)
    //   right_alive_mask = `u128::MAX`  (the `|| !masked` branch of `row_alive_masks`)
    // The driver passes those directly to the kernel, skipping the fold.
    let mut action = Emit { sink: EmitSink { level } };
    run_level_rows::<DENSE, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut action,
    )
}

/// The row loop of `Route::PlainDense` on a level where neither operand has
/// a multi-pair node, so that every cell is one pair against one:
/// [`run_level_rows_plain`] with the cell's column read and arm dispatch
/// taken out of it. g's pairs are read into one array before the rows; a
/// cell is then two grid reads and an emit. Same nodes
/// in the same order, and the work clock is charged per row what that
/// loop's cells and row charge it: a unit a cell of a live column, and the
/// row's width. `relay` says the level may be built as a description
/// ([`relay_rows`]): nothing filters its nodes.
pub(crate) fn run_level_rows_single(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    relay: bool,
) -> Result<(), OperationError> {
    let RowLoop { f_level, g_level, ctx, f_width, .. } = rows;
    let RowScratch { f_pairs: f_scratch, g_pairs: g_cols, node_idx: slab } = scratch;
    let lim = eng.limits();
    super::super::note_single_rows();
    let right_width = ctx.right_width;
    // Column `j`'s pair, a dead column's `NO_PRODUCT` on both sides; the f
    // rows' buffer decodes a column the table does not hold.
    g_cols.clear();
    lim.reserve_exact(g_cols, right_width)?;
    let dead = ChildPair::new(EncodedChildRef::from_raw(NO_PRODUCT), EncodedChildRef::from_raw(NO_PRODUCT));
    for j in 0..right_width {
        let pairs = column_pairs(ctx, g_level, j, f_scratch);
        debug_assert!(pairs.len() <= 1, "a g node of one pair at most");
        g_cols.push(pairs.first().copied().unwrap_or(dead));
    }
    let rows = SingleRows { f_level, ctx, f_width, g_cols: &g_cols[..] };
    if relay && !super::super::relay_forced_off() && relay_rows(eng, rows, slab, level)? {
        super::super::note_relayed();
        return Ok(());
    }
    let mut sink = EmitSink { level };
    // A level of a few columns keeps them in registers, and writes each
    // row's cells once.
    match right_width {
        1 => single_rows::<1>(eng, rows, f_scratch, slab, &mut sink),
        2 => single_rows::<2>(eng, rows, f_scratch, slab, &mut sink),
        3 => single_rows::<3>(eng, rows, f_scratch, slab, &mut sink),
        4 => single_rows::<4>(eng, rows, f_scratch, slab, &mut sink),
        _ => single_rows::<0>(eng, rows, f_scratch, slab, &mut sink),
    }
}

/// What the row loop of [`run_level_rows_single`] reads.
#[derive(Clone, Copy)]
struct SingleRows<'a> {
    f_level: &'a TddLevel,
    ctx: &'a CellCtx<'a>,
    f_width: usize,
    g_cols: &'a [ChildPair],
}

/// Build a level of one-pair cells whose two operand levels relay their
/// right children, where it can, and answer whether it did.
///
/// Where f's node `i` is `(a, b + i)`, an implicit level of one node digit,
/// and g's column `j` is `(c, e + j)` for every column, cell `(i, j)` is
/// the product of the left child's product `P = (a, c)` with the right
/// child's product `(b + i, e + j)`: live where that one is. Where the
/// right child's products in the window of the grid the level reads are
/// numbered in row-major order from some `v`, the level's node `n` is
/// `(P, v + n)`: the level is written as that description, of one node
/// digit, its cells as the window's renumbered from 0, the work clock
/// polled as the row loop polls it and the node arena's growth charged as
/// the row loop's pushes would charge it. The description stands for the
/// arenas the row loop would leave, whose scratch it does not use, so it
/// is written only where nothing bounds memory.
///
/// A level of fewer nodes than the floor, or whose window is not so
/// numbered, is left to the row loop, its cells perhaps written, which the
/// row loop writes again, and the level and the limits as they were.
fn relay_rows(
    eng: &Engine,
    rows: SingleRows<'_>,
    slab: &mut Vec<u32>,
    level: &mut TddLevel,
) -> Result<bool, OperationError> {
    let SingleRows { f_level, ctx, f_width, g_cols } = rows;
    let lim = eng.limits();
    let width = ctx.right_width;
    let (left, right) = (&ctx.sides.left, &ctx.sides.right);
    if !lim.memory_unbounded()
        || crate::diagram::stored_levels_forced()
        || left.plan.view.is_marginal()
        || right.plan.view.is_marginal()
        || width == 0
    {
        return Ok(false);
    }
    let Some(d) = f_level.implicit() else { return Ok(false) };
    let digit = crate::diagram::Digit { radix: f_width, left: 0, right: 1, node: 1 };
    if d.pairs_per_node() != 1 || d.nodes() != f_width || d.within() != 0 || d.digits() != [digit] {
        return Ok(false);
    }
    // A description's slots are below 2^31.
    let (a, b) = (d.first().0 as usize, d.first().1 as usize);
    let (c, e) = (g_cols[0].left.0, g_cols[0].right.0);
    if c == NO_PRODUCT
        || !g_cols.iter().enumerate().all(|(j, q)| q.left.0 == c && u64::from(q.right.0) == u64::from(e) + j as u64)
    {
        return Ok(false);
    }
    let p = match slab.get(left.base + a * left.stride as usize + c as usize) {
        Some(&p) if p != NO_PRODUCT => p,
        _ => return Ok(false),
    };
    let stride = right.stride as usize;
    let src = |i: usize| right.base + (b + i) * stride + e as usize;
    let (dst, cells) = (ctx.output_grid_base, f_width * width);
    let end = dst + cells;
    if src(f_width - 1) + width > slab.len() || (slab.len() < end && slab.capacity() < end) {
        return Ok(false);
    }
    if slab.len() < end {
        slab.resize(end, NO_PRODUCT);
    }
    debug_assert!(src(f_width - 1) + width <= dst || end <= src(0), "the child's grid and the level's are apart");
    if stride == width {
        slab.copy_within(src(0)..src(0) + cells, dst);
    } else {
        for i in 0..f_width {
            slab.copy_within(src(i)..src(i) + width, dst + i * width);
        }
    }
    let out = &mut slab[dst..end];
    let Some(&v) = out.iter().find(|&&x| x != NO_PRODUCT) else { return Ok(false) };
    let mut next = v;
    for run in out.chunks(1 << 12) {
        let ordered = run.iter().fold(true, |ordered, &x| {
            let live = x != NO_PRODUCT;
            let at = x == next;
            next += u32::from(live);
            ordered & (!live | at)
        });
        if !ordered {
            return Ok(false);
        }
    }
    let nodes = (next - v) as usize;
    if nodes < crate::diagram::floor().max(2) {
        return Ok(false);
    }
    let Some(described) = crate::diagram::ImplicitLevel::one_step(nodes, (i64::from(p), i64::from(v)), (0, 1)) else {
        return Ok(false);
    };
    if v != 0 {
        for x in out.iter_mut().filter(|x| **x != NO_PRODUCT) {
            *x -= v;
        }
    }
    // The row loop's polls: a row's units, a unit a live cell and its
    // width, added while the gate does not come due inside the row, and the
    // row's cells polled one at a time and then its width where it does.
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    let row_units = 2 * width as u64;
    let mut rows_left = f_width as u64;
    while rows_left > 0 {
        let short = (poll.due_in().saturating_sub(1) / row_units).min(rows_left);
        poll.add_short(short * row_units);
        rows_left -= short;
        if rows_left > 0 {
            poll = poll_each_then_due(poll, width as u64, width as u64)?;
            rows_left -= 1;
        }
    }
    poll.flush()?;
    debug_assert!(level.nodes().is_empty() && level.pairs.is_empty());
    let held = level.nodes.imply();
    let node_capacity = lim.charge_as_pushed::<EncodedNode>(held, nodes)?;
    let capacity = level.pairs.capacity();
    level.pairs.describe(described, capacity, node_capacity);
    Ok(true)
}

/// The row loop of [`run_level_rows_single`] on a level of `W` columns, or
/// of any number for `W = 0`.
///
/// With `W` columns a row's cells are found and then written once, a node
/// index or `NO_PRODUCT` each, in place of the reset and the writes of its
/// live cells; the row's single f pair is read off its node where the f
/// level stores its nodes and neither side is marginal.
#[inline(never)]
fn single_rows<const W: usize>(
    eng: &Engine,
    rows: SingleRows<'_>,
    f_scratch: &mut Vec<ChildPair>,
    slab: &mut Vec<u32>,
    sink: &mut EmitSink<'_>,
) -> Result<(), OperationError> {
    let SingleRows { f_level, ctx, f_width, g_cols } = rows;
    let lim = eng.limits();
    let right_width = ctx.right_width;
    debug_assert!(W == 0 || W == right_width);
    let width = if W == 0 { right_width } else { W };
    let live_cols = g_cols.iter().filter(|q| q.left.0 != NO_PRODUCT).count() as u64;
    // A row's charge: a unit a live cell and the row's width.
    let row_units = live_cols + width as u64;
    let left = DenseLookup { base: ctx.sides.left.base, stride: ctx.sides.left.stride };
    let right = DenseLookup { base: ctx.sides.right.base, stride: ctx.sides.right.stride };
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, f_level, f_width, left_view, right_view, f_scratch);
    // The f nodes, where a row's pair is its node's.
    let f_nodes = match f_level.stored() {
        Some(stored) if f_described.is_none() && !left_view.is_marginal() && !right_view.is_marginal() => Some(stored.nodes()),
        _ => None,
    };
    let g: [ChildPair; W] = std::array::from_fn(|j| g_cols[j]);

    // The resets of `run_level_rows`, where the rows' cells are not written
    // once: one fill of a small grid, a row at a time on a larger one.
    let slab_fill = f_width.saturating_mul(right_width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if W == 0 && slab_fill {
        let end = ctx.output_grid_base + f_width * right_width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            mark_dead(&mut slab[ctx.output_grid_base..end]);
        }
    }
    // With `W` columns and the f nodes stored, the grid extended over the
    // level's rows ahead of them, each row's cells written in place.
    let extended = W > 0 && f_nodes.is_some() && {
        let end = ctx.output_grid_base + f_width * width;
        if slab.len() < end && slab.capacity() >= end {
            slab.resize(end, NO_PRODUCT);
        }
        slab.len() >= end
    };
    let mut next = 0;
    'rows: while next < f_width {
        // The rows' nodes are written straight into the node arena, open
        // across rows, while it has room for a node a column; a row it has
        // no room for goes through the sink, which grows it.
        let mut out = sink.open_row(0, 0);
        // The rows the node arena has room for, read with no test of the
        // room, the grid's length or the gate a row: the gate's units
        // counted down here, and polled as below at the row it comes due.
        if extended
            && let (Some(nodes), Some(o)) = (f_nodes, out.as_mut())
            && o.node_room() >= width
        {
            let end = f_width.min(next + o.node_room() / width);
            let (mut due_at, mut due) = (poll.due_in(), poll.due_in());
            while next < end {
                let i = next;
                next += 1;
                let row_base = ctx.output_grid_base + i * width;
                let p = match nodes[i].kind() {
                    NodeKind::Inline(p) => Some(p),
                    _ => f_level.pairs_read(i, f_scratch).first().copied(),
                };
                let mut cells = [NO_PRODUCT; W];
                if let Some(p) = p {
                    let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
                    for (cell, q) in cells.iter_mut().zip(&g) {
                        if q.left.0 == NO_PRODUCT {
                            continue;
                        }
                        let lc = left.get_in_row(slab, lrow, q.left.0);
                        if lc == NO_PRODUCT {
                            continue;
                        }
                        let rc = right.get_in_row(slab, rrow, q.right.0);
                        if rc != NO_PRODUCT {
                            // Safety: the arena has room for a node a column
                            // of the rows up to `end`.
                            *cell = unsafe { o.single_node(lc, rc) };
                        }
                    }
                }
                debug_assert!(row_base + W <= slab.len(), "the grid is extended over the level");
                // Safety: the grid was extended over every row of the level.
                unsafe { slab.get_unchecked_mut(row_base..row_base + W) }.copy_from_slice(&cells);
                if p.is_none() {
                    continue;
                }
                if row_units < due {
                    due -= row_units;
                } else {
                    poll.add_short(due_at - due);
                    poll = poll_each_then_due(poll, live_cols, width as u64)?;
                    (due_at, due) = (poll.due_in(), poll.due_in());
                }
            }
            poll.add_short(due_at - due);
            continue 'rows;
        }
        while next < f_width {
            let i = next;
            next += 1;
            let row_base = ctx.output_grid_base + i * width;
            let p = match (f_nodes, f_described) {
                (Some(nodes), _) => match nodes[i].kind() {
                    NodeKind::Inline(p) => Some(p),
                    _ => f_level.pairs_read(i, f_scratch).first().copied(),
                },
                (None, Some(k)) => f_scratch[i * k..(i + 1) * k].first().copied(),
                (None, None) => f_level.pairs_view_decoded(i, f_scratch, left_view, right_view).first().copied(),
            };
            debug_assert!(f_level.pairs_view_decoded(i, &mut Vec::new(), left_view, right_view).len() <= 1, "an f node of one pair at most");
            if W == 0 {
                let fresh = extend_slab(slab, row_base, row_base + width);
                if !slab_fill && !fresh {
                    mark_dead(&mut slab[row_base..row_base + width]);
                }
            }
            // A dead f node's row stays dead.
            let Some(p) = p else {
                if W > 0 {
                    write_cells(slab, row_base, [NO_PRODUCT; W]);
                }
                continue;
            };
            let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
            // A cell's product: `NO_PRODUCT` where a side has none.
            let product = |node_idx: &[u32], q: &ChildPair| {
                if q.left.0 == NO_PRODUCT {
                    return None;
                }
                let lc = left.get_in_row(node_idx, lrow, q.left.0);
                if lc == NO_PRODUCT {
                    return None;
                }
                let rc = right.get_in_row(node_idx, rrow, q.right.0);
                (rc != NO_PRODUCT).then_some((lc, rc))
            };
            if let Some(o) = out.as_mut().filter(|o| o.has_room(0, width)) {
                if W > 0 {
                    let mut cells = [NO_PRODUCT; W];
                    for (cell, q) in cells.iter_mut().zip(&g) {
                        if let Some((lc, rc)) = product(slab, q) {
                            // Safety: the arena has room for a node a column.
                            *cell = unsafe { o.single_node(lc, rc) };
                        }
                    }
                    write_cells(slab, row_base, cells);
                } else {
                    for (j, q) in g_cols.iter().enumerate() {
                        if let Some((lc, rc)) = product(slab, q) {
                            // Safety: as above.
                            unsafe { o.single(slab, row_base + j, lc, rc) };
                        }
                    }
                }
                // A unit a live cell, polled as the row loop polls them,
                // in one add where the gate does not come due inside.
                if row_units < poll.due_in() {
                    poll.add_short(row_units);
                } else {
                    poll = poll_each_then_due(poll, live_cols, width as u64)?;
                }
                continue;
            }
            drop(out);
            if W > 0 {
                write_cells(slab, row_base, [NO_PRODUCT; W]);
            }
            for (j, q) in g_cols.iter().enumerate() {
                if let Some((lc, rc)) = product(slab, q) {
                    sink.single(eng, slab, row_base + j, lc, rc)?;
                }
            }
            if row_units < poll.due_in() {
                poll.add_short(row_units);
            } else {
                poll = poll_each_then_due(poll, live_cols, width as u64)?;
            }
            continue 'rows;
        }
    }
    poll.flush()?;
    Ok(())
}

/// Write `cells` at `start` of `slab`, extending it into the capacity a
/// grid's claim reserved where they reach past its end, any gap before them
/// `NO_PRODUCT`.
#[inline(always)]
pub(super) fn write_cells<const W: usize>(slab: &mut Vec<u32>, start: usize, cells: [u32; W]) {
    let end = start + W;
    let len = slab.len();
    if len >= end {
        slab[start..end].copy_from_slice(&cells);
        return;
    }
    debug_assert!(slab.capacity() >= end, "a grid's claim reserves its cells");
    if len < start {
        slab.resize(start, NO_PRODUCT);
    }
    let have = slab.len() - start;
    slab[start..].copy_from_slice(&cells[..have]);
    for (slot, &cell) in slab.spare_capacity_mut()[..W - have].iter_mut().zip(&cells[have..]) {
        slot.write(cell);
    }
    // Safety: the slots up to `end` were written above.
    unsafe { slab.set_len(end) };
}

/// Which plain-route child sides are [complete](super::super::products::Products::is_complete)
/// and read by arithmetic, and how a level whose two sides both are writes
/// its pairs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PlainLookups {
    /// Both sides read their grids.
    Grid,
    /// The left side is complete; the right reads its grid. `charged` as in
    /// [`PlainLookups::Complete`], for an arena reserved for every pair the
    /// level would write if none died.
    CompleteLeft { charged: usize },
    /// The right side is complete; the left reads its grid. `charged` as in
    /// [`PlainLookups::CompleteLeft`].
    CompleteRight { charged: usize },
    /// Both sides are complete, so no candidate dies and the level's pairs
    /// are known before its row loop. `charged` is the pair capacity the
    /// meters have been charged for when the arena was reserved to hold every
    /// pair, `usize::MAX` when it was not (see [`ReservedEmitSink`]).
    Complete { charged: usize },
}

/// The row loop of `Route::PlainDense` and `Route::Dense` on a level with at
/// least one [complete](super::super::products::Products::is_complete) child
/// side: [`CompleteLookup`] on each complete side, the grid on the other.
///
/// With both sides complete no candidate dies and no dead-pair mask can cull,
/// so the loop runs without the per-row mask fold (`DENSE`), with no
/// `NO_PRODUCT` test, and writes through a [`ReservedEmitSink`]; where the g
/// side is one node of one pair, every cell reads that pair
/// ([`EmitBesideOne`]) and no column is read. With one
/// side complete it is [`run_level_rows_plain::<false>`](run_level_rows_plain)
/// with that side's lookup replaced: the fold and the culls on the grid side
/// stay; on a level that built no masks they find nothing, as on
/// `Route::PlainDense` they would not be run. It writes through a
/// [`ReservedEmitSink`] too, its arena reserved for every pair the level
/// would write if none died.
pub(crate) fn run_level_rows_complete(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    lookups: PlainLookups,
) -> Result<(), OperationError> {
    let sides = rows.ctx.sides;
    let complete = |p: &ChildPlan<'_>| CompleteLookup { stride: p.stride };
    let grid = |p: &ChildPlan<'_>| DenseLookup { base: p.base, stride: p.stride };
    match lookups {
        PlainLookups::Complete { charged } => match lone_pair(rows, &mut *scratch.g_pairs) {
            Some(g) => run_level_rows::<true, _, _, _>(
                eng, rows, scratch,
                &complete(&sides.left), &complete(&sides.right),
                &mut EmitBesideOne { sink: ReservedEmitSink { level, charged }, g: [g] },
            ),
            None => run_level_rows::<true, _, _, _>(
                eng, rows, scratch,
                &complete(&sides.left), &complete(&sides.right),
                &mut Emit { sink: ReservedEmitSink { level, charged } },
            ),
        },
        PlainLookups::CompleteLeft { charged } => {
            let lone = lone_columns(eng, rows, &mut *scratch.g_pairs)?;
            let (left, right) = (complete(&sides.left), grid(&sides.right));
            match &lone {
                Some(lone) => run_level_rows::<false, _, _, _>(
                    eng, rows, scratch, &left, &right,
                    &mut EmitLoneColumns { sink: ReservedEmitSink { level, charged }, lone },
                ),
                None => run_level_rows::<false, _, _, _>(
                    eng, rows, scratch, &left, &right,
                    &mut Emit { sink: ReservedEmitSink { level, charged } },
                ),
            }
        }
        PlainLookups::CompleteRight { charged } => {
            let lone = lone_columns(eng, rows, &mut *scratch.g_pairs)?;
            let (left, right) = (grid(&sides.left), complete(&sides.right));
            match &lone {
                Some(lone) => run_level_rows::<false, _, _, _>(
                    eng, rows, scratch, &left, &right,
                    &mut EmitLoneColumns { sink: ReservedEmitSink { level, charged }, lone },
                ),
                None => run_level_rows::<false, _, _, _>(
                    eng, rows, scratch, &left, &right,
                    &mut Emit { sink: ReservedEmitSink { level, charged } },
                ),
            }
        }
        PlainLookups::Grid => unreachable!("a level with no complete side takes the grid row loop"),
    }
}

/// Each column's g pair, where every column of the level holds one: the
/// pair [`EmitLoneColumns`] reads a cell of the column against.
fn lone_columns(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: &mut Vec<ChildPair>,
) -> Result<Option<Vec<ChildPair>>, OperationError> {
    if super::super::lone_columns_forced_off() {
        return Ok(None);
    }
    let width = rows.ctx.right_width;
    let mut lone = Vec::new();
    eng.limits().reserve_exact(&mut lone, width)?;
    for j in 0..width {
        match column_pairs(rows.ctx, rows.g_level, j, scratch) {
            &[pair] => lone.push(pair),
            _ => return Ok(None),
        }
    }
    super::super::note_lone_columns();
    Ok(Some(lone))
}

/// [`Emit`] on a level with a complete side whose every column holds one g
/// pair, `lone[j]` column `j`'s: a cell walks its row's f pairs against that
/// pair as [`cell_pairs`] walks a cell of one g pair, with no column read
/// and no test of the cell's shape. A cell whose pairs fit the arena's room
/// is written there ([`ArenaCell`]); the others go through the sink.
struct EmitLoneColumns<'a, 'l> {
    sink: ReservedEmitSink<'l>,
    lone: &'a [ChildPair],
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for EmitLoneColumns<'_, '_> {
    const DENSE_SLAB: bool = true;
    const WHOLE_ROWS: bool = true;

    /// A row of one or two f pairs whose cells all fit the arenas' room is
    /// written there in one loop ([`lone_row`]).
    #[inline(always)]
    fn whole_row(&mut self, _eng: &Engine, row: WholeRow<'_, '_, L, R>) -> Result<usize, OperationError> {
        match row.f_pairs.len() {
            1 => lone_row::<1, L, R>(&mut self.sink, self.lone, row),
            2 => lone_row::<2, L, R>(&mut self.sink, self.lone, row),
            _ => Ok(0),
        }
    }

    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        let CellArgs { j, row_base, f_pairs, node_idx, left, right, gate, .. } = a;
        let q = self.lone[j];
        let (c, d) = (q.left.0, q.right.0);
        let grid_pos = row_base + j;
        let kills = left.kills() || right.kills();
        gate.poll(f_pairs.len() as u64)?;
        let grid = &*node_idx;
        // An f pair against the column's pair: `NO_PRODUCT` on both sides
        // where the left one dies.
        let cand = |p: &ChildPair| {
            let lc = left.get(grid, p.left.0, c);
            if left.kills() && lc == NO_PRODUCT {
                return (lc, lc);
            }
            (lc, right.get(grid, p.right.0, d))
        };
        if let [p] = f_pairs {
            let (lc, rc) = cand(p);
            if kills && rc == NO_PRODUCT {
                return Ok(());
            }
            return self.sink.single(eng, node_idx, grid_pos, lc, rc);
        }
        if let Some(mut cell) = self.sink.open_cell(f_pairs.len()) {
            for p in f_pairs {
                let (lc, rc) = cand(p);
                if !kills || rc != NO_PRODUCT {
                    // Safety: the cell was opened for every f pair of the row.
                    unsafe { cell.push(lc, rc) };
                }
            }
            return cell.close(eng, node_idx, grid_pos);
        }
        let start = self.sink.begin();
        push_kept(eng, &mut self.sink, f_pairs, kills, cand)?;
        self.sink.end(eng, node_idx, grid_pos, start)
    }
}

/// [`EmitLoneColumns`] on a row of `K` f pairs whose every cell the
/// arenas have room for ([`ArenaRow`]): each f pair's child rows read once
/// for the row, each cell's survivors written straight into the arena and
/// closed into its node, and the gate counted down a cell at a time and
/// polled at the cell that brings it due, as a poll a cell would charge it.
#[inline(always)]
fn lone_row<const K: usize, L: ChildLookup, R: ChildLookup>(
    sink: &mut ReservedEmitSink<'_>,
    lone: &[ChildPair],
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, node_idx, left, right, gate, .. } = row;
    let Ok(f) = <&[ChildPair; K]>::try_from(f_pairs) else { return Ok(0) };
    let Some(mut out) = sink.open_row(lone.len() * K, lone.len()) else { return Ok(0) };
    super::super::note_whole_row();
    let rows: [(L::Row, R::Row); K] = std::array::from_fn(|t| (left.row(f[t].left.0), right.row(f[t].right.0)));
    let units = K as u64;
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    for (j, q) in lone.iter().enumerate() {
        if units >= due {
            gate.add_short(due_at - due);
            gate.poll(units)?;
            (due_at, due) = (gate.due_in(), gate.due_in());
        } else {
            due -= units;
        }
        let (c, d) = (q.left.0, q.right.0);
        for &(l, r) in &rows {
            let lc = left.get_in_row(node_idx, l, c);
            if left.kills() && lc == NO_PRODUCT {
                continue;
            }
            let rc = right.get_in_row(node_idx, r, d);
            if right.kills() && rc == NO_PRODUCT {
                continue;
            }
            // Safety: the row was opened for `K` pairs a column.
            unsafe { out.push(lc, rc) };
        }
        // Safety: the row was opened for a node a column.
        unsafe { out.close(node_idx, row_base + j) };
    }
    gate.add_short(due_at - due);
    Ok(lone.len())
}

/// The most f pairs a row [`emit_row`] writes holds.
const ROW_PAIRS: usize = 8;

/// [`Emit`] on a row of at most [`ROW_PAIRS`] f pairs: its cells in one
/// loop ([`ArenaRow`]), each f pair's child rows read once for the row, up
/// to the first cell of more than one pair on both sides, or the first that
/// does not fit the arena's room, which go through [`process_cell`]. A cell
/// of one pair on a side is walked against the other side's pairs as
/// [`cell_pairs`] walks it, and the gate counted down by the cell's pairs
/// and polled at the cell that brings it due, as that walk's one poll a
/// cell charges it.
#[inline(always)]
fn emit_row<L: ChildLookup, R: ChildLookup, S: PairSink>(
    sink: &mut S,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, right_level_t, g_pairs_scratch, node_idx, left, right, gate } = row;
    let (k, width) = (f_pairs.len(), ctx.right_width);
    if k > ROW_PAIRS {
        return Ok(0);
    }
    let Some(mut out) = sink.open_row(0, width) else { return Ok(0) };
    let child_rows = |p: &ChildPair| (left.row(p.left.0), right.row(p.right.0));
    let mut rows = [child_rows(&f_pairs[0]); ROW_PAIRS];
    for (t, p) in f_pairs.iter().enumerate().skip(1) {
        rows[t] = child_rows(p);
    }
    let rows = &rows[..k];
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let mut j = 0;
    while j < width {
        let g_pairs = column_pairs(ctx, right_level_t, j, g_pairs_scratch);
        let m = g_pairs.len();
        if m == 0 {
            // A dead column's cell, as `process_cell` leaves it.
            node_idx[row_base + j] = NO_PRODUCT;
            j += 1;
            continue;
        }
        if (k > 1 && m > 1) || !out.fits(k * m) {
            break;
        }
        let units = (k * m) as u64;
        if units >= due {
            gate.add_short(due_at - due);
            gate.poll(units)?;
            (due_at, due) = (gate.due_in(), gate.due_in());
        } else {
            due -= units;
        }
        let mut cand = |(l, r): (L::Row, R::Row), q: &ChildPair| {
            let lc = left.get_in_row(node_idx, l, q.left.0);
            if left.kills() && lc == NO_PRODUCT {
                return;
            }
            let rc = right.get_in_row(node_idx, r, q.right.0);
            if right.kills() && rc == NO_PRODUCT {
                return;
            }
            // Safety: the cell fits its `k * m` candidates.
            unsafe { out.push(lc, rc) };
        };
        match g_pairs {
            [q] => rows.iter().for_each(|&f_rows| cand(f_rows, q)),
            _ => g_pairs.iter().for_each(|q| cand(rows[0], q)),
        }
        // Safety: the row was opened for a node a column.
        unsafe { out.close(node_idx, row_base + j) };
        j += 1;
    }
    gate.add_short(due_at - due);
    Ok(j)
}

/// The one pair of a level's g side, where it has one node and that node
/// one pair: the pair every cell of the level reads as its column's.
fn lone_pair(rows: RowLoop<'_>, scratch: &mut Vec<ChildPair>) -> Option<ChildPair> {
    if rows.ctx.right_width != 1 {
        return None;
    }
    match column_pairs(rows.ctx, rows.g_level, 0, scratch) {
        &[pair] => {
            super::super::note_lone_pair();
            Some(pair)
        }
        _ => None,
    }
}
