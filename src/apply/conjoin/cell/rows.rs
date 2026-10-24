//! The row loop every build route shares, and the four routes themselves.

use super::*;
use super::super::liveness::ROW_LIVE;
use crate::diagram::EncodedChildRef;

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
}

/// Fill small product grids once before traversal; reset larger grids row by
/// row to keep the current row cache-local. Both paths write the same values.
const DEAD_SLAB_FILL_MAX_CELLS: usize = 1 << 16;

/// Extend `slab` to `end` cells, `NO_PRODUCT`-filled, and say whether all
/// of `start..end` was new: the cells of a grid claimed past the slab's end
/// are first written here, a row as the row loop reaches it, into capacity
/// the claim reserved. A range the slab already reached into holds whatever
/// an earlier grid left there.
#[inline(always)]
fn extend_slab(slab: &mut Vec<u32>, start: usize, end: usize) -> bool {
    let len = slab.len();
    if len >= end {
        return false;
    }
    debug_assert!(slab.capacity() >= end, "a grid's claim reserves its cells");
    slab.resize(end, NO_PRODUCT);
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
/// the OR of that word of each of the mask's bits that some reach mask has.
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
fn described_rows(
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
    let mut poll = lim.gate_with(super::super::budget::DENSE_CELL_POLL_STRIDE);
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
            // their AND is the g nodes the reach scan below would find, and
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
            for j in 0..right_width {
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
/// `S`. Both dense routes are this action.
struct Emit<S> {
    sink: S,
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for Emit<S> {
    const DENSE_SLAB: bool = true;

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
/// Never streams: streaming marginal-child levels take the collapse-at-source
/// walker ([`run_level_rows_stream_count`]) unconditionally — there is no
/// materialize-then-convert fallback for them.
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

/// Sparse-output action: emit into the reused row scratch, then record each
/// surviving cell as a `ProductEntry` instead of leaving it in a dense slab.
struct SparseMargEmit<'a> {
    level: &'a mut TddLevel,
    product_list: &'a mut Vec<ProductEntry>,
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
        let lim = eng.limits();
        let row_pos = a.row_base + a.j;
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
            &mut EmitSink {
                level: &mut *self.level,
            },
            a.gate,
        )?;
        let nid = a.node_idx[row_pos];
        if nid != NO_PRODUCT {
            lim.try_push(
                self.product_list,
                ProductEntry {
                    f_idx: FNodeIdx(a.i as u32),
                    g_idx: GNodeIdx(a.j as u32),
                    prod_idx: ProductNodeIdx(nid),
                },
            )?;
        }
        Ok(())
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
            level,
            product_list,
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
/// Never streams: streaming levels take the collapse-at-source walker
/// ([`run_level_rows_stream_count`]) unconditionally — there is no post-cell
/// snapshot conversion.
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
/// row's width.
pub(crate) fn run_level_rows_single(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
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
    let g_cols = &g_cols[..];
    let live_cols = g_cols.iter().filter(|q| q.left.0 != NO_PRODUCT).count() as u64;
    let left = DenseLookup { base: ctx.sides.left.base, stride: ctx.sides.left.stride };
    let right = DenseLookup { base: ctx.sides.right.base, stride: ctx.sides.right.stride };
    let mut sink = EmitSink { level };
    let mut poll = lim.gate_with(super::super::budget::DENSE_CELL_POLL_STRIDE);

    // The resets of `run_level_rows`: one fill of a small grid, a row at a
    // time on a larger one.
    let slab_fill = f_width.saturating_mul(right_width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if slab_fill {
        let end = ctx.output_grid_base + f_width * right_width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            mark_dead(&mut slab[ctx.output_grid_base..end]);
        }
    }
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, f_level, f_width, left_view, right_view, f_scratch);
    for i in 0..f_width {
        let row_base = ctx.output_grid_base + i * right_width;
        let fresh = extend_slab(slab, row_base, row_base + right_width);
        let node_idx = &mut slab[..];
        if !slab_fill && !fresh {
            mark_dead(&mut node_idx[row_base..row_base + right_width]);
        }
        let f_pairs = match f_described {
            Some(k) => &f_scratch[i * k..(i + 1) * k],
            None => f_level.pairs_view_decoded(i, f_scratch, left_view, right_view),
        };
        // A dead f node's row stays dead.
        let Some(p) = f_pairs.first() else { continue };
        debug_assert!(f_pairs.len() == 1, "an f node of one pair at most");
        let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
        for (j, q) in g_cols.iter().enumerate() {
            if q.left.0 == NO_PRODUCT {
                continue;
            }
            let lc = left.get_in_row(node_idx, lrow, q.left.0);
            if lc == NO_PRODUCT {
                continue;
            }
            let rc = right.get_in_row(node_idx, rrow, q.right.0);
            if rc == NO_PRODUCT {
                continue;
            }
            sink.single(eng, node_idx, row_base + j, lc, rc)?;
        }
        poll.poll(live_cols + right_width as u64)?;
    }
    poll.flush()?;
    Ok(())
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
        PlainLookups::CompleteLeft { charged } => run_level_rows::<false, _, _, _>(
            eng, rows, scratch,
            &complete(&sides.left), &grid(&sides.right),
            &mut Emit { sink: ReservedEmitSink { level, charged } },
        ),
        PlainLookups::CompleteRight { charged } => run_level_rows::<false, _, _, _>(
            eng, rows, scratch,
            &grid(&sides.left), &complete(&sides.right),
            &mut Emit { sink: ReservedEmitSink { level, charged } },
        ),
        PlainLookups::Grid => unreachable!("a level with no complete side takes the grid row loop"),
    }
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
