//! The dense product build's row loop, its cell actions, and the
//! marginal, sparse-marginal and plain level entries that run it.

use super::super::liveness::ROW_LIVE;
use super::*;

/// Fold the per-row alive-column masks for one decoded f row.
///
/// left: pass-through ⇒ always alive (`MAX`); `!masked` ⇒ masks unused (`0`);
/// else the OR of the side's `live_cols` over the row's left refs. right mirrors
/// (`MAX` when pass-through OR `!masked`). Returns `None` when the row is
/// provably dead under the masks (every cell in it would be culled) — the caller
/// skips the whole row.
#[inline(always)]
pub(crate) fn row_alive_masks(ctx: &CellCtx<'_>, f_pairs: &[ChildPair]) -> Option<(u128, u128)> {
    let left_alive_mask: u128 = if ctx.sides.left.plan.is_passthrough() {
        u128::MAX // pass-through side: no grid; always alive
    } else if !ctx.masked {
        0u128
    } else {
        f_pairs.iter().fold(0u128, |acc, p1| {
            acc | ctx.sides.left.live_cols[p1.left.raw() as usize]
        })
    };
    if ctx.masked && left_alive_mask == 0 {
        return None;
    }

    let right_alive_mask: u128 = if ctx.sides.right.plan.is_passthrough() || !ctx.masked {
        u128::MAX
    } else {
        f_pairs.iter().fold(0u128, |acc, p1| {
            acc | ctx.sides.right.live_cols[p1.right.raw() as usize]
        })
    };
    if ctx.masked && right_alive_mask == 0 {
        return None;
    }

    Some((left_alive_mask, right_alive_mask))
}

// ============================= Row-loop driver =============================

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
    /// `ctx.output_grid_base + i * ctx.g_width` on a dense slab, `ctx.output_grid_base`
    /// otherwise, computed once per row by
    /// the driver for its `NO_PRODUCT` reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    pub(super) row_base: usize,
    /// Decoded pairs of f row `i` (never empty — empty rows are skipped).
    pub(super) f_pairs: &'a [ChildPair],
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) g_level: &'a TddLevel,
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
    /// Whether this action owns a dense `f_width × g_width` slab — one grid row per f row,
    /// so the per-row `NO_PRODUCT` resets tile the level's whole slab exactly once and
    /// can be replaced by a single fill (see [`run_level_rows`]). `false` for
    /// an action that writes every row into grid row 0 ([`Emit`] over a
    /// [`ListSink`], `rows_by_words` without `SLAB`), which must re-fill that row between rows.
    const DENSE_SLAB: bool;

    /// Whether the level has one column, which the action was built for: the
    /// row loop then knows its width without reading it.
    const ONE_COLUMN: bool = false;

    /// One cell of the row.
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError>;

    /// Whether the row loop offers the action [`whole_row`](Self::whole_row).
    const WHOLE_ROWS: bool = false;

    /// The cells of a row whose every column holds one, in column order, as
    /// [`cell`](Self::cell) a column would write them, for an action with a
    /// loop of its own for them: how many columns it wrote, leaving the
    /// cells from there to [`cell`](Self::cell).
    #[inline(always)]
    fn whole_row(
        &mut self,
        _eng: &Engine,
        _row: WholeRow<'_, '_, L, R>,
    ) -> Result<usize, OperationError> {
        Ok(0)
    }
}

/// What [`CellAction::whole_row`] reads of a row: [`CellArgs`] without the
/// column and the f row's index.
pub(super) struct WholeRow<'a, 'c, L, R> {
    pub(super) row_base: usize,
    pub(super) f_pairs: &'a [ChildPair],
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) g_level: &'a TddLevel,
    pub(super) g_pairs_scratch: &'a mut Vec<ChildPair>,
    pub(super) node_idx: &'a mut [u32],
    pub(super) left: &'a L,
    pub(super) right: &'a R,
    pub(super) gate: &'a mut crate::limits::PollGate<'c>,
}

/// The columns of `reach` (at most 64) whose reach meets `mask`, as the
/// bits of a word from the first.
#[inline(always)]
fn reaching(mask: u128, reach: &[u128]) -> u64 {
    debug_assert!(reach.len() <= 64);
    // From the last column down, a shift by one in place of a shift by the
    // column's index.
    reach
        .iter()
        .rev()
        .fold(0, |live, &r| live << 1 | u64::from(mask & r != 0))
}

/// Word `w` of the g nodes whose reach meets `mask`, read off the reach
/// columns `columns` of `words` words a bit (`liveness::build_reach_columns`):
/// the union of that word of each of the mask's bits that some reach mask has.
#[inline(always)]
fn reach_word(mask: u128, columns: &[u64], words: usize, w: usize) -> u64 {
    let bits = columns.len() / words;
    let mut m = if bits >= 128 {
        mask
    } else {
        mask & ((1u128 << bits) - 1)
    };
    let mut live = 0;
    while m != 0 {
        let b = m.trailing_zeros() as usize;
        m &= m - 1;
        live |= columns[b * words + w];
    }
    live
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

/// Whether [`run_level_rows`] culls columns per row on side `k` of a
/// level: the reach masks exist only on a masked level, a pass-through
/// side has no grid to be dead in, and `DENSE` is the regime where the
/// masks are not folded. A level that culls on either side never offers
/// its action a [whole row](CellAction::whole_row).
#[inline(always)]
pub(super) fn culls<const DENSE: bool, K: ChildLookup>(ctx: &CellCtx<'_>, k: &K) -> bool {
    !DENSE && ctx.masked && !k.passthrough() && k.kills()
}

/// The one row/cell loop of the dense product build.
///
/// `const DENSE` skips the per-row alive-mask fold on levels where it provably
/// cannot skip anything: `rows.ctx.masked` false and neither side a pass-through, where
/// `row_alive_masks` always returns `(0, u128::MAX)`, or neither side able to
/// [kill](ChildLookup::kills) a candidate, where no mask can clear. Only the
/// plain routes reach those regimes; every other entry folds.
///
/// `rows.ctx.g_width` and `rows.ctx.output_grid_base` are read from `rows.ctx` rather than passed alongside it, so
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
    let RowLoop {
        f_level,
        g_level,
        ctx,
        f_width,
        ..
    } = rows;
    let RowScratch {
        f_pairs: f_pairs_scratch,
        g_pairs: g_pairs_scratch,
        node_idx: slab,
    } = scratch;
    let lim = eng.limits();
    // Amortized wall-deadline/cancel poll: one read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    debug_assert!(
        !A::ONE_COLUMN || ctx.g_width == 1,
        "a one-column action runs on a level of one column"
    );
    let g_width = if A::ONE_COLUMN { 1 } else { ctx.g_width };
    let cull_left = culls::<DENSE, _>(ctx, left);
    let cull_right = culls::<DENSE, _>(ctx, right);
    // Whether a row may read its live columns off the reach columns: every
    // culling side built them.
    let (left_cols, right_cols) = (ctx.sides.left.reach_cols, ctx.sides.right.reach_cols);
    let by_columns = (cull_left || cull_right)
        && (!cull_left || !left_cols.is_empty())
        && (!cull_right || !right_cols.is_empty());
    let words = g_width.div_ceil(64);
    let culls = usize::from(cull_left) + usize::from(cull_right);
    // Whether a row reads its live columns off the row columns: every
    // culling side built them, and a side that culls nothing is read whole.
    let (left_rows, right_rows) = (ctx.sides.left.row_cols, ctx.sides.right.row_cols);
    let by_rows = (cull_left || cull_right) && ctx.sides.left.rows && ctx.sides.right.rows;
    debug_assert!(!by_rows || g_width <= super::super::liveness::ROW_COLUMNS_MAX_WIDTH);
    // A side read whole: every g node, and a live cell in the row.
    let whole = if by_rows {
        ROW_LIVE | ((1u64 << g_width) - 1)
    } else {
        0
    };

    // One slab fill instead of `f_width` row fills. On a dense-slab action the
    // per-row resets below tile `output_grid_base .. output_grid_base + f_width*g_width` exactly once each
    // (grid row `i` for f row `i`), so hoisting them into a single `fill` writes the same
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
    let slab_fill = A::DENSE_SLAB
        && !every_cell
        && f_width.saturating_mul(g_width) <= DEAD_SLAB_FILL_MAX_CELLS;
    // A side that cannot kill is a complete one ([`mark_dead_out_of_line`]).
    let reset = |cells: &mut [u32]| {
        if left.kills() && right.kills() {
            mark_dead(cells)
        } else {
            mark_dead_out_of_line(cells)
        }
    };
    if slab_fill {
        let end = ctx.output_grid_base + f_width * g_width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            reset(&mut slab[ctx.output_grid_base..end]);
        }
    }

    // An implicit f level's rows generated before the loop reads them; a
    // stored level's rows are slices of its arena.
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(
        lim,
        f_level,
        f_width,
        left_view,
        right_view,
        f_pairs_scratch,
    );
    for i in 0..f_width {
        let row_base = ctx.output_grid_base + if A::DENSE_SLAB { i * g_width } else { 0 };
        // A row the slab is extended over is `NO_PRODUCT` already.
        let fresh = extend_slab(slab, row_base, row_base + g_width);
        let node_idx = &mut slab[..];
        if !slab_fill && !every_cell && !fresh {
            reset(&mut node_idx[row_base..row_base + g_width]);
        }

        let f_pairs = match f_described {
            Some(k) => &f_pairs_scratch[i * k..(i + 1) * k],
            None => f_level.pairs_view_decoded(i, f_pairs_scratch, left_view, right_view),
        };
        // Empty pairs means dead (zero-containing) node — skip this row.
        if f_pairs.is_empty() {
            if every_cell && !fresh {
                reset(&mut node_idx[row_base..row_base + g_width]);
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
                row_base,
                f_pairs,
                ctx,
                g_level,
                g_pairs_scratch: &mut *g_pairs_scratch,
                node_idx: &mut *node_idx,
                left,
                right,
                gate: &mut poll,
            };
            first = action.whole_row(eng, row)?;
            if first == g_width {
                poll.poll(g_width as u64)?;
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
                    g_level,
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
            let l = if cull_left {
                f_pairs
                    .iter()
                    .fold(0, |l, p| l | left_rows[p.left.raw() as usize])
            } else {
                whole
            };
            let r = if cull_right {
                f_pairs
                    .iter()
                    .fold(0, |r, p| r | right_rows[p.right.raw() as usize])
            } else {
                whole
            };
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
            poll.poll(g_width as u64)?;
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
            let bits = if cull_left {
                left_alive_mask.count_ones()
            } else {
                0
            } + if cull_right {
                right_alive_mask.count_ones()
            } else {
                0
            };
            if by_columns && 2 * bits as usize * words < 3 * culls * g_width {
                super::super::note_reach_columns_row();
                for w in 0..words {
                    let n = (g_width - w * 64).min(64);
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
                poll.poll(g_width as u64)?;
                continue;
            }
            let mut j0 = 0;
            while j0 < g_width {
                let n = (g_width - j0).min(64);
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
            for j in first..g_width {
                cell(j)?;
            }
        }
        poll.poll(g_width as u64)?;
    }
    // The level's residual: what the gate holds is under one stride by
    // construction, and on a level narrower than a stride it is everything.
    poll.flush()?;
    Ok(())
}

/// Materializing action: each surviving cell becomes a product node in the
/// dense `f_width × g_width` output slab, written through the sink
/// `S`, or in grid row 0 where `S` records the cells ([`ListSink`]). Both
/// dense routes are this action, and the streaming route's with a sink
/// that folds each cell into the level's value column.
pub(super) struct Emit<S> {
    pub(super) sink: S,
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for Emit<S> {
    const DENSE_SLAB: bool = S::SLAB;
    /// A sink of grid row 0 opens no row.
    const WHOLE_ROWS: bool = S::SLAB;

    #[inline(always)]
    fn whole_row(
        &mut self,
        _eng: &Engine,
        row: WholeRow<'_, '_, L, R>,
    ) -> Result<usize, OperationError> {
        emit_row(&mut self.sink, row)
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        self.sink.row(a.i);
        process_cell::<_, _, _>(
            (eng, a.ctx),
            (a.j, a.row_base),
            a.f_pairs,
            (a.g_level, a.g_pairs_scratch),
            a.node_idx,
            (a.left, a.right),
            (&mut self.sink, a.gate),
        )
    }
}

/// The `Route::MarginalChild` row loop: a level with at least one marginal
/// child. Iterates rows `0..rows.f_width` in forward order, running the emit
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
    let mut inline = 0;
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut Emit {
            sink: EmitSink {
                level,
                charged: usize::MAX,
                inline: &mut inline,
            },
        },
    )?;
    level.leave_written_pairs(inline);
    Ok(())
}

/// The `Route::SparseMarg` row loop: an *exactly-one*-marginal-child level
/// whose output is structural (never a marginalization target).
///
/// Same emit kernel as [`run_level_rows_marginal`], but instead of a dense
/// `rows.f_width*rows.ctx.g_width` slab it reuses one `rows.ctx.g_width`-wide row scratch
/// at `rows.ctx.output_grid_base` and records each surviving cell in
/// `product_list`. The marginal child is a pass-through carrier that kills no
/// pair, so the structural sibling alone decides which cells are alive and a
/// dense slab would be mostly `NO_PRODUCT`. The parent densifies the
/// `product_list` through `ensure_grid` when it needs the grid.
///
/// Every cell is built at grid row 0, so the kernel's
/// `grid_pos == rows.ctx.output_grid_base + j`, then read back from the scratch
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
    let mut inline = 0;
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut Emit {
            sink: ListSink {
                emit: EmitSink {
                    level,
                    charged: usize::MAX,
                    inline: &mut inline,
                },
                product_list,
                base: rows.ctx.output_grid_base,
                row: 0,
            },
        },
    )?;
    level.leave_written_pairs(inline);
    Ok(())
}

/// The row loop of `Route::PlainDense` and `Route::Dense`: a level with no
/// marginal child. Iterates rows `0..rows.f_width` in forward order.
/// Runs the emit kernel with positional grid lookups on both sides (no
/// pass-through, no mask decode — the caller guarantees no marginal child).
///
/// `DENSE = true` asserts two level-invariant facts, `rows.ctx.masked ==
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
    let left = DenseLookup::new(&rows.ctx.sides.left);
    let right = DenseLookup::new(&rows.ctx.sides.right);
    let mut inline = 0;
    let mut action = Emit {
        sink: EmitSink {
            level,
            charged: usize::MAX,
            inline: &mut inline,
        },
    };
    run_level_rows::<DENSE, _, _, _>(eng, rows, scratch, &left, &right, &mut action)?;
    level.leave_written_pairs(inline);
    Ok(())
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
    let WholeRow {
        row_base,
        f_pairs,
        ctx,
        g_level,
        g_pairs_scratch,
        node_idx,
        left,
        right,
        gate,
    } = row;
    let (k, width) = (f_pairs.len(), ctx.g_width);
    if k > ROW_PAIRS {
        return Ok(0);
    }
    let Some(mut out) = sink.open_row(0, width, gate.limits().level_width_cap()) else {
        return Ok(0);
    };
    let child_rows = |p: &ChildPair| (left.row(p.left.0), right.row(p.right.0));
    let mut rows = [child_rows(&f_pairs[0]); ROW_PAIRS];
    for (t, p) in f_pairs.iter().enumerate().skip(1) {
        rows[t] = child_rows(p);
    }
    let rows = &rows[..k];
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let mut j = 0;
    while j < width {
        let g_pairs = column_pairs(ctx, g_level, j, g_pairs_scratch);
        let m = g_pairs.len();
        if m == 0 {
            // A dead column's cell, as `process_cell` leaves it.
            node_idx[row_base + j] = NO_PRODUCT;
            j += 1;
            continue;
        }
        // A cell of one pair against one is a node held inline, which
        // takes no room in the pair arena.
        if (k > 1 && m > 1) || (k * m > 1 && !out.fits(k * m)) {
            break;
        }
        super::rows_stream::charge(gate, &mut due_at, &mut due, (k * m) as u64)?;
        if k * m == 1 {
            let ((l, r), q) = (rows[0], &g_pairs[0]);
            let lc = left.get_in_row(node_idx, l, q.left.0);
            if !left.kills() || lc != NO_PRODUCT {
                let rc = right.get_in_row(node_idx, r, q.right.0);
                if !right.kills() || rc != NO_PRODUCT {
                    // Safety: the row was opened for a node a column.
                    unsafe { out.single(node_idx, row_base + j, lc, rc) };
                }
            }
            j += 1;
            continue;
        }
        // Plain loops keep the open row's write position in the row loop's
        // registers; a closure run through `for_each` may be compiled out of
        // line, and the row then passed to it in memory.
        if m == 1 {
            let q = &g_pairs[0];
            for &f_rows in rows {
                // Safety: the cell fits its `k * m` candidates.
                unsafe { push_candidate(&mut out, node_idx, (left, right), f_rows, q) };
            }
        } else {
            let f_rows = rows[0];
            for q in g_pairs {
                // Safety: as above.
                unsafe { push_candidate(&mut out, node_idx, (left, right), f_rows, q) };
            }
        }
        // Safety: the row was opened for a node a column.
        unsafe { out.close(node_idx, row_base + j) };
        j += 1;
    }
    gate.add_short(due_at - due);
    Ok(j)
}

/// Store in `out`'s open cell the candidate of the child rows `(l, r)`
/// against the g pair `q`, unless a side has no product there.
///
/// # Safety
///
/// The open cell fits the candidate.
#[inline(always)]
unsafe fn push_candidate<L: ChildLookup, R: ChildLookup>(
    out: &mut ArenaRow<'_>,
    node_idx: &[u32],
    (left, right): (&L, &R),
    (l, r): (L::Row, R::Row),
    q: &ChildPair,
) {
    let lc = left.get_in_row(node_idx, l, q.left.0);
    if left.kills() && lc == NO_PRODUCT {
        return;
    }
    let rc = right.get_in_row(node_idx, r, q.right.0);
    if right.kills() && rc == NO_PRODUCT {
        return;
    }
    // Safety: as the caller's.
    unsafe { out.push(lc, rc) };
}
