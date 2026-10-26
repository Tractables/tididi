//! The row loop of a masked level read by pairs.

use super::*;
use super::super::liveness::{ROW_COLUMNS_MAX_WIDTH, ROW_LIVE};
use crate::limits::PollGate;

/// The f pairs a row's ANDs are kept for on the stack; a longer row keeps
/// them in a buffer of the level's.
const FEW_PAIRS: usize = 16;

/// The most columns a level reads its rows' charges off a table of every
/// set of its columns, and the fewest table entries a row pays for.
const CHARGE_TABLE_COLUMNS: usize = 8;
const CHARGE_TABLE_ENTRIES_A_ROW: usize = 8;

/// The row loop of a masked level whose g level has at most
/// [`ROW_COLUMNS_MAX_WIDTH`] pairs in all (`liveness::pairs_fit`), read by
/// pairs.
///
/// Each side's row columns are then over g's pairs, numbered in column
/// order: bit `q` of child row `a`'s word is set iff the child grid's cell
/// at `a` and g pair `q`'s child on that side holds a product, and
/// [`ROW_LIVE`] iff the child row has a live cell. An f pair `(a, b)` and
/// g pair `q` make a live candidate iff the words of `a` on the left and
/// `b` on the right both have bit `q`: one AND decides an f pair against
/// every pair of g's level, where the cell walks read both child grids for
/// each candidate, and a cell holds a product iff the OR of its row's ANDs
/// has a bit of its column. A complete side kills nothing, and is read as
/// a word of every pair.
///
/// The rows, cells and pairs are those of [`run_level_rows`] on the
/// level's masks: a row is skipped where a side's fold has no live child
/// row, the cells entered are the columns with a pair in each side's fold,
/// the work clock is charged a unit a candidate of an entered cell and the
/// row's width a row, in one poll a row unless the gate comes due inside
/// it, and then as the row loop polls them, and a cell's surviving candidates
/// are written in the order of its walk, f pair by f pair and each against
/// the column's pairs in order (no column of so few pairs is grouped), one
/// pair alone as the one-pair cell writes it. A row visits its live cells
/// alone, the columns of the g pairs its ANDs have, all of them entered.
pub(crate) fn run_level_rows_pairs(
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
        PlainLookups::Grid => {
            pair_rows(eng, rows, scratch, &grid(&sides.left), &grid(&sides.right), &mut EmitSink { level })
        }
        PlainLookups::CompleteLeft { charged } => {
            pair_rows(
                eng, rows, scratch,
                &complete(&sides.left), &grid(&sides.right),
                &mut ReservedEmitSink { level, charged },
            )
        }
        PlainLookups::CompleteRight { charged } => {
            pair_rows(
                eng, rows, scratch,
                &grid(&sides.left), &complete(&sides.right),
                &mut ReservedEmitSink { level, charged },
            )
        }
        PlainLookups::Complete { .. } => unreachable!("a level of two complete sides builds no masks"),
    }
}

/// [`run_level_rows_pairs`] on its lookups and sink.
fn pair_rows<L, R, S>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    sink: &mut S,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let RowLoop { f_level, g_level, ctx, f_width, .. } = rows;
    let RowScratch { f_pairs: f_scratch, g_pairs: g_flat, node_idx: slab } = scratch;
    let lim = eng.limits();
    let right_width = ctx.right_width;
    debug_assert!(right_width <= ROW_COLUMNS_MAX_WIDTH);

    // g's pairs in column order; each column's as a mask of their bits, and
    // each pair's column. The f rows' buffer decodes a column the table
    // does not hold.
    g_flat.clear();
    lim.reserve_exact(g_flat, ROW_COLUMNS_MAX_WIDTH)?;
    let mut columns = Columns {
        g_flat: &[],
        cols: [0; ROW_COLUMNS_MAX_WIDTH],
        col_of: [0; ROW_COLUMNS_MAX_WIDTH],
        col_pairs: [0; ROW_COLUMNS_MAX_WIDTH],
        charge_of: [0; 1 << CHARGE_TABLE_COLUMNS],
        tabled: false,
    };
    for (j, col) in columns.cols.iter_mut().enumerate().take(right_width) {
        let start = g_flat.len();
        g_flat.extend_from_slice(column_pairs(ctx, g_level, j, f_scratch));
        debug_assert!(g_flat.len() <= ROW_COLUMNS_MAX_WIDTH, "a level read by pairs fits");
        *col = low(g_flat.len()) & !low(start);
        columns.col_of[start..g_flat.len()].fill(j as u8);
    }
    columns.g_flat = &g_flat[..];
    // Where every column holds one pair, pair `q` is column `q`'s, and a
    // word of pairs is a word of their columns.
    let one_pair = g_flat.len() == right_width && columns.cols[..right_width].iter().all(|c| c.count_ones() == 1);
    // The number of g pairs of a set of columns, which a row's charge
    // counts: off a table of every set where the level has few columns and
    // rows enough to pay for it, else column by column.
    for (n, col) in columns.col_pairs.iter_mut().zip(&columns.cols[..right_width]) {
        *n = col.count_ones() as u8;
    }
    columns.tabled = right_width <= CHARGE_TABLE_COLUMNS
        && (1 << right_width) <= f_width.saturating_mul(CHARGE_TABLE_ENTRIES_A_ROW);
    if columns.tabled {
        for set in 1..1usize << right_width {
            columns.charge_of[set] = columns.charge_of[set & (set - 1)] + columns.col_pairs[set.trailing_zeros() as usize];
        }
    }
    super::super::note_pair_rows(!left.kills() || !right.kills(), one_pair);
    let rows = PairRows { f_level, ctx, f_width, columns: &columns };
    // A level of two one-pair columns keeps them in registers and writes
    // each row's cells once.
    match (one_pair, right_width) {
        (true, 2) => pair_rows_of::<true, 2, _, _, _>(eng, rows, f_scratch, slab, left, right, sink),
        (true, _) => pair_rows_of::<true, 0, _, _, _>(eng, rows, f_scratch, slab, left, right, sink),
        (false, _) => pair_rows_of::<false, 0, _, _, _>(eng, rows, f_scratch, slab, left, right, sink),
    }
}

/// Poll `gate` for a row whose cells entered are the columns of the bits
/// of `entered`, of `col_pairs` pairs each or one each where `None`, then
/// its `width`, `units` in all, as the row loop polls them, where the gate
/// comes due inside the row ([`poll_row_in_cells`]). Takes and gives back
/// the gate by value, out of line, so that the row loop keeps its gate in
/// registers.
#[cold]
#[inline(never)]
fn poll_entered_due<'a>(
    mut gate: PollGate<'a>,
    ctx: &CellCtx<'_>,
    f_pairs: &[ChildPair],
    units: u64,
    width: usize,
    entered: u64,
    col_pairs: Option<&[u8; ROW_COLUMNS_MAX_WIDTH]>,
) -> Result<PollGate<'a>, OperationError> {
    let mut s = entered;
    let cells = std::iter::from_fn(|| {
        let j = (s != 0).then(|| s.trailing_zeros() as usize)?;
        s &= s - 1;
        Some((j, col_pairs.map_or(1, |n| usize::from(n[j]))))
    });
    poll_row_in_cells(&mut gate, ctx, f_pairs, units, width, cells)?;
    Ok(gate)
}

/// The low `n` bits.
#[inline(always)]
fn low(n: usize) -> u64 {
    (1u64 << n) - 1
}

/// A level's g pairs in column order, and what the row loop reads off them.
struct Columns<'a> {
    g_flat: &'a [ChildPair],
    /// Each column's pairs, as a mask of their bits.
    cols: [u64; ROW_COLUMNS_MAX_WIDTH],
    /// Each pair's column.
    col_of: [u8; ROW_COLUMNS_MAX_WIDTH],
    /// Each column's number of pairs.
    col_pairs: [u8; ROW_COLUMNS_MAX_WIDTH],
    /// The number of pairs of each set of columns, where `tabled`.
    charge_of: [u8; 1 << CHARGE_TABLE_COLUMNS],
    tabled: bool,
}

/// What the row loop of a level read by pairs reads.
#[derive(Clone, Copy)]
struct PairRows<'a> {
    f_level: &'a TddLevel,
    ctx: &'a CellCtx<'a>,
    f_width: usize,
    columns: &'a Columns<'a>,
}

/// The row loop of [`pair_rows`], `ONE` where every column holds one pair,
/// on a level of `W` columns, or of any number for `W = 0`.
///
/// With `W` columns, all of one pair, a row's cells are found and then
/// written once, a node index or `NO_PRODUCT` each, in place of the reset
/// and the writes of its live cells.
fn pair_rows_of<const ONE: bool, const W: usize, L, R, S>(
    eng: &Engine,
    rows: PairRows<'_>,
    f_scratch: &mut Vec<ChildPair>,
    slab: &mut Vec<u32>,
    left: &L,
    right: &R,
    sink: &mut S,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let PairRows { f_level, ctx, f_width, columns } = rows;
    let Columns { g_flat, cols, col_of, col_pairs, charge_of, tabled } = columns;
    let (g_flat, tabled) = (*g_flat, *tabled);
    let lim = eng.limits();
    let right_width = ctx.right_width;
    debug_assert!(W == 0 || (ONE && W == right_width));
    let width = if W == 0 { right_width } else { W };
    let pairs_in = |s: u64| -> u64 {
        if W > 0 {
            return (0..W).map(|j| s >> j & 1).sum();
        }
        if tabled {
            return u64::from(charge_of[s as usize]);
        }
        if ONE {
            return u64::from(s.count_ones());
        }
        let (mut s, mut n) = (s, 0u64);
        while s != 0 {
            n += u64::from(col_pairs[s.trailing_zeros() as usize]);
            s &= s - 1;
        }
        n
    };
    // The columns' pairs of a level of `W` one-pair columns.
    let g: [ChildPair; W] = std::array::from_fn(|j| g_flat[j]);
    let all = ROW_LIVE | low(g_flat.len());
    let (left_words, right_words) = (ctx.sides.left.row_cols, ctx.sides.right.row_cols);
    debug_assert!(!left.kills() || ctx.sides.left.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.left.0));
    debug_assert!(!right.kills() || ctx.sides.right.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.right.0));
    let word_left = |p: &ChildPair| if left.kills() { left_words[p.left.raw() as usize] } else { all };
    let word_right = |p: &ChildPair| if right.kills() { right_words[p.right.raw() as usize] } else { all };
    // The columns with a pair among the bits of `s`.
    let reached = |s: u64| {
        if ONE {
            return s & !ROW_LIVE;
        }
        let (mut s, mut out) = (s & !ROW_LIVE, 0u64);
        while s != 0 {
            let j = col_of[s.trailing_zeros() as usize] as usize;
            out |= 1 << j;
            s &= !cols[j];
        }
        out
    };
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    // The ANDs of a row's f pairs: of `FEW_PAIRS` pairs or fewer on the
    // stack, of more in a buffer.
    let mut few_ands = [0u64; FEW_PAIRS];
    let mut many: Vec<u64> = Vec::new();

    // The resets of `run_level_rows`: one fill of a small grid, a row at a
    // time on a larger one.
    let slab_fill = f_width.saturating_mul(width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if W == 0 && slab_fill {
        let end = ctx.output_grid_base + f_width * width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            mark_dead(&mut slab[ctx.output_grid_base..end]);
        }
    }
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, f_level, f_width, left_view, right_view, f_scratch);
    let mut next = 0;
    'rows: while next < f_width {
        // A row is written straight into the arenas, open across rows, where
        // its live cells' every candidate fits their room: each cell holds
        // at most an f pair's candidates against each of g's pairs. A row
        // without room goes through the sink, which grows them.
        let mut out = sink.open_row(0, 0);
        while next < f_width {
            let i = next;
            next += 1;
            let row_base = ctx.output_grid_base + i * width;
            if W == 0 {
                let fresh = extend_slab(slab, row_base, row_base + width);
                if !slab_fill && !fresh {
                    mark_dead(&mut slab[row_base..row_base + width]);
                }
            }
            let f_pairs = match f_described {
                Some(k) => &f_scratch[i * k..(i + 1) * k],
                None => f_level.pairs_view_decoded(i, f_scratch, left_view, right_view),
            };
            // A dead f node's row stays dead, as does a row with a side
            // without a live child row.
            if f_pairs.is_empty() {
                if W > 0 {
                    write_cells(slab, row_base, [NO_PRODUCT; W]);
                }
                continue;
            }
            // A row of two f pairs on two one-pair columns, the arenas with
            // room for its every candidate: the loop below unrolled, each f
            // pair's AND held in a register.
            if W == 2
                && let [p0, p1] = f_pairs
                && let Some(o) = out.as_mut().filter(|o| o.has_room(4, 2))
            {
                super::super::note_pair_row_unrolled();
                let (wl0, wr0, wl1, wr1) = (word_left(p0), word_right(p0), word_left(p1), word_right(p1));
                let (l, r) = (wl0 | wl1, wr0 | wr1);
                if l & r & ROW_LIVE == 0 {
                    write_cells(slab, row_base, [NO_PRODUCT; W]);
                    continue;
                }
                let entered = l & r & !ROW_LIVE;
                // Two f pairs a column entered, of bits 0 and 1 (a count
                // of ones without the popcount instruction is a dozen).
                let units = 2 * ((entered & 1) + (entered >> 1)) + 2;
                if units < poll.due_in() {
                    poll.add_short(units);
                } else {
                    poll = poll_entered_due(poll, ctx, f_pairs, units, 2, entered, None)?;
                }
                let (a0, a1) = (wl0 & wr0, wl1 & wr1);
                let mut row = [NO_PRODUCT; W];
                for (j, cell) in row.iter_mut().enumerate() {
                    if (a0 | a1) >> j & 1 == 0 {
                        continue;
                    }
                    let q = &g_flat[j];
                    for (p, and) in [(p0, a0), (p1, a1)] {
                        if and >> j & 1 != 0 {
                            // Safety: the arenas have room for a pair an f
                            // pair and a g pair.
                            unsafe { o.push(left.get(slab, p.left.0, q.left.0), right.get(slab, p.right.0, q.right.0)) };
                        }
                    }
                    // Safety: the arenas have room for a node a column.
                    if let Some(node) = unsafe { o.close_node() } {
                        *cell = node;
                    }
                }
                write_cells(slab, row_base, row);
                continue;
            }
            // Each side's fold, the g pairs some candidate of the row survives
            // against, and each f pair's AND.
            let few = f_pairs.len() <= FEW_PAIRS;
            if !few {
                many.clear();
                lim.reserve(&mut many, f_pairs.len())?;
            }
            let (mut l, mut r, mut live) = (0, 0, 0);
            for (k, p) in f_pairs.iter().enumerate() {
                let (wl, wr) = (word_left(p), word_right(p));
                let and = wl & wr;
                (l, r, live) = (l | wl, r | wr, live | and);
                if few {
                    few_ands[k] = and;
                } else {
                    many.push(and);
                }
            }
            if l & r & ROW_LIVE == 0 {
                if W > 0 {
                    write_cells(slab, row_base, [NO_PRODUCT; W]);
                }
                continue;
            }
            // The cells the masks enter, charged with the row's width; the
            // live ones among them.
            let entered = reached(l) & reached(r);
            let n1 = f_pairs.len();
            debug_assert_eq!(pairs_in(entered), u64::from(cols.iter().enumerate().filter(|&(j, _)| entered >> j & 1 != 0).fold(0, |a, (_, c)| a | c).count_ones()));
            // In one add where the gate does not come due inside the row,
            // and otherwise as the row loop polls the cells it enters.
            let units = n1 as u64 * pairs_in(entered) + width as u64;
            if units < poll.due_in() {
                poll.add_short(units);
            } else {
                poll = poll_entered_due(poll, ctx, f_pairs, units, width, entered, (!ONE).then_some(col_pairs))?;
            }
            let live = live & !ROW_LIVE;
            let ands = if few { &few_ands[..n1] } else { &many[..] };
            let mut cells = reached(live);
            debug_assert_eq!(cells & !entered, 0, "a live cell the masks do not enter");
            if let Some(o) = out.as_mut().filter(|o| o.has_room(n1 * g_flat.len(), width)) {
                if W > 0 {
                    let mut row = [NO_PRODUCT; W];
                    for (j, (cell, q)) in row.iter_mut().zip(&g).enumerate() {
                        if cells >> j & 1 == 0 {
                            continue;
                        }
                        for (p, &and) in f_pairs.iter().zip(ands) {
                            if and >> j & 1 != 0 {
                                // Safety: the arenas have room for a pair an
                                // f pair and a g pair.
                                unsafe { o.push(left.get(slab, p.left.0, q.left.0), right.get(slab, p.right.0, q.right.0)) };
                            }
                        }
                        // Safety: the arenas have room for a node a column.
                        if let Some(node) = unsafe { o.close_node() } {
                            *cell = node;
                        }
                    }
                    write_cells(slab, row_base, row);
                    continue;
                }
                let node_idx = &mut slab[..];
                while cells != 0 {
                    let j = cells.trailing_zeros() as usize;
                    cells &= cells - 1;
                    let col = cols[j];
                    for (p, &and) in f_pairs.iter().zip(ands) {
                        let mut m = and & col;
                        if ONE {
                            if m != 0 {
                                let q = &g_flat[j];
                                // Safety: the arenas have room for a pair an
                                // f pair and a g pair.
                                unsafe { o.push(left.get(node_idx, p.left.0, q.left.0), right.get(node_idx, p.right.0, q.right.0)) };
                            }
                            continue;
                        }
                        let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
                        while m != 0 {
                            let q = &g_flat[m.trailing_zeros() as usize];
                            m &= m - 1;
                            // Safety: as above.
                            unsafe { o.push(left.get_in_row(node_idx, lrow, q.left.0), right.get_in_row(node_idx, rrow, q.right.0)) };
                        }
                    }
                    // Safety: the arenas have room for a node a column.
                    unsafe { o.close(node_idx, row_base + j) };
                }
                continue;
            }
            drop(out);
            if W > 0 {
                write_cells(slab, row_base, [NO_PRODUCT; W]);
            }
            let node_idx = &mut slab[..];
            while cells != 0 {
                let j = cells.trailing_zeros() as usize;
                cells &= cells - 1;
                let col = cols[j];
                let grid_pos = row_base + j;
                if n1 == 1 && (ONE || col & (col - 1) == 0) {
                    let (p, q) = (&f_pairs[0], &g_flat[col.trailing_zeros() as usize]);
                    let lc = left.get(node_idx, p.left.0, q.left.0);
                    let rc = right.get(node_idx, p.right.0, q.right.0);
                    sink.single(eng, node_idx, grid_pos, lc, rc)?;
                    continue;
                }
                // Every candidate of the cell an AND keeps survives.
                let bound = if ONE { n1 } else { n1 * usize::from(col_pairs[j]) };
                if let Some(mut cell) = sink.open_cell(bound) {
                    for (p, &and) in f_pairs.iter().zip(ands) {
                        let mut m = and & col;
                        if ONE {
                            if m != 0 {
                                let q = &g_flat[j];
                                // Safety: the cell was opened for a pair an f pair.
                                unsafe { cell.push(left.get(node_idx, p.left.0, q.left.0), right.get(node_idx, p.right.0, q.right.0)) };
                            }
                            continue;
                        }
                        let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
                        while m != 0 {
                            let q = &g_flat[m.trailing_zeros() as usize];
                            m &= m - 1;
                            // Safety: the cell was opened for each f pair
                            // against each of the column's pairs.
                            unsafe { cell.push(left.get_in_row(node_idx, lrow, q.left.0), right.get_in_row(node_idx, rrow, q.right.0)) };
                        }
                    }
                    cell.close(eng, node_idx, grid_pos)?;
                    continue;
                }
                let start = sink.begin();
                for (p, &and) in f_pairs.iter().zip(ands) {
                    let mut m = and & col;
                    if m == 0 {
                        continue;
                    }
                    if ONE {
                        let q = &g_flat[j];
                        sink.pair(eng, left.get(node_idx, p.left.0, q.left.0), right.get(node_idx, p.right.0, q.right.0))?;
                        continue;
                    }
                    let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
                    while m != 0 {
                        let q = &g_flat[m.trailing_zeros() as usize];
                        m &= m - 1;
                        let lc = left.get_in_row(node_idx, lrow, q.left.0);
                        let rc = right.get_in_row(node_idx, rrow, q.right.0);
                        sink.pair(eng, lc, rc)?;
                    }
                }
                sink.end(eng, node_idx, grid_pos, start)?;
            }
            continue 'rows;
        }
    }
    poll.flush()?;
    Ok(())
}
