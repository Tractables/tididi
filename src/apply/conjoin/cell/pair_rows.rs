//! The row loop of a masked level read by pairs.

use super::*;
use super::super::liveness::{ROW_COLUMNS_MAX_WIDTH, ROW_LIVE};
use crate::limits::PollGate;
use crate::diagram::{NodeKind, StoredPairs};

/// The f pairs a row's intersections are kept for on the stack; a longer row keeps
/// them in a buffer of the level's.
const FEW_PAIRS: usize = 16;

/// The most columns a level reads its rows' charges off a table of every
/// set of its columns, and the fewest table entries a row pays for.
const CHARGE_TABLE_COLUMNS: usize = 8;
const CHARGE_TABLE_ENTRIES_A_ROW: usize = 8;

/// The fewest rows a byte of a level's pairs pays for a table of the
/// columns each set of that byte's pairs reaches.
const REACH_TABLE_ROWS_A_BYTE: usize = 64;

/// The row loop of a masked level whose g level has at most
/// [`ROW_COLUMNS_MAX_WIDTH`] pairs in all (`liveness::pairs_fit`), read by
/// pairs.
///
/// Each side's row columns are then over g's pairs, numbered in column
/// order: bit `q` of child row `a`'s word is set iff the child grid's cell
/// at `a` and g pair `q`'s child on that side holds a product, and
/// [`ROW_LIVE`] iff the child row has a live cell. An f pair `(a, b)` and
/// g pair `q` make a live candidate iff the words of `a` on the left and
/// `b` on the right both have bit `q`: one intersection decides an f pair against
/// every pair of g's level, where the cell walks read both child grids for
/// each candidate, and a cell holds a product iff the union of its row's intersections
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
/// alone, the columns of the g pairs its intersections have, all of them entered.
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
    let mut inline = 0;
    match lookups {
        PlainLookups::Grid => {
            let cells = rows.f_width.saturating_mul(rows.ctx.right_width);
            let most = ahead_most(rows.f_level, rows.g_level, cells);
            let mut sink = ProjectedEmitSink::new(eng.limits(), most, rows.f_width, level, &mut inline);
            pair_rows(eng, rows, scratch, &grid(&sides.left), &grid(&sides.right), &mut sink)?;
            sink.trim();
            Ok(())
        }
        PlainLookups::CompleteLeft { charged } => {
            pair_rows(
                eng, rows, scratch,
                &complete(&sides.left), &grid(&sides.right),
                &mut ReservedEmitSink { level, charged, inline: &mut inline },
            )
        }
        PlainLookups::CompleteRight { charged } => {
            pair_rows(
                eng, rows, scratch,
                &grid(&sides.left), &complete(&sides.right),
                &mut ReservedEmitSink { level, charged, inline: &mut inline },
            )
        }
        PlainLookups::Complete { .. } => unreachable!("a level of two complete sides builds no masks"),
    }?;
    level.leave_written_pairs(inline);
    Ok(())
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
    lim.reserve_exact(g_flat, (0..right_width).map(|j| g_level.pair_count_at(j)).sum())?;
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
    let g: [ChildPair; W] = std::array::from_fn(|j| g_flat[j]);
    let all = ROW_LIVE | low(g_flat.len());
    let (left_words, right_words) = (ctx.sides.left.row_cols, ctx.sides.right.row_cols);
    debug_assert!(!left.kills() || ctx.sides.left.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.left.0));
    debug_assert!(!right.kills() || ctx.sides.right.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.right.0));
    let mut reach_storage = [const { std::mem::MaybeUninit::<[u64; 256]>::uninit() }; 8];
    let reach_of = init_reach_tables::<ONE>(f_width, g_flat, col_of, &mut reach_storage);
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    let mut few_ands = [0u64; FEW_PAIRS];
    let mut many: Vec<u64> = Vec::new();

    let slab_fill = f_width.saturating_mul(width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if W == 0 && slab_fill && !extend_slab(slab, ctx.output_grid_base, ctx.output_grid_base + f_width * width) {
        mark_dead(&mut slab[ctx.output_grid_base..ctx.output_grid_base + f_width * width]);
    }
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, f_level, f_width, left_view, right_view, f_scratch);
    let (two_rows, two) = prepare_two_column_rows::<W>(f_level, f_width, (left_view, right_view), f_described, f_scratch, (left_words, right_words), g_flat, all, ctx.output_grid_base);
    let mut next = 0;
    'rows: while next < f_width {
        let mut out = sink.open_row(0, 0);
        while next < f_width {
            if let (Some(two), TwoRows::Described(k), Some(o)) = (&two, two_rows, out.as_mut()) {
                next = run_described_two_columns(two, &f_scratch[..f_width * k], k, (left, right), slab, o, (next, f_width), &mut poll);
                if next == f_width { break; }
            }
            let i = next;
            next += 1;
            let row_base = ctx.output_grid_base + i * width;
            if W == 0 && !extend_slab(slab, row_base, row_base + width) && !slab_fill {
                mark_dead(&mut slab[row_base..row_base + width]);
            }
            let f_pairs = match f_described {
                Some(k) => &f_scratch[i * k..(i + 1) * k],
                None => f_level.pairs_view_decoded(i, f_scratch, left_view, right_view),
            };
            if f_pairs.is_empty() {
                if W > 0 { write_cells(slab, row_base, [NO_PRODUCT; W]); }
                continue;
            }
            if let (Some(two), TwoRows::Stored(stored), Some(o)) = (&two, two_rows, out.as_mut())
                && f_pairs.len() <= 2
                && let Some(to) = try_stored_two_columns(two, stored, f_width, (left, right), slab, o, i, &mut poll)
            {
                next = to;
                continue;
            }
            let (l, r, live) = fold_row_pair_masks(lim, f_pairs, (left, right), (left_words, right_words), all, &mut few_ands, &mut many)?;
            if l & r & ROW_LIVE == 0 {
                if W > 0 { write_cells(slab, row_base, [NO_PRODUCT; W]); }
                continue;
            }
            let n1 = f_pairs.len();
            let (cells, next_poll) = charge_row_entered_units::<ONE, W>(ctx, f_pairs, (l, r, live), reach_of, (cols, col_of, col_pairs), (tabled, charge_of, width), poll)?;
            poll = next_poll;
            let ands = if n1 <= FEW_PAIRS { &few_ands[..n1] } else { &many[..] };
            if let Some(o) = out.as_mut().filter(|o| o.has_room(n1 * g_flat.len(), width)) {
                write_row_in_room::<ONE, W, L, R>(o, (f_pairs, ands), (g_flat, &g, cols), cells, slab, row_base, (left, right));
                continue;
            }
            drop(out);
            emit_row_via_sink::<ONE, W, L, R, S>(eng, sink, i, (f_pairs, ands), (g_flat, cols, col_pairs), cells, slab, row_base, (left, right))?;
            continue 'rows;
        }
    }
    poll.flush()?;
    Ok(())
}

/// Charge the work clock for the columns `l` and `r` enter and return the
/// bitmask of live columns `cells` for the row along with `poll`.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn charge_row_entered_units<'a, const ONE: bool, const W: usize>(
    ctx: &CellCtx<'_>,
    f_pairs: &[ChildPair],
    (l, r, live): (u64, u64, u64),
    reach_of: &[[u64; 256]],
    (cols, col_of, col_pairs): (
        &[u64; ROW_COLUMNS_MAX_WIDTH],
        &[u8; ROW_COLUMNS_MAX_WIDTH],
        &[u8; ROW_COLUMNS_MAX_WIDTH],
    ),
    (tabled, charge_of, width): (bool, &[u8; 1 << CHARGE_TABLE_COLUMNS], usize),
    mut poll: PollGate<'a>,
) -> Result<(u64, PollGate<'a>), OperationError> {
    let entered = reached_cols::<ONE>(l, reach_of, cols, col_of) & reached_cols::<ONE>(r, reach_of, cols, col_of);
    let units = f_pairs.len() as u64 * pairs_in::<ONE, W>(entered, tabled, charge_of, col_pairs) + width as u64;
    if units < poll.due_in() {
        poll.add_short(units);
    } else {
        poll = poll_entered_due(poll, ctx, f_pairs, units, width, entered, (!ONE).then_some(col_pairs))?;
    }
    let cells = reached_cols::<ONE>(live & !ROW_LIVE, reach_of, cols, col_of);
    debug_assert_eq!(cells & !entered, 0, "a live cell the masks do not enter");
    Ok((cells, poll))
}

/// Run [`two_column_rows`] from described row `next`, returning the advanced
/// row index `to`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_described_two_columns<L: ChildLookup, R: ChildLookup>(
    two: &TwoColumns<'_>,
    rows: &[ChildPair],
    k: usize,
    (left, right): (&L, &R),
    slab: &mut Vec<u32>,
    o: &mut ArenaRow<'_>,
    (next, f_width): (usize, usize),
    poll: &mut crate::limits::PollGate<'_>,
) -> usize {
    let (to, units, cursor) =
        two_column_rows(two, |j| described_two(rows, k, j), (left, right), slab, o.cursor(), (next, f_width, poll.due_in()));
    o.set_cursor(cursor);
    poll.add_short(units);
    to
}

/// Run [`two_column_rows`] from stored row `i`, returning `Some(to)` when at
/// least one row was written.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_stored_two_columns<L: ChildLookup, R: ChildLookup>(
    two: &TwoColumns<'_>,
    stored: StoredPairs<'_>,
    f_width: usize,
    (left, right): (&L, &R),
    slab: &mut Vec<u32>,
    o: &mut ArenaRow<'_>,
    i: usize,
    poll: &mut crate::limits::PollGate<'_>,
) -> Option<usize> {
    let (nodes, arena) = (&stored.nodes()[..f_width], stored.arena());
    let (to, units, cursor) =
        two_column_rows(two, |j| stored_two(nodes, arena, j), (left, right), slab, o.cursor(), (i, f_width, poll.due_in()));
    o.set_cursor(cursor);
    (to > i).then(|| {
        poll.add_short(units);
        to
    })
}

/// Count the total g-pairs across the columns set in `s`.
#[inline(always)]
fn pairs_in<const ONE: bool, const W: usize>(s: u64, tabled: bool, charge_of: &[u8; 1 << CHARGE_TABLE_COLUMNS], col_pairs: &[u8]) -> u64 {
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
}

/// Build the per-byte column reach lookup tables when `!ONE` and the level has
/// enough rows to amortize them.
#[inline(always)]
fn init_reach_tables<'a, const ONE: bool>(
    f_width: usize,
    g_flat: &[ChildPair],
    col_of: &[u8],
    reach_of: &'a mut [std::mem::MaybeUninit<[u64; 256]>; 8],
) -> &'a [[u64; 256]] {
    let bytes = g_flat.len().div_ceil(8);
    if !ONE && (f_width >= REACH_TABLE_ROWS_A_BYTE * bytes || super::super::reach_tables_forced()) {
        super::super::note_reach_tables();
        for (k, table) in reach_of.iter_mut().enumerate().take(bytes) {
            let table = table.write([0; 256]);
            for b in 1..256usize {
                let q = 8 * k + b.trailing_zeros() as usize;
                let reach = if q < g_flat.len() { 1u64 << col_of[q] } else { 0 };
                table[b] = table[b & (b - 1)] | reach;
            }
        }
        // Safety: the first `bytes` tables were written above.
        unsafe { std::slice::from_raw_parts(reach_of.as_ptr().cast::<[u64; 256]>(), bytes) }
    } else {
        &[]
    }
}

/// The columns with a pair among the bits of `s`.
#[inline(always)]
fn reached_cols<const ONE: bool>(s: u64, reach_of: &[[u64; 256]], cols: &[u64], col_of: &[u8]) -> u64 {
    if ONE {
        return s & !ROW_LIVE;
    }
    let mut s = s & !ROW_LIVE;
    if !reach_of.is_empty() {
        return reach_of.iter().fold(0, |out, table| {
            let byte = s as u8;
            s >>= 8;
            out | table[usize::from(byte)]
        });
    }
    let mut out = 0u64;
    while s != 0 {
        let j = col_of[s.trailing_zeros() as usize] as usize;
        out |= 1 << j;
        s &= !cols[j];
    }
    out
}

/// Prepare the optional two-column fast-path descriptors when `W == 2`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn prepare_two_column_rows<'a, const W: usize>(
    f_level: &'a TddLevel,
    f_width: usize,
    (left_view, right_view): (crate::diagram::ChildDecoder, crate::diagram::ChildDecoder),
    f_described: Option<usize>,
    f_scratch: &[ChildPair],
    (left_words, right_words): (&'a [u64], &'a [u64]),
    g_flat: &[ChildPair],
    all: u64,
    output_grid_base: usize,
) -> (TwoRows<'a>, Option<TwoColumns<'a>>) {
    let two_rows = match f_described {
        Some(k @ 1..=2) if W == 2 && f_scratch.len() >= f_width * k => TwoRows::Described(k),
        None if W == 2 && !left_view.is_marginal() && !right_view.is_marginal() => match f_level.stored() {
            Some(stored) if stored.nodes().len() >= f_width => TwoRows::Stored(stored),
            _ => TwoRows::Neither,
        },
        _ => TwoRows::Neither,
    };
    let two = (W == 2 && !matches!(two_rows, TwoRows::Neither)).then(|| TwoColumns {
        left_words,
        right_words,
        g: [g_flat[0], g_flat[1]],
        all,
        output_grid_base,
    });
    (two_rows, two)
}

/// Fold `f_pairs`'s left and right row words into `(l, r, live)` and record
/// each pair's intersection in `few_ands` (up to `FEW_PAIRS`) or `many`.
#[inline(always)]
fn fold_row_pair_masks<L: ChildLookup, R: ChildLookup>(
    lim: &crate::limits::Limits,
    f_pairs: &[ChildPair],
    (left, right): (&L, &R),
    (left_words, right_words): (&[u64], &[u64]),
    all: u64,
    few_ands: &mut [u64; FEW_PAIRS],
    many: &mut Vec<u64>,
) -> Result<(u64, u64, u64), OperationError> {
    let few = f_pairs.len() <= FEW_PAIRS;
    if !few {
        many.clear();
        lim.reserve(many, f_pairs.len())?;
    }
    let (mut l, mut r, mut live) = (0, 0, 0);
    for (k, p) in f_pairs.iter().enumerate() {
        let wl = if left.kills() { left_words[p.left.raw() as usize] } else { all };
        let wr = if right.kills() { right_words[p.right.raw() as usize] } else { all };
        let and = wl & wr;
        (l, r, live) = (l | wl, r | wr, live | and);
        if few {
            few_ands[k] = and;
        } else {
            many.push(and);
        }
    }
    Ok((l, r, live))
}

/// Write a row's live `cells` directly into `o` when `o` has enough room for
/// every candidate pair and node.
#[inline(always)]
fn write_row_in_room<const ONE: bool, const W: usize, L: ChildLookup, R: ChildLookup>(
    o: &mut ArenaRow<'_>,
    (f_pairs, ands): (&[ChildPair], &[u64]),
    (g_flat, g, cols): (&[ChildPair], &[ChildPair; W], &[u64]),
    mut cells: u64,
    slab: &mut Vec<u32>,
    row_base: usize,
    (left, right): (&L, &R),
) {
    if W > 0 {
        let mut row = [NO_PRODUCT; W];
        for (j, (cell, q)) in row.iter_mut().zip(g).enumerate() {
            if cells >> j & 1 == 0 {
                continue;
            }
            for (p, &and) in f_pairs.iter().zip(ands) {
                if and >> j & 1 != 0 {
                    // Safety: the arenas have room for a pair an f pair and a g pair.
                    unsafe { o.push(left.get(slab, p.left.0, q.left.0), right.get(slab, p.right.0, q.right.0)) };
                }
            }
            // Safety: the arenas have room for a node a column.
            if let Some(node) = unsafe { o.close_node() } {
                *cell = node;
            }
        }
        write_cells(slab, row_base, row);
        return;
    }
    let node_idx = &mut slab[..];
    while cells != 0 {
        let j = cells.trailing_zeros() as usize;
        cells &= cells - 1;
        let col = cols[j];
        let _: Result<(), std::convert::Infallible> = try_for_each_cell_candidate::<ONE, _, _, _>(
            (f_pairs, ands), (g_flat, j, col), node_idx, (left, right),
            |lc, rc| {
                // Safety: the arenas have room for a pair an f pair and a g pair.
                unsafe { o.push(lc, rc) };
                Ok(())
            },
        );
        // Safety: the arenas have room for a node a column.
        unsafe { o.close(node_idx, row_base + j) };
    }
}

/// Emit a row's live `cells` through `sink` when the open-row fast path lacks
/// room.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn emit_row_via_sink<const ONE: bool, const W: usize, L: ChildLookup, R: ChildLookup, S: PairSink>(
    eng: &Engine,
    sink: &mut S,
    i: usize,
    (f_pairs, ands): (&[ChildPair], &[u64]),
    (g_flat, cols, col_pairs): (&[ChildPair], &[u64], &[u8]),
    mut cells: u64,
    slab: &mut Vec<u32>,
    row_base: usize,
    (left, right): (&L, &R),
) -> Result<(), OperationError> {
    let n1 = f_pairs.len();
    sink.enter_row(eng, i)?;
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
        let bound = if ONE { n1 } else { n1 * usize::from(col_pairs[j]) };
        if let Some(mut cell) = sink.open_cell(bound) {
            let _: Result<(), std::convert::Infallible> = try_for_each_cell_candidate::<ONE, _, _, _>(
                (f_pairs, ands), (g_flat, j, col), node_idx, (left, right),
                |lc, rc| {
                    // Safety: the cell was opened for each f pair against each of the column's pairs.
                    unsafe { cell.push(lc, rc) };
                    Ok(())
                },
            );
            cell.close(eng, node_idx, grid_pos)?;
            continue;
        }
        let start = sink.begin();
        try_for_each_cell_candidate::<ONE, _, _, _>(
            (f_pairs, ands), (g_flat, j, col), node_idx, (left, right),
            |lc, rc| sink.pair(eng, lc, rc),
        )?;
        sink.end(eng, node_idx, grid_pos, start)?;
    }
    Ok(())
}

/// Call `emit(lc, rc)` for each live candidate of a cell at column `j` with
/// column mask `col` against `f_pairs` and their row masks `ands`.
#[inline(always)]
fn try_for_each_cell_candidate<const ONE: bool, L: ChildLookup, R: ChildLookup, E>(
    (f_pairs, ands): (&[ChildPair], &[u64]),
    (g_flat, j, col): (&[ChildPair], usize, u64),
    node_idx: &[u32],
    (left, right): (&L, &R),
    mut emit: impl FnMut(u32, u32) -> Result<(), E>,
) -> Result<(), E> {
    for (p, &and) in f_pairs.iter().zip(ands) {
        let mut m = and & col;
        if m == 0 {
            continue;
        }
        if ONE {
            let q = &g_flat[j];
            emit(left.get(node_idx, p.left.0, q.left.0), right.get(node_idx, p.right.0, q.right.0))?;
            continue;
        }
        let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
        while m != 0 {
            let q = &g_flat[m.trailing_zeros() as usize];
            m &= m - 1;
            emit(left.get_in_row(node_idx, lrow, q.left.0), right.get_in_row(node_idx, rrow, q.right.0))?;
        }
    }
    Ok(())
}

/// What the rows of a level of two one-pair columns read in
/// [`two_column_rows`].
struct TwoColumns<'a> {
    /// Each side's row columns, read where the side's lookups kill.
    left_words: &'a [u64],
    right_words: &'a [u64],
    /// The two columns' g pairs.
    g: [ChildPair; 2],
    /// The word of a side whose lookups kill nothing.
    all: u64,
    output_grid_base: usize,
}

/// Where [`two_column_rows`] reads f's rows: the scratch rows of `k` pairs
/// a described level was decoded into, the stored nodes, or neither.
#[derive(Clone, Copy)]
enum TwoRows<'a> {
    Described(usize),
    Stored(StoredPairs<'a>),
    Neither,
}

/// Row `i` of `rows`, rows of `k` pairs, one or two, as its first and last
/// pair and their count.
#[inline(always)]
fn described_two(rows: &[ChildPair], k: usize, i: usize) -> Option<(ChildPair, ChildPair, u64)> {
    debug_assert!((i + 1) * k <= rows.len());
    // Safety: the caller reads the rows below `rows.len() / k`.
    unsafe { Some((*rows.get_unchecked(i * k), *rows.get_unchecked(i * k + k - 1), k as u64)) }
}

/// Node `i` of a stored level's `nodes`, whose pairs are in `arena`, as
/// [`described_two`] gives a row, `None` for a node of more than two pairs.
#[inline(always)]
fn stored_two(nodes: &[EncodedNode], arena: &[ChildPair], i: usize) -> Option<(ChildPair, ChildPair, u64)> {
    debug_assert!(i < nodes.len());
    // Safety: the caller reads the nodes below `nodes.len()`.
    match unsafe { nodes.get_unchecked(i) }.kind() {
        NodeKind::Inline(p) => Some((p, p, 1)),
        NodeKind::Multi { start, len: 2 } => match arena[start as usize..][..2] {
            [p, q] => Some((p, q, 2)),
            _ => None,
        },
        _ => None,
    }
}

/// The rows of a level of two one-pair columns from `next` on, `f_row(i)`
/// row `i`'s f pairs where there are at most two of them, while its cells
/// fit `out`'s room and its units fit under `due_in` with those of the rows
/// before it, written at `c`. Rows, cells, pairs, nodes and units are those
/// the row loop of [`pair_rows_of`] writes and charges. Returns the first
/// row left to that loop, the units of the rows written, which the caller
/// adds to its gate, and the write position past them.
///
/// The rows are taken in runs that fit the room and the gate at their most,
/// four pairs, two nodes and six units a row, so a row tests neither, and
/// their cells are written past the slab's length, which is set once.
#[inline(never)]
fn two_column_rows<L, R>(
    two: &TwoColumns<'_>,
    f_row: impl Fn(usize) -> Option<(ChildPair, ChildPair, u64)>,
    (left, right): (&L, &R),
    slab: &mut Vec<u32>,
    mut c: RowCursor,
    (mut next, end, due_in): (usize, usize, u64),
) -> (usize, u64, RowCursor)
where
    L: ChildLookup,
    R: ChildLookup,
{
    let [q0, q1] = two.g;
    let word_left = |p: ChildPair| {
        if left.kills() {
            debug_assert!((p.left.raw() as usize) < two.left_words.len());
            // Safety: an f pair's left child is a row of the left child grid,
            // which has a row column each.
            unsafe { *two.left_words.get_unchecked(p.left.raw() as usize) }
        } else {
            two.all
        }
    };
    let word_right = |p: ChildPair| {
        if right.kills() {
            debug_assert!((p.right.raw() as usize) < two.right_words.len());
            // Safety: as on the left.
            unsafe { *two.right_words.get_unchecked(p.right.raw() as usize) }
        } else {
            two.all
        }
    };
    // The grid's rows from `next` on are written past the slab's length,
    // into the capacity its claim reserved, a gap before them filled dead
    // as `write_cells` fills one.
    let first = two.output_grid_base + 2 * next;
    if slab.len() < first {
        slab.resize(first, NO_PRODUCT);
    }
    assert!(slab.capacity() >= two.output_grid_base + 2 * end, "a grid's claim reserves its cells");
    let filled = slab.len();
    let cells = slab.as_mut_ptr();
    let mut added = 0u64;
    'runs: loop {
        let rows = (c.pair_room() / 4).min(c.node_room() / 2).min(((due_in - added).saturating_sub(1) / 6) as usize);
        let stop = end.min(next + rows);
        if stop == next {
            break;
        }
        while next < stop {
            let Some((p0, p1, n)) = f_row(next) else {
                break 'runs;
            };
            // A row of one pair has no second intersection.
            let second = 0u64.wrapping_sub(n - 1);
            let (wl0, wr0) = (word_left(p0), word_right(p0));
            let (wl1, wr1) = (word_left(p1) & second, word_right(p1) & second);
            let both = (wl0 | wl1) & (wr0 | wr1);
            let mut row = [NO_PRODUCT; 2];
            if both & ROW_LIVE != 0 {
                // The row's f pairs against each column entered, and its
                // width.
                let entered = both & 3;
                added += n * ((entered & 1) + (entered >> 1)) + 2;
                if n == 2 {
                    super::super::note_pair_row_unrolled();
                }
                // Safety: the child grids are below the slab's length, and
                // no cell is written while this view is read.
                let grid = unsafe { std::slice::from_raw_parts(cells, filled) };
                row = eval_two_column_cells(&mut c, grid, (left, right), (p0, p1), [q0, q1], (wl0 & wr0, wl1 & wr1));
            }
            // Safety: the claim reserved the grid's cells to `end`.
            unsafe { cells.add(two.output_grid_base + 2 * next).cast::<[u32; 2]>().write_unaligned(row) };
            next += 1;
        }
    }
    let written = two.output_grid_base + 2 * next;
    if slab.len() < written {
        // Safety: the cells below `written` past the length were written
        // above.
        unsafe { slab.set_len(written) };
    }
    (next, added, c)
}

/// Evaluate the two columns `q` of a row in [`two_column_rows`] with pair
/// intersections `(a0, a1)` and f pairs `(p0, p1)`.
#[inline(always)]
fn eval_two_column_cells<L: ChildLookup, R: ChildLookup>(
    c: &mut RowCursor,
    grid: &[u32],
    (left, right): (&L, &R),
    (p0, p1): (ChildPair, ChildPair),
    q: [ChildPair; 2],
    (a0, a1): (u64, u64),
) -> [u32; 2] {
    let mut row = [NO_PRODUCT; 2];
    for (j, (cell, q)) in row.iter_mut().zip(q).enumerate() {
        let (b0, b1) = (a0 >> j & 1 != 0, a1 >> j & 1 != 0);
        let get = |p: ChildPair| (left.get(grid, p.left.0, q.left.0), right.get(grid, p.right.0, q.right.0));
        // Safety: the run's rows fit the room, four pairs and two nodes a row.
        unsafe {
            if b0 && b1 {
                let ((l0, r0), (l1, r1)) = (get(p0), get(p1));
                *cell = c.pair_node(l0, r0, l1, r1);
            } else if b0 || b1 {
                let (l, r) = get(if b0 { p0 } else { p1 });
                *cell = c.single_node(l, r);
            }
        }
    }
    row
}
