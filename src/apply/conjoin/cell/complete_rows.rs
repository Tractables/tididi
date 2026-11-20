//! A plain level with a complete child side: its lookups, its entry, and
//! the action that reads one pair a column.

use super::*;
use crate::limits::Transient;

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
    /// pair, `usize::MAX` when it was not (see [`EmitSink`]).
    Complete { charged: usize },
}

impl PlainLookups {
    /// The pair capacity the meters have been charged for, `usize::MAX`
    /// where no arena was reserved.
    pub(crate) fn charged(self) -> usize {
        match self {
            PlainLookups::Grid => usize::MAX,
            PlainLookups::CompleteLeft { charged }
            | PlainLookups::CompleteRight { charged }
            | PlainLookups::Complete { charged } => charged,
        }
    }
}

/// `body` with `l` and `r` the lookups `lookups` names for `sides`, and the
/// optional `const` true where both sides are complete; compiled once for
/// each variant listed, any other unreachable.
macro_rules! with_lookups {
    (@sides Grid, $s:expr) => {
        (&DenseLookup::new(&$s.left), &DenseLookup::new(&$s.right))
    };
    (@sides CompleteLeft, $s:expr) => {
        (&CompleteLookup::new(&$s.left), &DenseLookup::new(&$s.right))
    };
    (@sides CompleteRight, $s:expr) => {
        (&DenseLookup::new(&$s.left), &CompleteLookup::new(&$s.right))
    };
    (@sides Complete, $s:expr) => {
        (&CompleteLookup::new(&$s.left), &CompleteLookup::new(&$s.right))
    };
    (@both Complete) => {
        true
    };
    (@both $arm:ident) => {
        false
    };
    ($lookups:expr, $sides:expr, [$($arm:ident),+], |$l:ident, $r:ident| $body:expr) => {
        with_lookups!($lookups, $sides, [$($arm),+], |$l, $r, const _BOTH| $body)
    };
    ($lookups:expr, $sides:expr, [$($arm:ident),+], |$l:ident, $r:ident, const $both:ident| $body:expr) => {
        match $lookups {
            $(PlainLookups::$arm { .. } => {
                const $both: bool = with_lookups!(@both $arm);
                let ($l, $r) = with_lookups!(@sides $arm, $sides);
                $body
            })+
            #[allow(unreachable_patterns)]
            other => unreachable!("{other:?} has no row loop here"),
        }
    };
}
pub(super) use with_lookups;

/// The row loop of `Route::PlainDense` and `Route::Dense` on a level with at
/// least one [complete](super::super::products::Products::is_complete) child
/// side: [`CompleteLookup`] on each complete side, the grid on the other.
///
/// With both sides complete no candidate dies and no dead-pair mask can cull,
/// so the loop runs without the per-row mask fold (`DENSE`), with no
/// `NO_PRODUCT` test, and writes through an [`EmitSink`]; where every g
/// node has one pair, a cell reads its column's ([`EmitLoneColumns`]). With one
/// side complete it is [`run_level_rows_plain::<false>`](run_level_rows_plain)
/// with that side's lookup replaced: the fold and the culls on the grid side
/// stay; on a level that built no masks they find nothing, as on
/// `Route::PlainDense` they would not be run. It writes through an
/// [`EmitSink`] too, its arena reserved for every pair the level
/// would write if none died.
pub(crate) fn run_level_rows_complete(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    lookups: PlainLookups,
) -> Result<(), OperationError> {
    let mut inline = 0;
    let sink = EmitSink {
        level,
        charged: lookups.charged(),
        inline: &mut inline,
    };
    // A level with no complete side takes the grid row loop.
    with_lookups!(
        lookups,
        rows.ctx.sides,
        [Complete, CompleteLeft, CompleteRight],
        |l, r, const BOTH| run_complete_or_lone::<BOTH, _, _>(eng, rows, scratch, (l, r), sink)
    )?;
    level.leave_written_pairs(inline);
    Ok(())
}

/// [`run_level_rows_complete`] through [`EmitLoneColumns`] when every g node
/// has one pair, or [`Emit`] otherwise.
#[inline(always)]
fn run_complete_or_lone<const DENSE: bool, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    (left, right): (&L, &R),
    sink: EmitSink<'_>,
) -> Result<(), OperationError> {
    match lone_columns(eng, rows, &mut *scratch.g_pairs)? {
        Some(lone) => run_level_rows::<DENSE, _, _, _>(
            eng,
            rows,
            scratch,
            left,
            right,
            &mut EmitLoneColumns { sink, lone: &lone },
        ),
        None => {
            run_level_rows::<DENSE, _, _, _>(eng, rows, scratch, left, right, &mut Emit { sink })
        }
    }
}

/// Each column's g pair, where every column of the level holds one: the
/// pair [`EmitLoneColumns`] reads a cell of the column against.
fn lone_columns<'e>(
    eng: &'e Engine,
    rows: RowLoop<'_>,
    scratch: &mut Vec<ChildPair>,
) -> Result<Option<Transient<'e, Vec<ChildPair>>>, OperationError> {
    if super::super::lone_columns_forced_off() {
        return Ok(None);
    }
    let width = rows.ctx.g_width;
    let mut lone = Transient::new(eng.limits(), Vec::new());
    eng.limits().reserve_exact(&mut *lone, width)?;
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
/// pair, `lone[j]` column `j`'s: a cell is [`cell_pairs`] on its row's f
/// pairs and that pair, with no column read.
struct EmitLoneColumns<'a, 'l> {
    sink: EmitSink<'l>,
    lone: &'a [ChildPair],
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for EmitLoneColumns<'_, '_> {
    const DENSE_SLAB: bool = true;
    const WHOLE_ROWS: bool = true;
    const ROW_RUNS: bool = true;

    /// A row of one or two f pairs whose cells all fit the arenas' room is
    /// written there in one loop ([`lone_row`]).
    #[inline(always)]
    fn whole_row(
        &mut self,
        _eng: &Engine,
        row: WholeRow<'_, '_, L, R>,
    ) -> Result<usize, OperationError> {
        // A row of one cell costs less as that cell: `lone_row` opens its
        // arena row and gate count for every row.
        if self.lone.len() == 1 {
            return Ok(0);
        }
        match row.f_pairs.len() {
            1 => lone_row::<1, L, R>(&mut self.sink, self.lone, row),
            2 => lone_row::<2, L, R>(&mut self.sink, self.lone, row),
            _ => Ok(0),
        }
    }

    /// The level's rows of one or two f pairs written through one arena
    /// row and one gate count ([`lone_rows`]).
    #[inline(always)]
    fn whole_rows(
        &mut self,
        _eng: &Engine,
        rows: WholeRows<'_, '_, L, R>,
    ) -> Result<usize, OperationError> {
        if self.lone.len() == 1 {
            return Ok(0);
        }
        lone_rows(&mut self.sink, self.lone, rows)
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        cell_pairs::<_, _, _>(
            (eng, a.ctx),
            (a.j, a.row_base + a.j),
            a.f_pairs,
            std::slice::from_ref(&self.lone[a.j]),
            a.node_idx,
            (a.left, a.right),
            (&mut self.sink, a.gate),
        )
    }
}

/// [`EmitLoneColumns`] on a row of `K`, one or two, f pairs whose every
/// cell the arenas have room for ([`ArenaRow`]): each f pair's child rows
/// read once for the row, each cell's survivors written straight into the
/// arena through a [`RowCursor`] held in registers and closed into its
/// node, and the gate charged for the row as a poll a cell would charge
/// it.
#[inline(always)]
fn lone_row<const K: usize, L: ChildLookup, R: ChildLookup>(
    sink: &mut EmitSink<'_>,
    lone: &[ChildPair],
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    const { assert!(K == 1 || K == 2, "a whole row holds one or two f pairs") };
    let WholeRow {
        row_base,
        f_pairs,
        node_idx,
        left,
        right,
        gate,
        ..
    } = row;
    let Ok(f) = <&[ChildPair; K]>::try_from(f_pairs) else {
        return Ok(0);
    };
    let width = lone.len();
    // A cell of one f pair is a node held inline: it writes no pair to the
    // pair arena.
    let width_cap = gate.limits().level_width_cap();
    let Some(mut out) = sink.open_row(if K == 1 { 0 } else { width * K }, width, width_cap) else {
        return Ok(0);
    };
    super::super::note_whole_row();
    let rows = lone_child_rows((left, right), f);
    // The row's write position held in registers across its cells.
    let mut cursor = out.cursor();
    // The row's cells, tested once here rather than at each store.
    assert!(row_base <= node_idx.len() && width <= node_idx.len() - row_base);
    let row_units = (K * width) as u64;
    if row_units < gate.due_in() {
        gate.add_short(row_units);
    } else {
        poll_due(gate, (K as u64, width as u64), 0)?;
    }
    // Safety: the row was opened for its cells, which `node_idx` holds.
    unsafe { lone_cells::<K, _, _>(&mut cursor, (left, right), rows, (node_idx, row_base), 0, lone) };
    out.set_cursor(cursor);
    Ok(width)
}

/// `times` polls of `work` units each, then one of `after`: the charge of a
/// lone row that brings the gate due. Out of line, as inlined its loop
/// moves the registers of the row loop's other paths.
#[cold]
#[inline(never)]
fn poll_due(
    gate: &mut crate::limits::PollGate<'_>,
    (work, times): (u64, u64),
    after: u64,
) -> Result<(), OperationError> {
    gate.poll_times(work, times)?;
    gate.poll(after)
}

/// The child rows of a lone row's first and last f pair.
type LoneRows<L, R> = (
    (<L as ChildLookup>::Row, <R as ChildLookup>::Row),
    (<L as ChildLookup>::Row, <R as ChildLookup>::Row),
);

/// [`LoneRows`] of the f pairs `f`, one or two.
#[inline(always)]
fn lone_child_rows<L: ChildLookup, R: ChildLookup>(
    (left, right): (&L, &R),
    f: &[ChildPair],
) -> LoneRows<L, R> {
    let (first, last) = (f[0], f[f.len() - 1]);
    (
        (left.row(first.left.0), right.row(first.right.0)),
        (left.row(last.left.0), right.row(last.right.0)),
    )
}

/// The cells from column `start` of a lone row of `K` f pairs whose child
/// rows are `rows`, one a pair of `cells`, each written through `cursor`
/// and its node stored at `row_base` plus its column of `node_idx`.
///
/// # Safety
///
/// The cursor has room for a node a cell and, with two f pairs, two pairs
/// a cell, and `node_idx` holds `row_base + start + cells.len()` cells.
#[inline(always)]
unsafe fn lone_cells<const K: usize, L: ChildLookup, R: ChildLookup>(
    cursor: &mut RowCursor,
    (left, right): (&L, &R),
    ((l0, r0), (l1, r1)): LoneRows<L, R>,
    (node_idx, row_base): (&mut [u32], usize),
    start: usize,
    cells: &[ChildPair],
) {
    // The cursor in a local: read and written through `cursor`, its node
    // count is stored back at every cell, as the arena writes might alias
    // it.
    let held = cursor;
    let mut local = *held;
    let cursor = &mut local;
    // A bounded range: zipped with a slice it keeps one count, where an
    // open one keeps two.
    for (j, q) in (start..start + cells.len()).zip(cells) {
        let (c, d) = (q.left.0, q.right.0);
        // An f pair's product with the column's pair: `None` where a side
        // has none.
        let product = |node_idx: &[u32], l: L::Row, r: R::Row| {
            let lc = left.get_in_view(node_idx, l, c);
            if left.kills() && lc == NO_PRODUCT {
                return None;
            }
            let rc = right.get_in_view(node_idx, r, d);
            (!right.kills() || rc != NO_PRODUCT).then_some((lc, rc))
        };
        let first = product(node_idx, l0, r0);
        // Safety: as the caller's.
        let node = unsafe {
            if K == 1 {
                first.map(|(lc, rc)| cursor.single_node(lc, rc))
            } else {
                match (first, product(node_idx, l1, r1)) {
                    (Some((a, b)), Some((c, d))) => Some(cursor.pair_node(a, b, c, d)),
                    (Some((a, b)), None) | (None, Some((a, b))) => Some(cursor.single_node(a, b)),
                    (None, None) => None,
                }
            }
        };
        if let Some(node) = node {
            // Safety: as the caller's.
            unsafe { *node_idx.get_unchecked_mut(row_base + j) = node };
        }
    }
    *held = local;
}

/// Rows of fewer cells than this are written by [`lone_rows`] in its own
/// loop; wider ones through [`lone_cells_apart`]. Inline in the row loop
/// the cell loop's cursor and slab pointers live on the stack, one load a
/// cell and four more a cell written; a call costs less than that only on
/// rows of at least this many cells.
const INLINE_LONE_CELLS: usize = 8;

/// [`lone_cells`] from column 0 in a function of its own, so its loop has
/// the registers to itself.
///
/// # Safety
///
/// As [`lone_cells`].
#[inline(never)]
unsafe fn lone_cells_apart<const K: usize, L: ChildLookup, R: ChildLookup>(
    cursor: &mut RowCursor,
    sides: (&L, &R),
    rows: LoneRows<L, R>,
    row: (&mut [u32], usize),
    cells: &[ChildPair],
) {
    // Safety: as the caller's.
    unsafe { lone_cells::<K, L, R>(cursor, sides, rows, row, 0, cells) }
}

/// [`EmitLoneColumns`] on a level's rows from the first, each as
/// [`lone_row`] and the row loop's poll after it would write and charge it,
/// all through one [`RowCursor`] and their units counted down from one
/// `due_in`. The first row of more than two f pairs, or that the arenas
/// have no room for, is left to the row loop with the rows after it.
#[inline(always)]
fn lone_rows<L: ChildLookup, R: ChildLookup>(
    sink: &mut EmitSink<'_>,
    lone: &[ChildPair],
    rows: WholeRows<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRows {
        f_level,
        f_width,
        f_described,
        f_pairs_scratch,
        views,
        dense,
        ctx,
        node_idx,
        whole_grid,
        left,
        right,
        gate,
    } = rows;
    let width = lone.len();
    let base = ctx.output_grid_base;
    // An open grid's cells, tested once here rather than at each store; a
    // row of a grid opened a row at a time is in the slab once opened.
    assert!(
        !whole_grid
            || base <= node_idx.len()
                && f_width.checked_mul(width).is_some_and(|n| n <= node_idx.len() - base)
    );
    let width_cap = gate.limits().level_width_cap();
    let Some(mut out) = sink.open_row(0, 0, width_cap) else {
        return Ok(0);
    };
    let mut cursor = out.cursor();
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let mut i = 0;
    while i < f_width {
        let row_base = base + i * width;
        if !whole_grid && !extend_slab(node_idx, row_base, row_base + width) {
            mark_dead(&mut node_idx[row_base..row_base + width]);
        }
        let f = f_row(f_level, f_described, f_pairs_scratch, i, views);
        if f.is_empty() || (!dense && row_alive_masks(ctx, f).is_none()) {
            i += 1;
            continue;
        }
        let k = f.len();
        let pairs = if k == 1 { 0 } else { 2 * width };
        if k > 2 || cursor.node_room() < width || cursor.pair_room() < pairs {
            break;
        }
        // The row's cells, `k` units each, and the row loop's unit a cell.
        let units = (width * (k + 1)) as u64;
        if units < due {
            due -= units;
        } else {
            gate.add_short(due_at - due);
            poll_due(gate, (k as u64, width as u64), width as u64)?;
            (due_at, due) = (gate.due_in(), gate.due_in());
        }
        super::super::note_whole_row();
        let rows = lone_child_rows((left, right), f);
        let row = (&mut node_idx[..], row_base);
        // Safety: the room was tested above, and the row's cells are in
        // the slab.
        unsafe {
            let (c, sides) = (&mut cursor, (left, right));
            match (k == 1, width < INLINE_LONE_CELLS) {
                (true, true) => lone_cells::<1, _, _>(c, sides, rows, row, 0, lone),
                (false, true) => lone_cells::<2, _, _>(c, sides, rows, row, 0, lone),
                (true, false) => lone_cells_apart::<1, _, _>(c, sides, rows, row, lone),
                (false, false) => lone_cells_apart::<2, _, _>(c, sides, rows, row, lone),
            }
        }
        i += 1;
    }
    out.set_cursor(cursor);
    gate.add_short(due_at - due);
    Ok(i)
}
