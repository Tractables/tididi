//! A plain level with a complete child side: its lookups, its entry, and
//! the actions that read a lone g pair or one pair a column.

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

/// The row loop of `Route::PlainDense` and `Route::Dense` on a level with at
/// least one [complete](super::super::products::Products::is_complete) child
/// side: [`CompleteLookup`] on each complete side, the grid on the other.
///
/// With both sides complete no candidate dies and no dead-pair mask can cull,
/// so the loop runs without the per-row mask fold (`DENSE`), with no
/// `NO_PRODUCT` test, and writes through a [`EmitSink`]; where the g
/// side is one node of one pair, every cell reads that pair
/// ([`EmitBesideOne`]) and no column is read, and where every g node has
/// one pair, a cell reads its column's ([`EmitLoneColumns`]). With one
/// side complete it is [`run_level_rows_plain::<false>`](run_level_rows_plain)
/// with that side's lookup replaced: the fold and the culls on the grid side
/// stay; on a level that built no masks they find nothing, as on
/// `Route::PlainDense` they would not be run. It writes through a
/// [`EmitSink`] too, its arena reserved for every pair the level
/// would write if none died.
pub(crate) fn run_level_rows_complete(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    lookups: PlainLookups,
) -> Result<(), OperationError> {
    let sides = rows.ctx.sides;
    let (complete, grid) = (CompleteLookup::new, DenseLookup::new);
    let mut inline = 0;
    match lookups {
        PlainLookups::Complete { charged } => match lone_pair(rows, &mut *scratch.g_pairs) {
            Some(g) => run_level_rows::<true, _, _, _>(
                eng,
                rows,
                scratch,
                &complete(&sides.left),
                &complete(&sides.right),
                &mut EmitBesideOne {
                    sink: EmitSink {
                        level,
                        charged,
                        inline: &mut inline,
                    },
                    g: [g],
                },
            ),
            None => run_complete_or_lone::<true, _, _>(
                eng,
                rows,
                scratch,
                (&complete(&sides.left), &complete(&sides.right)),
                EmitSink {
                    level,
                    charged,
                    inline: &mut inline,
                },
            ),
        },
        PlainLookups::CompleteLeft { charged } => run_complete_or_lone::<false, _, _>(
            eng,
            rows,
            scratch,
            (&complete(&sides.left), &grid(&sides.right)),
            EmitSink {
                level,
                charged,
                inline: &mut inline,
            },
        ),
        PlainLookups::CompleteRight { charged } => run_complete_or_lone::<false, _, _>(
            eng,
            rows,
            scratch,
            (&grid(&sides.left), &complete(&sides.right)),
            EmitSink {
                level,
                charged,
                inline: &mut inline,
            },
        ),
        PlainLookups::Grid => unreachable!("a level with no complete side takes the grid row loop"),
    }?;
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

    /// A row of one or two f pairs whose cells all fit the arenas' room is
    /// written there in one loop ([`lone_row`]).
    #[inline(always)]
    fn whole_row(
        &mut self,
        _eng: &Engine,
        row: WholeRow<'_, '_, L, R>,
    ) -> Result<usize, OperationError> {
        match row.f_pairs.len() {
            1 => lone_row::<1, L, R>(&mut self.sink, self.lone, row),
            2 => lone_row::<2, L, R>(&mut self.sink, self.lone, row),
            _ => Ok(0),
        }
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
/// node, and the gate polled at the cell that brings it due, as a poll a
/// cell would charge it, and counted down at once for the cells between.
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
    let (l0, r0) = (left.row(f[0].left.0), right.row(f[0].right.0));
    let (l1, r1) = (left.row(f[K - 1].left.0), right.row(f[K - 1].right.0));
    // The row's write position held in registers across its cells.
    let mut cursor = out.cursor();
    let units = K as u64;
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    // The row's cells, tested once here rather than at each store.
    assert!(row_base <= node_idx.len() && width <= node_idx.len() - row_base);
    let mut start = 0;
    while start < width {
        // Cell `start` charged to the gate, which polls if it comes due,
        // then the cells after it that leave it short of due, charged
        // together: their loop holds no gate count.
        super::rows_stream::charge(gate, &mut due_at, &mut due, units)?;
        let short = (due.saturating_sub(1) / units).min((width - start - 1) as u64);
        due -= short * units;
        let end = start + 1 + short as usize;
        for (j, q) in (start..end).zip(&lone[start..end]) {
            let (c, d) = (q.left.0, q.right.0);
            // An f pair's product with the column's pair: `None` where a side
            // has none.
            let product = |node_idx: &[u32], l: L::Row, r: R::Row| {
                let lc = left.get_in_row(node_idx, l, c);
                if left.kills() && lc == NO_PRODUCT {
                    return None;
                }
                let rc = right.get_in_row(node_idx, r, d);
                (!right.kills() || rc != NO_PRODUCT).then_some((lc, rc))
            };
            let first = product(node_idx, l0, r0);
            // Safety: the row was opened for a node a column and, with two f
            // pairs, for two pairs a column.
            let node = unsafe {
                if K == 1 {
                    first.map(|(lc, rc)| cursor.single_node(lc, rc))
                } else {
                    match (first, product(node_idx, l1, r1)) {
                        (Some((a, b)), Some((c, d))) => Some(cursor.pair_node(a, b, c, d)),
                        (Some((a, b)), None) | (None, Some((a, b))) => {
                            Some(cursor.single_node(a, b))
                        }
                        (None, None) => None,
                    }
                }
            };
            if let Some(node) = node {
                // Safety: `j` is below `width`, which the row's cells fit.
                unsafe { *node_idx.get_unchecked_mut(row_base + j) = node };
            }
        }
        start = end;
    }
    out.set_cursor(cursor);
    gate.add_short(due_at - due);
    Ok(width)
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
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        cell_pairs::<_, _, _>(
            (eng, a.ctx),
            (a.j, a.row_base + a.j),
            a.f_pairs,
            &self.g,
            a.node_idx,
            (a.left, a.right),
            (&mut self.sink, a.gate),
        )
    }
}

/// The one pair of a level's g side, where it has one node and that node
/// one pair: the pair every cell of the level reads as its column's.
fn lone_pair(rows: RowLoop<'_>, scratch: &mut Vec<ChildPair>) -> Option<ChildPair> {
    if rows.ctx.g_width != 1 {
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
