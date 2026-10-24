//! The row loop of a masked level read by pairs.

use super::*;
use super::super::liveness::{ROW_COLUMNS_MAX_WIDTH, ROW_LIVE};

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
/// row's width a row, and a cell's surviving candidates are written in the
/// order of its walk, f pair by f pair and each against the column's pairs
/// in order (no column of so few pairs is grouped), one pair alone as the
/// one-pair cell writes it.
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
            super::super::note_pair_rows(false);
            pair_rows(eng, rows, scratch, &grid(&sides.left), &grid(&sides.right), &mut EmitSink { level })
        }
        PlainLookups::CompleteLeft { charged } => {
            super::super::note_pair_rows(true);
            pair_rows(
                eng, rows, scratch,
                &complete(&sides.left), &grid(&sides.right),
                &mut ReservedEmitSink { level, charged },
            )
        }
        PlainLookups::CompleteRight { charged } => {
            super::super::note_pair_rows(true);
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
    let low = |n: usize| (1u64 << n) - 1;

    // g's pairs in column order; each column's as a mask of their bits, and
    // each pair's column. The f rows' buffer decodes a column the table
    // does not hold.
    g_flat.clear();
    lim.reserve_exact(g_flat, ROW_COLUMNS_MAX_WIDTH)?;
    let mut cols = [0u64; ROW_COLUMNS_MAX_WIDTH];
    let mut col_of = [0u8; ROW_COLUMNS_MAX_WIDTH];
    for (j, col) in cols.iter_mut().enumerate().take(right_width) {
        let start = g_flat.len();
        g_flat.extend_from_slice(column_pairs(ctx, g_level, j, f_scratch));
        debug_assert!(g_flat.len() <= ROW_COLUMNS_MAX_WIDTH, "a level read by pairs fits");
        *col = low(g_flat.len()) & !low(start);
        col_of[start..g_flat.len()].fill(j as u8);
    }
    let g_flat = &g_flat[..];
    let all = ROW_LIVE | low(g_flat.len());
    let (left_words, right_words) = (ctx.sides.left.row_cols, ctx.sides.right.row_cols);
    debug_assert!(!left.kills() || ctx.sides.left.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.left.0));
    debug_assert!(!right.kills() || ctx.sides.right.reach.iter().zip(g_flat).all(|(&r, q)| r == 1u128 << q.right.0));
    let word_left = |p: &ChildPair| if left.kills() { left_words[p.left.raw() as usize] } else { all };
    let word_right = |p: &ChildPair| if right.kills() { right_words[p.right.raw() as usize] } else { all };
    // The columns with a pair among the bits of `s`.
    let reached = |s: u64| {
        let (mut s, mut out) = (s & !ROW_LIVE, 0u64);
        while s != 0 {
            let j = col_of[s.trailing_zeros() as usize] as usize;
            out |= 1 << j;
            s &= !cols[j];
        }
        out
    };
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
        // A dead f node's row stays dead, as does a row with a side
        // without a live child row.
        if f_pairs.is_empty() {
            continue;
        }
        // Each side's fold, and the g pairs some candidate of the row
        // survives against.
        let (l, r, live) = f_pairs.iter().fold((0, 0, 0), |(l, r, live), p| {
            let (wl, wr) = (word_left(p), word_right(p));
            (l | wl, r | wr, live | (wl & wr))
        });
        if l & r & ROW_LIVE == 0 {
            continue;
        }
        // The cells the masks enter.
        let entered = reached(l) & reached(r);
        let live = live & !ROW_LIVE;
        let n1 = f_pairs.len() as u64;
        let mut cells = entered;
        while cells != 0 {
            let j = cells.trailing_zeros() as usize;
            cells &= cells - 1;
            let col = cols[j];
            poll.poll(n1 * u64::from(col.count_ones()))?;
            if live & col == 0 {
                continue;
            }
            let grid_pos = row_base + j;
            if n1 == 1 && col.count_ones() == 1 {
                let (p, q) = (&f_pairs[0], &g_flat[col.trailing_zeros() as usize]);
                let lc = left.get(node_idx, p.left.0, q.left.0);
                let rc = right.get(node_idx, p.right.0, q.right.0);
                sink.single(eng, node_idx, grid_pos, lc, rc)?;
                continue;
            }
            let start = sink.begin();
            for p in f_pairs {
                let mut m = word_left(p) & word_right(p) & col;
                if m == 0 {
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
        poll.poll(right_width as u64)?;
    }
    poll.flush()?;
    Ok(())
}
