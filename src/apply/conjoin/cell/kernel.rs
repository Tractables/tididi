//! One cell of the product: the pair walk, and the polls that charge a cell
//! or a row as that walk does.
//!
//! The `#[inline(always)]` in this directory and in `child_lookup` are not
//! decoration: dropping them costs about a tenth of a percent of the
//! instructions a mid-size exact count retires, and the loss shows up as work
//! moving out of `process_cell` into the `CellAction` impls. Anything added
//! here that the walk calls per pair belongs in the same regime.

use super::*;
use crate::limits::PollGate;

/// Push the surviving candidates of `items` in order: `cand` maps an item to
/// its `(lc, rc)`, and a candidate survives when neither side is
/// [`NO_PRODUCT`]. `cand` reads the right side only for a live left side, as
/// a walk that tests each side before the next would, and returns
/// `NO_PRODUCT` for it otherwise; so the right side alone says whether a
/// candidate survives. With `kills` false no candidate can die (neither
/// side's lookup [kills](ChildLookup::kills)), and every one is kept without
/// a test.
///
/// While the sink has [room](PairSink::room), survivors are written without
/// a capacity test: each is stored at the buffer's end and the length, held
/// in a register for the run, steps past it. A dead candidate is skipped by a
/// branch, not stored: in a counting diagram's conjunction most candidates
/// die, as few as one in a hundred survives, and a branch that mostly goes
/// one way costs less than a store of every candidate. A run holds no more
/// candidates than the room, so no write passes the buffer's capacity. Once
/// the room is used up, the next survivor goes through [`PairSink::pair`],
/// which grows the buffer; that is the survivor at which pushing each
/// survivor in turn would grow it, so the buffer's capacity, and every meter
/// charged from it, changes at the same pairs as before.
#[inline(always)]
pub(super) fn push_kept<T, S: PairSink>(
    eng: &Engine,
    sink: &mut S,
    items: &[T],
    kills: bool,
    mut cand: impl FnMut(&T) -> (u32, u32),
) -> Result<(), OperationError> {
    let mut rest = items;
    while !rest.is_empty() {
        let room = sink.room();
        if room == 0 {
            let (lc, rc) = cand(&rest[0]);
            rest = &rest[1..];
            debug_assert!(
                lc != NO_PRODUCT || rc == NO_PRODUCT,
                "a dead left side stands for the right"
            );
            debug_assert!(
                kills || rc != NO_PRODUCT,
                "a side that kills nothing answered no product"
            );
            if !kills || rc != NO_PRODUCT {
                sink.pair(eng, lc, rc)?;
            }
            continue;
        }
        let (run, tail) = rest.split_at(room.min(rest.len()));
        rest = tail;
        let v = sink.buf();
        debug_assert!(
            run.len() <= v.capacity() - v.len(),
            "a sink's room is within its buffer's capacity"
        );
        let out = v.as_mut_ptr();
        let mut len = v.len();
        for item in run {
            let (lc, rc) = cand(item);
            debug_assert!(
                lc != NO_PRODUCT || rc == NO_PRODUCT,
                "a dead left side stands for the right"
            );
            debug_assert!(
                kills || rc != NO_PRODUCT,
                "a side that kills nothing answered no product"
            );
            if !kills || rc != NO_PRODUCT {
                // Safety: `len` starts at the buffer's length and steps at
                // most once per item of `run`, which holds no more items than
                // the sink's room, itself within the spare capacity, so the
                // write stays inside the allocation.
                unsafe {
                    out.add(len).write(raw_pair(lc, rc))
                };
                len += 1;
            }
        }
        // Safety: slots below `len` hold the old contents and one written
        // pair per survivor.
        unsafe { v.set_len(len) };
    }
    Ok(())
}

/// The one-sided product walk: one operand contributes a single pair, the other
/// is swept. `ITER_F` says which — `true` sweeps `f_pairs` against the lone g
/// pair (N×1), `false` sweeps `g_pairs` against the lone f pair (1×N). The
/// grid lookup is ordered `(f field, g field)` in both directions.
///
/// The pair count is known before the sweep, so the whole cell is charged to
/// the work clock in one go: one branch per cell rather than one per pair.
#[inline(always)]
fn cell_one_sided<const ITER_F: bool, L, R, S>(
    eng: &Engine,
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
    node_idx: &mut [u32],
    grid_pos: usize,
    (left, right): (&L, &R),
    (sink, gate): (&mut S, &mut PollGate),
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let n = if ITER_F { f_pairs.len() } else { g_pairs.len() };
    gate.poll(n as u64)?;
    // A cell the arena has room for is written straight into it.
    if let Some(mut cell) = sink.open_cell(n) {
        if ITER_F {
            let (c, d) = (g_pairs[0].left.0, g_pairs[0].right.0);
            for p1 in f_pairs {
                let lc = left.get(node_idx, p1.left.0, c);
                if left.kills() && lc == NO_PRODUCT {
                    continue;
                }
                let rc = right.get(node_idx, p1.right.0, d);
                if !right.kills() || rc != NO_PRODUCT {
                    // Safety: the cell was opened for a pair a candidate.
                    unsafe { cell.push(lc, rc) };
                }
            }
        } else {
            let p1 = &f_pairs[0];
            let (lrow, rrow) = (left.row(p1.left.0), right.row(p1.right.0));
            for p2 in g_pairs {
                let lc = left.get_in_row(node_idx, lrow, p2.left.0);
                if left.kills() && lc == NO_PRODUCT {
                    continue;
                }
                let rc = right.get_in_row(node_idx, rrow, p2.right.0);
                if !right.kills() || rc != NO_PRODUCT {
                    // Safety: as above.
                    unsafe { cell.push(lc, rc) };
                }
            }
        }
        return cell.close(eng, node_idx, grid_pos);
    }
    let cell_start = sink.begin();
    let kills = left.kills() || right.kills();
    if ITER_F {
        // N×1: the row changes per pair, so each lookup resolves its own.
        let (c, d) = (g_pairs[0].left.0, g_pairs[0].right.0);
        let node_idx = &*node_idx;
        push_kept(eng, sink, f_pairs, kills, |p1| {
            let lc = left.get(node_idx, p1.left.0, c);
            if left.kills() && lc == NO_PRODUCT {
                return (lc, lc);
            }
            (lc, right.get(node_idx, p1.right.0, d))
        })?;
    } else {
        // 1×N: one f pair fixes both child rows for the whole sweep.
        let p1 = &f_pairs[0];
        let lrow = left.row(p1.left.0);
        let rrow = right.row(p1.right.0);
        let node_idx = &*node_idx;
        push_kept(eng, sink, g_pairs, kills, |p2| {
            let lc = left.get_in_row(node_idx, lrow, p2.left.0);
            if left.kills() && lc == NO_PRODUCT {
                return (lc, lc);
            }
            (lc, right.get_in_row(node_idx, rrow, p2.right.0))
        })?;
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// The general product walk: every f pair against every g pair, with the
/// dead-pair pre-filter culling rows and columns that cannot contribute.
///
/// On a column the column table grouped (see [`RightColumns::runs`]) and an
/// f row of at least [`GROUPED_MIN_PAIRS`] pairs, the walk takes both sides a
/// run of shared `.left` at a time.
#[inline(always)]
fn cell_two_sided<L, R, S>(
    (eng, ctx): (&Engine, &CellCtx<'_>),
    (j, grid_pos): (usize, usize),
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
    node_idx: &mut [u32],
    (left, right): (&L, &R),
    (sink, gate): (&mut S, &mut PollGate),
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    // ── N×M (both levels multi-pair; masks built for every side that can
    // kill where `ctx.masked`, the level's grid big enough to pay for them)
    // ──────
    // The whole-cell dead test belongs to the row loop, which applies it to
    // every arm before the cell is entered; what is left here is the per-`p1`
    // form of it, which only this arm can use.
    let cell_start = sink.begin();
    if f_pairs.len() >= GROUPED_MIN_PAIRS
        && let Some(g_runs) = ctx.right_cols.and_then(|cols| cols.runs(j))
    {
        // ── Grouped N×M: run-length groups by shared `.left` ──────────
        // Emits the same pair multiset as the general double loop, in
        // grouped order. The word-cells path (`word_cells.rs`) emits a
        // grouped cell's candidates in this same order, and
        // `the_levels_read_by_vectors_build_what_the_row_loop_builds`
        // requires each node's pairs in identical order from both, so a
        // change to this walk's order must be made there too. The column
        // table groups only levels with no pass-through side.
        debug_assert!(!left.passthrough() && !right.passthrough());
        let n2 = g_pairs.len();
        let n1 = f_pairs.len();
        let mut p1_idx = 0;
        while p1_idx < n1 {
            let p1_left = f_pairs[p1_idx].left;
            let g1_start = p1_idx;
            p1_idx += 1;
            while p1_idx < n1 && f_pairs[p1_idx].left == p1_left {
                p1_idx += 1;
            }
            let g1 = &f_pairs[g1_start..p1_idx];
            gate.poll((g1.len() * n2) as u64)?;

            if ctx.masked
                && left.kills()
                && ctx.sides.left.live_cols[p1_left.raw() as usize] & ctx.sides.left.reach[j] == 0
            {
                continue;
            }

            let mut g2_start = 0;
            for &g2_end in g_runs {
                let g2 = &g_pairs[g2_start..g2_end as usize];
                g2_start = g2_end as usize;
                let lc = left.get(node_idx, p1_left.0, g2[0].left.0);
                if left.kills() && lc == NO_PRODUCT {
                    continue;
                }

                for p1 in g1 {
                    if ctx.masked
                        && right.kills()
                        && ctx.sides.right.live_cols[p1.right.raw() as usize]
                            & ctx.sides.right.reach[j]
                            == 0
                    {
                        continue;
                    }
                    let rrow = right.row(p1.right.0);
                    let node_idx = &*node_idx;
                    // `lc` is live here, so only the right side can kill.
                    push_kept(eng, sink, g2, right.kills(), |p2| {
                        (lc, right.get_in_row(node_idx, rrow, p2.right.0))
                    })?;
                }
            }
        }
    } else {
        // ── General N×M ───────────────────────────────────────────────
        let kills = left.kills() || right.kills();
        for p1 in f_pairs {
            gate.poll(g_pairs.len() as u64)?;
            if ctx.masked
                && !left.passthrough()
                && left.kills()
                && ctx.sides.left.live_cols[p1.left.raw() as usize] & ctx.sides.left.reach[j] == 0
            {
                continue;
            }
            if ctx.masked
                && !right.passthrough()
                && right.kills()
                && ctx.sides.right.live_cols[p1.right.raw() as usize] & ctx.sides.right.reach[j]
                    == 0
            {
                continue;
            }
            let lrow = left.row(p1.left.0);
            let rrow = right.row(p1.right.0);
            let node_idx = &*node_idx;
            push_kept(eng, sink, g_pairs, kills, |p2| {
                let lc = left.get_in_row(node_idx, lrow, p2.left.0);
                if left.kills() && lc == NO_PRODUCT {
                    return (lc, lc);
                }
                (lc, right.get_in_row(node_idx, rrow, p2.right.0))
            })?;
        }
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// Merged per-cell product walk — one kernel for every dense cell action.
///
/// Arms: 1×1 (single-pair fast path via `sink.single`), N×1 / 1×N (one side
/// single — [`cell_one_sided`], one loop in both directions), N×M (reach-mask
/// culls, grouped by shared `.left` on a long f row against a column the
/// column table grouped). A cell in the N×M arm has both levels multi-pair
/// (both sides having >1 pairs); its culls read the liveness/reach arrays
/// only where the level built them (`ctx.masked`).
///
/// `ChildLookup::passthrough()` is a constant `false` on `DenseLookup`, so the
/// plain instantiations carry no pass-through branch in the inner loops; and
/// `ChildLookup::kills()` is a constant `false` on `CompleteLookup`, so a
/// complete side carries no `NO_PRODUCT` test and no mask cull.
///
/// The `g_pairs_scratch` lifetime is independent from `node_idx`: `right_level`
/// borrows from a separate `Tdd` operand, and `pairs_view_decoded` borrows
/// `g_pairs_scratch` as the decode buffer — neither aliases the output slab.
///
/// Sink allocation can return [`OperationError::OverBudget`]; every arm polls
/// for [`OperationError::Stopped`], the scratch sink included.
///
/// Keep input slices and lookup geometry as direct parameters so the optimizer
/// retains their aliasing information within the pair loops.
///
/// Inlined into the row loop: most cells are 1×1 or a handful of pairs, and
/// an out-of-line call would cost such a cell more than its pairs do — the
/// arguments spilled and reloaded, the column table and the work-clock gate
/// read back through memory at every cell.
#[inline(always)]
pub(crate) fn process_cell<L, R, S>(
    (eng, ctx): (&Engine, &CellCtx<'_>),
    (j, row_base): (usize, usize),
    f_pairs: &[ChildPair],
    (right_level, g_pairs_scratch): (&TddLevel, &mut Vec<ChildPair>),
    node_idx: &mut [u32],
    (left, right): (&L, &R),
    (sink, gate): (&mut S, &mut PollGate),
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let g_pairs = column_pairs(ctx, right_level, j, g_pairs_scratch);
    // A dead column's cell holds no product. Its row was reset to that
    // already, except where every live cell is written and no row is reset
    // ([`run_level_rows`](super::rows::run_level_rows)).
    if g_pairs.is_empty() {
        node_idx[row_base + j] = NO_PRODUCT;
        return Ok(());
    }
    // `row_base` is the row's flat slab offset, already computed by the row loop
    // (`ctx.output_grid_base + grid_row * ctx.right_width`) for its `NO_PRODUCT` reset — reuse it instead of
    // re-deriving the same product per cell.
    cell_pairs(
        (eng, ctx),
        (j, row_base + j),
        f_pairs,
        g_pairs,
        node_idx,
        (left, right),
        (sink, gate),
    )
}

/// Column `j`'s pairs: the g node's, read from the per-level
/// [`RightColumns`] table where there is one, else decoded into `scratch`.
///
/// Everything about resolving them (masks, encoding, range) depends only on
/// `j` and the level, so the table holds it and this is two loads. `None`
/// is the fallback for the levels the table declines (marginal-encoded g,
/// or an arena the budget rejected) and for the levels that do not build
/// one, whose rows read their columns too few times again: re-derive per
/// cell, and walk an N×M cell ungrouped.
#[inline(always)]
pub(crate) fn column_pairs<'a>(
    ctx: &'a CellCtx<'_>,
    right_level: &'a TddLevel,
    j: usize,
    scratch: &'a mut Vec<ChildPair>,
) -> &'a [ChildPair] {
    match ctx.right_cols {
        Some(cols) => cols.get(j),
        None => right_level.pairs_view_decoded(
            j,
            scratch,
            ctx.sides.left.plan.view,
            ctx.sides.right.plan.view,
        ),
    }
}

/// Poll `gate` for the walk of cell `j`, the f pairs `f_pairs` against a
/// column of `n2` g pairs, as [`cell_pairs`] polls it: in one poll unless
/// the gate comes due inside the cell, and then once for each piece the
/// walk polls, an f pair's candidates or, on a cell it groups, those of a
/// run of f pairs sharing a left child. The clock then reads what the
/// walk's does at the poll that comes due, so that a stop falls at the
/// same work. For a route that charges a cell's candidates without
/// walking it.
#[inline(always)]
pub(crate) fn poll_cell_as_walked(
    gate: &mut PollGate,
    ctx: &CellCtx<'_>,
    j: usize,
    f_pairs: &[ChildPair],
    n2: usize,
) -> Result<(), OperationError> {
    let n1 = f_pairs.len();
    let units = (n1 * n2) as u64;
    if units < gate.due_in() || n1 == 1 || n2 <= 1 {
        return gate.poll(units);
    }
    poll_cell_in_pieces(gate, ctx, j, f_pairs, n2)
}

/// [`poll_cell_as_walked`] for a cell the gate comes due inside.
#[cold]
#[inline(never)]
fn poll_cell_in_pieces(
    gate: &mut PollGate,
    ctx: &CellCtx<'_>,
    j: usize,
    f_pairs: &[ChildPair],
    n2: usize,
) -> Result<(), OperationError> {
    let n1 = f_pairs.len();
    if n1 >= GROUPED_MIN_PAIRS && ctx.right_cols.and_then(|cols| cols.runs(j)).is_some() {
        let mut start = 0;
        while start < n1 {
            let left = f_pairs[start].left;
            let end = f_pairs[start..]
                .iter()
                .position(|p| p.left != left)
                .map_or(n1, |k| start + k);
            gate.poll(((end - start) * n2) as u64)?;
            start = end;
        }
        return Ok(());
    }
    gate.poll_times(n2 as u64, n1 as u64)
}

/// Poll `gate` for a row of the row loop over a level without masks or
/// over the cells the masks enter, `units` in all: the cells at the
/// columns `cells` gives, each with its column's g pairs, in column
/// order, and then the row's `width`. In one poll unless the gate comes
/// due inside the row, and then cell by cell as the row loop polls them
/// ([`poll_cell_as_walked`]).
#[inline(always)]
pub(crate) fn poll_row_as_walked(
    gate: &mut PollGate,
    ctx: &CellCtx<'_>,
    f_pairs: &[ChildPair],
    units: u64,
    width: usize,
    cells: impl Iterator<Item = (usize, usize)>,
) -> Result<(), OperationError> {
    if units < gate.due_in() {
        return gate.poll(units);
    }
    poll_row_in_cells(gate, ctx, f_pairs, units, width, cells)
}

/// [`poll_row_as_walked`] for a row the gate comes due inside: the
/// cells `cells` gives polled as the row loop polls them, then the row's
/// `width`.
#[cold]
#[inline(never)]
pub(crate) fn poll_row_in_cells(
    gate: &mut PollGate,
    ctx: &CellCtx<'_>,
    f_pairs: &[ChildPair],
    units: u64,
    width: usize,
    cells: impl Iterator<Item = (usize, usize)>,
) -> Result<(), OperationError> {
    let mut polled = 0u64;
    for (j, n2) in cells {
        if n2 != 0 {
            poll_cell_as_walked(gate, ctx, j, f_pairs, n2)?;
            polled += (f_pairs.len() * n2) as u64;
        }
    }
    debug_assert_eq!(
        polled + width as u64,
        units,
        "a row's cells charge other than its units"
    );
    gate.poll(width as u64)
}

/// Poll `gate` for `each` units polled one at a time and then `then` in
/// one poll, as the row loop polls a row of `each` one-pair cells and its
/// width, where the gate comes due inside them. Takes and gives back the
/// gate by value, out of line, so that a row loop that polls nothing else
/// out of line keeps its gate in registers.
#[cold]
#[inline(never)]
pub(crate) fn poll_each_then_due(
    mut gate: PollGate<'_>,
    each: u64,
    then: u64,
) -> Result<PollGate<'_>, OperationError> {
    gate.poll_each(each)?;
    gate.poll(then)?;
    Ok(gate)
}

/// The product walk of one live cell, `f_pairs` against `g_pairs`, both
/// nonempty, written at `grid_pos`: the arms of [`process_cell`], which
/// reads column `j`'s pairs for it.
#[inline(always)]
pub(crate) fn cell_pairs<L, R, S>(
    (eng, ctx): (&Engine, &CellCtx<'_>),
    (j, grid_pos): (usize, usize),
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
    node_idx: &mut [u32],
    (left, right): (&L, &R),
    (sink, gate): (&mut S, &mut PollGate),
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    if f_pairs.len() == 1 && g_pairs.len() == 1 {
        // ── 1×1 ──────────────────────────────────────────────────────────
        gate.poll(1)?;
        let p1 = &f_pairs[0];
        let p2 = &g_pairs[0];
        let lc = left.get(node_idx, p1.left.0, p2.left.0);
        if !left.kills() || lc != NO_PRODUCT {
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if !right.kills() || rc != NO_PRODUCT {
                sink.single(eng, node_idx, grid_pos, lc, rc)?;
            }
        }
    } else if g_pairs.len() == 1 {
        cell_one_sided::<true, _, _, _>(
            eng,
            f_pairs,
            g_pairs,
            node_idx,
            grid_pos,
            (left, right),
            (sink, gate),
        )?;
    } else if f_pairs.len() == 1 {
        cell_one_sided::<false, _, _, _>(
            eng,
            f_pairs,
            g_pairs,
            node_idx,
            grid_pos,
            (left, right),
            (sink, gate),
        )?;
    } else {
        cell_two_sided(
            (eng, ctx),
            (j, grid_pos),
            f_pairs,
            g_pairs,
            node_idx,
            (left, right),
            (sink, gate),
        )?;
    }
    Ok(())
}
