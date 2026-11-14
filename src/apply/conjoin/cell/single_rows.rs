//! A plain level of one-pair cells: its loop over an array of g's pairs,
//! and the relay that writes the level as a description.

use super::*;
use crate::diagram::{EncodedChildRef, NodeKind};

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
    let RowLoop {
        f_level,
        g_level,
        ctx,
        f_width,
        ..
    } = rows;
    let RowScratch {
        f_pairs: f_scratch,
        g_pairs: g_cols,
        node_idx: slab,
    } = scratch;
    let lim = eng.limits();
    super::super::note_single_rows();
    let g_width = ctx.g_width;
    // Column `j`'s pair, a dead column's `NO_PRODUCT` on both sides; the f
    // rows' buffer decodes a column the table does not hold.
    g_cols.clear();
    lim.reserve_exact(g_cols, g_width)?;
    let dead = ChildPair::new(
        EncodedChildRef::from_raw(NO_PRODUCT),
        EncodedChildRef::from_raw(NO_PRODUCT),
    );
    for j in 0..g_width {
        let pairs = column_pairs(ctx, g_level, j, f_scratch);
        debug_assert!(pairs.len() <= 1, "a g node of one pair at most");
        g_cols.push(pairs.first().copied().unwrap_or(dead));
    }
    let rows = SingleRows {
        f_level,
        ctx,
        f_width,
        g_cols: &g_cols[..],
    };
    if relay && !super::super::relay_forced_off() && relay_rows(eng, rows, slab, level)? {
        super::super::note_relayed();
        return Ok(());
    }
    let mut inline = 0;
    let mut sink = EmitSink {
        level: &mut *level,
        charged: usize::MAX,
        inline: &mut inline,
    };
    // A level of a few columns keeps them in registers, and writes each
    // row's cells once.
    match g_width {
        1 => single_rows::<1>(eng, rows, f_scratch, slab, &mut sink),
        2 => single_rows::<2>(eng, rows, f_scratch, slab, &mut sink),
        3 => single_rows::<3>(eng, rows, f_scratch, slab, &mut sink),
        4 => single_rows::<4>(eng, rows, f_scratch, slab, &mut sink),
        _ => single_rows::<0>(eng, rows, f_scratch, slab, &mut sink),
    }?;
    level.leave_written_pairs(inline);
    Ok(())
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
    let SingleRows {
        f_level,
        ctx,
        f_width,
        g_cols,
    } = rows;
    let lim = eng.limits();
    let width = ctx.g_width;
    let (left, right) = (&ctx.sides.left, &ctx.sides.right);
    if !lim.memory_unbounded()
        || crate::diagram::stored_levels_forced()
        || left.plan.view.is_marginal()
        || right.plan.view.is_marginal()
        || width == 0
    {
        return Ok(false);
    }
    let Some(d) = f_level.implicit() else {
        return Ok(false);
    };
    let digit = crate::diagram::Digit {
        radix: f_width,
        left: 0,
        right: 1,
        node: 1,
    };
    if d.pairs_per_node() != 1 || d.nodes() != f_width || d.within() != 0 || d.digits() != [digit] {
        return Ok(false);
    }
    // A description's slots are below 2^31.
    let (a, b) = (d.first().0 as usize, d.first().1 as usize);
    let (c, e) = (g_cols[0].left.0, g_cols[0].right.0);
    if c == NO_PRODUCT
        || !g_cols
            .iter()
            .enumerate()
            .all(|(j, q)| q.left.0 == c && u64::from(q.right.0) == u64::from(e) + j as u64)
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
    debug_assert!(
        src(f_width - 1) + width <= dst || end <= src(0),
        "the child's grid and the level's are apart"
    );
    if stride == width {
        slab.copy_within(src(0)..src(0) + cells, dst);
    } else {
        for i in 0..f_width {
            slab.copy_within(src(i)..src(i) + width, dst + i * width);
        }
    }
    let out = &mut slab[dst..end];
    let Some(&v) = out.iter().find(|&&x| x != NO_PRODUCT) else {
        return Ok(false);
    };
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
    // Past the width cap the row loop writes the nodes, and refuses them.
    if nodes < crate::diagram::floor().max(2) || nodes > lim.level_width_cap() {
        return Ok(false);
    }
    let Some(described) =
        crate::diagram::ImplicitLevel::one_step(nodes, (i64::from(p), i64::from(v)), (0, 1))
    else {
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
    let SingleRows {
        f_level,
        ctx,
        f_width,
        g_cols,
    } = rows;
    let lim = eng.limits();
    let g_width = ctx.g_width;
    debug_assert!(W == 0 || W == g_width);
    let width = if W == 0 { g_width } else { W };
    let live_cols = g_cols.iter().filter(|q| q.left.0 != NO_PRODUCT).count() as u64;
    // A row's charge: a unit a live cell and the row's width.
    let row_units = live_cols + width as u64;
    let left = DenseLookup::new(&ctx.sides.left);
    let right = DenseLookup::new(&ctx.sides.right);
    let mut poll = lim.gate_with(super::super::budget::dense_cell_poll_stride());
    let (left_view, right_view) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let f_described = described_rows(lim, f_level, f_width, left_view, right_view, f_scratch);
    // The f nodes, where a row's pair is its node's.
    let f_nodes = match f_level.stored() {
        Some(stored)
            if f_described.is_none() && !left_view.is_marginal() && !right_view.is_marginal() =>
        {
            Some(stored.nodes())
        }
        _ => None,
    };
    let g: [ChildPair; W] = std::array::from_fn(|j| g_cols[j]);
    // With `W` columns and the f nodes stored, the grid extended over the
    // level's rows ahead of them, each row's cells written in place.
    let extended = W > 0 && f_nodes.is_some() && {
        let end = ctx.output_grid_base + f_width * width;
        if slab.len() < end && slab.capacity() >= end {
            slab.resize(end, NO_PRODUCT);
        }
        slab.len() >= end
    };
    // A cell's product: `None` where a side has none.
    let product = |node_idx: &[u32], (lrow, rrow): (usize, usize), q: &ChildPair| {
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
    let mut next = 0;
    'rows: while next < f_width {
        // The rows' nodes are written straight into the node arena, open
        // across rows, while it has room for a node a column; a row it has
        // no room for goes through the sink, which grows it.
        let mut out = sink.open_row(0, 0, eng.limits().level_width_cap());
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
                    let at = (left.row(p.left.0), right.row(p.right.0));
                    for (cell, q) in cells.iter_mut().zip(&g) {
                        if let Some((lc, rc)) = product(slab, at, q) {
                            // Safety: the arena has room for a node a column
                            // of the rows up to `end`.
                            *cell = unsafe { o.single_node(lc, rc) };
                        }
                    }
                }
                debug_assert!(
                    row_base + W <= slab.len(),
                    "the grid is extended over the level"
                );
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
                (None, None) => f_level
                    .pairs_view_decoded(i, f_scratch, left_view, right_view)
                    .first()
                    .copied(),
            };
            debug_assert!(
                f_level
                    .pairs_view_decoded(i, &mut Vec::new(), left_view, right_view)
                    .len()
                    <= 1,
                "an f node of one pair at most"
            );
            if W == 0 {
                open_row(slab, row_base, row_base + width);
            }
            // A dead f node's row stays dead.
            let Some(p) = p else {
                if W > 0 {
                    write_cells(slab, row_base, [NO_PRODUCT; W]);
                }
                continue;
            };
            let at = (left.row(p.left.0), right.row(p.right.0));
            if let Some(o) = out.as_mut().filter(|o| o.has_room(0, width)) {
                if W > 0 {
                    let mut cells = [NO_PRODUCT; W];
                    for (cell, q) in cells.iter_mut().zip(&g) {
                        if let Some((lc, rc)) = product(slab, at, q) {
                            // Safety: the arena has room for a node a column.
                            *cell = unsafe { o.single_node(lc, rc) };
                        }
                    }
                    write_cells(slab, row_base, cells);
                } else {
                    for (j, q) in g_cols.iter().enumerate() {
                        if let Some((lc, rc)) = product(slab, at, q) {
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
                if let Some((lc, rc)) = product(slab, at, q) {
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
