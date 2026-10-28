//! The streaming collapse route: a level whose per-cell result is a count or a
//! weight rather than a set of emitted pairs.

use super::rows::{Emit, run_level_rows};
use super::word_cells::WordScratch;
use super::*;
use crate::diagram::{EncodedChildRef, ScaledFloat, WeightStore};
use crate::value::{plain_floats, CheckedProducts, FloatCol, PlainProducts, StreamChild};

// A test sums its streamed float levels a cell at a time, as the oracle
// for the row sums, or reads and keeps child rows wherever it can.
#[cfg(test)]
use crate::apply::conjoin::tests::{cells_only, eager};

#[cfg(not(test))]
#[inline(always)]
fn cells_only() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn eager() -> bool {
    false
}

impl<F: ValueDomain> StreamState<'_, F> {
    /// Fold one cell into the output column and record its slot in the grid,
    /// where the parent's pairs take it as a bare slot reference: a column
    /// past the slots a reference can name is refused.
    #[inline(always)]
    fn fold_cell(
        &mut self,
        eng: &Engine,
        pairs: &[ChildPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), OperationError> {
        let slot = eng.limits().next_slot(F::col_len(self.counts))?;
        let v = F::fold_cell(pairs, &self.left, &self.right, self.store);
        F::push_col(eng, self.counts, v)?;
        node_idx[grid_pos] = slot;
        Ok(())
    }
}

/// Fold a streaming target's cells directly into an integer, weighted or
/// scaled-float column.
/// The child lookups handle both structural and marginal children. Child views
/// borrow their levels only during this call; `stream_state` owns the output
/// column so the caller can borrow the levels mutably again to commit it.
pub(crate) fn run_level_rows_stream_count<L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    stream_state: &mut StreamLevelState,
    env: StreamEnv<'_>,
) -> Result<(), OperationError> {
    match stream_state {
        StreamLevelState::Weighted(counts) => {
            stream_level::<WeightFold, L, R>(eng, rows, scratch, left, right, counts, env)
        }
        StreamLevelState::Float(values) => stream_float_level(eng, rows, scratch, left, right, values, env),
        StreamLevelState::Int(counts) => {
            stream_level::<IntFold, L, R>(eng, rows, scratch, left, right, counts, env)
        }
    }
}

/// [`stream_level`] on a scaled-float level, each cell summed in floats as
/// its pairs are found: [`PlainSink`] where the two child columns keep
/// every product a float, [`CheckedSum`] where they read as floats, and
/// [`FoldSink`] otherwise.
fn stream_float_level<L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    values: &mut FloatCol,
    env: StreamEnv<'_>,
) -> Result<(), OperationError> {
    let st = attach_children::<FloatFold>(env, rows.children, values);
    if let Some(products) = PlainProducts::of(&st.left, &st.right)
        && let Some(out) = st.counts.narrow_under(products.scale)
    {
        let sink = PlainSink { products, out, sum: 0.0, live: false, none: Vec::new() };
        if cells_only() {
            return run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut Emit { sink });
        }
        let pools = &eng.scratch.apply;
        let (mut ones, mut floats, mut flags, mut slots) = (pools.cell_pairs.checkout(eng), pools.sum_floats.checkout(eng), pools.sum_flags.checkout(eng), pools.row_slots.checkout(eng));
        let bufs = SumBuffers { ones: &mut ones, floats: &mut floats, flags: &mut flags, slots: &mut slots };
        let mut action = sum_rows(eng, rows, left, right, sink, bufs, &mut *scratch.g_pairs)?;
        return run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action);
    }
    let mut st = st;
    let mut cell_pairs = eng.scratch.apply.cell_pairs.checkout(eng);
    let StreamState { left: lchild, right: rchild, counts, store } = &mut st;
    let pools = &eng.scratch.apply;
    let (mut lbuf, mut rbuf) = (pools.left_floats.checkout(eng), pools.right_floats.checkout(eng));
    if !cells_only()
        && let Some(l) = plain_floats(eng, lchild, &mut lbuf)?
        && let Some(r) = plain_floats(eng, rchild, &mut rbuf)?
    {
        let products = CheckedProducts::new(l, lchild.view, r, rchild.view);
        let sum = CheckedSum { products, left: lchild, right: rchild, counts, store, cell: &mut cell_pairs };
        let pools = &eng.scratch.apply;
        let (mut ones, mut floats, mut flags, mut slots) = (pools.cell_pairs.checkout(eng), pools.sum_floats.checkout(eng), pools.sum_flags.checkout(eng), pools.row_slots.checkout(eng));
        let bufs = SumBuffers { ones: &mut ones, floats: &mut floats, flags: &mut flags, slots: &mut slots };
        let mut action = sum_rows(eng, rows, left, right, sum, bufs, &mut *scratch.g_pairs)?;
        return run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action);
    }
    let mut action = Emit { sink: FoldSink { fold: &mut st, cell: CollectSink { out: &mut cell_pairs } } };
    run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action)
}

/// Bind the child columns and collapse this level into its value column.
fn stream_level<F: ValueDomain, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    counts: &mut F::Col,
    env: StreamEnv<'_>,
) -> Result<(), OperationError> {
    let mut st = attach_children::<F>(env, rows.children, counts);
    let mut cell_pairs = eng.scratch.apply.cell_pairs.checkout(eng);
    let mut action = Emit { sink: FoldSink { fold: &mut st, cell: CollectSink { out: &mut cell_pairs } } };
    run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action)
}

/// [`run_level_rows_stream_count`] on a level whose cells are read by g-pair
/// vectors ([`super::word_cells::words_pay_marginal`]): the same column, each
/// cell's pairs folded in the order the row loop folds them, and the same
/// work. A side carried through kills nothing.
#[expect(clippy::too_many_arguments)]
pub(crate) fn run_level_rows_stream_words(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    words: &mut WordScratch,
    f_rows: Sides<usize>,
    marginal_children: bool,
    stream_state: &mut StreamLevelState,
    env: StreamEnv<'_>,
) -> Result<(), OperationError> {
    super::super::note_word_cells(super::super::WordNote::Stream);
    let sides = &rows.ctx.sides;
    if marginal_children {
        super::super::note_word_cells(super::super::WordNote::StreamMarginal);
        let (left, right) = (MarginalLookup::new(&sides.left), MarginalLookup::new(&sides.right));
        stream_words(eng, rows, scratch, words, f_rows, &left, &right, stream_state, env)
    } else {
        let left = DenseLookup { base: sides.left.base, stride: sides.left.stride };
        let right = DenseLookup { base: sides.right.base, stride: sides.right.stride };
        stream_words(eng, rows, scratch, words, f_rows, &left, &right, stream_state, env)
    }
}

/// [`run_level_rows_stream_words`] on one pair of lookups.
#[expect(clippy::too_many_arguments)]
fn stream_words<L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    words: &mut WordScratch,
    f_rows: Sides<usize>,
    left: &L,
    right: &R,
    stream_state: &mut StreamLevelState,
    env: StreamEnv<'_>,
) -> Result<(), OperationError> {
    let mut cell_pairs = eng.scratch.apply.cell_pairs.checkout(eng);
    match stream_state {
        StreamLevelState::Weighted(counts) => {
            let mut st = attach_children::<WeightFold>(env, rows.children, counts);
            let sink = FoldSink { fold: &mut st, cell: CollectSink { out: &mut cell_pairs } };
            super::word_cells::rows_by_words::<true, _, _, _>(eng, rows, scratch, words, f_rows, left, right, sink)
        }
        StreamLevelState::Float(values) => {
            let mut st = attach_children::<FloatFold>(env, rows.children, values);
            if let Some(products) = PlainProducts::of(&st.left, &st.right)
                && let Some(out) = st.counts.narrow_under(products.scale)
            {
                let sink = PlainSink { products, out, sum: 0.0, live: false, none: Vec::new() };
                return super::word_cells::rows_by_words::<true, _, _, _>(eng, rows, scratch, words, f_rows, left, right, sink);
            }
            let sink = FoldSink { fold: &mut st, cell: CollectSink { out: &mut cell_pairs } };
            super::word_cells::rows_by_words::<true, _, _, _>(eng, rows, scratch, words, f_rows, left, right, sink)
        }
        StreamLevelState::Int(counts) => {
            let mut st = attach_children::<IntFold>(env, rows.children, counts);
            let sink = FoldSink { fold: &mut st, cell: CollectSink { out: &mut cell_pairs } };
            super::word_cells::rows_by_words::<true, _, _, _>(eng, rows, scratch, words, f_rows, left, right, sink)
        }
    }
}

/// Collapse at the source: each live cell's pairs gathered into a reused
/// scratch `Vec<ChildPair>` ([`CollectSink`]) and folded into the column as
/// the cell ends, never touching the output level.
struct FoldSink<'a, 'data, F: ValueDomain> {
    fold: &'a mut StreamState<'data, F>,
    /// Reused across all cells: bounds the transient peak to one cell's pairs.
    cell: CollectSink<'a>,
}

impl<F: ValueDomain> PairSink for FoldSink<'_, '_, F> {
    #[inline(always)]
    fn single(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.cell.out.clear();
        self.cell.single(eng, node_idx, grid_pos, lc, rc)?;
        self.fold.fold_cell(eng, self.cell.out, node_idx, grid_pos)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.cell.out.clear();
        self.cell.begin()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.cell.pair(eng, lc, rc)
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.cell.buf()
    }

    /// An empty cell stays `NO_PRODUCT` (no slot), as the emit walk
    /// produces no node for zero pairs.
    #[inline(always)]
    fn end(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, _start: usize) -> Result<(), OperationError> {
        if self.cell.out.is_empty() {
            return Ok(());
        }
        self.fold.fold_cell(eng, self.cell.out, node_idx, grid_pos)
    }
}

/// The fold of a scaled-float level whose two child columns are plain
/// ([`PlainProducts`]): each pair's product added to its cell's sum as the
/// row loop finds the pair, in the order [`FoldSink`] would collect it, and
/// each cell's sum appended to the column as the float a column under the
/// products' scale holds for it. The values are those [`FoldSink`] folds,
/// bit for bit, with no pair list between the walk and the fold.
struct PlainSink<'a> {
    products: PlainProducts<'a>,
    /// The output column's floats, under `products.scale`.
    out: &'a mut Vec<f64>,
    /// The open cell's sum, and whether a pair reached it.
    sum: f64,
    live: bool,
    /// The buffer [`PairSink::buf`] names, never written: the sink has no
    /// room, so every pair comes through [`PairSink::pair`].
    none: Vec<ChildPair>,
}

impl PlainSink<'_> {
    /// Append the cell's value `v` and record its slot, as
    /// [`StreamState::fold_cell`] does.
    #[inline(always)]
    fn push(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, v: f64) -> Result<(), OperationError> {
        let lim = eng.limits();
        let len = self.out.len();
        let slot = lim.next_slot(len)?;
        if len == self.out.capacity() {
            lim.reserve(self.out, 1)?;
        }
        self.out.push(v);
        node_idx[grid_pos] = slot;
        Ok(())
    }
}

impl PairSink for PlainSink<'_> {
    /// A one-pair cell's sum is its product: adding a nonnegative float to
    /// zero gives it back.
    #[inline(always)]
    fn single(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, lc: u32, rc: u32) -> Result<(), OperationError> {
        let v = self.products.product(lc, rc);
        self.push(eng, node_idx, grid_pos, v)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.sum = 0.0;
        self.live = false;
        0
    }

    #[inline(always)]
    fn pair(&mut self, _eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.sum += self.products.product(lc, rc);
        self.live = true;
        Ok(())
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        &mut self.none
    }

    #[inline(always)]
    fn room(&mut self) -> usize {
        0
    }

    /// An empty cell stays `NO_PRODUCT`, as with [`FoldSink`].
    #[inline(always)]
    fn end(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, _start: usize) -> Result<(), OperationError> {
        if !self.live {
            return Ok(());
        }
        let v = self.sum;
        self.push(eng, node_idx, grid_pos, v)
    }
}

/// What [`read_columns`] writes for a column without pairs: no pair side a
/// grid reads has bit 31 set.
const DEAD_COLUMN: u32 = u32::MAX;

/// The level's columns as [`sum_row`] reads them: into `ones` each column's
/// one pair, [`DEAD_COLUMN`] on both sides for a column without pairs, where
/// no column has two pairs or more and the level has two columns or more
/// (empty otherwise); the pairs of all the columns; and whether every pair
/// names a cell of the two child grids.
fn read_columns<'a>(
    eng: &Engine,
    rows: RowLoop<'_>,
    ones: &'a mut Vec<ChildPair>,
    scratch: &mut Vec<ChildPair>,
) -> Result<(&'a [ChildPair], usize, bool), OperationError> {
    ones.clear();
    let width = rows.ctx.right_width;
    let (sl, sr) = (rows.ctx.sides.left.stride, rows.ctx.sides.right.stride);
    let mut one = width >= 2;
    if one {
        eng.limits().reserve(ones, width)?;
    }
    let dead = ChildPair { left: EncodedChildRef::from_raw(DEAD_COLUMN), right: EncodedChildRef::from_raw(DEAD_COLUMN) };
    let (mut pairs, mut fits) = (0, true);
    for j in 0..width {
        let g = column_pairs(rows.ctx, rows.g_level, j, scratch);
        pairs += g.len();
        fits &= g.iter().all(|q| q.left.0 < sl && q.right.0 < sr);
        if one {
            match g {
                [] => ones.push(dead),
                &[q] => ones.push(q),
                _ => {
                    one = false;
                    ones.clear();
                }
            }
        }
    }
    Ok((&ones[..], pairs, fits))
}

/// A row reads its child rows as floats ([`ValueSide`]) where that reads at
/// most this many child cells for each pair it sums.
const VALUE_ROW_READS: usize = 2;

/// What [`ValueSide`] holds for a child row it has not read.
const NO_SLOT: u32 = u32::MAX;

/// A child side's grid rows read as the floats [`sum_row`] multiplies, a
/// slot a row: one float for each column of the child's grid, and one
/// past them that a [`DEAD_COLUMN`] reads. A cell holding no node reads
/// zero, which adds nothing to a sum of nonnegative floats, and is marked
/// dead. Where the level's f pairs come back to child rows they have left
/// ([`value_rows`]), the rows read first keep a slot of their own for the
/// level, as many as it has room for; a last slot holds the other row read
/// last.
struct ValueSide<'b> {
    /// The child grid's columns.
    n: usize,
    /// The slots' floats and marks, `n + 1` a slot.
    vals: &'b mut [f64],
    dead: &'b mut [u8],
    /// Each slot's least nonzero float (infinite where none is), its
    /// greatest, and one where a cell of it is dead.
    stats: &'b mut [f64],
    /// Each child row's kept slot, [`NO_SLOT`] where it has none, where
    /// the side keeps rows; empty where it keeps none.
    slot_of: &'b mut [u32],
    /// The slots rows may keep, and those handed out.
    cap: u32,
    used: u32,
    /// The row the last slot (`cap`) holds, [`NO_SLOT`] before the first.
    at: u32,
}

/// A child row as a [`ValueSide`] holds it.
#[derive(Clone, Copy)]
struct RowFloats<'r> {
    vals: &'r [f64],
    dead: &'r [u8],
    any_dead: bool,
    lo: f64,
    hi: f64,
}

impl<'b> ValueSide<'b> {
    fn new(n: usize, vals: &'b mut [f64], dead: &'b mut [u8], stats: &'b mut [f64], slot_of: &'b mut [u32]) -> Self {
        let cap = (stats.len() / 3 - 1) as u32;
        ValueSide { n, vals, dead, stats, slot_of, cap, used: 0, at: NO_SLOT }
    }

    /// Whether row `at` has a kept slot or would get one.
    #[inline(always)]
    fn kept(&self, at: u32) -> bool {
        !self.slot_of.is_empty() && (self.used < self.cap || self.slot_of[at as usize] != NO_SLOT)
    }

    /// The cells a read of child row `at` reads, counted where it reads a
    /// row it may have to read again: none where the side holds the row or
    /// keeps it.
    #[inline(always)]
    fn cost(&self, at: u32) -> usize {
        if self.at == at || self.kept(at) { 0 } else { self.n }
    }

    /// The slot of child row `at` of `side`, read first where the side
    /// does not hold it, each cell as `factor` reads its node. A cell
    /// `factor` names no float for is marked dead: no pair reads it.
    #[inline(always)]
    fn read<C: ChildLookup>(&mut self, side: &C, node_idx: &[u32], at: u32, factor: impl Fn(u32) -> Option<f64>) -> usize {
        let s = if self.kept(at) {
            let s = self.slot_of[at as usize];
            if s != NO_SLOT {
                return s as usize;
            }
            let s = self.used;
            self.used += 1;
            self.slot_of[at as usize] = s;
            s as usize
        } else {
            if self.at == at {
                return self.cap as usize;
            }
            self.at = at;
            self.cap as usize
        };
        let (n, w) = (self.n, self.n + 1);
        let (vals, dead) = (&mut self.vals[s * w..s * w + w], &mut self.dead[s * w..s * w + w]);
        let row = side.row(at);
        let (mut any, mut lo, mut hi) = (0u8, f64::INFINITY, 0.0f64);
        for (q, (v, d)) in vals[..n].iter_mut().zip(&mut dead[..n]).enumerate() {
            let c = side.get_in_row(node_idx, row, q as u32);
            let x = if side.kills() && c == NO_PRODUCT { None } else { factor(c) };
            let f = x.unwrap_or(0.0);
            *v = f;
            *d = u8::from(x.is_none());
            any |= *d;
            if f != 0.0 {
                (lo, hi) = (lo.min(f), hi.max(f));
            }
        }
        (vals[n], dead[n]) = (0.0, 1);
        self.stats[3 * s..3 * s + 3].copy_from_slice(&[lo, hi, f64::from(any)]);
        s
    }

    /// The row slot `s` holds.
    #[inline(always)]
    fn row(&self, s: usize) -> RowFloats<'_> {
        let w = self.n + 1;
        let st = &self.stats[3 * s..3 * s + 3];
        RowFloats { vals: &self.vals[s * w..s * w + w], dead: &self.dead[s * w..s * w + w], any_dead: st[2] != 0.0, lo: st[0], hi: st[1] }
    }
}

/// Whether every product of a float of row `l` and a float of row `r` is
/// its float, as `S` tests a product: the least and the greatest are,
/// and a product's rounding keeps the order of the products. Rows of
/// nonnegative floats.
#[inline(always)]
fn rows_exact<S: RowSum>(l: RowFloats<'_>, r: RowFloats<'_>) -> bool {
    !S::CHECKED || l.lo == f64::INFINITY || r.lo == f64::INFINITY || (S::times(l.lo, r.lo).1 && S::times(l.hi, r.hi).1)
}

/// A row sum's flag: a pair of the cell lives.
const LIVE: u8 = 1;
/// A row sum's flag: a product of the cell is not its float.
const INEXACT: u8 = 2;

/// [`sum_row`]'s child rows read as floats, and a row's sums against
/// one-pair columns with their flags ([`LIVE`], [`INEXACT`]).
struct ValueRows<'b> {
    left: ValueSide<'b>,
    right: ValueSide<'b>,
    sums: &'b mut [f64],
    flags: &'b mut [u8],
}

/// How a level's f pairs read one child side's rows, in the order the row
/// loop reaches them: one past the greatest row, the reads of a side
/// holding one row (each change of row), and the rows read.
#[derive(Clone, Copy, Default)]
struct SideReads {
    rows: usize,
    changes: usize,
    distinct: usize,
}

/// The bytes the rows a level keeps may take beyond its output grid's
/// (`KEEP_GRID_SHARE` times those), within a quarter of the memory
/// headroom.
const KEEP_MIN_BYTES: usize = 64 << 20;
const KEEP_GRID_SHARE: usize = 1;

/// The level's [`ValueRows`], where both child sides are read from grids
/// and every column pair names a cell of them (`fits`), with sums for
/// `ones` columns and `pairs` column pairs in all. A side keeps the rows
/// it reads where its f pairs read each of its rows twice or more on
/// average, coming back to it after leaving it: as many as fit
/// [`KEEP_MIN_BYTES`] or the output grid's bytes, the left side first.
#[expect(clippy::too_many_arguments)]
fn value_rows<'b, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    left: &L,
    right: &R,
    (fits, ones, pairs): (bool, usize, usize),
    floats: &'b mut Vec<f64>,
    flags: &'b mut Vec<u8>,
    slots: &'b mut Vec<u32>,
    scratch: &mut Vec<ChildPair>,
) -> Result<Option<ValueRows<'b>>, OperationError> {
    let (sl, sr) = (rows.ctx.sides.left.stride as usize, rows.ctx.sides.right.stride as usize);
    if !fits || left.passthrough() || right.passthrough() || sl == 0 || sr == 0 {
        return Ok(None);
    }
    let lim = eng.limits();
    let (reads, f_pairs) = side_reads(lim, rows, slots, scratch)?;
    let grid = rows.f_width.saturating_mul(rows.ctx.right_width).saturating_mul(4);
    let mut room = KEEP_MIN_BYTES.max(grid.saturating_mul(KEEP_GRID_SHARE)).min((lim.headroom() / 4) as usize);
    // The rows a side keeps, out of the room left: a kept row costs its
    // floats, marks and statistics, and the side a slot index a row.
    let mut keep = |r: SideReads, n: usize| {
        let pays = (r.changes >= 2 * r.distinct && r.distinct * n <= f_pairs.saturating_mul(pairs).saturating_mul(VALUE_ROW_READS))
            || (eager() && r.changes > r.distinct);
        let kept = if pays && 4 * r.rows < room { r.distinct.min((room - 4 * r.rows) / (9 * n + 33)) } else { 0 };
        if kept > 0 {
            room -= 4 * r.rows + kept * (9 * n + 33);
        }
        kept
    };
    let (kl, kr) = (keep(reads.left, sl), keep(reads.right, sr));
    let (ls, rs) = (kl + 1, kr + 1);
    let (wl, wr) = ((sl + 1) * ls, (sr + 1) * rs);
    let nf = wl + 3 * ls + wr + 3 * rs + ones;
    lim.reserve(floats, nf)?;
    floats.resize(nf, 0.0);
    let nd = wl + wr + ones;
    lim.reserve(flags, nd)?;
    flags.resize(nd, 0);
    let (nl, nr) = (if kl > 0 { reads.left.rows } else { 0 }, if kr > 0 { reads.right.rows } else { 0 });
    let ns = nl + nr;
    slots.clear();
    lim.reserve(slots, ns)?;
    slots.resize(ns, NO_SLOT);
    let (lv, rest) = floats.split_at_mut(wl);
    let (lst, rest) = rest.split_at_mut(3 * ls);
    let (rv, rest) = rest.split_at_mut(wr);
    let (rst, sums) = rest.split_at_mut(3 * rs);
    let (ld, rest) = flags.split_at_mut(wl);
    let (rd, flags) = rest.split_at_mut(wr);
    let (lmap, rmap) = slots.split_at_mut(nl);
    Ok(Some(ValueRows { left: ValueSide::new(sl, lv, ld, lst, lmap), right: ValueSide::new(sr, rv, rd, rst, rmap), sums, flags }))
}

/// Each child side's [`SideReads`] over the level's f pairs, and the f
/// pairs. `slots` is scratch.
fn side_reads(
    lim: &crate::limits::Limits,
    rows: RowLoop<'_>,
    slots: &mut Vec<u32>,
    scratch: &mut Vec<ChildPair>,
) -> Result<(Sides<SideReads>, usize), OperationError> {
    let (lview, rview) = (rows.ctx.sides.left.plan.view, rows.ctx.sides.right.plan.view);
    let mut reads = Sides { left: SideReads::default(), right: SideReads::default() };
    let (mut last, mut f_pairs) = ((NO_SLOT, NO_SLOT), 0);
    for i in 0..rows.f_width {
        for p in rows.f_level.pairs_view_decoded(i, scratch, lview, rview) {
            let (a, b) = (p.left.0, p.right.0);
            reads.left.changes += usize::from(a != last.0);
            reads.right.changes += usize::from(b != last.1);
            reads.left.rows = reads.left.rows.max(a as usize + 1);
            reads.right.rows = reads.right.rows.max(b as usize + 1);
            last = (a, b);
            f_pairs += 1;
        }
    }
    // The rows read, counted where a side changes rows often enough to
    // keep them.
    let count = |r: &SideReads| r.changes >= 4 || (eager() && r.changes >= 2);
    let (cl, cr) = (count(&reads.left), count(&reads.right));
    if cl || cr {
        let (nl, nr) = (if cl { reads.left.rows } else { 0 }, if cr { reads.right.rows } else { 0 });
        slots.clear();
        lim.reserve(slots, nl + nr)?;
        slots.resize(nl + nr, 0);
        let (ml, mr) = slots.split_at_mut(nl);
        for i in 0..rows.f_width {
            for p in rows.f_level.pairs_view_decoded(i, scratch, lview, rview) {
                if cl {
                    let m = &mut ml[p.left.0 as usize];
                    reads.left.distinct += usize::from(*m == 0);
                    *m = 1;
                }
                if cr {
                    let m = &mut mr[p.right.0 as usize];
                    reads.right.distinct += usize::from(*m == 0);
                    *m = 1;
                }
            }
        }
    }
    for (c, r) in [(cl, &mut reads.left), (cr, &mut reads.right)] {
        if !c {
            r.distinct = r.changes;
        }
    }
    Ok((reads, f_pairs))
}

/// How [`sum_row`] forms a cell's products and appends its sum, and the
/// sink the row loop's own cells go to.
trait RowSum {
    /// Whether a product's float can differ from the product, so that a
    /// cell holding such a product is summed through [`cells`](Self::cells)
    /// instead.
    const CHECKED: bool;
    /// The sink of the cells the row loop walks itself.
    type Cells: PairSink;
    fn cells(&mut self) -> &mut Self::Cells;
    /// The product of the values `lc` and `rc` name, and whether the float
    /// is that product.
    fn product(&self, lc: u32, rc: u32) -> (f64, bool);
    /// The left factor of [`product`](Self::product) for `lc`, `None`
    /// where `lc` names no value: `product(lc, rc)` is
    /// `times(left_factor(lc), right_factor(rc))`.
    fn left_factor(&self, lc: u32) -> Option<f64>;
    /// The right factor of [`product`](Self::product) for `rc`.
    fn right_factor(&self, rc: u32) -> Option<f64>;
    /// The product of two factors, and whether the float is that product.
    fn times(a: f64, b: f64) -> (f64, bool);
    /// Append a cell's sum `v` of products each its float.
    fn push(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, v: f64) -> Result<(), OperationError>;
}

impl<'a> RowSum for PlainSink<'a> {
    const CHECKED: bool = false;
    type Cells = PlainSink<'a>;

    #[inline(always)]
    fn cells(&mut self) -> &mut PlainSink<'a> {
        self
    }

    #[inline(always)]
    fn product(&self, lc: u32, rc: u32) -> (f64, bool) {
        (self.products.product(lc, rc), true)
    }

    #[inline(always)]
    fn left_factor(&self, lc: u32) -> Option<f64> {
        self.products.left_factor(lc)
    }

    #[inline(always)]
    fn right_factor(&self, rc: u32) -> Option<f64> {
        self.products.right_factor(rc)
    }

    #[inline(always)]
    fn times(a: f64, b: f64) -> (f64, bool) {
        (a * b, true)
    }

    #[inline(always)]
    fn push(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, v: f64) -> Result<(), OperationError> {
        PlainSink::push(self, eng, node_idx, grid_pos, v)
    }
}

/// The sums of a scaled-float level whose child columns read as floats
/// under one scale each ([`plain_floats`]) but whose products a float may
/// not hold ([`CheckedProducts`]): a cell whose every product is its float
/// and whose sum stays finite is summed in floats, the value
/// [`FloatFold::fold`] gives it; any other cell is collected and folded as
/// [`FoldSink`] folds it, which is also how the row loop's own cells are
/// summed.
struct CheckedSum<'a, 'data> {
    products: CheckedProducts<'a>,
    left: &'a StreamChild<'data, FloatFold>,
    right: &'a StreamChild<'data, FloatFold>,
    counts: &'a mut FloatCol,
    store: &'a WeightStore,
    /// The open cell's pairs, reused across cells.
    cell: &'a mut Vec<ChildPair>,
}

impl CheckedSum<'_, '_> {
    /// Append the value `v` at `grid_pos`, as [`StreamState::fold_cell`]
    /// appends a cell's.
    #[inline(always)]
    fn append(&mut self, eng: &Engine, v: ScaledFloat, node_idx: &mut [u32], grid_pos: usize) -> Result<(), OperationError> {
        let slot = eng.limits().next_slot(self.counts.len())?;
        self.counts.push(eng, v)?;
        node_idx[grid_pos] = slot;
        Ok(())
    }

    /// Fold the collected cell and append it, as [`StreamState::fold_cell`] does.
    #[inline(always)]
    fn fold(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize) -> Result<(), OperationError> {
        let v = FloatFold::fold_cell(self.cell, self.left, self.right, self.store);
        self.append(eng, v, node_idx, grid_pos)
    }
}

impl PairSink for CheckedSum<'_, '_> {
    #[inline(always)]
    fn single(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.cell.clear();
        self.pair(eng, lc, rc)?;
        self.fold(eng, node_idx, grid_pos)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.cell.clear();
        0
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        eng.limits().try_push(self.cell, ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc)))
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.cell
    }

    #[inline(always)]
    fn end(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, _start: usize) -> Result<(), OperationError> {
        if self.cell.is_empty() {
            return Ok(());
        }
        self.fold(eng, node_idx, grid_pos)
    }
}

impl<'a, 'data> RowSum for CheckedSum<'a, 'data> {
    const CHECKED: bool = true;
    type Cells = Self;

    #[inline(always)]
    fn cells(&mut self) -> &mut Self {
        self
    }

    #[inline(always)]
    fn product(&self, lc: u32, rc: u32) -> (f64, bool) {
        self.products.product(lc, rc)
    }

    #[inline(always)]
    fn left_factor(&self, lc: u32) -> Option<f64> {
        self.products.left_factor(lc)
    }

    #[inline(always)]
    fn right_factor(&self, rc: u32) -> Option<f64> {
        self.products.right_factor(rc)
    }

    #[inline(always)]
    fn times(a: f64, b: f64) -> (f64, bool) {
        CheckedProducts::times(a, b)
    }

    #[inline(always)]
    fn push(&mut self, eng: &Engine, node_idx: &mut [u32], grid_pos: usize, v: f64) -> Result<(), OperationError> {
        let v = self.products.value(v);
        self.append(eng, v, node_idx, grid_pos)
    }
}

/// The streaming action of a scaled-float level summed in floats: a row of
/// one f pair, and a row of f pairs against columns of one pair each,
/// summed in a loop of its own ([`sum_row`]); any other row a cell at a
/// time, as [`Emit`] sums it.
struct SumRows<'o, S> {
    sum: S,
    /// Each column's one pair ([`read_columns`]), or empty.
    ones: &'o [ChildPair],
    /// The pairs of all the level's columns.
    pairs: usize,
    values: Option<ValueRows<'o>>,
}

/// The pooled buffers a [`SumRows`] holds: its one-pair columns, and its
/// [`ValueRows`]' floats, marks and slots.
struct SumBuffers<'o> {
    ones: &'o mut Vec<ChildPair>,
    floats: &'o mut Vec<f64>,
    flags: &'o mut Vec<u8>,
    slots: &'o mut Vec<u32>,
}

/// The [`SumRows`] of a level summed by `sum`.
fn sum_rows<'o, S: RowSum, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    left: &L,
    right: &R,
    sum: S,
    bufs: SumBuffers<'o>,
    scratch: &mut Vec<ChildPair>,
) -> Result<SumRows<'o, S>, OperationError> {
    let SumBuffers { ones, floats, flags, slots } = bufs;
    let (ones, pairs, fits) = read_columns(eng, rows, ones, scratch)?;
    let values = value_rows(eng, rows, left, right, (fits, ones.len(), pairs), floats, flags, slots, scratch)?;
    Ok(SumRows { sum, ones, pairs, values })
}

impl<L: ChildLookup, R: ChildLookup, S: RowSum> CellAction<L, R> for SumRows<'_, S> {
    const DENSE_SLAB: bool = true;
    const WHOLE_ROWS: bool = true;

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
            self.sum.cells(),
            a.gate,
        )
    }

    #[inline(always)]
    fn whole_row(&mut self, eng: &Engine, row: WholeRow<'_, '_, L, R>) -> Result<usize, OperationError> {
        sum_row(&mut self.sum, self.ones, self.pairs, self.values.as_mut(), eng, row)
    }
}

/// The columns [`ones_row`] sums at a time.
const SUM_COLUMNS: usize = 128;

/// Charge `units` to `gate` as one poll would, counting down `due` from
/// `due_at`, the gate's `due_in` when the count began.
#[inline(always)]
fn charge(gate: &mut crate::limits::PollGate, due_at: &mut u64, due: &mut u64, units: u64) -> Result<(), OperationError> {
    if units >= *due {
        gate.add_short(*due_at - *due);
        gate.poll(units)?;
        (*due_at, *due) = (gate.due_in(), gate.due_in());
    } else {
        *due -= units;
    }
    Ok(())
}

/// The cells of a row as [`process_cell`] sums them, each cell's products
/// added in the order it adds them and each cell charged and appended in
/// column order as it charges and appends them: a row of one f pair a cell
/// at a time, its sum held in a register, and a row of f pairs against
/// one-pair columns ([`read_columns`]) f pair after f pair. Where `values`
/// holds the level's [`ValueRows`] and reading an f pair's child rows reads
/// at most [`VALUE_ROW_READS`] cells a pair it sums, the pairs read their
/// floats off those rows. A cell a checked sum cannot hold goes through its
/// cell sink, its pairs in the same order. Any other row is left to the
/// cells (0 columns written).
#[inline(always)]
fn sum_row<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    ones: &[ChildPair],
    pairs: usize,
    values: Option<&mut ValueRows<'_>>,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    if row.f_pairs.len() == 1 {
        let p1 = row.f_pairs[0];
        if let Some(vr) = values
            && (vr.left.cost(p1.left.0) + vr.right.cost(p1.right.0) <= VALUE_ROW_READS * pairs || eager())
        {
            let ls = vr.left.read(row.left, row.node_idx, p1.left.0, |c| sum.left_factor(c));
            let rs = vr.right.read(row.right, row.node_idx, p1.right.0, |c| sum.right_factor(c));
            let (l, r) = (vr.left.row(ls), vr.right.row(rs));
            return match (l.any_dead || r.any_dead, !rows_exact::<S>(l, r)) {
                (true, true) => one_pair_row_values::<true, true, _, _, _>(sum, l, r, eng, row),
                (true, false) => one_pair_row_values::<true, false, _, _, _>(sum, l, r, eng, row),
                (false, true) => one_pair_row_values::<false, true, _, _, _>(sum, l, r, eng, row),
                (false, false) => one_pair_row_values::<false, false, _, _, _>(sum, l, r, eng, row),
            };
        }
        return one_pair_row(sum, eng, row);
    }
    if ones.is_empty() {
        return Ok(0);
    }
    match values {
        Some(vr) => ones_row_values(sum, ones, vr, eng, row),
        None => ones_row(sum, ones, eng, row),
    }
}

/// [`sum_row`] on a row of one f pair, its products read off the grids.
#[inline(always)]
fn one_pair_row<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, right_level_t, g_pairs_scratch, node_idx, left, right, gate } = row;
    let width = ctx.right_width;
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let p1 = f_pairs[0];
    let (lrow, rrow) = (left.row(p1.left.0), right.row(p1.right.0));
    for j in 0..width {
        let g = column_pairs(ctx, right_level_t, j, g_pairs_scratch);
        if g.is_empty() {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, g.len() as u64)?;
        let (mut acc, mut live, mut exact) = (0.0f64, false, true);
        for q in g {
            let lc = left.get_in_row(node_idx, lrow, q.left.0);
            if left.kills() && lc == NO_PRODUCT {
                continue;
            }
            let rc = right.get_in_row(node_idx, rrow, q.right.0);
            if right.kills() && rc == NO_PRODUCT {
                continue;
            }
            let (p, ok) = sum.product(lc, rc);
            acc += p;
            live = true;
            exact &= ok;
        }
        if !live {
            continue;
        }
        if !S::CHECKED || (exact && acc < f64::INFINITY) {
            sum.push(eng, node_idx, row_base + j, acc)?;
            continue;
        }
        refold(sum, eng, node_idx, row_base + j, f_pairs, g, left, right)?;
    }
    gate.add_short(due_at - due);
    Ok(width)
}

/// [`sum_row`] on a row of one f pair whose child rows are `l` and `r`;
/// `TRACK` where a cell of them is dead, and `CHECK` where a product of
/// them may not be its float ([`rows_exact`]).
#[inline(always)]
fn one_pair_row_values<const TRACK: bool, const CHECK: bool, L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    l: RowFloats<'_>,
    r: RowFloats<'_>,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, right_level_t, g_pairs_scratch, node_idx, left, right, gate } = row;
    let width = ctx.right_width;
    let (lv, ld, rv, rd) = (l.vals, l.dead, r.vals, r.dead);
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    for j in 0..width {
        let g = column_pairs(ctx, right_level_t, j, g_pairs_scratch);
        if g.is_empty() {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, g.len() as u64)?;
        let (mut acc, mut flags) = (0.0f64, 0u8);
        for q in g {
            let (i, t) = (q.left.0 as usize, q.right.0 as usize);
            // Safety: every column pair names a cell of the two child grids
            // (`read_columns`), and a row holds a float and a mark for each
            // cell of its child row.
            let (a, b) = unsafe { (*lv.get_unchecked(i), *rv.get_unchecked(t)) };
            let (p, ok) = S::times(a, b);
            acc += p;
            if TRACK {
                // Safety: as above.
                flags |= LIVE ^ unsafe { *ld.get_unchecked(i) | *rd.get_unchecked(t) };
            }
            if CHECK && !ok {
                flags |= INEXACT;
            }
        }
        if TRACK && flags & LIVE == 0 {
            continue;
        }
        if !S::CHECKED || (flags & INEXACT == 0 && acc < f64::INFINITY) {
            sum.push(eng, node_idx, row_base + j, acc)?;
            continue;
        }
        refold(sum, eng, node_idx, row_base + j, f_pairs, g, left, right)?;
    }
    gate.add_short(due_at - due);
    Ok(width)
}

/// [`sum_row`] on a row of f pairs against one-pair columns, its products
/// read off the grids, [`SUM_COLUMNS`] columns at a time.
#[inline(always)]
fn ones_row<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    ones: &[ChildPair],
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, node_idx, left, right, gate, .. } = row;
    let width = ctx.right_width;
    let k = f_pairs.len();
    debug_assert_eq!(ones.len(), width);
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let mut sums = [0.0f64; SUM_COLUMNS];
    let mut flags = [0u8; SUM_COLUMNS];
    let mut j0 = 0;
    while j0 < width {
        let block = &ones[j0..width.min(j0 + SUM_COLUMNS)];
        let n = block.len();
        sums[..n].fill(0.0);
        flags[..n].fill(0);
        for p1 in f_pairs {
            add_grid(sum, &mut sums[..n], &mut flags[..n], block, node_idx, left, right, p1.left.0, p1.right.0);
        }
        for (x, q) in block.iter().enumerate() {
            let j = j0 + x;
            if q.left.0 == DEAD_COLUMN {
                node_idx[row_base + j] = NO_PRODUCT;
                continue;
            }
            charge(gate, &mut due_at, &mut due, k as u64)?;
            if flags[x] & LIVE == 0 {
                continue;
            }
            if !S::CHECKED || (flags[x] & INEXACT == 0 && sums[x] < f64::INFINITY) {
                sum.push(eng, node_idx, row_base + j, sums[x])?;
                continue;
            }
            refold(sum, eng, node_idx, row_base + j, f_pairs, std::slice::from_ref(q), left, right)?;
        }
        j0 += n;
    }
    gate.add_short(due_at - due);
    Ok(width)
}

/// [`sum_row`] on a row of f pairs against one-pair columns, the row's
/// sums in `vr`, each f pair's products read off its child rows where
/// reading them reads at most [`VALUE_ROW_READS`] cells a column.
#[inline(always)]
fn ones_row_values<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    ones: &[ChildPair],
    vr: &mut ValueRows<'_>,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, node_idx, left, right, gate, .. } = row;
    let width = ctx.right_width;
    let k = f_pairs.len();
    debug_assert_eq!(ones.len(), width);
    let ValueRows { left: lside, right: rside, sums, flags } = vr;
    let (sums, flags) = (&mut sums[..width], &mut flags[..width]);
    sums.fill(0.0);
    flags.fill(0);
    for p1 in f_pairs {
        let (a, b) = (p1.left.0, p1.right.0);
        if lside.cost(a) + rside.cost(b) > VALUE_ROW_READS * width && !eager() {
            add_grid(sum, sums, flags, ones, node_idx, left, right, a, b);
            continue;
        }
        let ls = lside.read(left, node_idx, a, |c| sum.left_factor(c));
        let rs = rside.read(right, node_idx, b, |c| sum.right_factor(c));
        let (l, r) = (lside.row(ls), rside.row(rs));
        match (l.any_dead || r.any_dead, !rows_exact::<S>(l, r)) {
            (true, true) => add_values::<true, true, S>(sums, flags, ones, l, r),
            (true, false) => add_values::<true, false, S>(sums, flags, ones, l, r),
            (false, true) => add_values::<false, true, S>(sums, flags, ones, l, r),
            (false, false) => add_values::<false, false, S>(sums, flags, ones, l, r),
        }
    }
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    for (j, q) in ones.iter().enumerate() {
        if q.left.0 == DEAD_COLUMN {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, k as u64)?;
        if flags[j] & LIVE == 0 {
            continue;
        }
        if !S::CHECKED || (flags[j] & INEXACT == 0 && sums[j] < f64::INFINITY) {
            sum.push(eng, node_idx, row_base + j, sums[j])?;
            continue;
        }
        refold(sum, eng, node_idx, row_base + j, f_pairs, std::slice::from_ref(q), left, right)?;
    }
    gate.add_short(due_at - due);
    Ok(width)
}

/// Add f pair `(a, b)`'s product to each one-pair column's sum, read off
/// the grids; a column without pairs is skipped.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn add_grid<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &S,
    sums: &mut [f64],
    flags: &mut [u8],
    ones: &[ChildPair],
    node_idx: &[u32],
    left: &L,
    right: &R,
    a: u32,
    b: u32,
) {
    let (lrow, rrow) = (left.row(a), right.row(b));
    for ((s, f), q) in sums.iter_mut().zip(flags.iter_mut()).zip(ones) {
        if q.left.0 == DEAD_COLUMN {
            continue;
        }
        let lc = left.get_in_row(node_idx, lrow, q.left.0);
        if left.kills() && lc == NO_PRODUCT {
            continue;
        }
        let rc = right.get_in_row(node_idx, rrow, q.right.0);
        if right.kills() && rc == NO_PRODUCT {
            continue;
        }
        let (p, ok) = sum.product(lc, rc);
        *s += p;
        *f |= if S::CHECKED && !ok { LIVE | INEXACT } else { LIVE };
    }
}

/// Add an f pair's product to each one-pair column's sum, read off its
/// child rows `l` and `r`: a dead cell's zero adds nothing, and a column
/// without pairs reads the rows' last floats. `TRACK` where a cell of the
/// rows is dead, and `CHECK` where a product of them may not be its float.
#[inline(always)]
fn add_values<const TRACK: bool, const CHECK: bool, S: RowSum>(sums: &mut [f64], flags: &mut [u8], ones: &[ChildPair], l: RowFloats<'_>, r: RowFloats<'_>) {
    let (sl, sr) = (l.vals.len() - 1, r.vals.len() - 1);
    for ((s, f), q) in sums.iter_mut().zip(flags.iter_mut()).zip(ones) {
        let (i, t) = ((q.left.0 as usize).min(sl), (q.right.0 as usize).min(sr));
        // Safety: `i <= sl` and `t <= sr`, and each row holds a float and a
        // mark for its child row's cells and one past them.
        let (a, b) = unsafe { (*l.vals.get_unchecked(i), *r.vals.get_unchecked(t)) };
        let (p, ok) = S::times(a, b);
        *s += p;
        let live = if TRACK {
            // Safety: as above.
            LIVE ^ unsafe { *l.dead.get_unchecked(i) | *r.dead.get_unchecked(t) }
        } else {
            LIVE
        };
        *f |= if CHECK && !ok { live | INEXACT } else { live };
    }
}

/// Fold the cell at `pos`, the f pairs `f_pairs` against the column pairs
/// `g`, through `sum`'s cell sink, its pairs in the order the row loop
/// walks them: a cell whose float sum a checked sum cannot hold.
#[expect(clippy::too_many_arguments)]
#[inline(never)]
fn refold<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    eng: &Engine,
    node_idx: &mut [u32],
    pos: usize,
    f_pairs: &[ChildPair],
    g: &[ChildPair],
    left: &L,
    right: &R,
) -> Result<(), OperationError> {
    let cells = sum.cells();
    let start = cells.begin();
    for p1 in f_pairs {
        for q in g {
            let lc = left.get(node_idx, p1.left.0, q.left.0);
            if left.kills() && lc == NO_PRODUCT {
                continue;
            }
            let rc = right.get(node_idx, p1.right.0, q.right.0);
            if !right.kills() || rc != NO_PRODUCT {
                cells.pair(eng, lc, rc)?;
            }
        }
    }
    cells.end(eng, node_idx, pos, start)
}
