//! The streaming collapse route: a level whose per-cell result is a count or a
//! weight rather than a set of emitted pairs.

use super::rows::{Emit, run_level_rows};
use super::word_cells::WordScratch;
use super::super::marginal_plan::Carrier;
use super::*;
use crate::diagram::{EncodedChildRef, ScaledFloat, WeightStore};
use crate::value::{plain_floats, CheckedProducts, FloatCol, PlainProducts, StreamChild};

// A test sums its streamed float levels a cell at a time, as the oracle
// for the row sums, or reads and keeps child rows and sums blocks of f
// pairs and of rows wherever it can, and counts the blocks.
#[cfg(test)]
use crate::apply::conjoin::tests::{cells_only, eager, note_block, note_lanes};

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

#[cfg(not(test))]
#[inline(always)]
fn note_block() {}

#[cfg(not(test))]
#[inline(always)]
fn note_lanes() {}

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
        let mut next = pools.cell_pairs.checkout(eng);
        let bufs = SumBuffers { ones: &mut ones, floats: &mut floats, flags: &mut flags, slots: &mut slots, next: &mut next };
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
        let mut next = pools.cell_pairs.checkout(eng);
        let bufs = SumBuffers { ones: &mut ones, floats: &mut floats, flags: &mut flags, slots: &mut slots, next: &mut next };
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

/// Column `j`'s pairs: where `ONES`, its one pair off `ones`, the level's
/// columns as [`read_columns`] reads them, and none for a [`DEAD_COLUMN`];
/// else as [`column_pairs`] reads them.
#[inline(always)]
fn pairs_at<'a, const ONES: bool>(ones: &'a [ChildPair], ctx: &'a CellCtx<'_>, level: &'a TddLevel, j: usize, scratch: &'a mut Vec<ChildPair>) -> &'a [ChildPair] {
    if ONES {
        let q = &ones[j];
        return if q.left.0 == DEAD_COLUMN { &[] } else { std::slice::from_ref(q) };
    }
    column_pairs(ctx, level, j, scratch)
}

/// The level's columns as [`sum_row`] reads them: into `ones` each column's
/// one pair, [`DEAD_COLUMN`] on both sides for a column without pairs, where
/// no column has two pairs or more and the level has two columns or more
/// (empty otherwise); the pairs of all the columns; and one past the
/// greatest field of each side of a pair.
fn read_columns<'a>(
    eng: &Engine,
    rows: RowLoop<'_>,
    ones: &'a mut Vec<ChildPair>,
    scratch: &mut Vec<ChildPair>,
) -> Result<(&'a [ChildPair], usize, Sides<usize>), OperationError> {
    ones.clear();
    let width = rows.ctx.right_width;
    let mut one = width >= 2;
    if one {
        eng.limits().reserve(ones, width)?;
    }
    let dead = ChildPair { left: EncodedChildRef::from_raw(DEAD_COLUMN), right: EncodedChildRef::from_raw(DEAD_COLUMN) };
    let (mut pairs, mut bound) = (0, Sides { left: 0, right: 0 });
    for j in 0..width {
        let g = column_pairs(rows.ctx, rows.g_level, j, scratch);
        pairs += g.len();
        for q in g {
            bound.left = bound.left.max(q.left.0 as usize + 1);
            bound.right = bound.right.max(q.right.0 as usize + 1);
        }
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
    Ok((&ones[..], pairs, bound))
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
/// level, as many as it has room for; the last slots, one for each f pair
/// of a [`BLOCK`] or one, hold the other rows read last. A side that carries
/// the g pair's field through has no grid: one row, the value of each of
/// its fields, serves the whole level. A side whose rows are narrow against
/// the level's pairs also lays the rows of a block of [`LANES`] rows side by
/// side ([`lay`](Self::lay)).
struct ValueSide<'b> {
    /// The floats a row holds.
    n: usize,
    /// Whether every f pair reads the same row: the side carries the g
    /// pair's field.
    fixed: bool,
    /// The slots' floats and marks, `n + 1` a slot.
    vals: &'b mut [f64],
    dead: &'b mut [u8],
    /// Each slot's least nonzero float (infinite where none is or the
    /// read finds none, [`read`](Self::read)), its greatest, and one where
    /// a cell of it is dead.
    stats: &'b mut [f64],
    /// Each child row's kept slot, [`NO_SLOT`] where it has none, where
    /// the side keeps rows; empty where it keeps none.
    slot_of: &'b mut [u32],
    /// The slots rows may keep, and those handed out.
    cap: u32,
    used: u32,
    /// The last slots (from `cap`): the rows they hold, [`NO_SLOT`] before
    /// the first, how many there are, those a row being summed reads, and
    /// the next to take.
    at: [u32; BLOCK],
    spares: usize,
    pinned: u8,
    next: usize,
    /// The rows of a block laid side by side, float `q` of lane `k` at
    /// `q * LANES + k`, and bit `k` of mark `q` set where that cell is
    /// dead; empty where the side lays none.
    lanes: &'b mut [f64],
    marks: &'b mut [u8],
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
    #[expect(clippy::too_many_arguments)]
    fn new(
        n: usize,
        fixed: bool,
        spares: usize,
        vals: &'b mut [f64],
        dead: &'b mut [u8],
        stats: &'b mut [f64],
        slot_of: &'b mut [u32],
        (lanes, marks): (&'b mut [f64], &'b mut [u8]),
    ) -> Self {
        let cap = (stats.len() / 3 - spares) as u32;
        ValueSide { n, fixed, vals, dead, stats, slot_of, cap, used: 0, at: [NO_SLOT; BLOCK], spares, pinned: 0, next: 0, lanes, marks }
    }

    /// Whether the side lays rows side by side.
    #[inline(always)]
    fn lays(&self) -> bool {
        !self.marks.is_empty()
    }

    /// Lay the row slot `s` holds as lane `k`.
    #[inline(always)]
    fn lay(&mut self, s: usize, k: usize) {
        let w = self.n + 1;
        let (vals, dead) = (&self.vals[s * w..s * w + w], &self.dead[s * w..s * w + w]);
        let bit = 1u8 << k;
        for ((lane, mark), (&v, &d)) in self.lanes.chunks_exact_mut(LANES).zip(self.marks.iter_mut()).zip(vals.iter().zip(dead)) {
            lane[k] = v;
            *mark = (*mark & !bit) | (d << k);
        }
    }

    /// Read child rows `at`, the rows of a block's lanes, for
    /// [`lane_values`]: into one slot, returned, where the side keeps the
    /// same row for every lane, else each read and laid as its lane. Also
    /// each lane's least and greatest nonzero floats, read where `stats`,
    /// and whether a cell of its row is dead.
    #[inline(always)]
    fn read_lanes<C: ChildLookup>(&mut self, side: &C, node_idx: &[u32], at: &[u32], lay: bool, stats: bool, factor: impl Fn(u32) -> Option<f64>) -> (usize, [(f64, f64, bool); LANES]) {
        let mut found = [(f64::INFINITY, 0.0, false); LANES];
        let mut s = 0;
        for (k, &a) in at.iter().enumerate() {
            if k == 0 || lay {
                self.unpin();
                s = self.read(side, node_idx, a, stats, &factor);
                let r = self.row(s);
                found[k] = (r.lo, r.hi, r.any_dead);
                if lay {
                    self.lay(s, k);
                }
            } else {
                found[k] = found[0];
            }
        }
        (s, found)
    }

    /// Whether row `at` has a kept slot or would get one.
    #[inline(always)]
    fn kept(&self, at: u32) -> bool {
        !self.slot_of.is_empty() && (self.used < self.cap || self.slot_of[at as usize] != NO_SLOT)
    }

    /// The last slot holding row `at`, if one does.
    #[inline(always)]
    fn spare(&self, at: u32) -> Option<usize> {
        self.at[..self.spares].iter().position(|&x| x == at)
    }

    /// The cells a read of child row `at` reads, counted where it reads a
    /// row it may have to read again: none where the side holds the row or
    /// keeps it.
    #[inline(always)]
    fn cost(&self, at: u32) -> usize {
        let at = if self.fixed { 0 } else { at };
        if self.kept(at) || self.spare(at).is_some() { 0 } else { self.n }
    }

    /// Free the last slots for the next row's reads.
    #[inline(always)]
    fn unpin(&mut self) {
        self.pinned = 0;
    }

    /// The slot of child row `at` of `side`, read first where the side
    /// does not hold it, each cell as `factor` reads its node. A cell
    /// `factor` names no float for is marked dead: no pair reads it. Where
    /// `stats`, also the row's least and greatest nonzero floats, which
    /// only a checked sum reads ([`rows_exact`]). A row read into a last
    /// slot holds it until [`unpin`](Self::unpin); at most `spares` rows
    /// are read between.
    #[inline(always)]
    fn read<C: ChildLookup>(&mut self, side: &C, node_idx: &[u32], at: u32, stats: bool, factor: impl Fn(u32) -> Option<f64>) -> usize {
        let at = if self.fixed { 0 } else { at };
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
            if let Some(k) = self.spare(at) {
                self.pinned |= 1 << k;
                return self.cap as usize + k;
            }
            let mut k = self.next;
            while self.pinned & (1 << k) != 0 {
                k = (k + 1) % self.spares;
            }
            self.next = (k + 1) % self.spares;
            self.pinned |= 1 << k;
            self.at[k] = at;
            self.cap as usize + k
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
            if stats && f != 0.0 {
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
    spans_exact::<S>((l.lo, l.hi), (r.lo, r.hi))
}

/// [`rows_exact`] for rows whose least and greatest nonzero floats are
/// `l` and `r`.
#[inline(always)]
fn spans_exact<S: RowSum>(l: (f64, f64), r: (f64, f64)) -> bool {
    !S::CHECKED || l.0 == f64::INFINITY || r.0 == f64::INFINITY || (S::times(l.0, r.0).1 && S::times(l.1, r.1).1)
}

/// A row sum's flag: a pair of the cell lives.
const LIVE: u8 = 1;
/// A row sum's flag: a product of the cell is not its float.
const INEXACT: u8 = 2;

/// [`sum_row`]'s child rows read as floats, a row's sums against one-pair
/// columns with their flags ([`LIVE`], [`INEXACT`]), and the sums of rows
/// a block of rows summed ahead of their turn.
struct ValueRows<'b> {
    left: ValueSide<'b>,
    right: ValueSide<'b>,
    sums: &'b mut [f64],
    flags: &'b mut [u8],
    ahead: Ahead<'b>,
    gather: Gather<'b>,
}

/// The f pairs of a row against one-pair columns [`ones_row_values`] adds
/// at a time, loading and storing each column's sum once for them.
const BLOCK: usize = 4;

/// The column pairs a level has where [`sum_row`] sums blocks of f pairs
/// or of rows.
const BLOCK_MIN_PAIRS: usize = 256;

/// The rows of one f pair [`sum_row`] sums at a time ([`block_rows`]),
/// walking each column's pairs once for all of them, and the fewest it
/// sums so.
const LANES: usize = 8;
const LANES_MIN: usize = 4;

/// The pairs a level's columns hold on average where [`sum_row`] sums
/// blocks of rows: fewer, and a cell's own work outweighs its pairs'.
const LANE_COLUMN_PAIRS: usize = 8;

/// The share of the level's column pairs up to which a side's rows are
/// narrow enough to lay side by side ([`ValueSide::lay`]): its floats at
/// most the pairs over this.
const LANE_ROW_SHARE: usize = 6;

/// The bytes a level's blocks of rows may take: the sums held ahead,
/// each side's rows laid side by side, and a [`Gather`].
const LANE_MAX_BYTES: usize = 64 << 20;

/// The floats of a side's rows from which a level's blocks of rows read
/// that side's row gathered ([`Gather`]): about what a core's second-level
/// cache holds, past which reading a float of the row a pair is a miss.
const GATHER_MIN_FLOATS: usize = 1 << 15;

/// The row every row of a block reads on a side that does not lay its
/// rows, gathered in the order a block's walk of the level's column pairs
/// reads it: pair `w` of the walk reads float `w`, so a wide row is read
/// in order rather than at each pair's field. Gathered for the second
/// block in a row that reads the same row, and kept while blocks read it.
struct Gather<'b> {
    /// Each column pair's float and dead mark, empty on a level that
    /// gathers no row.
    vals: &'b mut [f64],
    dead: &'b mut [u8],
    /// The side gathered: the right one where true.
    right: bool,
    /// The row gathered, and the row the last block read on that side.
    held: Option<u32>,
    last: Option<u32>,
}

impl Gather<'_> {
    /// Row `at` of `row`'s floats on side `right` gathered, where the
    /// level gathers that side and the last block read the same row;
    /// `None` otherwise. The level's columns are walked as
    /// [`lane_values`] walks them.
    #[inline(always)]
    fn rows(&mut self, right: bool, at: u32, row: RowFloats<'_>, ctx: &CellCtx<'_>, level: &TddLevel, scratch: &mut Vec<ChildPair>) -> Option<Gathered<'_>> {
        if self.vals.is_empty() || self.right != right {
            return None;
        }
        if self.held != Some(at) {
            if self.last != Some(at) {
                self.last = Some(at);
                return None;
            }
            self.gather(right, row, ctx, level, scratch);
            self.held = Some(at);
        }
        Some(Gathered { vals: self.vals, dead: self.dead })
    }

    /// Gather `row` on side `right` along the level's column pairs.
    #[inline(never)]
    fn gather(&mut self, right: bool, row: RowFloats<'_>, ctx: &CellCtx<'_>, level: &TddLevel, scratch: &mut Vec<ChildPair>) {
        let mut w = 0;
        for j in 0..ctx.right_width {
            for q in column_pairs(ctx, level, j, scratch) {
                let i = if right { q.right.0 } else { q.left.0 } as usize;
                (self.vals[w], self.dead[w]) = (row.vals[i], row.dead[i]);
                w += 1;
            }
        }
        debug_assert_eq!(w, self.vals.len(), "a gather holds a float for each column pair");
    }
}

/// The cell sums and flags of the rows after the first of a block of
/// rows, a row's width each, empty on a level that sums no such blocks:
/// rows `rows`, of which the first `got` have been appended, `held` in all.
struct Ahead<'b> {
    sums: &'b mut [f64],
    flags: &'b mut [u8],
    rows: [(usize, ChildPair); LANES - 1],
    got: usize,
    held: usize,
}

impl Ahead<'_> {
    /// Where the sums of row `i` of f pair `p` are held, as the next held
    /// row; none held after it otherwise.
    #[inline(always)]
    fn take(&mut self, i: usize, p: ChildPair) -> Option<usize> {
        if self.got < self.held && self.rows[self.got] == (i, p) {
            self.got += 1;
            return Some(self.got - 1);
        }
        (self.got, self.held) = (0, 0);
        None
    }
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

/// The floats a row of a child side holds, where the side's pair fields
/// `bound` allows value rows: a grid side's columns, where every field
/// names one of them, and the fields of a side carrying the g pair's field,
/// where reading them reads at most [`VALUE_ROW_READS`] times the level's
/// `pairs`. Also whether every f pair reads the same row.
fn row_width(side: &ChildPlan<'_>, passthrough: bool, bound: usize, pairs: usize) -> Option<(usize, bool)> {
    let n = side.stride as usize;
    match side.plan.carrier {
        None if !passthrough && n > 0 && bound <= n => Some((n, false)),
        Some(Carrier::G) if passthrough && bound <= pairs.saturating_mul(VALUE_ROW_READS) => Some((bound.max(1), true)),
        _ => None,
    }
}

/// The level's [`ValueRows`], where each child side's pair fields allow
/// them ([`row_width`]), with sums for `ones` columns and `pairs` column
/// pairs in all. A side read from its grid keeps the rows it reads where
/// its f pairs read each of its rows twice or more on average, coming back
/// to it after leaving it: as many as fit [`KEEP_MIN_BYTES`] or the output
/// grid's bytes, the left side first.
#[expect(clippy::too_many_arguments)]
fn value_rows<'b, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    left: &L,
    right: &R,
    (bound, ones, pairs): (Sides<usize>, usize, usize),
    floats: &'b mut Vec<f64>,
    flags: &'b mut Vec<u8>,
    slots: &'b mut Vec<u32>,
    scratch: &mut Vec<ChildPair>,
) -> Result<Option<ValueRows<'b>>, OperationError> {
    let sides = &rows.ctx.sides;
    let (Some((sl, lfixed)), Some((sr, rfixed))) = (
        row_width(&sides.left, left.passthrough(), bound.left, pairs),
        row_width(&sides.right, right.passthrough(), bound.right, pairs),
    ) else {
        return Ok(None);
    };
    let lim = eng.limits();
    let (reads, f_pairs) = side_reads(lim, rows, Sides { left: !lfixed, right: !rfixed }, slots, scratch)?;
    let grid = rows.f_width.saturating_mul(rows.ctx.right_width).saturating_mul(4);
    let mut room = KEEP_MIN_BYTES.max(grid.saturating_mul(KEEP_GRID_SHARE)).min((lim.headroom() / 4) as usize);
    // The rows a side keeps, out of the room left: a kept row costs its
    // floats, marks and statistics, and the side a slot index a row.
    let mut keep = |r: SideReads, n: usize, fixed: bool| {
        let pays = !fixed
            && ((r.changes >= 2 * r.distinct && r.distinct * n <= f_pairs.saturating_mul(pairs).saturating_mul(VALUE_ROW_READS))
                || (eager() && r.changes > r.distinct));
        let kept = if pays && 4 * r.rows < room { r.distinct.min((room - 4 * r.rows) / (9 * n + 33)) } else { 0 };
        if kept > 0 {
            room -= 4 * r.rows + kept * (9 * n + 33);
        }
        kept
    };
    let (kl, kr) = (keep(reads.left, sl, lfixed), keep(reads.right, sr, rfixed));
    // A level of wide enough rows sums a block of f pairs at a time, a
    // side holding a row for each. A level of rows of one f pair whose
    // columns hold enough pairs sums a block of rows at a time: a side
    // whose rows are narrow against the level's pairs lays the block's
    // rows side by side, the narrower side first, and the level holds the
    // sums of the block's rows after the first, all within
    // [`LANE_MAX_BYTES`].
    let width = rows.ctx.right_width;
    let spares = if pairs >= BLOCK_MIN_PAIRS || eager() { BLOCK } else { 1 };
    let ahead_bytes = 9 * (LANES - 1) * width;
    let blocks = spares == BLOCK
        && rows.f_width >= 2
        && width > 0
        && (pairs >= LANE_COLUMN_PAIRS * width || eager())
        && ahead_bytes <= LANE_MAX_BYTES;
    let mut budget = if blocks { LANE_MAX_BYTES - ahead_bytes } else { 0 };
    let mut lays = |n: usize, fixed: bool| {
        let bytes = (n + 1) * (8 * LANES + 1);
        let on = !fixed && (n.saturating_mul(LANE_ROW_SHARE) <= pairs || eager()) && bytes <= budget;
        if on {
            budget -= bytes;
        }
        on
    };
    let (ll, rl) = if sl <= sr {
        let l = lays(sl, lfixed);
        (l, lays(sr, rfixed))
    } else {
        let r = lays(sr, rfixed);
        (lays(sl, lfixed), r)
    };
    // A side that does not lay its rows, where the other does, reads the
    // same row for every row of a block: a wide one is gathered.
    let shared = match (ll, rl) {
        (true, false) => Some((true, sr)),
        (false, true) => Some((false, sl)),
        _ => None,
    };
    let (gright, gw) = match shared {
        Some((right, n)) if (n >= GATHER_MIN_FLOATS || eager()) && 9 * pairs <= budget => (right, pairs),
        _ => (false, 0),
    };
    let aw = if ll || rl { (LANES - 1) * width } else { 0 };
    let (ml, mr) = (if ll { sl + 1 } else { 0 }, if rl { sr + 1 } else { 0 });
    let (ls, rs) = (kl + spares, kr + spares);
    let (wl, wr) = ((sl + 1) * ls, (sr + 1) * rs);
    let nf = wl + 3 * ls + wr + 3 * rs + ones + aw + LANES * (ml + mr) + gw;
    lim.reserve(floats, nf)?;
    floats.resize(nf, 0.0);
    let nd = wl + wr + ones + aw + ml + mr + gw;
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
    let (rst, rest) = rest.split_at_mut(3 * rs);
    let (sums, rest) = rest.split_at_mut(ones);
    let (ahead_sums, rest) = rest.split_at_mut(aw);
    let (llanes, rest) = rest.split_at_mut(LANES * ml);
    let (rlanes, gvals) = rest.split_at_mut(LANES * mr);
    let (ld, rest) = flags.split_at_mut(wl);
    let (rd, rest) = rest.split_at_mut(wr);
    let (flags, rest) = rest.split_at_mut(ones);
    let (ahead_flags, rest) = rest.split_at_mut(aw);
    let (lmarks, rest) = rest.split_at_mut(ml);
    let (rmarks, gdead) = rest.split_at_mut(mr);
    let (lmap, rmap) = slots.split_at_mut(nl);
    let left = ValueSide::new(sl, lfixed, spares, lv, ld, lst, lmap, (llanes, lmarks));
    let right = ValueSide::new(sr, rfixed, spares, rv, rd, rst, rmap, (rlanes, rmarks));
    let none = ChildPair { left: EncodedChildRef::from_raw(DEAD_COLUMN), right: EncodedChildRef::from_raw(DEAD_COLUMN) };
    let ahead = Ahead { sums: ahead_sums, flags: ahead_flags, rows: [(0, none); LANES - 1], got: 0, held: 0 };
    let gather = Gather { vals: gvals, dead: gdead, right: gright, held: None, last: None };
    Ok(Some(ValueRows { left, right, sums, flags, ahead, gather }))
}

/// Each child side's [`SideReads`] over the level's f pairs, its rows
/// counted where it `may` keep them, and the f pairs. `slots` is scratch.
fn side_reads(
    lim: &crate::limits::Limits,
    rows: RowLoop<'_>,
    may: Sides<bool>,
    slots: &mut Vec<u32>,
    scratch: &mut Vec<ChildPair>,
) -> Result<(Sides<SideReads>, usize), OperationError> {
    let mut reads = Sides { left: SideReads::default(), right: SideReads::default() };
    let (mut last, mut f_pairs) = ((NO_SLOT, NO_SLOT), 0);
    each_f_pair(rows, scratch, |p| {
        let (a, b) = (p.left.0, p.right.0);
        reads.left.changes += usize::from(a != last.0);
        reads.right.changes += usize::from(b != last.1);
        reads.left.rows = reads.left.rows.max(a as usize + 1);
        reads.right.rows = reads.right.rows.max(b as usize + 1);
        last = (a, b);
        f_pairs += 1;
    });
    // The rows read, counted where a side changes rows often enough to
    // keep them.
    let count = |r: &SideReads| r.changes >= 4 || (eager() && r.changes >= 2);
    let (cl, cr) = (may.left && count(&reads.left), may.right && count(&reads.right));
    if cl || cr {
        let (nl, nr) = (if cl { reads.left.rows } else { 0 }, if cr { reads.right.rows } else { 0 });
        slots.clear();
        lim.reserve(slots, nl + nr)?;
        slots.resize(nl + nr, 0);
        let (ml, mr) = slots.split_at_mut(nl);
        each_f_pair(rows, scratch, |p| {
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
        });
    }
    for (c, r) in [(cl, &mut reads.left), (cr, &mut reads.right)] {
        if !c {
            r.distinct = r.changes;
        }
    }
    Ok((reads, f_pairs))
}

/// `each` on the pairs of the level's f rows in order, as the rows'
/// [`pairs_view_decoded`](TddLevel::pairs_view_decoded) gives them, read
/// with one cursor along an implicit level, which a read by index finds
/// each node's first pair for afresh. Pairs are decoded only where a child
/// is marginal.
fn each_f_pair(rows: RowLoop<'_>, scratch: &mut Vec<ChildPair>, mut each: impl FnMut(ChildPair)) {
    let (lview, rview) = (rows.ctx.sides.left.plan.view, rows.ctx.sides.right.plan.view);
    let mut cursor = None;
    if !lview.is_marginal() && !rview.is_marginal() {
        for i in 0..rows.f_width {
            rows.f_level.pairs_read_next(&mut cursor, i, scratch).iter().for_each(|&p| each(p));
        }
    } else {
        for i in 0..rows.f_width {
            for &p in rows.f_level.pairs_read_next(&mut cursor, i, scratch) {
                each(crate::diagram::decoded(p, lview, rview));
            }
        }
    }
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
        node_idx[grid_pos] = self.counts.push_offset(eng, v, self.products.scale())?;
        Ok(())
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
    peek: Peek<'o>,
}

/// The pooled buffers a [`SumRows`] holds: its one-pair columns, its
/// [`ValueRows`]' floats, marks and slots, and the f pairs of the rows
/// after a row.
struct SumBuffers<'o> {
    ones: &'o mut Vec<ChildPair>,
    floats: &'o mut Vec<f64>,
    flags: &'o mut Vec<u8>,
    slots: &'o mut Vec<u32>,
    next: &'o mut Vec<ChildPair>,
}

/// The level's rows as [`sum_row`] reads those after a row: the f pairs
/// of a block's rows after its first, and the first row a block may
/// start at.
struct Peek<'o> {
    f_level: &'o TddLevel,
    f_width: usize,
    scratch: &'o mut Vec<ChildPair>,
    next: [ChildPair; LANES - 1],
    from: usize,
}

/// The sides of a block of rows that lay its rows side by side; a side
/// that does not reads the same row for all of them.
#[derive(Clone, Copy)]
enum Lay {
    Left,
    Right,
    Both,
}

impl Peek<'_> {
    /// The rows of a block starting at row `i` of f pair `p1`, at most
    /// [`LANES`] and at least [`LANES_MIN`], their f pairs after `p1` in
    /// `next`, and the sides that lay them: each row is one of one f pair
    /// the row loop hands to [`sum_row`], reading its child rows reads at
    /// most [`VALUE_ROW_READS`] cells a pair, and a side whose rows differ
    /// lays them.
    #[inline(never)]
    fn block(&mut self, ctx: &CellCtx<'_>, i: usize, p1: ChildPair, vr: &ValueRows<'_>, pairs: usize) -> Option<(usize, Lay)> {
        if i < self.from {
            return None;
        }
        let (lview, rview) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
        let (lside, rside) = (&vr.left, &vr.right);
        let (mut lsame, mut rsame) = (true, true);
        let mut m = 1;
        while m < LANES && i + m < self.f_width {
            let f = self.f_level.pairs_view_decoded(i + m, self.scratch, lview, rview);
            let &[p] = f else {
                break;
            };
            if row_alive_masks(ctx, f).is_none() || (lside.cost(p.left.0) + rside.cost(p.right.0) > VALUE_ROW_READS * pairs && !eager()) {
                break;
            }
            let (l, r) = (lsame && (lside.fixed || p.left == p1.left), rsame && (rside.fixed || p.right == p1.right));
            if (!l && !lside.lays()) || (!r && !rside.lays()) {
                break;
            }
            (lsame, rsame) = (l, r);
            self.next[m - 1] = p;
            m += 1;
        }
        let lay = match (lsame, rsame) {
            (true, false) => Some(Lay::Right),
            (false, true) => Some(Lay::Left),
            (false, false) => Some(Lay::Both),
            (true, true) => None,
        };
        match lay {
            Some(lay) if m >= LANES_MIN || (eager() && m >= 2) => Some((m, lay)),
            _ => {
                self.from = i + m;
                None
            }
        }
    }
}

/// The [`SumRows`] of a level summed by `sum`.
fn sum_rows<'o, 'r: 'o, S: RowSum, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'r>,
    left: &L,
    right: &R,
    sum: S,
    bufs: SumBuffers<'o>,
    scratch: &mut Vec<ChildPair>,
) -> Result<SumRows<'o, S>, OperationError> {
    let SumBuffers { ones, floats, flags, slots, next } = bufs;
    let (ones, pairs, bound) = read_columns(eng, rows, ones, scratch)?;
    let values = value_rows(eng, rows, left, right, (bound, ones.len(), pairs), floats, flags, slots, scratch)?;
    let none = ChildPair { left: EncodedChildRef::from_raw(DEAD_COLUMN), right: EncodedChildRef::from_raw(DEAD_COLUMN) };
    let peek = Peek { f_level: rows.f_level, f_width: rows.f_width, scratch: next, next: [none; LANES - 1], from: 0 };
    Ok(SumRows { sum, ones, pairs, values, peek })
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
        sum_row(&mut self.sum, self.ones, self.pairs, self.values.as_mut(), &mut self.peek, eng, row)
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
/// floats off those rows, and a row of one f pair followed by such rows
/// sums with them up to [`LANES`] at a time ([`block_rows`]), on a level
/// with room for their sums. A cell a checked sum cannot hold goes through its
/// cell sink, its pairs in the same order. Any other row is left to the
/// cells (0 columns written).
#[inline(always)]
fn sum_row<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    ones: &[ChildPair],
    pairs: usize,
    values: Option<&mut ValueRows<'_>>,
    peek: &mut Peek<'_>,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    if row.f_pairs.len() == 1 {
        let p1 = row.f_pairs[0];
        if let Some(vr) = values {
            if !vr.ahead.sums.is_empty() {
                let width = row.ctx.right_width;
                let i = (row.row_base - row.ctx.output_grid_base) / width;
                if let Some(k) = vr.ahead.take(i, p1) {
                    let at = k * width..(k + 1) * width;
                    return ahead_row(sum, &vr.ahead.sums[at.clone()], &vr.ahead.flags[at], eng, row);
                }
                if (vr.left.cost(p1.left.0) + vr.right.cost(p1.right.0) <= VALUE_ROW_READS * pairs || eager())
                    && let Some((m, lay)) = peek.block(row.ctx, i, p1, vr, pairs)
                {
                    return block_rows(sum, i, p1, &peek.next[..m - 1], lay, vr, eng, row);
                }
            }
            if vr.left.cost(p1.left.0) + vr.right.cost(p1.right.0) > VALUE_ROW_READS * pairs && !eager() {
                return one_pair_row(sum, eng, row);
            }
            vr.left.unpin();
            vr.right.unpin();
            let ls = vr.left.read(row.left, row.node_idx, p1.left.0, S::CHECKED, |c| sum.left_factor(c));
            let rs = vr.right.read(row.right, row.node_idx, p1.right.0, S::CHECKED, |c| sum.right_factor(c));
            let (l, r) = (vr.left.row(ls), vr.right.row(rs));
            return match (l.any_dead || r.any_dead, !rows_exact::<S>(l, r), ones.is_empty()) {
                (true, true, true) => one_pair_row_values::<true, true, false, _, _, _>(sum, ones, l, r, eng, row),
                (true, false, true) => one_pair_row_values::<true, false, false, _, _, _>(sum, ones, l, r, eng, row),
                (false, true, true) => one_pair_row_values::<false, true, false, _, _, _>(sum, ones, l, r, eng, row),
                (false, false, true) => one_pair_row_values::<false, false, false, _, _, _>(sum, ones, l, r, eng, row),
                (true, true, false) => one_pair_row_values::<true, true, true, _, _, _>(sum, ones, l, r, eng, row),
                (true, false, false) => one_pair_row_values::<true, false, true, _, _, _>(sum, ones, l, r, eng, row),
                (false, true, false) => one_pair_row_values::<false, true, true, _, _, _>(sum, ones, l, r, eng, row),
                (false, false, false) => one_pair_row_values::<false, false, true, _, _, _>(sum, ones, l, r, eng, row),
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
/// `TRACK` where a cell of them is dead, `CHECK` where a product of them
/// may not be its float ([`rows_exact`]), and `ONES` where each column's
/// pairs are read off `ones` ([`pairs_at`]).
#[inline(always)]
fn one_pair_row_values<const TRACK: bool, const CHECK: bool, const ONES: bool, L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    ones: &[ChildPair],
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
        let g = pairs_at::<ONES>(ones, ctx, right_level_t, j, g_pairs_scratch);
        if g.is_empty() {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, g.len() as u64)?;
        let (mut acc, mut flags) = (0.0f64, 0u8);
        for q in g {
            let (i, t) = (q.left.0 as usize, q.right.0 as usize);
            // Safety: every pair field is below its side's row width
            // (`row_width`), and a row holds a float and a mark for each.
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

/// [`sum_row`] on row `i` of f pair `p1` and the rows of f pairs `next`
/// after it, a block of at most [`LANES`]: their child rows read as
/// floats, those of a side `lay` names laid side by side, each column's
/// pairs walked once for the block, row `i`'s cells appended and the
/// others' sums held until the row loop reaches their rows.
#[expect(clippy::too_many_arguments)]
#[inline(never)]
fn block_rows<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    i: usize,
    p1: ChildPair,
    next: &[ChildPair],
    lay: Lay,
    vr: &mut ValueRows<'_>,
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    note_lanes();
    let ValueRows { left: lside, right: rside, ahead, gather, .. } = vr;
    let m = 1 + next.len();
    let (mut la, mut ra) = ([p1.left.0; LANES], [p1.right.0; LANES]);
    for (k, p) in next.iter().enumerate() {
        (la[k + 1], ra[k + 1]) = (p.left.0, p.right.0);
        ahead.rows[k] = (i + 1 + k, *p);
    }
    (ahead.got, ahead.held) = (0, m - 1);
    let (lays, rays) = match lay {
        Lay::Left => (true, false),
        Lay::Right => (false, true),
        Lay::Both => (true, true),
    };
    let (ls, lst) = lside.read_lanes(row.left, row.node_idx, &la[..m], lays, S::CHECKED, |c| sum.left_factor(c));
    let (rs, rst) = rside.read_lanes(row.right, row.node_idx, &ra[..m], rays, S::CHECKED, |c| sum.right_factor(c));
    let track = (0..m).any(|k| lst[k].2 || rst[k].2);
    let check = (0..m).any(|k| !spans_exact::<S>((lst[k].0, lst[k].1), (rst[k].0, rst[k].1)));
    let (sums, flags) = (&mut *ahead.sums, &mut *ahead.flags);
    let (lrow, rrow) = (lside.row(ls), rside.row(rs));
    let (lone, rone) = (Shared { vals: lrow.vals, dead: lrow.dead }, Shared { vals: rrow.vals, dead: rrow.dead });
    let (llaid, rlaid) = (Laid { vals: &*lside.lanes, marks: &*lside.marks }, Laid { vals: &*rside.lanes, marks: &*rside.marks });
    let out = (m, sums, flags);
    let (ctx, level) = (row.ctx, row.right_level_t);
    match lay {
        Lay::Left => match gather.rows(true, if rside.fixed { 0 } else { ra[0] }, rrow, ctx, level, &mut *row.g_pairs_scratch) {
            Some(rg) => lanes_by(track, check, sum, llaid, rg, out, eng, row),
            None => lanes_by(track, check, sum, llaid, rone, out, eng, row),
        },
        Lay::Right => match gather.rows(false, if lside.fixed { 0 } else { la[0] }, lrow, ctx, level, &mut *row.g_pairs_scratch) {
            Some(lg) => lanes_by(track, check, sum, lg, rlaid, out, eng, row),
            None => lanes_by(track, check, sum, lone, rlaid, out, eng, row),
        },
        Lay::Both => lanes_by(track, check, sum, llaid, rlaid, out, eng, row),
    }
}

/// A side's floats for the [`LANES`] rows of a block, as [`lane_values`]
/// reads them.
trait Lanes: Copy {
    /// Cell `i`'s floats, one a lane, and its dead marks, bit `k` for
    /// lane `k`, read by pair `w` of the walk of the level's column pairs.
    /// Every pair field of the level is a cell.
    fn at(self, i: usize, w: usize) -> ([f64; LANES], u8);
}

/// One row every lane reads.
#[derive(Clone, Copy)]
struct Shared<'r> {
    vals: &'r [f64],
    dead: &'r [u8],
}

/// Rows laid side by side ([`ValueSide::lay`]).
#[derive(Clone, Copy)]
struct Laid<'r> {
    vals: &'r [f64],
    marks: &'r [u8],
}

impl Lanes for Shared<'_> {
    #[inline(always)]
    fn at(self, i: usize, _w: usize) -> ([f64; LANES], u8) {
        // Safety: every pair field is below its side's row width
        // (`row_width`), and a row holds a float and a mark for each.
        let (v, d) = unsafe { (*self.vals.get_unchecked(i), *self.dead.get_unchecked(i)) };
        ([v; LANES], 0u8.wrapping_sub(d))
    }
}

impl Lanes for Laid<'_> {
    #[inline(always)]
    fn at(self, i: usize, _w: usize) -> ([f64; LANES], u8) {
        // Safety: as for `Shared`; laid rows hold `LANES` floats and one
        // mark for each cell of a row.
        unsafe { (self.vals.as_ptr().add(i * LANES).cast::<[f64; LANES]>().read(), *self.marks.get_unchecked(i)) }
    }
}

/// One row every lane reads, gathered ([`Gather`]).
#[derive(Clone, Copy)]
struct Gathered<'r> {
    vals: &'r [f64],
    dead: &'r [u8],
}

impl Lanes for Gathered<'_> {
    #[inline(always)]
    fn at(self, _i: usize, w: usize) -> ([f64; LANES], u8) {
        // Safety: the walk reads as many pairs as the gather holds floats
        // and marks, the level's column pairs.
        let (v, d) = unsafe { (*self.vals.get_unchecked(w), *self.dead.get_unchecked(w)) };
        ([v; LANES], 0u8.wrapping_sub(d))
    }
}

/// [`lane_values`] where a cell of the rows is dead (`track`) and where a
/// product of them may not be its float (`check`).
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn lanes_by<A: Lanes, B: Lanes, L: ChildLookup, R: ChildLookup, S: RowSum>(
    track: bool,
    check: bool,
    sum: &mut S,
    l: A,
    r: B,
    out: (usize, &mut [f64], &mut [u8]),
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    match (track, check) {
        (true, true) => lane_values::<true, true, _, _, _, _, _>(sum, l, r, out, eng, row),
        (true, false) => lane_values::<true, false, _, _, _, _, _>(sum, l, r, out, eng, row),
        (false, true) => lane_values::<false, true, _, _, _, _, _>(sum, l, r, out, eng, row),
        (false, false) => lane_values::<false, false, _, _, _, _, _>(sum, l, r, out, eng, row),
    }
}

/// [`block_rows`]' sums: the block's `m` rows, the lanes of `l` and `r`,
/// summed column by column, each cell's products in the order
/// [`one_pair_row_values`] adds them; the first row's cells appended, the
/// others' sums and flags held in `sums` and `flags`, a row's width each.
/// `TRACK` where a cell of the rows is dead, and `CHECK` where a product
/// of them may not be its float.
#[inline(always)]
fn lane_values<const TRACK: bool, const CHECK: bool, A: Lanes, B: Lanes, L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    l: A,
    r: B,
    (m, sums, flags): (usize, &mut [f64], &mut [u8]),
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, right_level_t, g_pairs_scratch, node_idx, left, right, gate } = row;
    let width = ctx.right_width;
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    let mut w = 0;
    for j in 0..width {
        let g = column_pairs(ctx, right_level_t, j, g_pairs_scratch);
        if g.is_empty() {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, g.len() as u64)?;
        let (mut acc, mut live, mut bad) = ([0.0f64; LANES], if TRACK { 0u8 } else { u8::MAX }, 0u8);
        for q in g {
            let (a, da) = l.at(q.left.0 as usize, w);
            let (b, db) = r.at(q.right.0 as usize, w);
            w += 1;
            for k in 0..LANES {
                let (p, ok) = S::times(a[k], b[k]);
                acc[k] += p;
                if CHECK {
                    bad |= u8::from(!ok) << k;
                }
            }
            if TRACK {
                live |= !(da | db);
            }
        }
        for k in 1..m {
            sums[(k - 1) * width + j] = acc[k];
            flags[(k - 1) * width + j] = (((live >> k) & 1) * LIVE) | (((bad >> k) & 1) * INEXACT);
        }
        if live & 1 == 0 {
            continue;
        }
        if !S::CHECKED || (bad & 1 == 0 && acc[0] < f64::INFINITY) {
            sum.push(eng, node_idx, row_base + j, acc[0])?;
            continue;
        }
        refold(sum, eng, node_idx, row_base + j, f_pairs, g, left, right)?;
    }
    gate.add_short(due_at - due);
    Ok(width)
}

/// [`sum_row`] on a row a block summed ahead ([`block_rows`]): its cells'
/// sums and flags `sums` and `flags` appended in column order, charged as
/// [`one_pair_row_values`] charges them.
#[inline(never)]
fn ahead_row<L: ChildLookup, R: ChildLookup, S: RowSum>(
    sum: &mut S,
    sums: &[f64],
    flags: &[u8],
    eng: &Engine,
    row: WholeRow<'_, '_, L, R>,
) -> Result<usize, OperationError> {
    let WholeRow { row_base, f_pairs, ctx, right_level_t, g_pairs_scratch, node_idx, left, right, gate } = row;
    let width = ctx.right_width;
    let (mut due_at, mut due) = (gate.due_in(), gate.due_in());
    for j in 0..width {
        let g = column_pairs(ctx, right_level_t, j, g_pairs_scratch);
        if g.is_empty() {
            node_idx[row_base + j] = NO_PRODUCT;
            continue;
        }
        charge(gate, &mut due_at, &mut due, g.len() as u64)?;
        let (acc, fl) = (sums[j], flags[j]);
        if fl & LIVE == 0 {
            continue;
        }
        if !S::CHECKED || (fl & INEXACT == 0 && acc < f64::INFINITY) {
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
/// reading them reads at most [`VALUE_ROW_READS`] cells a column, a
/// [`BLOCK`] of such f pairs at a time on a level with a side's rows for
/// them.
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
    let ValueRows { left: lside, right: rside, sums, flags, .. } = vr;
    let (sums, flags) = (&mut sums[..width], &mut flags[..width]);
    sums.fill(0.0);
    flags.fill(0);
    let reads = |lside: &ValueSide<'_>, rside: &ValueSide<'_>, p: &ChildPair| lside.cost(p.left.0) + rside.cost(p.right.0) <= VALUE_ROW_READS * width || eager();
    let mut at = 0;
    while at < k {
        let block = f_pairs.get(at..at + BLOCK).filter(|b| lside.spares == BLOCK && b.iter().all(|p| reads(lside, rside, p)));
        if let Some(block) = block {
            note_block();
            lside.unpin();
            rside.unpin();
            let (mut ls, mut rs) = ([0usize; BLOCK], [0usize; BLOCK]);
            for (x, p) in block.iter().enumerate() {
                ls[x] = lside.read(left, node_idx, p.left.0, S::CHECKED, |c| sum.left_factor(c));
                rs[x] = rside.read(right, node_idx, p.right.0, S::CHECKED, |c| sum.right_factor(c));
            }
            let (l, r) = (ls.map(|s| lside.row(s)), rs.map(|s| rside.row(s)));
            let track = l.iter().chain(&r).any(|x| x.any_dead);
            let check = (0..BLOCK).any(|x| !rows_exact::<S>(l[x], r[x]));
            match (track, check) {
                (true, true) => add_block::<true, true, S>(sums, flags, ones, l, r),
                (true, false) => add_block::<true, false, S>(sums, flags, ones, l, r),
                (false, true) => add_block::<false, true, S>(sums, flags, ones, l, r),
                (false, false) => add_block::<false, false, S>(sums, flags, ones, l, r),
            }
            at += BLOCK;
            continue;
        }
        let p1 = &f_pairs[at];
        at += 1;
        let (a, b) = (p1.left.0, p1.right.0);
        if !reads(lside, rside, p1) {
            add_grid(sum, sums, flags, ones, node_idx, left, right, a, b);
            continue;
        }
        lside.unpin();
        rside.unpin();
        let ls = lside.read(left, node_idx, a, S::CHECKED, |c| sum.left_factor(c));
        let rs = rside.read(right, node_idx, b, S::CHECKED, |c| sum.right_factor(c));
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

/// [`add_values`] for a block of f pairs, their child rows `l` and `r`:
/// each column's sum loaded and stored once for the block, its products
/// added in the block's order.
#[inline(always)]
fn add_block<const TRACK: bool, const CHECK: bool, S: RowSum>(sums: &mut [f64], flags: &mut [u8], ones: &[ChildPair], l: [RowFloats<'_>; BLOCK], r: [RowFloats<'_>; BLOCK]) {
    let (sl, sr) = (l[0].vals.len() - 1, r[0].vals.len() - 1);
    let (lv, rv) = (l.map(|x| x.vals), r.map(|x| x.vals));
    let (ld, rd) = (l.map(|x| x.dead), r.map(|x| x.dead));
    for ((s, f), q) in sums.iter_mut().zip(flags.iter_mut()).zip(ones) {
        let (i, t) = ((q.left.0 as usize).min(sl), (q.right.0 as usize).min(sr));
        let (mut acc, mut fl) = (*s, *f);
        for k in 0..BLOCK {
            // Safety: `i <= sl` and `t <= sr`, and each row of a side holds
            // as many floats and marks as the first.
            let (a, b) = unsafe { (*lv[k].get_unchecked(i), *rv[k].get_unchecked(t)) };
            let (p, ok) = S::times(a, b);
            acc += p;
            let live = if TRACK {
                // Safety: as above.
                LIVE ^ unsafe { *ld[k].get_unchecked(i) | *rd[k].get_unchecked(t) }
            } else {
                LIVE
            };
            fl |= if CHECK && !ok { live | INEXACT } else { live };
        }
        (*s, *f) = (acc, fl);
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
