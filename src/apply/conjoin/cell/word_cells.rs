//! The cells of a level read by g-pair vectors.

use super::*;
use crate::limits::Limits;

/// A child row whose vector is not built yet ([`WordScratch::slots`]).
const UNBUILT: u32 = u32::MAX;

/// The fewest candidates, f pairs times g pairs, a level has for its
/// cells to be read by vectors ([`words_pay`]).
const WORDS_MIN_CANDIDATES: usize = 1 << 12;

/// The most words a level's vectors take in all ([`words_pay`]).
const WORDS_MAX_VECTOR_WORDS: usize = 1 << 22;

/// Scratch of the levels whose cells are read by g-pair vectors
/// ([`run_level_rows_words`]), kept with the conjunction's workspace.
#[derive(Default)]
pub(crate) struct WordScratch {
    /// g's pairs in column order, each one's column, and where each column
    /// starts.
    g_flat: Vec<ChildPair>,
    col_of: Vec<u32>,
    col_start: Vec<u32>,
    /// Each child row's slot among its side's vectors, [`UNBUILT`] before
    /// a row of the level first names it, and the vectors, a slot's words
    /// one after another: bit `q` set where the child cell at the row and
    /// g pair `q`'s child on that side holds a product.
    slots: Sides<Vec<u32>>,
    vectors: Sides<Vec<u64>>,
    /// The current row's live candidates, `k << 32 | q` for its `k`th f pair
    /// and g pair `q`, in that order, and sorted by column.
    entries: Vec<u64>,
    sorted: Vec<u64>,
    /// A count a column while a row's candidates are sorted, and zero
    /// between rows; the columns counted.
    counts: Vec<u32>,
    touched: Vec<u32>,
    /// Each f pair's first f pair of equal `.left`, on a row of a grouped
    /// cell; a grouped cell's candidates keyed in the grouped walk's order.
    runs_of: Vec<u32>,
    grouped: Vec<(u64, u64)>,
}

impl WordScratch {
    pub(in crate::apply::conjoin) fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.g_flat);
        visit(&mut self.col_of);
        visit(&mut self.col_start);
        visit(&mut self.slots.left);
        visit(&mut self.slots.right);
        visit(&mut self.vectors.left);
        visit(&mut self.vectors.right);
        visit(&mut self.entries);
        visit(&mut self.sorted);
        visit(&mut self.counts);
        visit(&mut self.touched);
        visit(&mut self.runs_of);
        visit(&mut self.grouped);
    }
}

/// Whether a level's cells are read by g-pair vectors: `f` and `g` hold
/// candidates enough, each f pair against each g pair, that building the
/// vectors of the child rows the f pairs name on each side that kills,
/// `f_rows` rows a side at most, and ANDing a pair's two, costs under a
/// quarter of them. A level on whose cells the masks cull candidates may
/// cost less than its candidates; the vectors read only live ones.
pub(crate) fn words_pay(f: &TddLevel, g: &TddLevel, f_rows: Sides<usize>, kills: Sides<bool>) -> bool {
    let bound = |l: &TddLevel| l.pairs.len() + l.node_count();
    if bound(f).saturating_mul(bound(g)) < WORDS_MIN_CANDIDATES {
        return false;
    }
    let (fp, gp) = (f.live_pairs(), g.live_pairs());
    let candidates = fp.saturating_mul(gp);
    if candidates < WORDS_MIN_CANDIDATES || gp > u32::MAX as usize {
        return false;
    }
    let words = gp.div_ceil(64);
    let rows = |n: usize, k: bool| if k { n.min(fp) } else { 0 };
    let rows = rows(f_rows.left, kills.left) + rows(f_rows.right, kills.right);
    if rows.saturating_mul(words) > WORDS_MAX_VECTOR_WORDS {
        return false;
    }
    let build = rows.saturating_mul(gp) + f_rows.left + f_rows.right;
    let reads = fp.saturating_mul(words).saturating_mul(2) + fp;
    build.saturating_add(reads).saturating_mul(4) < candidates
}

/// The row loop of a plain level whose cells are read by g-pair vectors
/// ([`words_pay`]): it builds what [`run_level_rows`] builds on the level's
/// lookups, charging what it charges and writing each cell's pairs in the
/// order its walk writes them, but reads the child grids only for live
/// candidates.
///
/// Each child row an f pair names on a side whose lookup kills gets a
/// vector over g's pairs, a bit a pair in column order, set where the
/// child cell at the row and the pair's child on that side holds a
/// product. An f pair and a g pair are a live candidate iff both vectors of
/// the f pair have the g pair's bit (a complete side's vector is every
/// bit), so a row ANDs its pairs' vectors, sorts the live candidates by
/// column, and its cells read their own off the sorted list.
///
/// A level without masks enters only its live cells
/// ([`run_live_cells`]); on a masked level, the row loop enters the cells
/// the masks leave, and one whose candidates are all dead is charged and
/// left dead.
pub(crate) fn run_level_rows_words(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    words: &mut WordScratch,
    level: &mut TddLevel,
    lookups: PlainLookups,
    f_rows: Sides<usize>,
) -> Result<(), OperationError> {
    let sides = rows.ctx.sides;
    let complete = |p: &ChildPlan<'_>| CompleteLookup { stride: p.stride };
    let grid = |p: &ChildPlan<'_>| DenseLookup { base: p.base, stride: p.stride };
    match lookups {
        PlainLookups::Grid => {
            super::super::note_word_cells(super::super::WordNote::Grids);
            rows_by_words(eng, rows, scratch, words, f_rows, &grid(&sides.left), &grid(&sides.right), EmitSink { level })
        }
        PlainLookups::CompleteLeft { charged } => {
            super::super::note_word_cells(super::super::WordNote::Complete);
            rows_by_words(eng, rows, scratch, words, f_rows, &complete(&sides.left), &grid(&sides.right), ReservedEmitSink { level, charged })
        }
        PlainLookups::CompleteRight { charged } => {
            super::super::note_word_cells(super::super::WordNote::Complete);
            rows_by_words(eng, rows, scratch, words, f_rows, &grid(&sides.left), &complete(&sides.right), ReservedEmitSink { level, charged })
        }
        PlainLookups::Complete { .. } => unreachable!("a level of two complete sides kills no candidate"),
    }
}

/// [`run_level_rows_words`] on one pair of lookups.
#[expect(clippy::too_many_arguments)]
fn rows_by_words<L: ChildLookup, R: ChildLookup, S: PairSink>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    words: &mut WordScratch,
    f_rows: Sides<usize>,
    left: &L,
    right: &R,
    sink: S,
) -> Result<(), OperationError> {
    let mut action = EmitByWords::new(eng, rows, &mut *scratch.g_pairs, words, f_rows, sink)?;
    if rows.ctx.masked {
        run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action)?;
    } else {
        run_live_cells(eng, rows, scratch, left, right, &mut action)?;
    }
    action.finish();
    Ok(())
}

/// The row loop of a level without masks read by g-pair vectors, which
/// enters only the cells that hold a live candidate. A row of the row loop
/// over such a level enters every cell, charges each a unit a candidate
/// and charges a unit a column; this charges the same, a row at a time.
fn run_live_cells<L: ChildLookup, R: ChildLookup, S: PairSink>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    action: &mut EmitByWords<'_, S>,
) -> Result<(), OperationError> {
    super::super::note_word_cells(super::super::WordNote::LiveCells);
    let RowLoop { f_level, ctx, f_width, .. } = rows;
    let RowScratch { f_pairs: f_scratch, node_idx: slab, .. } = scratch;
    let lim = eng.limits();
    let mut poll = lim.gate_with(super::super::budget::DENSE_CELL_POLL_STRIDE);
    let width = ctx.right_width;
    let candidates_a_pair = action.w.g_flat.len() as u64;
    // The row loop's resets: a fill of a small grid, a row at a time on a
    // large one.
    let slab_fill = f_width.saturating_mul(width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if slab_fill {
        let end = ctx.output_grid_base + f_width * width;
        if !extend_slab(slab, ctx.output_grid_base, end) {
            mark_dead(&mut slab[ctx.output_grid_base..end]);
        }
    }
    let (lv, rv) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    let described = described_rows(lim, f_level, f_width, lv, rv, f_scratch);
    for i in 0..f_width {
        let row_base = ctx.output_grid_base + i * width;
        let fresh = extend_slab(slab, row_base, row_base + width);
        if !slab_fill && !fresh {
            mark_dead(&mut slab[row_base..row_base + width]);
        }
        let f_pairs = match described {
            Some(k) => &f_scratch[i * k..(i + 1) * k],
            None => f_level.pairs_view_decoded(i, f_scratch, lv, rv),
        };
        if f_pairs.is_empty() {
            continue;
        }
        action.read_row(i, f_pairs, slab, left, right)?;
        while let Some(j) = action.next_column() {
            let (from, to) = action.take_column(j);
            action.emit(eng, i, j, row_base + j, f_pairs, ctx, slab, left, right, from, to)?;
        }
        poll.poll(f_pairs.len() as u64 * candidates_a_pair + width as u64)?;
    }
    poll.flush()?;
    Ok(())
}

/// The emit action of [`run_level_rows_words`].
struct EmitByWords<'w, S> {
    sink: S,
    w: &'w mut WordScratch,
    lim: &'w Limits,
    /// Words a vector.
    words: usize,
    /// The row whose candidates `w.sorted` holds, and the next of them a
    /// cell takes; the row `w.runs_of` was read for.
    row: usize,
    cursor: usize,
    runs_row: usize,
}

impl<'w, S: PairSink> EmitByWords<'w, S> {
    /// Read g's pairs in column order and clear the vectors.
    fn new(
        eng: &'w Engine,
        rows: RowLoop<'_>,
        decode: &mut Vec<ChildPair>,
        w: &'w mut WordScratch,
        f_rows: Sides<usize>,
        sink: S,
    ) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let RowLoop { g_level, ctx, .. } = rows;
        let width = ctx.right_width;
        w.g_flat.clear();
        w.col_of.clear();
        w.col_start.clear();
        lim.reserve_exact(&mut w.col_start, width + 1)?;
        w.col_start.push(0);
        for j in 0..width {
            let pairs = column_pairs(ctx, g_level, j, decode);
            lim.reserve(&mut w.g_flat, pairs.len())?;
            lim.reserve(&mut w.col_of, pairs.len())?;
            w.g_flat.extend_from_slice(pairs);
            w.col_of.extend(std::iter::repeat_n(j as u32, pairs.len()));
            w.col_start.push(w.g_flat.len() as u32);
        }
        for (slots, n) in [(&mut w.slots.left, f_rows.left), (&mut w.slots.right, f_rows.right)] {
            slots.clear();
            lim.reserve_exact(slots, n)?;
            slots.resize(n, UNBUILT);
        }
        w.vectors.left.clear();
        w.vectors.right.clear();
        w.counts.clear();
        lim.reserve_exact(&mut w.counts, width)?;
        w.counts.resize(width, 0);
        let words = w.g_flat.len().div_ceil(64);
        Ok(EmitByWords { sink, w, lim, words, row: usize::MAX, cursor: 0, runs_row: usize::MAX })
    }

    /// Every candidate of the last row was in a cell the loop entered.
    fn finish(&self) {
        debug_assert!(self.row == usize::MAX || self.cursor == self.w.sorted.len(), "a live candidate in a cell the row loop did not enter");
    }

    /// The slot of child row `r`'s vector on one side, built on first use:
    /// bit `q` where the child cell at `r` and `field` of g pair `q` holds a
    /// product.
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn slot<K: ChildLookup>(
        lim: &Limits,
        slots: &mut [u32],
        vectors: &mut Vec<u64>,
        g_flat: &[ChildPair],
        words: usize,
        lookup: &K,
        node_idx: &[u32],
        r: u32,
        field: impl Fn(&ChildPair) -> u32,
    ) -> Result<usize, OperationError> {
        let s = slots[r as usize];
        if s != UNBUILT {
            return Ok(s as usize);
        }
        let s = vectors.len() / words;
        slots[r as usize] = s as u32;
        lim.reserve(vectors, words)?;
        let row = lookup.row(r);
        for chunk in g_flat.chunks(64) {
            let word = chunk.iter().enumerate().fold(0u64, |word, (b, q)| {
                word | u64::from(lookup.get_in_row(node_idx, row, field(q)) != NO_PRODUCT) << b
            });
            vectors.push(word);
        }
        Ok(s)
    }

    /// Row `i`'s live candidates into `w.sorted`, by column.
    #[inline(always)]
    fn read_row<L: ChildLookup, R: ChildLookup>(
        &mut self,
        i: usize,
        f_pairs: &[ChildPair],
        node_idx: &[u32],
        left: &L,
        right: &R,
    ) -> Result<(), OperationError> {
        let (lim, words) = (self.lim, self.words);
        let w = &mut *self.w;
        debug_assert!(self.row == usize::MAX || self.cursor == w.sorted.len(), "a live candidate in a cell the row loop did not enter");
        w.entries.clear();
        for (k, p) in f_pairs.iter().enumerate() {
            let ls = if left.kills() {
                Self::slot(lim, &mut w.slots.left, &mut w.vectors.left, &w.g_flat, words, left, node_idx, p.left.0, |q| q.left.0)?
            } else {
                0
            };
            let rs = if right.kills() {
                Self::slot(lim, &mut w.slots.right, &mut w.vectors.right, &w.g_flat, words, right, node_idx, p.right.0, |q| q.right.0)?
            } else {
                0
            };
            let lv = if left.kills() { &w.vectors.left[ls * words..(ls + 1) * words] } else { &[][..] };
            let rv = if right.kills() { &w.vectors.right[rs * words..(rs + 1) * words] } else { &[][..] };
            let key = (k as u64) << 32;
            for word in 0..words {
                let mut m = match (left.kills(), right.kills()) {
                    (true, true) => lv[word] & rv[word],
                    (true, false) => lv[word],
                    (false, true) => rv[word],
                    (false, false) => unreachable!("a level of two complete sides kills no candidate"),
                };
                while m != 0 {
                    let q = (word * 64) as u64 + u64::from(m.trailing_zeros());
                    m &= m - 1;
                    lim.try_push(&mut w.entries, key | q)?;
                }
            }
        }
        // In order of f pair, then g pair: by column already on a row of
        // one f pair. Otherwise a counting sort, stable, by column.
        w.sorted.clear();
        if f_pairs.len() == 1 {
            std::mem::swap(&mut w.entries, &mut w.sorted);
        } else {
            let col = |e: u64| w.col_of[(e & 0xffff_ffff) as usize] as usize;
            w.touched.clear();
            for &e in &w.entries {
                let c = col(e);
                if w.counts[c] == 0 {
                    lim.try_push(&mut w.touched, c as u32)?;
                }
                w.counts[c] += 1;
            }
            w.touched.sort_unstable();
            let mut at = 0;
            for &c in &w.touched {
                let n = w.counts[c as usize];
                w.counts[c as usize] = at;
                at += n;
            }
            lim.reserve_exact(&mut w.sorted, w.entries.len())?;
            w.sorted.resize(w.entries.len(), 0);
            for &e in &w.entries {
                let c = col(e);
                w.sorted[w.counts[c] as usize] = e;
                w.counts[c] += 1;
            }
            for &c in &w.touched {
                w.counts[c as usize] = 0;
            }
        }
        self.row = i;
        self.cursor = 0;
        Ok(())
    }

    /// The column of the current row's next live candidate.
    #[inline(always)]
    fn next_column(&self) -> Option<usize> {
        let e = *self.w.sorted.get(self.cursor)?;
        Some(self.w.col_of[(e & 0xffff_ffff) as usize] as usize)
    }

    /// The current row's live candidates in column `j`, which no column
    /// before it has left: `w.sorted[from..to]`.
    #[inline(always)]
    fn take_column(&mut self, j: usize) -> (usize, usize) {
        let w = &*self.w;
        let from = self.cursor;
        let mut to = from;
        while to < w.sorted.len() && w.col_of[(w.sorted[to] & 0xffff_ffff) as usize] as usize == j {
            to += 1;
        }
        debug_assert!(
            to == w.sorted.len() || w.col_of[(w.sorted[to] & 0xffff_ffff) as usize] as usize > j,
            "a live candidate in a cell the row loop did not enter"
        );
        self.cursor = to;
        (from, to)
    }

    /// Write the cell at `grid_pos`, row `i` and column `j`, of the live
    /// candidates `w.sorted[from..to]`, at least one, in the order the
    /// cell's walk writes them.
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn emit<L: ChildLookup, R: ChildLookup>(
        &mut self,
        eng: &Engine,
        i: usize,
        j: usize,
        grid_pos: usize,
        f_pairs: &[ChildPair],
        ctx: &CellCtx<'_>,
        node_idx: &mut [u32],
        left: &L,
        right: &R,
        from: usize,
        to: usize,
    ) -> Result<(), OperationError> {
        let w = &mut *self.w;
        let start = w.col_start[j] as usize;
        let (n1, n2) = (f_pairs.len(), w.col_start[j + 1] as usize - start);
        let q_of = |e: u64| (e & 0xffff_ffff) as usize;
        let product = |e: u64, node_idx: &[u32]| {
            let (p, q) = (&f_pairs[(e >> 32) as usize], &w.g_flat[q_of(e)]);
            let lc = left.get(node_idx, p.left.0, q.left.0);
            let rc = right.get(node_idx, p.right.0, q.right.0);
            debug_assert!(lc != NO_PRODUCT && rc != NO_PRODUCT, "a candidate read live is live");
            (lc, rc)
        };
        if n1 == 1 && n2 == 1 {
            let (lc, rc) = product(w.sorted[from], node_idx);
            return self.sink.single(eng, node_idx, grid_pos, lc, rc);
        }
        let begin = self.sink.begin();
        let runs = if n1 >= GROUPED_MIN_PAIRS && n2 > 1 { ctx.right_cols.and_then(|cols| cols.runs(j)) } else { None };
        match runs {
            None => {
                for &e in &w.sorted[from..to] {
                    let (lc, rc) = product(e, node_idx);
                    self.sink.pair(eng, lc, rc)?;
                }
            }
            Some(runs) => {
                // The grouped walk's order: by f pair run of equal `.left`,
                // then g pair run of equal `.left`, then f pair, then g pair.
                super::super::note_word_cells(super::super::WordNote::Grouped);
                if self.runs_row != i {
                    self.runs_row = i;
                    w.runs_of.clear();
                    self.lim.reserve_exact(&mut w.runs_of, n1)?;
                    let mut first = 0;
                    for (k, p) in f_pairs.iter().enumerate() {
                        if p.left != f_pairs[first].left {
                            first = k;
                        }
                        w.runs_of.push(first as u32);
                    }
                }
                w.grouped.clear();
                self.lim.reserve(&mut w.grouped, to - from)?;
                for &e in &w.sorted[from..to] {
                    let local = (q_of(e) - start) as u32;
                    let run = runs.partition_point(|&end| end <= local) as u64;
                    w.grouped.push((u64::from(w.runs_of[(e >> 32) as usize]) << 32 | run, e));
                }
                w.grouped.sort_unstable();
                for &(_, e) in &w.grouped {
                    let (lc, rc) = product(e, node_idx);
                    self.sink.pair(eng, lc, rc)?;
                }
            }
        }
        self.sink.end(eng, node_idx, grid_pos, begin)
    }
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for EmitByWords<'_, S> {
    const DENSE_SLAB: bool = true;

    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        let n2 = (self.w.col_start[a.j + 1] - self.w.col_start[a.j]) as usize;
        // A dead column's cell, as `process_cell` leaves it.
        if n2 == 0 {
            a.node_idx[a.row_base + a.j] = NO_PRODUCT;
            return Ok(());
        }
        // The cell walk's charge: a unit a candidate.
        a.gate.poll((a.f_pairs.len() * n2) as u64)?;
        if a.i != self.row {
            self.read_row(a.i, a.f_pairs, a.node_idx, a.left, a.right)?;
        }
        let (from, to) = self.take_column(a.j);
        if from == to {
            return Ok(());
        }
        self.emit(eng, a.i, a.j, a.row_base + a.j, a.f_pairs, a.ctx, a.node_idx, a.left, a.right, from, to)
    }
}
