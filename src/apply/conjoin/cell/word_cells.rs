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

/// A row reads its cells by vectors where at most one of this many of its
/// candidates is live; any other row walks its cells.
const LIVE_SHARE: u64 = 4;

/// The candidates a level's rows count before the level stops reading
/// rows by vectors, if more than one of [`LIVE_SHARE`] of them was live.
const PROBATION: u64 = 1 << 16;

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
    /// The current row's pairs' slots on each side.
    row_slots: Vec<(u32, u32)>,
    /// The current row's live candidates, `k << 32 | q` for its `k`th f pair
    /// and g pair `q`, in that order, and sorted by column.
    entries: Vec<u64>,
    sorted: Vec<u64>,
    /// A count a column while a row's candidates are sorted, and zero
    /// between rows; the columns counted.
    counts: Vec<u32>,
    touched: Vec<u32>,
    /// Each f pair's first f pair of equal `.left`, on a row of a grouped
    /// cell; each g pair's run of equal `.left` in its column, on a level
    /// with grouped columns; a grouped cell's candidates in the grouped
    /// walk's order, and a count a run while they are sorted into it.
    runs_of: Vec<u32>,
    run_of: Vec<u32>,
    grouped: Vec<u64>,
    run_counts: Vec<u32>,
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
        visit(&mut self.row_slots);
        visit(&mut self.entries);
        visit(&mut self.sorted);
        visit(&mut self.counts);
        visit(&mut self.touched);
        visit(&mut self.runs_of);
        visit(&mut self.run_of);
        visit(&mut self.grouped);
        visit(&mut self.run_counts);
    }
}

/// Whether a level's cells are read by g-pair vectors: `f` and `g` hold
/// candidates enough, each f pair against each g pair, that building the
/// vectors of the child rows the f pairs name on each side that kills,
/// `f_rows` rows a side at most, and ANDing a pair's two, costs under a
/// quarter of them.
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
/// candidates. A level without masks enters only the cells that hold one;
/// on a masked level, the row loop enters the cells its masks leave.
///
/// Each child row an f pair names on a side whose lookup kills gets a
/// vector over g's pairs, a bit a pair in column order, set where the
/// child cell at the row and the pair's child on that side holds a
/// product. An f pair and a g pair are a live candidate iff both vectors of
/// the f pair have the g pair's bit (a complete side's vector is every
/// bit). A row counts its live candidates off its pairs' vectors; where few
/// of them are live it sorts them by column and writes its live cells off
/// the sorted list, and otherwise walks its cells as the row loop does
/// ([`LIVE_SHARE`]). A level whose rows hold mostly live candidates walks
/// the rest of them ([`PROBATION`]).
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

/// [`run_level_rows_words`] on one pair of lookups. A row of the row loop
/// over a level without masks enters every cell, charges each a unit a
/// candidate and charges a unit a column; a row read by vectors charges
/// the same in one poll.
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
    let mut a = ByWords::new(eng, rows, &mut *scratch.g_pairs, words, f_rows, sink)?;
    if rows.ctx.masked {
        run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut a)?;
        a.finish();
        return Ok(());
    }
    let RowLoop { f_level, g_level, ctx, f_width, .. } = rows;
    let RowScratch { f_pairs: f_scratch, g_pairs: g_scratch, node_idx: slab } = scratch;
    let lim = eng.limits();
    let mut poll = lim.gate_with(super::super::budget::DENSE_CELL_POLL_STRIDE);
    let width = ctx.right_width;
    let gp = a.w.g_flat.len() as u64;
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
        let candidates = f_pairs.len() as u64 * gp;
        if a.take_row(i, f_pairs, slab, left, right)? {
            while let Some(j) = a.next_column() {
                let (from, to) = a.take_column(j);
                a.emit(eng, i, j, row_base + j, f_pairs, ctx, slab, left, right, from, to)?;
            }
            poll.poll(candidates + width as u64)?;
        } else {
            for j in 0..width {
                process_cell(eng, j, row_base, f_pairs, ctx, g_level, g_scratch, slab, left, right, &mut a.sink, &mut poll)?;
            }
            poll.poll(width as u64)?;
        }
    }
    poll.flush()?;
    a.finish();
    Ok(())
}

/// The vectors and the sink of [`rows_by_words`].
struct ByWords<'w, S> {
    sink: S,
    w: &'w mut WordScratch,
    lim: &'w Limits,
    /// Words a vector.
    words: usize,
    /// The candidates of the rows counted so far, and the live ones;
    /// whether the level still counts its rows.
    seen: u64,
    live: u64,
    on: bool,
    /// The current row, and whether it is read by vectors.
    row: usize,
    row_words: bool,
    /// The next of the current row's live candidates a cell takes; the row
    /// `w.runs_of` was read for.
    cursor: usize,
    runs_row: usize,
}

/// Word `word` of the live candidates of an f pair whose vectors are in
/// `slots`, on a level whose sides kill as `kills` says.
#[inline(always)]
fn anded(w: &WordScratch, words: usize, kills: (bool, bool), slots: (u32, u32), word: usize) -> u64 {
    let at = |v: &[u64], s: u32| v[s as usize * words + word];
    match kills {
        (true, true) => at(&w.vectors.left, slots.0) & at(&w.vectors.right, slots.1),
        (true, false) => at(&w.vectors.left, slots.0),
        (false, true) => at(&w.vectors.right, slots.1),
        (false, false) => unreachable!("a level of two complete sides kills no candidate"),
    }
}

impl<'w, S: PairSink> ByWords<'w, S> {
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
        // Each g pair's run in its column, where the column table grouped it.
        w.run_of.clear();
        w.run_counts.clear();
        if let Some(cols) = ctx.right_cols {
            let mut most = 0;
            for j in 0..width {
                if let Some(runs) = cols.runs(j) {
                    let start = w.col_start[j] as usize;
                    if w.run_of.is_empty() {
                        lim.reserve_exact(&mut w.run_of, w.g_flat.len())?;
                        w.run_of.resize(w.g_flat.len(), 0);
                    }
                    let mut from = 0;
                    for (r, &end) in runs.iter().enumerate() {
                        w.run_of[start + from..start + end as usize].fill(r as u32);
                        from = end as usize;
                    }
                    most = most.max(runs.len());
                }
            }
            lim.reserve_exact(&mut w.run_counts, most)?;
            w.run_counts.resize(most, 0);
        }
        let words = w.g_flat.len().div_ceil(64);
        Ok(ByWords { sink, w, lim, words, seen: 0, live: 0, on: true, row: usize::MAX, row_words: false, cursor: 0, runs_row: usize::MAX })
    }

    /// Every live candidate of the last row read by vectors was in a cell
    /// the loop entered.
    fn finish(&self) {
        debug_assert!(!self.row_words || self.cursor == self.w.sorted.len(), "a live candidate in a cell the row loop did not enter");
    }

    /// Count row `i`, of pairs `f_pairs`, and read it by vectors where few
    /// of its candidates are live: whether it is.
    #[inline(always)]
    fn take_row<L: ChildLookup, R: ChildLookup>(
        &mut self,
        i: usize,
        f_pairs: &[ChildPair],
        node_idx: &[u32],
        left: &L,
        right: &R,
    ) -> Result<bool, OperationError> {
        self.finish();
        self.row = i;
        self.row_words = false;
        if !self.on {
            return Ok(false);
        }
        let candidates = f_pairs.len() as u64 * self.w.g_flat.len() as u64;
        let live = self.count_row(f_pairs, node_idx, left, right)?;
        self.seen += candidates;
        self.live += live;
        if self.seen >= PROBATION && self.live * LIVE_SHARE > self.seen {
            self.on = false;
        }
        if live * LIVE_SHARE > candidates {
            return Ok(false);
        }
        super::super::note_word_cells(super::super::WordNote::Row);
        self.read_row(f_pairs.len(), left.kills(), right.kills())?;
        self.row_words = true;
        Ok(true)
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
    ) -> Result<u32, OperationError> {
        let s = slots[r as usize];
        if s != UNBUILT {
            return Ok(s);
        }
        let s = (vectors.len() / words) as u32;
        slots[r as usize] = s;
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

    /// The live candidates of a row of pairs `f_pairs`, whose vectors this
    /// builds where they are not yet.
    #[inline(always)]
    fn count_row<L: ChildLookup, R: ChildLookup>(
        &mut self,
        f_pairs: &[ChildPair],
        node_idx: &[u32],
        left: &L,
        right: &R,
    ) -> Result<u64, OperationError> {
        let (lim, words, kills) = (self.lim, self.words, (left.kills(), right.kills()));
        let w = &mut *self.w;
        w.row_slots.clear();
        lim.reserve(&mut w.row_slots, f_pairs.len())?;
        let mut live = 0u64;
        for p in f_pairs {
            let ls = if kills.0 {
                Self::slot(lim, &mut w.slots.left, &mut w.vectors.left, &w.g_flat, words, left, node_idx, p.left.0, |q| q.left.0)?
            } else {
                0
            };
            let rs = if kills.1 {
                Self::slot(lim, &mut w.slots.right, &mut w.vectors.right, &w.g_flat, words, right, node_idx, p.right.0, |q| q.right.0)?
            } else {
                0
            };
            w.row_slots.push((ls, rs));
            for word in 0..words {
                live += u64::from(anded(w, words, kills, (ls, rs), word).count_ones());
            }
        }
        Ok(live)
    }

    /// The live candidates of the row of `n1` pairs that
    /// [`count_row`](Self::count_row) counted last, into `w.sorted` by
    /// column.
    #[inline(always)]
    fn read_row(&mut self, n1: usize, left_kills: bool, right_kills: bool) -> Result<(), OperationError> {
        let (lim, words, kills) = (self.lim, self.words, (left_kills, right_kills));
        let w = &mut *self.w;
        debug_assert_eq!(w.row_slots.len(), n1, "a row is read after it is counted");
        w.entries.clear();
        for k in 0..n1 {
            let slots = w.row_slots[k];
            let key = (k as u64) << 32;
            for word in 0..words {
                let mut m = anded(w, words, kills, slots, word);
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
        if n1 == 1 {
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
                let node_idx = &*node_idx;
                push_kept(eng, &mut self.sink, &w.sorted[from..to], false, |&e| product(e, node_idx))?;
            }
            Some(runs) => {
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
                order_grouped(self.lim, &w.sorted[from..to], &w.runs_of, &w.run_of, &mut w.run_counts[..runs.len()], &mut w.grouped)?;
                let node_idx = &*node_idx;
                push_kept(eng, &mut self.sink, &w.grouped, false, |&e| product(e, node_idx))?;
            }
        }
        self.sink.end(eng, node_idx, grid_pos, begin)
    }
}

impl<L: ChildLookup, R: ChildLookup, S: PairSink> CellAction<L, R> for ByWords<'_, S> {
    const DENSE_SLAB: bool = true;

    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    /// A cell the masks leave: on a row read by vectors, its live
    /// candidates off the sorted list, after the charge of the cell's walk,
    /// a unit a candidate; on any other row, the cell's walk.
    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        if a.i != self.row {
            self.take_row(a.i, a.f_pairs, a.node_idx, a.left, a.right)?;
        }
        if !self.row_words {
            return process_cell(
                eng, a.j, a.row_base, a.f_pairs, a.ctx, a.right_level_t, a.g_pairs_scratch,
                a.node_idx, a.left, a.right, &mut self.sink, a.gate,
            );
        }
        let n2 = (self.w.col_start[a.j + 1] - self.w.col_start[a.j]) as usize;
        // A dead column's cell, as `process_cell` leaves it.
        if n2 == 0 {
            a.node_idx[a.row_base + a.j] = NO_PRODUCT;
            return Ok(());
        }
        a.gate.poll((a.f_pairs.len() * n2) as u64)?;
        let (from, to) = self.take_column(a.j);
        if from == to {
            return Ok(());
        }
        self.emit(eng, a.i, a.j, a.row_base + a.j, a.f_pairs, a.ctx, a.node_idx, a.left, a.right, from, to)
    }
}

/// The live candidates `cell` of a grouped cell, `k << 32 | q` in order
/// of f pair and then g pair, into `out` in the grouped walk's order: by
/// run of f pairs of equal `.left` (`runs_of`, each f pair's run), then
/// run of g pairs of equal `.left` (`run_of`, each g pair's run in its
/// column), then f pair, then g pair. The candidates of one f run are
/// consecutive already, and each run's are sorted by g run: by counts,
/// a count a g run of the column in `counts`, zero, where they are many
/// against the runs, and by comparison otherwise.
fn order_grouped(
    lim: &Limits,
    cell: &[u64],
    runs_of: &[u32],
    run_of: &[u32],
    counts: &mut [u32],
    out: &mut Vec<u64>,
) -> Result<(), OperationError> {
    let k_of = |e: u64| (e >> 32) as usize;
    let q_of = |e: u64| (e & 0xffff_ffff) as usize;
    out.clear();
    lim.reserve(out, cell.len())?;
    let mut b0 = 0;
    while b0 < cell.len() {
        let run = runs_of[k_of(cell[b0])];
        let mut b1 = b0 + 1;
        while b1 < cell.len() && runs_of[k_of(cell[b1])] == run {
            b1 += 1;
        }
        let block = &cell[b0..b1];
        let base = out.len();
        if block.len() * 2 >= counts.len() {
            for &e in block {
                counts[run_of[q_of(e)] as usize] += 1;
            }
            let mut at = base as u32;
            for c in counts.iter_mut() {
                let n = *c;
                *c = at;
                at += n;
            }
            out.resize(base + block.len(), 0);
            for &e in block {
                let c = &mut counts[run_of[q_of(e)] as usize];
                out[*c as usize] = e;
                *c += 1;
            }
            counts.fill(0);
        } else {
            out.extend_from_slice(block);
            out[base..].sort_unstable_by_key(|&e| (run_of[q_of(e)], e));
        }
        b0 = b1;
    }
    Ok(())
}
