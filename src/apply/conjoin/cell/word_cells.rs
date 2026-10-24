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

/// A row counted by vectors writes its live cells off its live candidates
/// where at most one of this many candidates of its live cells is live;
/// otherwise it walks its live cells alone.
const CELL_SHARE: u64 = 8;

/// A row writes its live cells off f-pair bits, a word per 64 f pairs a
/// live cell, where they take at most this many words a live candidate
/// and at most [`CELL_BITS_MAX_WORDS`] in all; otherwise off its live
/// candidates sorted by column.
const BITS_SHARE: u64 = 2;
const CELL_BITS_MAX_WORDS: usize = 1 << 22;

/// The candidates a level's rows count before the level stops reading
/// rows by vectors, if more than one of [`LIVE_SHARE`] of them was live.
const PROBATION: u64 = 1 << 16;

/// The f nodes, and the g nodes, a level's sample takes at most, spread
/// over the level ([`sample_cells`]).
const SAMPLE_NODES: usize = 32;

/// The candidates of a sampled cell read at most, and the fewest of its
/// f pairs and of its g pairs they take where it holds more
/// ([`sample_cells`]); a level's sample reads at most one in
/// `SAMPLE_SHARE` of its candidates.
const SAMPLE_CELL: usize = 64;
const SAMPLE_SIDE: usize = 8;
const SAMPLE_SHARE: usize = 8;

/// A level with masks is read by vectors where at most one of this many
/// of its sampled candidates is live ([`words_pay`]).
const MASKED_LIVE_SHARE: u128 = 10;

/// The row loop's cost of a level without masks, in about a cycle a unit:
/// a cell, a candidate and a live candidate; and the vectors': a child
/// cell read or a vector word built, an f pair's vector word read, a live
/// cell, a live candidate and a cell ([`words_pay`]). Fitted to levels run
/// both ways.
const WALK_COST: [u128; 3] = [18, 2, 9];
const WORDS_COST: [u128; 5] = [5, 10, 31, 6, 2];

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
    /// Each vector's summary, a slot's words one after another: bit `b` of
    /// summary word `s` set where the vector's word `64 s + b` is not zero.
    summaries: Sides<Vec<u64>>,
    /// On each side that kills, g's pairs by their child there: the
    /// children some pair names, ascending, where each one's pairs start
    /// in `by_child`, and the pairs, by child and then in column order.
    /// A count a child while they are sorted.
    named: Sides<Vec<u32>>,
    named_start: Sides<Vec<u32>>,
    by_child: Sides<Vec<u32>>,
    child_counts: Vec<u32>,
    /// The current row's pairs' slots on each side; the OR of its pairs'
    /// live candidates; its live cells' columns, ascending, and as bits.
    row_slots: Vec<(u32, u32)>,
    union: Vec<u64>,
    union_sum: Vec<u64>,
    live_cols: Vec<u32>,
    col_live: Vec<u64>,
    /// Each live cell's slot among the current row's (its index in
    /// `live_cols`), read at the row's live columns alone; a slot's words
    /// one after another, bit `k` set where the row's `k`th f pair has a
    /// live candidate in the cell.
    col_slot: Vec<u32>,
    cell_bits: Vec<u64>,
    /// A cell's live candidates, `k << 32 | q` for the row's `k`th f pair
    /// and g pair `q`, in that order, on a row read off its bits; the row's,
    /// and sorted by column, on a row read off its sorted candidates.
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
    /// A level's sampled g pairs ([`sample_cells`]), node after node, and
    /// its sampled g nodes.
    sample: Vec<ChildPair>,
    sample_cols: Vec<SampledColumn>,
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
        visit(&mut self.summaries.left);
        visit(&mut self.summaries.right);
        visit(&mut self.union_sum);
        visit(&mut self.named.left);
        visit(&mut self.named.right);
        visit(&mut self.named_start.left);
        visit(&mut self.named_start.right);
        visit(&mut self.by_child.left);
        visit(&mut self.by_child.right);
        visit(&mut self.child_counts);
        visit(&mut self.row_slots);
        visit(&mut self.union);
        visit(&mut self.live_cols);
        visit(&mut self.col_live);
        visit(&mut self.col_slot);
        visit(&mut self.cell_bits);
        visit(&mut self.entries);
        visit(&mut self.sorted);
        visit(&mut self.counts);
        visit(&mut self.touched);
        visit(&mut self.runs_of);
        visit(&mut self.run_of);
        visit(&mut self.grouped);
        visit(&mut self.run_counts);
        visit(&mut self.sample);
        visit(&mut self.sample_cols);
    }
}

/// Whether a level's cells are read by g-pair vectors. Its candidates,
/// each f pair against each g pair, are enough, and the vectors of the
/// child rows its f pairs name on each side that kills, `f_rows` rows a
/// side at most, fit [`WORDS_MAX_VECTOR_WORDS`]. Then a sample of its cells
/// ([`sample_cells`]) estimates its live candidates and live cells. A
/// level whose sampled candidates are mostly in grouped cells is walked:
/// the grouped walk reads a left cell once a run of f pairs against a run
/// of g pairs. A level with masks is read by vectors where g's nodes are
/// not all one pair (the masks cull most of the f pairs of a column of
/// one), at most one in [`MASKED_LIVE_SHARE`] of its candidates is live,
/// and building and ANDing its vectors costs under a quarter of them; a
/// level without, where the vectors cost less than the row loop by
/// [`WORDS_COST`] and [`WALK_COST`], the sampled rows that would walk
/// their cells ([`LIVE_SHARE`]) priced as the row loop.
pub(crate) fn words_pay(
    eng: &Engine,
    rows: RowLoop<'_>,
    w: &mut WordScratch,
    lookups: PlainLookups,
    f_rows: Sides<usize>,
    node_idx: &[u32],
) -> bool {
    let (f, g) = (rows.f_level, rows.g_level);
    let sides = rows.ctx.sides;
    let kills = Sides {
        left: !matches!(lookups, PlainLookups::CompleteLeft { .. }),
        right: !matches!(lookups, PlainLookups::CompleteRight { .. }),
    };
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
    let rows_of = |n: usize, k: bool| if k { n.min(fp) } else { 0 };
    let (rows_l, rows_r) = (rows_of(f_rows.left, kills.left), rows_of(f_rows.right, kills.right));
    if (rows_l + rows_r).saturating_mul(words) > WORDS_MAX_VECTOR_WORDS || (rows.ctx.masked && !g.has_multi_pair()) {
        return false;
    }
    let complete = |p: &ChildPlan<'_>| CompleteLookup { stride: p.stride };
    let grid = |p: &ChildPlan<'_>| DenseLookup { base: p.base, stride: p.stride };
    let sampled = rows.f_width.min(SAMPLE_NODES) * rows.ctx.right_width.min(SAMPLE_NODES);
    let budget = (candidates / SAMPLE_SHARE / sampled.max(1)).clamp(1, SAMPLE_CELL);
    let sample = match lookups {
        PlainLookups::Grid => sample_cells(eng, rows, w, &grid(&sides.left), &grid(&sides.right), node_idx, budget),
        PlainLookups::CompleteLeft { .. } => sample_cells(eng, rows, w, &complete(&sides.left), &grid(&sides.right), node_idx, budget),
        PlainLookups::CompleteRight { .. } => sample_cells(eng, rows, w, &grid(&sides.left), &complete(&sides.right), node_idx, budget),
        PlainLookups::Complete { .. } => return false,
    };
    let Ok(s) = sample else { return false };
    if s.candidates == 0 || (!rows.ctx.masked && s.grouped * 2 > s.candidates) {
        return false;
    }
    let side = |rows: usize, named: usize, k: bool| {
        if k { rows.saturating_mul(named.min(gp) + words).saturating_add(gp + named) } else { 0 }
    };
    let (named_l, named_r) = (sides.left.stride as usize, sides.right.stride as usize);
    let build = (side(rows_l, named_l, kills.left) + side(rows_r, named_r, kills.right) + f_rows.left + f_rows.right) as u128;
    let reads = fp as u128 * words as u128;
    let (candidates, cells) = (candidates as u128, rows.f_width as u128 * rows.ctx.right_width as u128);
    let live = s.live as u128 * candidates / s.candidates as u128;
    if rows.ctx.masked {
        return live * MASKED_LIVE_SHARE < candidates && (build + 2 * reads + fp as u128) * 4 < candidates;
    }
    let live_cells = s.live_cells as u128 * cells / s.cells.max(1) as u128;
    let walked = s.walked;
    let (w_cells, w_candidates) = (walked.cells as u128 * cells / s.cells.max(1) as u128, walked.candidates as u128 * candidates / s.candidates as u128);
    let w_live = walked.live as u128 * candidates / s.candidates as u128;
    let w_live_cells = walked.live_cells as u128 * cells / s.cells.max(1) as u128;
    let walk_cost = |cells: u128, candidates: u128, live: u128| {
        [cells, candidates, live].iter().zip(WALK_COST).map(|(x, c)| x * c).sum::<u128>()
    };
    let walk = walk_cost(cells, candidates, live);
    let rest = [build, reads, live_cells - w_live_cells.min(live_cells), live - w_live.min(live), cells - w_cells.min(cells)];
    let vectors = rest.iter().zip(WORDS_COST).map(|(x, c)| x * c).sum::<u128>() + walk_cost(w_cells, w_candidates, w_live);
    vectors < walk
}

/// A level's sampled cells ([`sample_cells`]): the candidates they hold,
/// the live ones (estimated in a cell whose candidates were sampled) and
/// those in grouped cells; the cells, and the live ones among them; and
/// the same of the sampled f nodes whose sampled cells hold more than one
/// in [`LIVE_SHARE`] live candidates, whose rows would walk their cells.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CellSample {
    pub(crate) candidates: u64,
    pub(crate) live: u64,
    pub(crate) grouped: u64,
    pub(crate) cells: u64,
    pub(crate) live_cells: u64,
    pub(crate) walked: SampledRows,
}

/// The sampled cells of the sampled f nodes whose rows would walk them
/// ([`CellSample::walked`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SampledRows {
    pub(crate) candidates: u64,
    pub(crate) live: u64,
    pub(crate) cells: u64,
    pub(crate) live_cells: u64,
}

/// A sampled g node: where its sampled pairs start in
/// [`WordScratch::sample`], its pairs, and whether its column is grouped.
#[derive(Clone, Copy)]
struct SampledColumn {
    start: u32,
    pairs: u32,
    grouped: bool,
}

/// `m` of `n` indices spread evenly, all of them where `n <= m`.
fn spread(n: usize, m: usize) -> impl Iterator<Item = usize> {
    let m = m.min(n);
    (0..m).map(move |k| k * n / m)
}

/// The cells of up to [`SAMPLE_NODES`] f nodes against up to as many g
/// nodes, spread over each level, read off the child grids: in each, its
/// candidates where they are at most `budget` ([`SAMPLE_CELL`] at most),
/// and otherwise as many, f pairs against g pairs spread over each
/// node's, at least [`SAMPLE_SIDE`] of a side that has them. A cell is
/// live where a sampled candidate is, and grouped where the walk groups it
/// (an f row of [`GROUPED_MIN_PAIRS`] pairs against a grouped column).
fn sample_cells<L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    w: &mut WordScratch,
    left: &L,
    right: &R,
    node_idx: &[u32],
    budget: usize,
) -> Result<CellSample, OperationError> {
    let lim = eng.limits();
    let RowLoop { f_level, g_level, ctx, f_width, .. } = rows;
    let (lv, rv) = (ctx.sides.left.plan.view, ctx.sides.right.plan.view);
    w.sample.clear();
    w.sample_cols.clear();
    lim.reserve(&mut w.sample, SAMPLE_NODES * SAMPLE_CELL)?;
    lim.reserve(&mut w.sample_cols, SAMPLE_NODES)?;
    for j in spread(ctx.right_width, SAMPLE_NODES) {
        let pairs = column_pairs(ctx, g_level, j, &mut w.g_flat);
        let grouped = ctx.right_cols.is_some_and(|c| c.runs(j).is_some());
        w.sample_cols.push(SampledColumn { start: w.sample.len() as u32, pairs: pairs.len() as u32, grouped });
        w.sample.extend(spread(pairs.len(), SAMPLE_CELL).map(|k| pairs[k]));
    }
    let mut s = CellSample::default();
    let mut picks = [0usize; SAMPLE_CELL];
    for i in spread(f_width, SAMPLE_NODES) {
        let pairs = f_level.pairs_view_decoded(i, &mut w.g_flat, lv, rv);
        let n1 = pairs.len() as u64;
        let row = SampledRows { candidates: s.candidates, live: s.live, cells: s.cells, live_cells: s.live_cells };
        let mut f_picked = 0;
        for k in spread(pairs.len(), SAMPLE_CELL) {
            picks[f_picked] = k;
            f_picked += 1;
        }
        for (b, col) in w.sample_cols.iter().enumerate() {
            let end = w.sample_cols.get(b + 1).map_or(w.sample.len(), |c| c.start as usize);
            let g_picked = &w.sample[col.start as usize..end];
            let held = n1 * u64::from(col.pairs);
            s.cells += 1;
            s.candidates += held;
            if held == 0 {
                continue;
            }
            if n1 as usize >= GROUPED_MIN_PAIRS && col.grouped {
                s.grouped += held;
            }
            // As many f pairs as leave a pair to each, at least
            // `SAMPLE_SIDE` of g's, and as many g pairs as leave a pair
            // to each of those.
            let s1 = f_picked.min((budget / g_picked.len()).max(SAMPLE_SIDE.min(budget)));
            let s2 = g_picked.len().min(budget / s1);
            let mut found = 0u64;
            for a in spread(f_picked, s1) {
                let p = pairs[picks[a]];
                let (lrow, rrow) = (left.row(p.left.0), right.row(p.right.0));
                for b in spread(g_picked.len(), s2) {
                    let q = g_picked[b];
                    let live = (!left.kills() || left.get_in_row(node_idx, lrow, q.left.0) != NO_PRODUCT)
                        && (!right.kills() || right.get_in_row(node_idx, rrow, q.right.0) != NO_PRODUCT);
                    found += u64::from(live);
                }
            }
            s.live += found * held / (s1 * s2) as u64;
            s.live_cells += u64::from(found > 0);
        }
        let (candidates, live) = (s.candidates - row.candidates, s.live - row.live);
        if live * LIVE_SHARE > candidates {
            let w = &mut s.walked;
            w.candidates += candidates;
            w.live += live;
            w.cells += s.cells - row.cells;
            w.live_cells += s.live_cells - row.live_cells;
        }
    }
    Ok(s)
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
/// bit). A row counts its live candidates off its pairs' vectors, and
/// walks its cells as the row loop does where many of them are live
/// ([`LIVE_SHARE`]). Otherwise it walks its live cells alone where many of
/// their candidates are live ([`CELL_SHARE`]). Elsewhere it marks, for
/// each live cell, the f pairs with a live candidate there, a bit an f
/// pair, and writes each live cell off its bits: f pair by f pair, each
/// one's live g pairs in the cell read off its vectors again (a column of
/// one pair needs no reading). A row whose bits would take many words
/// against its live candidates ([`BITS_SHARE`]) sorts them by column
/// instead and writes its live cells off the sorted list. A level whose
/// rows hold mostly live candidates walks the rest of them
/// ([`PROBATION`]).
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
    let mut a = ByWords::new(eng, rows, &mut *scratch.g_pairs, words, f_rows, (left.kills(), right.kills()), sink)?;
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
        match a.take_row(i, f_pairs, slab, left, right)? {
            RowMode::Bits => {
                for k in 0..a.w.live_cols.len() {
                    let j = a.w.live_cols[k] as usize;
                    a.emit_bits(eng, i, j, row_base + j, f_pairs, ctx, slab, left, right)?;
                }
                poll.poll(candidates + width as u64)?;
            }
            RowMode::Sorted => {
                while let Some(j) = a.next_column() {
                    let (from, to) = a.take_column(j);
                    a.emit_sorted(eng, i, j, row_base + j, f_pairs, ctx, slab, left, right, from, to)?;
                }
                poll.poll(candidates + width as u64)?;
            }
            RowMode::Cells => {
                // The walks charge their cells; the row charges the rest.
                let mut walked = 0u64;
                for k in 0..a.w.live_cols.len() {
                    let j = a.w.live_cols[k] as usize;
                    walked += f_pairs.len() as u64 * u64::from(a.w.col_start[j + 1] - a.w.col_start[j]);
                    process_cell(eng, j, row_base, f_pairs, ctx, g_level, g_scratch, slab, left, right, &mut a.sink, &mut poll)?;
                }
                poll.poll(candidates - walked + width as u64)?;
            }
            RowMode::Walk => {
                for j in 0..width {
                    process_cell(eng, j, row_base, f_pairs, ctx, g_level, g_scratch, slab, left, right, &mut a.sink, &mut poll)?;
                }
                poll.poll(width as u64)?;
            }
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
    /// Words a vector, and words a vector's summary.
    words: usize,
    sum_words: usize,
    /// The candidates of the rows counted so far, and the live ones;
    /// whether the level still counts its rows.
    seen: u64,
    live: u64,
    on: bool,
    /// The current row, and how it is read.
    row: usize,
    mode: RowMode,
    /// Words a live cell's f-pair bits on the current row; the live cells
    /// written so far on a row read off its bits; the next of the current
    /// row's sorted candidates a cell takes; the row `w.runs_of` was read
    /// for.
    cell_words: usize,
    written: usize,
    cursor: usize,
    runs_row: usize,
}

/// How a row of a level read by vectors is read: by its cells' walks; by
/// the walks of its live cells alone, the others left dead; off its live
/// cells' f-pair bits; or off its live candidates sorted by column.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RowMode {
    Walk,
    Cells,
    Bits,
    Sorted,
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

/// The words of the live candidates of an f pair whose vectors are in
/// `slots` that are not zero, each with its index, in order: those its
/// vectors' summaries leave.
#[inline(always)]
fn live_words(
    w: &WordScratch,
    (words, sum_words): (usize, usize),
    kills: (bool, bool),
    slots: (u32, u32),
    mut each: impl FnMut(usize, u64),
) {
    let at = |v: &[u64], s: u32, k: usize| v[s as usize * sum_words + k];
    for k in 0..sum_words {
        let mut sum = match kills {
            (true, true) => at(&w.summaries.left, slots.0, k) & at(&w.summaries.right, slots.1, k),
            (true, false) => at(&w.summaries.left, slots.0, k),
            (false, true) => at(&w.summaries.right, slots.1, k),
            (false, false) => unreachable!("a level of two complete sides kills no candidate"),
        };
        while sum != 0 {
            let word = k * 64 + sum.trailing_zeros() as usize;
            sum &= sum - 1;
            let m = anded(w, words, kills, slots, word);
            if m != 0 {
                each(word, m);
            }
        }
    }
}

/// [`live_words`] within g pairs `start..end`, at least one: the words
/// of the live candidates there of an f pair whose vectors are in `slots`,
/// each cut to the range, that are not zero, with their indices, in
/// order. A range of a few words reads them; a longer one reads the
/// words the summaries leave.
#[inline(always)]
fn live_words_between(
    w: &WordScratch,
    (words, sum_words): (usize, usize),
    kills: (bool, bool),
    slots: (u32, u32),
    (start, end): (usize, usize),
    mut each: impl FnMut(usize, u64),
) {
    let (w0, w1) = (start / 64, (end - 1) / 64);
    // Bits `lo..hi` of a word, `lo < hi <= 64`.
    let span = |lo: usize, hi: usize| (u64::MAX >> (64 - (hi - lo))) << lo;
    let cut = |word: usize| span(if word == w0 { start % 64 } else { 0 }, if word == w1 { end - word * 64 } else { 64 });
    if w1 - w0 < 4 {
        for word in w0..=w1 {
            let m = anded(w, words, kills, slots, word) & cut(word);
            if m != 0 {
                each(word, m);
            }
        }
        return;
    }
    let at = |v: &[u64], s: u32, k: usize| v[s as usize * sum_words + k];
    for k in w0 / 64..=w1 / 64 {
        let mut sum = match kills {
            (true, true) => at(&w.summaries.left, slots.0, k) & at(&w.summaries.right, slots.1, k),
            (true, false) => at(&w.summaries.left, slots.0, k),
            (false, true) => at(&w.summaries.right, slots.1, k),
            (false, false) => unreachable!("a level of two complete sides kills no candidate"),
        };
        sum &= span(if k == w0 / 64 { w0 % 64 } else { 0 }, if k == w1 / 64 { w1 % 64 + 1 } else { 64 });
        while sum != 0 {
            let word = k * 64 + sum.trailing_zeros() as usize;
            sum &= sum - 1;
            let m = anded(w, words, kills, slots, word) & cut(word);
            if m != 0 {
                each(word, m);
            }
        }
    }
}

impl<'w, S: PairSink> ByWords<'w, S> {
    /// Read g's pairs in column order, sort them by child on each side
    /// that kills, and clear the vectors.
    fn new(
        eng: &'w Engine,
        rows: RowLoop<'_>,
        decode: &mut Vec<ChildPair>,
        w: &'w mut WordScratch,
        f_rows: Sides<usize>,
        kills: (bool, bool),
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
        let strides = (ctx.sides.left.stride as usize, ctx.sides.right.stride as usize);
        if kills.0 {
            by_child(lim, &w.g_flat, strides.0, |q| q.left.0, &mut w.child_counts, &mut w.named.left, &mut w.named_start.left, &mut w.by_child.left)?;
        }
        if kills.1 {
            by_child(lim, &w.g_flat, strides.1, |q| q.right.0, &mut w.child_counts, &mut w.named.right, &mut w.named_start.right, &mut w.by_child.right)?;
        }
        w.vectors.left.clear();
        w.vectors.right.clear();
        w.summaries.left.clear();
        w.summaries.right.clear();
        w.col_slot.clear();
        lim.reserve_exact(&mut w.col_slot, width)?;
        w.col_slot.resize(width, 0);
        w.counts.clear();
        lim.reserve_exact(&mut w.counts, width)?;
        w.counts.resize(width, 0);
        w.col_live.clear();
        lim.reserve_exact(&mut w.col_live, width.div_ceil(64))?;
        w.col_live.resize(width.div_ceil(64), 0);
        w.live_cols.clear();
        lim.reserve_exact(&mut w.live_cols, width)?;
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
        let sum_words = words.div_ceil(64);
        w.union.clear();
        lim.reserve_exact(&mut w.union, words)?;
        w.union.resize(words, 0);
        w.union_sum.clear();
        lim.reserve_exact(&mut w.union_sum, sum_words)?;
        w.union_sum.resize(sum_words, 0);
        Ok(ByWords { sink, w, lim, words, sum_words, seen: 0, live: 0, on: true, row: usize::MAX, mode: RowMode::Walk, cell_words: 0, written: 0, cursor: 0, runs_row: usize::MAX })
    }

    /// Every live cell of the last row read by vectors was in a cell the
    /// loop entered; clear the row's live columns.
    fn finish(&mut self) {
        debug_assert!(self.mode != RowMode::Bits || self.written == self.w.live_cols.len(), "a live cell the row loop did not enter");
        debug_assert!(self.mode != RowMode::Sorted || self.cursor == self.w.sorted.len(), "a live candidate in a cell the row loop did not enter");
        if self.mode != RowMode::Walk {
            for &j in &self.w.live_cols {
                self.w.col_live[j as usize / 64] = 0;
            }
        }
    }

    /// Count row `i`, of pairs `f_pairs`, and choose how to read it: by
    /// its cells' walks where many of its candidates are live; otherwise
    /// by the walks of its live cells where many of their candidates are
    /// live, and where few are off its live cells' f-pair bits, or off its
    /// sorted live candidates where the bits would take many words more.
    #[inline(always)]
    fn take_row<L: ChildLookup, R: ChildLookup>(
        &mut self,
        i: usize,
        f_pairs: &[ChildPair],
        node_idx: &[u32],
        left: &L,
        right: &R,
    ) -> Result<RowMode, OperationError> {
        self.finish();
        self.row = i;
        self.mode = RowMode::Walk;
        if !self.on {
            return Ok(RowMode::Walk);
        }
        let candidates = f_pairs.len() as u64 * self.w.g_flat.len() as u64;
        let live = self.count_row(f_pairs, node_idx, left, right)?;
        self.seen += candidates;
        self.live += live;
        if self.seen >= PROBATION && self.live * LIVE_SHARE > self.seen {
            self.on = false;
        }
        if live * LIVE_SHARE > candidates {
            return Ok(RowMode::Walk);
        }
        super::super::note_word_cells(super::super::WordNote::Row);
        let in_live_cells = self.live_cells(f_pairs.len());
        if live * CELL_SHARE > in_live_cells {
            super::super::note_word_cells(super::super::WordNote::Cells);
            self.mode = RowMode::Cells;
            return Ok(RowMode::Cells);
        }
        let bits = self.w.live_cols.len().saturating_mul(f_pairs.len().div_ceil(64));
        let bits_pay = bits <= CELL_BITS_MAX_WORDS && bits as u64 <= live * BITS_SHARE;
        self.mode = if bits_pay && !super::super::word_rows_sorted_forced() {
            self.cell_bits(f_pairs.len(), left.kills(), right.kills())?;
            RowMode::Bits
        } else {
            super::super::note_word_cells(super::super::WordNote::Sorted);
            self.read_row(f_pairs.len(), left.kills(), right.kills())?;
            RowMode::Sorted
        };
        Ok(self.mode)
    }

    /// The columns of the live cells of the row of `n1` pairs that
    /// [`count_row`](Self::count_row) counted last, into `w.live_cols` and
    /// `w.col_live`: the candidates of those cells.
    #[inline(always)]
    fn live_cells(&mut self, n1: usize) -> u64 {
        let w = &mut *self.w;
        w.live_cols.clear();
        let mut candidates = 0u64;
        // The first g pair of a column not yet taken.
        let mut next = 0usize;
        for k in 0..w.union_sum.len() {
            let mut sum = w.union_sum[k];
            while sum != 0 {
                let word = k * 64 + sum.trailing_zeros() as usize;
                sum &= sum - 1;
                if (word + 1) * 64 <= next {
                    continue;
                }
                let mut m = w.union[word];
                if word * 64 < next {
                    m &= !((1u64 << (next - word * 64)) - 1);
                }
                while m != 0 {
                    let q = word * 64 + m.trailing_zeros() as usize;
                    let j = w.col_of[q] as usize;
                    // `new` reserved a column a live cell.
                    w.live_cols.push(j as u32);
                    w.col_live[j / 64] |= 1 << (j % 64);
                    next = w.col_start[j + 1] as usize;
                    candidates += (next - w.col_start[j] as usize) as u64;
                    if next >= (word + 1) * 64 {
                        break;
                    }
                    m &= !((1u64 << (next - word * 64)) - 1);
                }
            }
        }
        candidates * n1 as u64
    }

    /// The slot of child row `r`'s vector on one side, built on first use:
    /// bit `q` where the child cell at `r` and g pair `q`'s child on that
    /// side holds a product, and its summary. The build reads the row's
    /// cell at each child some g pair names (`named`, ascending) and sets
    /// the bits of the pairs naming a live one (`by_child` from
    /// `named_start`).
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn slot<K: ChildLookup>(
        lim: &Limits,
        slots: &mut [u32],
        vectors: &mut Vec<u64>,
        summaries: &mut Vec<u64>,
        named: &[u32],
        named_start: &[u32],
        by_child: &[u32],
        words: usize,
        lookup: &K,
        node_idx: &[u32],
        r: u32,
    ) -> Result<u32, OperationError> {
        let s = slots[r as usize];
        if s != UNBUILT {
            return Ok(s);
        }
        let base = vectors.len();
        slots[r as usize] = (base / words) as u32;
        lim.reserve(vectors, words)?;
        vectors.resize(base + words, 0);
        let v = &mut vectors[base..];
        let row = lookup.row(r);
        for (k, &c) in named.iter().enumerate() {
            if lookup.get_in_row(node_idx, row, c) != NO_PRODUCT {
                for &q in &by_child[named_start[k] as usize..named_start[k + 1] as usize] {
                    v[q as usize / 64] |= 1 << (q % 64);
                }
            }
        }
        let sum_base = summaries.len();
        lim.reserve(summaries, words.div_ceil(64))?;
        summaries.resize(sum_base + words.div_ceil(64), 0);
        for (word, &x) in v.iter().enumerate() {
            summaries[sum_base + word / 64] |= u64::from(x != 0) << (word % 64);
        }
        Ok(slots[r as usize])
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
        let (lim, words, kills) = (self.lim, (self.words, self.sum_words), (left.kills(), right.kills()));
        let w = &mut *self.w;
        w.row_slots.clear();
        lim.reserve(&mut w.row_slots, f_pairs.len())?;
        for k in 0..w.union_sum.len() {
            let mut sum = std::mem::take(&mut w.union_sum[k]);
            while sum != 0 {
                w.union[k * 64 + sum.trailing_zeros() as usize] = 0;
                sum &= sum - 1;
            }
        }
        let mut live = 0u64;
        for p in f_pairs {
            let ls = if kills.0 {
                Self::slot(lim, &mut w.slots.left, &mut w.vectors.left, &mut w.summaries.left, &w.named.left, &w.named_start.left, &w.by_child.left, words.0, left, node_idx, p.left.0)?
            } else {
                0
            };
            let rs = if kills.1 {
                Self::slot(lim, &mut w.slots.right, &mut w.vectors.right, &mut w.summaries.right, &w.named.right, &w.named_start.right, &w.by_child.right, words.0, right, node_idx, p.right.0)?
            } else {
                0
            };
            w.row_slots.push((ls, rs));
            let (mut union, mut union_sum) = (std::mem::take(&mut w.union), std::mem::take(&mut w.union_sum));
            live_words(w, words, kills, (ls, rs), |word, m| {
                live += u64::from(m.count_ones());
                union[word] |= m;
                union_sum[word / 64] |= 1 << (word % 64);
            });
            (w.union, w.union_sum) = (union, union_sum);
        }
        Ok(live)
    }

    /// The live candidates of the row of `n1` pairs that
    /// [`count_row`](Self::count_row) counted last, into `w.sorted` by
    /// column.
    #[inline(always)]
    fn read_row(&mut self, n1: usize, left_kills: bool, right_kills: bool) -> Result<(), OperationError> {
        let (lim, words, kills) = (self.lim, (self.words, self.sum_words), (left_kills, right_kills));
        let w = &mut *self.w;
        debug_assert_eq!(w.row_slots.len(), n1, "a row is read after it is counted");
        let mut entries = std::mem::take(&mut w.entries);
        entries.clear();
        let mut grown = Ok(());
        for k in 0..n1 {
            let slots = w.row_slots[k];
            let key = (k as u64) << 32;
            live_words(w, words, kills, slots, |word, mut m| {
                while m != 0 && grown.is_ok() {
                    let q = (word * 64) as u64 + u64::from(m.trailing_zeros());
                    m &= m - 1;
                    if entries.len() == entries.capacity() {
                        grown = lim.reserve(&mut entries, 1);
                        if grown.is_err() {
                            break;
                        }
                    }
                    entries.push(key | q);
                }
            });
        }
        w.entries = entries;
        grown?;
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

    /// The f-pair bits of the live cells of the row of `n1` pairs that
    /// [`count_row`](Self::count_row) counted last, into `w.cell_bits`: an
    /// f pair's bit is set in each live cell where one of its live words
    /// has a bit of the cell's column.
    #[inline(always)]
    fn cell_bits(&mut self, n1: usize, left_kills: bool, right_kills: bool) -> Result<(), OperationError> {
        let (words, kills) = ((self.words, self.sum_words), (left_kills, right_kills));
        let w = &mut *self.w;
        debug_assert_eq!(w.row_slots.len(), n1, "a row is read after it is counted");
        let cell_words = n1.div_ceil(64);
        for (s, &j) in w.live_cols.iter().enumerate() {
            w.col_slot[j as usize] = s as u32;
        }
        let mut bits = std::mem::take(&mut w.cell_bits);
        bits.clear();
        let n = w.live_cols.len() * cell_words;
        let grown = self.lim.reserve_exact(&mut bits, n);
        if grown.is_ok() {
            bits.resize(n, 0);
            for k in 0..n1 {
                let (at, bit) = (k / 64, 1u64 << (k % 64));
                live_words(w, words, kills, w.row_slots[k], |word, mut m| {
                    while m != 0 {
                        let j = w.col_of[word * 64 + m.trailing_zeros() as usize] as usize;
                        bits[w.col_slot[j] as usize * cell_words + at] |= bit;
                        // Past the column's other pairs in the word.
                        let end = w.col_start[j + 1] as usize - word * 64;
                        if end >= 64 {
                            break;
                        }
                        m &= !((1u64 << end) - 1);
                    }
                });
            }
        }
        w.cell_bits = bits;
        grown?;
        self.cell_words = cell_words;
        self.written = 0;
        Ok(())
    }

    /// Write the live cell at `grid_pos`, row `i` and column `j`, off its
    /// f-pair bits: f pair by f pair, each one's live g pairs in the cell
    /// in order, read off its vectors where the column has more than one
    /// pair ([`write_cell`](Self::write_cell)).
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn emit_bits<L: ChildLookup, R: ChildLookup>(
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
    ) -> Result<(), OperationError> {
        self.written += 1;
        let (lim, words, kills) = (self.lim, (self.words, self.sum_words), (left.kills(), right.kills()));
        let w = &mut *self.w;
        let (start, end) = (w.col_start[j] as usize, w.col_start[j + 1] as usize);
        let s = w.col_slot[j] as usize;
        let bits = &w.cell_bits[s * self.cell_words..(s + 1) * self.cell_words];
        let mut entries = std::mem::take(&mut w.entries);
        entries.clear();
        let mut grown = Ok(());
        if end - start == 1 {
            // The column's one g pair against each f pair whose bit is set.
            grown = lim.reserve(&mut entries, bits.iter().map(|b| b.count_ones() as usize).sum());
            if grown.is_ok() {
                for (at, &b) in bits.iter().enumerate() {
                    let mut b = b;
                    while b != 0 {
                        entries.push(((at * 64) as u64 + u64::from(b.trailing_zeros())) << 32 | start as u64);
                        b &= b - 1;
                    }
                }
            }
        } else {
            for (at, &b) in bits.iter().enumerate() {
                let mut b = b;
                while b != 0 && grown.is_ok() {
                    let k = at * 64 + b.trailing_zeros() as usize;
                    b &= b - 1;
                    let key = (k as u64) << 32;
                    live_words_between(w, words, kills, w.row_slots[k], (start, end), |word, mut m| {
                        let n = m.count_ones() as usize;
                        if entries.capacity() - entries.len() < n {
                            grown = lim.reserve(&mut entries, n);
                            if grown.is_err() {
                                return;
                            }
                        }
                        while m != 0 {
                            entries.push(key | (word * 64) as u64 | u64::from(m.trailing_zeros()));
                            m &= m - 1;
                        }
                    });
                }
            }
        }
        w.entries = entries;
        grown?;
        self.write_cell(eng, i, j, grid_pos, f_pairs, ctx, node_idx, left, right, None)
    }

    /// Write the cell at `grid_pos`, row `i` and column `j`, of the sorted
    /// live candidates `w.sorted[from..to]`, at least one
    /// ([`write_cell`](Self::write_cell)).
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn emit_sorted<L: ChildLookup, R: ChildLookup>(
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
        self.write_cell(eng, i, j, grid_pos, f_pairs, ctx, node_idx, left, right, Some((from, to)))
    }

    /// Write the cell at `grid_pos`, row `i` and column `j`, of its live
    /// candidates, `w.sorted[from..to]` where `sorted` names them and
    /// `w.entries` otherwise, in the order the cell's walk writes them.
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn write_cell<L: ChildLookup, R: ChildLookup>(
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
        sorted: Option<(usize, usize)>,
    ) -> Result<(), OperationError> {
        let w = &mut *self.w;
        let (n1, n2) = (f_pairs.len(), (w.col_start[j + 1] - w.col_start[j]) as usize);
        let cell = match sorted {
            Some((from, to)) => &w.sorted[from..to],
            None => &w.entries[..],
        };
        let q_of = |e: u64| (e & 0xffff_ffff) as usize;
        let product = |e: u64, node_idx: &[u32]| {
            let (p, q) = (&f_pairs[(e >> 32) as usize], &w.g_flat[q_of(e)]);
            let lc = left.get(node_idx, p.left.0, q.left.0);
            let rc = right.get(node_idx, p.right.0, q.right.0);
            debug_assert!(lc != NO_PRODUCT && rc != NO_PRODUCT, "a candidate read live is live");
            (lc, rc)
        };
        if n1 == 1 && n2 == 1 {
            let (lc, rc) = product(cell[0], node_idx);
            return self.sink.single(eng, node_idx, grid_pos, lc, rc);
        }
        let begin = self.sink.begin();
        let runs = if n1 >= GROUPED_MIN_PAIRS && n2 > 1 { ctx.right_cols.and_then(|cols| cols.runs(j)) } else { None };
        match runs {
            None => {
                let node_idx = &*node_idx;
                push_kept(eng, &mut self.sink, cell, false, |&e| product(e, node_idx))?;
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
                order_grouped(self.lim, cell, &w.runs_of, &w.run_of, &mut w.run_counts[..runs.len()], &mut w.grouped)?;
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

    /// A cell the masks leave: on a row read by vectors, after the charge
    /// of the cell's walk, a unit a candidate, the cell off its f-pair bits
    /// or its sorted live candidates where it is live; on any other row,
    /// the cell's walk.
    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        if a.i != self.row {
            self.take_row(a.i, a.f_pairs, a.node_idx, a.left, a.right)?;
        }
        let walk = match self.mode {
            RowMode::Walk => true,
            RowMode::Cells => self.w.col_live[a.j / 64] >> (a.j % 64) & 1 != 0,
            RowMode::Bits | RowMode::Sorted => false,
        };
        if walk {
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
        match self.mode {
            RowMode::Bits if self.w.col_live[a.j / 64] >> (a.j % 64) & 1 != 0 => {
                self.emit_bits(eng, a.i, a.j, a.row_base + a.j, a.f_pairs, a.ctx, a.node_idx, a.left, a.right)
            }
            RowMode::Sorted => {
                let (from, to) = self.take_column(a.j);
                if from == to {
                    return Ok(());
                }
                self.emit_sorted(eng, a.i, a.j, a.row_base + a.j, a.f_pairs, a.ctx, a.node_idx, a.left, a.right, from, to)
            }
            // A dead cell.
            _ => Ok(()),
        }
    }
}

/// g's pairs `g_flat` by their child on one side, `field`, below
/// `width`: the children named, ascending, into `named`; where each one's
/// pairs start, and one past the last, into `named_start`; the pairs, by
/// child and then in order, into `by_child`. A counting sort, `counts` a
/// child, where the children are not many more than the pairs; a
/// comparison sort otherwise.
#[expect(clippy::too_many_arguments)]
fn by_child(
    lim: &Limits,
    g_flat: &[ChildPair],
    width: usize,
    field: impl Fn(&ChildPair) -> u32,
    counts: &mut Vec<u32>,
    named: &mut Vec<u32>,
    named_start: &mut Vec<u32>,
    by_child: &mut Vec<u32>,
) -> Result<(), OperationError> {
    named.clear();
    named_start.clear();
    by_child.clear();
    lim.reserve_exact(by_child, g_flat.len())?;
    if width <= 4 * g_flat.len() + 4096 {
        counts.clear();
        lim.reserve_exact(counts, width)?;
        counts.resize(width, 0);
        for q in g_flat {
            counts[field(q) as usize] += 1;
        }
        let mut at = 0u32;
        for (c, n) in counts.iter_mut().enumerate() {
            if *n > 0 {
                lim.try_push(named, c as u32)?;
                lim.try_push(named_start, at)?;
                let k = *n;
                *n = at;
                at += k;
            }
        }
        lim.try_push(named_start, at)?;
        by_child.resize(g_flat.len(), 0);
        for (q, p) in g_flat.iter().enumerate() {
            let n = &mut counts[field(p) as usize];
            by_child[*n as usize] = q as u32;
            *n += 1;
        }
    } else {
        by_child.extend(0..g_flat.len() as u32);
        by_child.sort_by_key(|&q| field(&g_flat[q as usize]));
        for (k, &q) in by_child.iter().enumerate() {
            let c = field(&g_flat[q as usize]);
            if named.last() != Some(&c) {
                lim.try_push(named, c)?;
                lim.try_push(named_start, k as u32)?;
            }
        }
        lim.try_push(named_start, g_flat.len() as u32)?;
    }
    Ok(())
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
