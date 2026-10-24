//! The streaming collapse route: a level whose per-cell result is a count or a
//! weight rather than a set of emitted pairs.

use super::rows::{Emit, run_level_rows};
use super::word_cells::WordScratch;
use super::*;

impl<F: ValueDomain> StreamState<'_, F> {
    /// Fold one cell into the output column and record its slot in the grid.
    #[inline(always)]
    fn fold_cell(
        &mut self,
        eng: &Engine,
        pairs: &[ChildPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), OperationError> {
        let v = F::fold_cell(pairs, &self.left, &self.right, self.store);
        let cell_idx = F::col_len(self.counts);
        F::push_col(eng, self.counts, v)?;
        node_idx[grid_pos] = cell_idx as u32;
        Ok(())
    }
}

/// Fold a streaming target's cells directly into an integer or weighted column.
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
        StreamLevelState::Int(counts) => {
            stream_level::<IntFold, L, R>(eng, rows, scratch, left, right, counts, env)
        }
    }
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
