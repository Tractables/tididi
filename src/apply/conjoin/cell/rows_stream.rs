//! The streaming collapse route: a level whose per-cell result is a count or a
//! weight rather than a set of emitted pairs.

use super::rows::{CellAction, CellArgs, run_level_rows};
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
    let mut cell_pairs = eng.apply().cell_pairs.checkout(eng.limits());
    let mut action = StreamCollapse { fold: &mut st, cell_pairs: &mut cell_pairs };
    run_level_rows::<false, _, _, _>(eng, rows, scratch, left, right, &mut action)
}

/// Collapse-at-source action: enumerate each alive cell's surviving `(lc, rc)`
/// refs into a reused scratch `Vec<ChildPair>` and feed them straight to the
/// fold, never touching `level`.
struct StreamCollapse<'a, 'data, F: ValueDomain> {
    fold: &'a mut StreamState<'data, F>,
    /// Reused across all cells — bounds the transient peak to one cell's pairs.
    cell_pairs: &'a mut Vec<ChildPair>,
}

impl<L: ChildLookup, R: ChildLookup, F: ValueDomain> CellAction<L, R> for StreamCollapse<'_, '_, F> {
    /// Collapse walks may visit marginal-encoded operand nodes.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per f row (the collapsed scalars live in the
    /// streaming column; `node_idx` still carries this level's cell→slot map).
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        self.cell_pairs.clear();
        process_cell::<_, _, _>(
            eng,
            a.j,
            a.row_base,
            a.inputs1,
            a.ctx,
            a.right_level_t,
            a.inputs2_scratch,
            a.node_idx,
            a.left,
            a.right,
            &mut CollectSink {
                out: self.cell_pairs,
            },
            a.gate,
        )?;
        // An empty cell stays `NO_PRODUCT` (no slot) — mirrors the emit walk, where
        // `emit_product_node` produces no node for zero pairs. Same `row_base + j`
        // the kernel used, not a second derivation of it.
        if !self.cell_pairs.is_empty() {
            self.fold
                .fold_cell(eng, self.cell_pairs, a.node_idx, a.row_base + a.j)?;
        }
        Ok(())
    }
}
