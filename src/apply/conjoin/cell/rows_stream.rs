//! The streaming collapse route: a level whose per-cell result is a count or a
//! weight rather than a set of emitted pairs.

use super::rows::{CellAction, CellArgs, run_level_rows};
use super::*;

/// Per-cell scalar fold for the streaming collapse walker
/// ([`stream_collapse_rows`]): resolves one alive cell's collected pairs to a
/// single scalar and records it in the streaming state, remapping
/// `node_idx[grid_pos]` from DEAD to the new slot index. ONE impl, generic
/// over the value kind, so the ONE row/cell loop serves both the integer count
/// fold and the weighted (`BigRational`) fold (D2 stage 1).
pub(crate) trait StreamCellFold {
    fn fold_cell(
        &mut self,
        eng: &Engine,
        pairs: &[InputPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), ApplyError>;
}

/// The single source of truth for the fold / column-push / `node_idx` remap
/// step, for both value kinds.
///
/// Growth past the output column's initial `k1.max(k2)` reserve must stay
/// fallible — the column can grow up to alive cells (≤ k1*k2), well past the
/// upfront reserve. The push discipline is the value kind's: `CountVec::push`
/// stores a `Count::Big` as the `STREAM_OVERFLOW` sentinel with the exact
/// `BigUint` in the lazily-built, `None`-backfilled side table; the weighted
/// column is an ordinary `Vec` whose per-pair transient is budget-charged by
/// [`CollectSink`] instead.
impl<F: StreamPayload> StreamCellFold for StreamState<'_, F> {
    #[inline(always)]
    fn fold_cell(
        &mut self,
        eng: &Engine,
        pairs: &[InputPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), ApplyError> {
        let v = F::fold_cell(pairs, &self.left, &self.right, self.ws);
        let cell_idx = F::col_len::<ApplyBudget>(self.counts);
        F::push_col::<ApplyBudget>(eng, self.counts, v)?;
        node_idx[grid_pos] = cell_idx as u32;
        Ok(())
    }
}

/// Streaming collapse-at-source entry — the ONE driver for streaming-
/// marginalize levels, generic over the child lookups:
///
/// - Marginal-child shapes (Route A): at least one child marginal
///   (`MargLookup` sides — a `MargLookup` degrades to the plain dense grid
///   read on a non-pass-through side, so both-marginal, and
///   one-marginal × leaf all route here with the same lookups). The level is
///   a marginalize target whose every alive cell collapses to a scalar
///   `Σ left × right` — there is no downstream structure to keep.
/// - Plain shape (Route B): a streaming target with no marginal child
///   (leaf children at the lowest levels), served by `DenseLookup` sides.
///
/// Monomorphizes the per-cell fold once per level on the state's value kind —
/// integer or weighted — binds that kind's two child column VIEWS
/// ([`attach_children`]; the columns are read in place in `left_level` /
/// `right_level`, never copied), then runs the shared [`stream_collapse_rows`]
/// loop. Which side is marginal is carried by the views themselves
/// (`StreamChild::is_marg`), so the fold needs no shape-specific wiring.
///
/// The views live only for this call: `stream_state` owns the output column and
/// outlives them, so the caller can retake `&mut levels` to commit it.
///
/// This is the ONLY streaming build path — there is no materialize-then-fold
/// alternative to fall back on: [`bothmarg_collapse_enabled`] disables
/// streaming *eligibility* rather than switching routes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_level_rows_stream_count<L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    stream_state: &mut StreamLevelState,
    left_idx: usize,
    right_idx: usize,
    vtree: &crate::vtree::Vtree,
    left_level: &TddLevel,
    right_level: &TddLevel,
    computed: &[Option<CountVec<ApplyBudget>>],
    computed_weights: &[Option<Vec<WeightVal>>],
    ws: Option<&crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    match stream_state {
        StreamLevelState::Weighted(counts) => {
            let mut st = attach_children::<WeightFold>(
                eng,
                left_idx,
                right_idx,
                vtree,
                left_level,
                right_level,
                computed_weights,
                counts,
                ws,
            )?;
            stream_collapse_rows(
                eng,
                k1,
                c1_level_t,
                c2_level_t,
                cell_ctx,
                inputs1_scratch,
                inputs2_scratch,
                node_idx,
                left,
                right,
                &mut st,
            )
        }
        StreamLevelState::Int(counts) => {
            let mut st = attach_children::<IntFold>(
                eng,
                left_idx,
                right_idx,
                vtree,
                left_level,
                right_level,
                computed,
                counts,
                None,
            )?;
            stream_collapse_rows(
                eng,
                k1,
                c1_level_t,
                c2_level_t,
                cell_ctx,
                inputs1_scratch,
                inputs2_scratch,
                node_idx,
                left,
                right,
                &mut st,
            )
        }
    }
}

/// Collapse-at-source action: enumerate each alive cell's surviving `(lc, rc)`
/// refs into a reused scratch `Vec<InputPair>` and feed them straight to the
/// fold, never touching `level`.
struct StreamCollapse<'a, F> {
    fold: &'a mut F,
    /// Reused across all cells — bounds the transient peak to one cell's pairs.
    cell_pairs: Vec<InputPair>,
}

impl<L: ChildLookup, R: ChildLookup, F: StreamCellFold> CellAction<L, R> for StreamCollapse<'_, F> {
    /// Collapse walks may visit marginal-encoded operand nodes.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row (the collapsed scalars live in the
    /// streaming column; `node_idx` still carries this level's cell→slot map).
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        self.cell_pairs.clear();
        process_cell::<_, _, _>(
            eng,
            a.j,
            a.row_base,
            a.inputs1,
            a.left_alive_mask,
            a.right_alive_mask,
            a.ctx,
            a.c2_level_t,
            a.inputs2_scratch,
            a.node_idx,
            a.left,
            a.right,
            &mut CollectSink {
                out: &mut self.cell_pairs,
            },
            a.gate,
        )?;
        // An empty cell stays DEAD (no slot) — mirrors the emit walk, where
        // `emit_product_node` produces no node for zero pairs. Same `row_base + j`
        // the kernel used, not a second derivation of it.
        if !self.cell_pairs.is_empty() {
            self.fold
                .fold_cell(eng, &self.cell_pairs, a.node_idx, a.row_base + a.j)?;
        }
        Ok(())
    }
}

/// The collapse route's entry into the shared row loop (see
/// [`run_level_rows_stream_count`] for the route/shape documentation).
///
/// Count-identical to the materializing emit walk by construction — same
/// per-cell pair multiset (the kernel IS the emit walk minus node
/// materialization; its row/reach culls prune only provably-dead pairs, and
/// the ≥64×64 grouped N×M path emits the same multiset in a different order
/// under an order-independent fold).
#[allow(clippy::too_many_arguments)]
fn stream_collapse_rows<L: ChildLookup, R: ChildLookup, F: StreamCellFold>(
    eng: &Engine,
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    fold: &mut F,
) -> Result<(), ApplyError> {
    // A4: the per-cell scratch is pooled, not rebuilt from empty at every
    // streaming level — `cell` clears it before each cell, so pooled capacity can
    // carry nothing but capacity. Returned on the error path too, under the
    // module's byte cap, so one huge level can't park its arena in the pool.
    let mut action = StreamCollapse {
        fold,
        cell_pairs: pool_take(&eng.apply().cell_pairs),
    };
    let result = run_level_rows::<false, _, _, _>(
        eng,
        k1,
        c1_level_t,
        c2_level_t,
        cell_ctx,
        inputs1_scratch,
        inputs2_scratch,
        node_idx,
        left,
        right,
        &mut action,
    );
    pool_put_bounded(
        &eng.apply().cell_pairs,
        std::mem::take(&mut action.cell_pairs),
        MAX_LEVEL_ARENA_BYTES,
    );
    result
}
