//! The row loop every build route shares, and the four routes themselves.

use super::*;

// ============================= Row-loop driver =============================
//
// The four build routes — marginal-child emit (`run_level_rows_marg`),
// sparse-marg emit (`run_level_rows_marg_sparse`), streaming collapse
// (`stream_collapse_rows`) and plain emit (`run_level_rows_plain`) — differ
// only in what they do per row and per cell. Everything around that (the
// per-row DEAD reset, the c1 pair decode, the empty/dead-row skips, the alive
// masks, the k2 column sweep, the between-cell poll) is ONE loop body, living
// once in `run_level_rows` and parameterized by a [`CellAction`] — the same
// shape as `process_cell`, which is one product walk parameterized by a
// [`PairSink`].

/// Everything one cell of the row loop needs, bundled and passed by value.
///
/// A bundle rather than a dozen `cell` parameters: restating the parameters in
/// every impl measured about twice the added source lines at identical codegen.
struct CellArgs<'a, 'c, L, R> {
    /// Column index — the c2 node.
    j: usize,
    /// TRUE c1 row index — NOT necessarily the grid row: the sparse-marg route
    /// builds every row at grid row 0 and needs this as the product entry's
    /// `c1_idx`.
    i: usize,
    /// Flat slab offset of the grid row this cell writes into —
    /// `ctx.t_base + CellAction::grid_row(i) * ctx.k2`, computed ONCE per row by
    /// the driver for its DEAD reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    row_base: usize,
    /// Decoded pairs of c1 row `i` (never empty — empty rows are skipped).
    inputs1: &'a [InputPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &'a CellCtx<'c>,
    c2_level_t: &'a TddLevel,
    inputs2_scratch: &'a mut Vec<InputPair>,
    node_idx: &'a mut [u32],
    left: &'a L,
    right: &'a R,
}

/// Per-row / per-cell action of the shared row loop ([`run_level_rows`]).
///
/// All hooks are `#[inline(always)]` in impls, so each instantiation
/// monomorphizes to what the hand-written per-route loop produced and the hooks
/// three of the four routes leave at their empty defaults vanish entirely.
trait CellAction<L: ChildLookup, R: ChildLookup> {
    /// Whether the driver debug-asserts that each c1 row node is structurally
    /// internal before decoding its pairs. Mirrors [`PairSink::ASSERT_INTERNAL`],
    /// and for the same reason cannot be a shared unconditional assert: the
    /// collapse and marginal-child routes legitimately walk marginal-encoded
    /// operand nodes.
    const ASSERT_INTERNAL: bool;

    /// Whether this action owns a DENSE `k1 × k2` slab — one grid row per c1 row,
    /// so the per-row DEAD resets tile the level's whole slab exactly once and
    /// can be replaced by a single fill (see [`run_level_rows`]). FALSE for
    /// [`SparseMargEmit`], which reuses grid row 0 for EVERY structural row and
    /// therefore must re-fill that one row between rows.
    const DENSE_SLAB: bool;

    /// Output grid row for c1 row `i`. Drives BOTH the per-row DEAD reset and
    /// the kernel's `grid_pos` (through `CellArgs::row_base`), so the two can
    /// never drift. Deliberately has no default — "row `i` of a dense slab" vs
    /// "the one reused row scratch" is exactly the distinction a default would
    /// paper over.
    fn grid_row(&self, i: usize) -> usize;

    /// Fires once per LIVE row, after the empty-row skip and after the alive
    /// masks resolve — the seam for per-row state an action needs pinned before
    /// its first cell but only for rows that are actually built.
    #[inline(always)]
    fn begin_row(&mut self, _i: usize, _inputs1: &[InputPair]) {}

    /// One cell of the row.
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError>;

    /// Fires once after the last row, on the success path only — a failing cell
    /// short-circuits out of the driver, so an action's end-of-level bookkeeping
    /// is skipped for a level the apply is abandoning.
    #[inline(always)]
    fn finish(&mut self, _k1: usize, _ctx: &CellCtx<'_>, _node_idx: &[u32]) {}
}

/// Size gate for [`run_level_rows`]'s one-shot DEAD slab fill (A3). At or below
/// this many cells the whole `k1 × k2` slab is filled once before the row loop;
/// above it the fill stays row-wise so the reset of the row about to be built
/// keeps that row in L1. THE gate constant — defined once, read once.
const DEAD_SLAB_FILL_MAX_CELLS: usize = 1 << 16;

/// The ONE row/cell loop of the dense product build.
///
/// `const DENSE` skips the per-row alive-mask fold on levels where it provably
/// cannot skip anything: `ctx.nxm` false and neither side a pass-through, where
/// `row_alive_masks` always returns `(0, u128::MAX)`. Only the plain route
/// reaches that regime; the other three always fold.
///
/// `k2` and `t_base` are read from `ctx` rather than passed alongside it, so
/// the row reset and the kernel's `grid_pos` derive from the same values by
/// construction.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_level_rows<const DENSE: bool, L, R, A>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    action: &mut A,
) -> Result<(), ApplyError>
where
    L: ChildLookup,
    R: ChildLookup,
    A: CellAction<L, R>,
{
    // Amortized wall-deadline/cancel poll: one TLS read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = crate::limits::PollTicker::new(super::super::budget::DENSE_CELL_POLL_STRIDE);
    let k2 = ctx.k2;

    // A3 — one slab fill instead of `k1` row fills. On a dense-slab action the
    // per-row resets below tile `t_base .. t_base + k1*k2` exactly once each
    // (`grid_row(i) == i`), so hoisting them into a single `fill` writes the same
    // cells with the same byte pattern; only the order changes, and nothing reads
    // this level's slab between rows (the child lookups read the CHILD levels'
    // disjoint slabs). Above the size gate the per-row form stays: interleaving
    // the reset with the row's cell work is what keeps the active row L1-resident
    // on a large grid.
    let slab_fill = A::DENSE_SLAB && k1.saturating_mul(k2) <= DEAD_SLAB_FILL_MAX_CELLS;
    if slab_fill {
        node_idx[ctx.t_base..ctx.t_base + k1 * k2].fill(DEAD);
    }

    for i in 0..k1 {
        let row_base = ctx.t_base + action.grid_row(i) * k2;
        if !slab_fill {
            node_idx[row_base..row_base + k2].fill(DEAD);
        }

        if A::ASSERT_INTERNAL {
            debug_assert!(c1_level_t.nodes[i].is_internal()
                || c1_level_t.nodes[i].b == u32::MAX,
                "expected internal node at internal vtree position: t_base={} i={i} k1={k1} node_a={:#x} node_b={:#x}",
                ctx.t_base, c1_level_t.nodes[i].a, c1_level_t.nodes[i].b);
        }

        let inputs1 = c1_level_t.pairs_view_decoded(
            i, inputs1_scratch, ctx.left_mask, ctx.right_mask,
        );
        // Empty pairs means dead (ZERO-containing) node — skip this row.
        if inputs1.is_empty() { continue; }

        let (left_alive_mask, right_alive_mask) = if DENSE {
            (0u128, u128::MAX)
        } else {
            match row_alive_masks(ctx, inputs1) {
                Some(masks) => masks,
                // Row skip: if c1[i]'s pairs all reference dead child rows,
                // no cell in this row can produce output.
                None => continue,
            }
        };

        action.begin_row(i, inputs1);

        for j in 0..k2 {
            action.cell(CellArgs {
                j, i, row_base, inputs1, left_alive_mask, right_alive_mask,
                ctx, c2_level_t, left, right,
                inputs2_scratch: &mut *inputs2_scratch,
                node_idx: &mut *node_idx,
            })?;
        }
        // One `tick_by(k2)` per row instead of `tick()` per cell: the ticker only
        // meters accumulated work, so the same total is booked either way. A row
        // wider than the stride now polls once rather than once per stride's worth
        // of cells — the poll is an idempotent deadline read, so firing once per
        // crossing is equivalent, and a row that skipped the `j` loop (empty or
        // dead) books nothing, exactly as before (`k2 == 0` books nothing either).
        poll.tick_by(k2 as u64)?;
    }
    action.finish(k1, ctx, node_idx);
    Ok(())
}

/// Route A action: materialize each cell as a product node in the dense
/// `k1 × k2` output slab.
struct MargEmit<'a> {
    level: &'a mut TddLevel,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for MargEmit<'_> {
    /// A marginal-child level's c1 rows may be marginal-encoded.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )
    }
}

/// Route A row-loop: forward-order scatter for levels with at least one marginal child.
///
/// Called when `marg_child_dispatch` is true. Iterates rows 0..k1 in forward order
/// (lever-8 reverse iteration is a no-op on this path — see the per-level comment),
/// running the emit kernel for each (i,j) cell.
///
/// Never streams: streaming marginal-child levels take the collapse-at-source
/// walker ([`run_level_rows_stream_count`]) unconditionally — there is no
/// materialize-then-convert fallback for them.
///
/// Takes `c1_level_t` as a pre-taken immutable borrow into c1.levels[t_idx] so the
/// caller can keep its `vtree = &c1.vtree` borrow live simultaneously.
pub(crate) fn run_level_rows_marg(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
) -> Result<(), ApplyError> {
    let left = MargLookup::left(cell_ctx);
    let right = MargLookup::right(cell_ctx);
    run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        &left, &right,
        &mut MargEmit { level },
    )
}

/// Per-cell scalar fold for the streaming collapse walker
/// ([`stream_collapse_rows`]): resolves one alive cell's collected pairs to a
/// single scalar and records it in the streaming state, remapping
/// `node_idx[grid_pos]` from DEAD to the new slot index. ONE impl, generic
/// over the value kind, so the ONE row/cell loop serves both the integer count
/// fold and the weighted (`BigRational`) fold (D2 stage 1).
pub(crate) trait StreamCellFold {
    fn fold_cell(
        &mut self,
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
        pairs: &[InputPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), ApplyError> {
        let v = F::fold_cell(pairs, &self.left, &self.right, self.ws);
        let cell_idx = F::col_len::<ApplyBudget>(self.counts);
        F::push_col::<ApplyBudget>(self.counts, v)?;
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
                left_idx, right_idx, vtree, left_level, right_level, computed_weights, counts, ws,
            )?;
            stream_collapse_rows(
                k1, c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch, node_idx, left, right, &mut st,
            )
        }
        StreamLevelState::Int(counts) => {
            let mut st = attach_children::<IntFold>(
                left_idx, right_idx, vtree, left_level, right_level, computed, counts, None,
            )?;
            stream_collapse_rows(
                k1, c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch, node_idx, left, right, &mut st,
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

impl<L: ChildLookup, R: ChildLookup, F: StreamCellFold> CellAction<L, R>
    for StreamCollapse<'_, F>
{
    /// Collapse walks may visit marginal-encoded operand nodes.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row (the collapsed scalars live in the
    /// streaming column; `node_idx` still carries this level's cell→slot map).
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        self.cell_pairs.clear();
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut CollectSink { out: &mut self.cell_pairs },
        )?;
        // An empty cell stays DEAD (no slot) — mirrors the emit walk, where
        // `emit_product_node` produces no node for zero pairs. Same `row_base + j`
        // the kernel used, not a second derivation of it.
        if !self.cell_pairs.is_empty() {
            self.fold.fold_cell(&self.cell_pairs, a.node_idx, a.row_base + a.j)?;
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
        cell_pairs: pool_take(&super::super::SCRATCH_CELL_PAIRS),
    };
    let result = run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        left, right,
        &mut action,
    );
    pool_put_bounded(
        &super::super::SCRATCH_CELL_PAIRS,
        std::mem::take(&mut action.cell_pairs),
        MAX_LEVEL_ARENA_BYTES,
    );
    result
}

/// Sparse-output action: emit into the reused row scratch, then record each
/// surviving cell as a `ProductEntry` instead of leaving it in a dense slab.
struct SparseMargEmit<'a> {
    level: &'a mut TddLevel,
    product_list: &'a mut Vec<ProductEntry>,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for SparseMargEmit<'_> {
    /// A marginal-child level's c1 rows may be marginal-encoded.
    const ASSERT_INTERNAL: bool = false;

    /// NOT a dense slab — every structural row is rebuilt in the SAME `k2`-wide
    /// scratch row, so its DEAD reset must fire between rows and cannot be
    /// hoisted into a one-shot slab fill.
    const DENSE_SLAB: bool = false;

    /// ONE reused `k2`-wide row scratch instead of a dense slab: every
    /// structural row builds at grid row 0, so `row_base == ctx.t_base`.
    /// The true row index survives only in `CellArgs::i`, which the product
    /// entry's `c1_idx` reads below.
    #[inline(always)]
    fn grid_row(&self, _i: usize) -> usize { 0 }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        let row_pos = a.row_base + a.j;
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )?;
        let nid = a.node_idx[row_pos];
        if nid != DEAD {
            try_push(self.product_list, ProductEntry {
                c1_idx: C1NodeIdx(a.i as u32),
                c2_idx: C2NodeIdx(a.j as u32),
                prod_idx: ProdNodeIdx(nid),
            })?;
        }
        Ok(())
    }
}

/// Sparse-output variant of Route A for an *exactly-one*-marginal-child level
/// whose output is structural (never a marginalize target).
///
/// Identical per-cell math to [`run_level_rows_marg`] — it runs the same emit
/// kernel — but instead of writing into a dense `k1*k2` slab it reuses a single
/// `k2`-wide row scratch (`cell_ctx.t_base .. +k2`) and records each surviving
/// cell into `product_list`. The marginal child is a pass-through carrier (it
/// never kills a pair), so the *structural* sibling alone governs which cells
/// are alive; the dense slab the other path allocates is therefore mostly DEAD
/// and pure overhead. The grandparent densifies the emitted `product_list`
/// lazily via `ensure_grid`, reproducing exactly the grid the dense path would
/// have built.
///
/// Every cell is built at grid row 0 so the kernel's
/// `grid_pos == cell_ctx.t_base + j` (one row), then read back from the scratch
/// and, if alive, pushed as
/// `ProductEntry { c1_idx: row i, c2_idx: col j, prod_idx: node }`.
///
/// No streaming: the `use_sparse_marg` gate excludes marginalize targets
/// explicitly (`!is_marg_target`), so a streaming target never routes here.
/// (A one-marginal-child marginalize target does exist; it takes the streaming
/// dispatch, not this sparse path.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_level_rows_marg_sparse(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    product_list: &mut Vec<ProductEntry>,
) -> Result<(), ApplyError> {
    let left = MargLookup::left(cell_ctx);
    let right = MargLookup::right(cell_ctx);
    run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        &left, &right,
        &mut SparseMargEmit { level, product_list },
    )
}

/// Route B action: materializing emit.
struct PlainEmit<'a> {
    level: &'a mut TddLevel,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for PlainEmit<'_> {
    /// Route B's operand invariant: no marginal child, so every c1 row node at
    /// an internal vtree position is structurally internal.
    const ASSERT_INTERNAL: bool = true;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )
    }
}

/// Route B row-loop: forward pass for plain (non-marginal-child) levels.
///
/// Called when `marg_child_dispatch` is false. Iterates rows in forward order.
/// Runs the emit kernel with the caller's plain lookups (no pass-through, no
/// mask decode — the caller guarantees no marginal child).
///
/// `const DENSE: bool` selects the branch-hoisted fast path for the common case where
/// both level-invariant guards hold simultaneously:
///   1. `cell_ctx.nxm == false` — no liveness-mask filtering.
///   2. `!cell_ctx.left_passthrough && !cell_ctx.right_passthrough` — no pass-through sides.
///
/// When `DENSE = true` the inner loop is free of branches on those constants;
/// when `DENSE = false`, nxm row-skip checks are active.
///
/// Never streams: streaming levels take the collapse-at-source walker
/// ([`run_level_rows_stream_count`]) unconditionally — there is no post-cell
/// snapshot conversion.
#[inline(always)]
pub(crate) fn run_level_rows_plain<const DENSE: bool, L: ChildLookup, R: ChildLookup>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    left_lookup: &L,
    right_lookup: &R,
) -> Result<(), ApplyError> {
    // When DENSE, the alive masks are constants (nxm is false, no pass-through):
    //   left_alive_mask  = 0u128      (the !nxm branch of `row_alive_masks`)
    //   right_alive_mask = u128::MAX  (the `|| !nxm` branch of `row_alive_masks`)
    // The driver passes those directly to the kernel, skipping the fold.
    let mut action = PlainEmit { level };
    run_level_rows::<DENSE, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        left_lookup, right_lookup,
        &mut action,
    )
}
