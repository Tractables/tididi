//! The row loop every build route shares, and the four routes themselves.

use super::*;

// ============================= Row-loop driver =============================
//
// The four build routes — marginal-child emit (`run_level_rows_marginal`),
// sparse-marginal emit (`run_level_rows_marginal_sparse`), streaming collapse
// (`run_level_rows_stream_count`) and plain emit (`run_level_rows_plain`) — differ
// only in what they do per row and per cell. Everything around that (the
// per-row `NO_PRODUCT` reset, the f pair decode, the empty/dead-row skips, the alive
// masks, the right_width column sweep, the between-cell poll) is one loop body, living
// once in `run_level_rows` and parameterized by a [`CellAction`] — the same
// shape as `process_cell`, which is one product walk parameterized by a
// [`PairSink`].

/// The per-level references every build route reads, bundled and passed by
/// value.
///
/// Built once per level, after the output level and its two children have been
/// split apart, and threaded down the route chain unchanged.
#[derive(Clone, Copy)]
pub(crate) struct RowLoop<'a> {
    /// `f`'s level at this vtree node: one row per node.
    pub(crate) f_level: &'a TddLevel,
    /// `g`'s level at this vtree node: one column per node.
    pub(crate) g_level: &'a TddLevel,
    /// The output level's two children, read in place by the streaming fold.
    pub(crate) children: Sides<&'a TddLevel>,
    pub(crate) ctx: &'a CellCtx<'a>,
    /// Number of rows — `f`'s width at this vtree node.
    pub(crate) f_width: usize,
}

/// The three buffers the row loop writes through.
///
/// Kept apart from [`RowLoop`] so the shared half stays `Copy`: these are the
/// exclusive borrows, and a route that hands them on moves the bundle.
pub(crate) struct RowScratch<'a> {
    /// Decode buffer for the current row's `f` pairs.
    pub inputs1: &'a mut Vec<ChildPair>,
    /// Decode buffer for the current cell's `g` pairs.
    pub inputs2: &'a mut Vec<ChildPair>,
    /// The product-grid slab.
    pub(crate) node_idx: &'a mut [u32],
}

/// Everything one cell of the row loop needs, bundled and passed by value.
///
/// A bundle rather than a dozen `cell` parameters, which every impl would
/// otherwise have to restate.
pub(super) struct CellArgs<'a, 'c, L, R> {
    /// Column index — the g node.
    pub(super) j: usize,
    /// The f row index itself — not necessarily the grid row: the sparse-marginal route
    /// builds every row at grid row 0 and needs this as the product entry's
    /// `left_idx`.
    pub(super) i: usize,
    /// Flat slab offset of the grid row this cell writes into —
    /// `ctx.output_grid_base + CellAction::grid_row(i) * ctx.right_width`, computed once per row by
    /// the driver for its `NO_PRODUCT` reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    pub(super) row_base: usize,
    /// Decoded pairs of f row `i` (never empty — empty rows are skipped).
    pub(super) inputs1: &'a [ChildPair],
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) right_level_t: &'a TddLevel,
    pub(super) inputs2_scratch: &'a mut Vec<ChildPair>,
    pub(super) node_idx: &'a mut [u32],
    pub(super) left: &'a L,
    pub(super) right: &'a R,
    /// The level's one work-clock gate. Every cell charges the pairs it walks
    /// into it, and the row loop adds one unit per cell; it is flushed when the
    /// level ends. Per-cell gates cannot do this — a gate narrower than its
    /// stride charges nothing at all, which is what a one-sided cell always is.
    pub(super) gate: &'a mut crate::limits::PollGate<'c>,
}

/// Per-row / per-cell action of the shared row loop ([`run_level_rows`]).
///
/// All hooks are `#[inline(always)]` in impls, so each instantiation
/// monomorphizes to what the hand-written per-route loop produced and the hooks
/// three of the four routes leave at their empty defaults vanish entirely.
pub(super) trait CellAction<L: ChildLookup, R: ChildLookup> {
    /// Whether the driver debug-asserts that each f row node is structurally
    /// internal before decoding its pairs. Mirrors [`PairSink::ASSERT_INTERNAL`],
    /// and for the same reason cannot be a shared unconditional assert: the
    /// collapse and marginal-child routes legitimately walk marginal-encoded
    /// operand nodes.
    const ASSERT_INTERNAL: bool;

    /// Whether this action owns a dense `left_width × right_width` slab — one grid row per f row,
    /// so the per-row `NO_PRODUCT` resets tile the level's whole slab exactly once and
    /// can be replaced by a single fill (see [`run_level_rows`]). `false` for
    /// [`SparseMargEmit`], which reuses grid row 0 for every structural row and
    /// therefore must re-fill that one row between rows.
    const DENSE_SLAB: bool;

    /// Output grid row for f row `i`. Drives both the per-row `NO_PRODUCT` reset and
    /// the kernel's `grid_pos` (through `CellArgs::row_base`), so the two can
    /// never drift.
    fn grid_row(&self, i: usize) -> usize;

    /// One cell of the row.
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError>;
}

/// Fill small product grids once before traversal; reset larger grids row by
/// row to keep the current row cache-local. Both paths write the same values.
const DEAD_SLAB_FILL_MAX_CELLS: usize = 1 << 16;

/// The one row/cell loop of the dense product build.
///
/// `const DENSE` skips the per-row alive-mask fold on levels where it provably
/// cannot skip anything: `ctx.both_multi_pair` false and neither side a pass-through, where
/// `row_alive_masks` always returns `(0, u128::MAX)`. Only the plain route
/// reaches that regime; the other three always fold.
///
/// `right_width` and `output_grid_base` are read from `ctx` rather than passed alongside it, so
/// the row reset and the kernel's `grid_pos` derive from the same values by
/// construction.
#[inline(always)]
pub(super) fn run_level_rows<const DENSE: bool, L, R, A>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    left: &L,
    right: &R,
    action: &mut A,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    A: CellAction<L, R>,
{
    let RowLoop { f_level: left_level_t, g_level: right_level_t, ctx, f_width: left_width, .. } = rows;
    // Named once, ahead of the row loop, so the loop body indexes locals rather
    // than reaching back through the bundle at every cell.
    let RowScratch { inputs1: inputs1_scratch, inputs2: inputs2_scratch, node_idx } = scratch;
    let lim = eng.limits();
    // Amortized wall-deadline/cancel poll: one read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = lim.gate_with(super::super::budget::DENSE_CELL_POLL_STRIDE);
    let right_width = ctx.right_width;
    // Whether the per-column cull below can fire at all: the reach masks exist
    // only where both levels are multi-pair, a pass-through side has no grid to
    // be dead in, and `DENSE` is the regime where the masks are not folded.
    let cull_left = !DENSE && ctx.both_multi_pair && !left.passthrough();
    let cull_right = !DENSE && ctx.both_multi_pair && !right.passthrough();

    // One slab fill instead of `left_width` row fills. On a dense-slab action the
    // per-row resets below tile `output_grid_base .. output_grid_base + left_width*right_width` exactly once each
    // (`grid_row(i) == i`), so hoisting them into a single `fill` writes the same
    // cells with the same byte pattern; only the order changes, and nothing reads
    // this level's slab between rows (the child lookups read the child levels'
    // disjoint slabs). Above the size gate the per-row form stays: interleaving
    // the reset with the row's cell work is what keeps the active row L1-resident
    // on a large grid.
    let slab_fill = A::DENSE_SLAB && left_width.saturating_mul(right_width) <= DEAD_SLAB_FILL_MAX_CELLS;
    if slab_fill {
        node_idx[ctx.output_grid_base..ctx.output_grid_base + left_width * right_width].fill(NO_PRODUCT);
    }

    for i in 0..left_width {
        let row_base = ctx.output_grid_base + action.grid_row(i) * right_width;
        if !slab_fill {
            node_idx[row_base..row_base + right_width].fill(NO_PRODUCT);
        }

        if A::ASSERT_INTERNAL {
            debug_assert!(
                left_level_t.nodes[i].is_internal() || left_level_t.nodes[i].b == u32::MAX,
                "expected internal node at internal vtree position: output_grid_base={} i={i} left_width={left_width} node_a={:#x} node_b={:#x}",
                ctx.output_grid_base,
                left_level_t.nodes[i].a,
                left_level_t.nodes[i].b
            );
        }

        let inputs1 =
            left_level_t.pairs_view_decoded(i, inputs1_scratch, ctx.sides.left.plan.view, ctx.sides.right.plan.view);
        // Empty pairs means dead (zero-containing) node — skip this row.
        if inputs1.is_empty() {
            continue;
        }

        let (left_alive_mask, right_alive_mask) = if DENSE {
            (0u128, u128::MAX)
        } else {
            match row_alive_masks(ctx, inputs1) {
                Some(masks) => masks,
                // Row skip: if f[i]'s pairs all reference dead child rows,
                // no cell in this row can produce output.
                None => continue,
            }
        };


        for j in 0..right_width {
            // The cell's own dead-cell test, taken before the cell is entered.
            // A column whose g node reaches no live child of this row cannot
            // produce anything, whatever arm the cell would have taken, so the
            // cheapest place to find that out is here — ahead of the column's
            // pair slice, the empty test and the arm dispatch. Both sides read
            // the row's mask, which is the union over the row's references, so
            // a clear intersection proves every cell of the column dead.
            if cull_left && left_alive_mask & ctx.sides.left.reach[j] == 0 {
                continue;
            }
            if cull_right && right_alive_mask & ctx.sides.right.reach[j] == 0 {
                continue;
            }
            action.cell(
                eng,
                CellArgs {
                    j,
                    i,
                    row_base,
                    inputs1,
                    ctx,
                    right_level_t,
                    left,
                    right,
                    inputs2_scratch: &mut *inputs2_scratch,
                    node_idx: &mut *node_idx,
                    gate: &mut poll,
                },
            )?;
        }
        // One `tick_by(right_width)` per row instead of `tick()` per cell: the ticker only
        // meters accumulated work, so the same total is booked either way. A row
        // wider than the stride now polls once rather than once per stride's worth
        // of cells — the poll is an idempotent deadline read, so firing once per
        // crossing is equivalent, and a row that skipped the `j` loop (empty or
        // dead) books nothing, exactly as before (`right_width == 0` books nothing either).
        poll.poll(right_width as u64)?;
    }
    // The level's residual: what the gate holds is under one stride by
    // construction, and on a level narrower than a stride it is everything.
    poll.flush()?;
    Ok(())
}

/// Materializing action: each surviving cell becomes a product node in the
/// dense `left_width × right_width` output slab. Both dense routes are this action; they differ
/// only in `ASSERT_INTERNAL`, which route B can afford and route A cannot — a
/// marginal-child level's f rows may be marginal-encoded.
struct Emit<'a, const ASSERT_INTERNAL: bool> {
    level: &'a mut TddLevel,
}

impl<const A: bool, L: ChildLookup, R: ChildLookup> CellAction<L, R> for Emit<'_, A> {
    const ASSERT_INTERNAL: bool = A;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per f row.
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
            a.inputs1,
            a.ctx,
            a.right_level_t,
            a.inputs2_scratch,
            a.node_idx,
            a.left,
            a.right,
            &mut EmitSink {
                level: &mut *self.level,
            },
            a.gate,
        )
    }
}

/// Route A row-loop: forward-order scatter for levels with at least one marginal child.
///
/// The `Route::MarginalChild` row loop. Iterates rows
/// `0..left_width` in forward order, running the emit kernel for each cell.
///
/// Never streams: streaming marginal-child levels take the collapse-at-source
/// walker ([`run_level_rows_stream_count`]) unconditionally — there is no
/// materialize-then-convert fallback for them.
///
/// `rows.f_level` is a pre-taken immutable borrow into `f`'s level array so the
/// caller can keep its `vtree = &f.vtree` borrow live simultaneously.
pub(crate) fn run_level_rows_marginal(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
) -> Result<(), OperationError> {
    let left = MarginalLookup::new(&rows.ctx.sides.left);
    let right = MarginalLookup::new(&rows.ctx.sides.right);
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut Emit::<false> { level },
    )
}

/// Sparse-output action: emit into the reused row scratch, then record each
/// surviving cell as a `ProductEntry` instead of leaving it in a dense slab.
struct SparseMargEmit<'a> {
    level: &'a mut TddLevel,
    product_list: &'a mut Vec<ProductEntry>,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for SparseMargEmit<'_> {
    /// A marginal-child level's f rows may be marginal-encoded.
    const ASSERT_INTERNAL: bool = false;

    /// Not a dense slab — every structural row is rebuilt in the same `right_width`-wide
    /// scratch row, so its `NO_PRODUCT` reset must fire between rows and cannot be
    /// hoisted into a one-shot slab fill.
    const DENSE_SLAB: bool = false;

    /// One reused `right_width`-wide row scratch instead of a dense slab: every
    /// structural row builds at grid row 0, so `row_base == ctx.output_grid_base`.
    /// The true row index survives only in `CellArgs::i`, which the product
    /// entry's `left_idx` reads below.
    #[inline(always)]
    fn grid_row(&self, _i: usize) -> usize {
        0
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
        let lim = eng.limits();
        let row_pos = a.row_base + a.j;
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
            &mut EmitSink {
                level: &mut *self.level,
            },
            a.gate,
        )?;
        let nid = a.node_idx[row_pos];
        if nid != NO_PRODUCT {
            lim.try_push(
                self.product_list,
                ProductEntry {
                    left_idx: LeftNodeIdx(a.i as u32),
                    right_idx: RightNodeIdx(a.j as u32),
                    prod_idx: ProductNodeIdx(nid),
                },
            )?;
        }
        Ok(())
    }
}

/// Sparse-output variant of Route A for an *exactly-one*-marginal-child level
/// whose output is structural (never a marginalization target).
///
/// Same emit kernel as [`run_level_rows_marginal`], but instead of a dense
/// `left_width*right_width` slab it reuses one `right_width`-wide row scratch
/// at `cell_ctx.output_grid_base` and records each surviving cell in
/// `product_list`. The marginal child is a pass-through carrier that kills no
/// pair, so the structural sibling alone decides which cells are alive and a
/// dense slab would be mostly `NO_PRODUCT`. The parent densifies the
/// `product_list` through `ensure_grid` when it needs the grid.
///
/// Every cell is built at grid row 0, so the kernel's
/// `grid_pos == cell_ctx.output_grid_base + j`, then read back from the scratch
/// and, if alive, pushed as
/// `ProductEntry { left_idx: row i, right_idx: col j, prod_idx: node }`.
///
/// Never streams: [`Route::SparseMarg`](crate::apply::conjoin::route::Route::SparseMarg)
/// is chosen only for a level that is not a marginalization target.
pub(crate) fn run_level_rows_marginal_sparse(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    product_list: &mut Vec<ProductEntry>,
) -> Result<(), OperationError> {
    let left = MarginalLookup::new(&rows.ctx.sides.left);
    let right = MarginalLookup::new(&rows.ctx.sides.right);
    run_level_rows::<false, _, _, _>(
        eng,
        rows,
        scratch,
        &left,
        &right,
        &mut SparseMargEmit {
            level,
            product_list,
        },
    )
}

/// Route B row-loop: forward pass for plain (non-marginal-child) levels.
///
/// The row loop of the routes with no marginal child. Iterates rows in forward order.
/// Runs the emit kernel with the caller's plain lookups (no pass-through, no
/// mask decode — the caller guarantees no marginal child).
///
/// `DENSE = true` asserts two level-invariant facts, `cell_ctx.both_multi_pair
/// == false` and no pass-through side, so the inner loop carries no branch on
/// them; `DENSE = false` keeps the `both_multi_pair` row-skip checks.
///
/// Never streams: streaming levels take the collapse-at-source walker
/// ([`run_level_rows_stream_count`]) unconditionally — there is no post-cell
/// snapshot conversion.
#[inline(always)]
pub(crate) fn run_level_rows_plain<const DENSE: bool, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    left_lookup: &L,
    right_lookup: &R,
) -> Result<(), OperationError> {
    // When `DENSE`, the alive masks are constants (both_multi_pair is false, no pass-through):
    //   left_alive_mask  = 0u128      (the !both_multi_pair branch of `row_alive_masks`)
    //   right_alive_mask = `u128::MAX`  (the `|| !both_multi_pair` branch of `row_alive_masks`)
    // The driver passes those directly to the kernel, skipping the fold.
    let mut action = Emit::<true> { level };
    run_level_rows::<DENSE, _, _, _>(
        eng,
        rows,
        scratch,
        left_lookup,
        right_lookup,
        &mut action,
    )
}
