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
pub(super) struct CellArgs<'a, 'c, L, R> {
    /// Column index — the c2 node.
    pub(super) j: usize,
    /// TRUE c1 row index — NOT necessarily the grid row: the sparse-marg route
    /// builds every row at grid row 0 and needs this as the product entry's
    /// `c1_idx`.
    pub(super) i: usize,
    /// Flat slab offset of the grid row this cell writes into —
    /// `ctx.t_base + CellAction::grid_row(i) * ctx.k2`, computed ONCE per row by
    /// the driver for its DEAD reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    pub(super) row_base: usize,
    /// Decoded pairs of c1 row `i` (never empty — empty rows are skipped).
    pub(super) inputs1: &'a [InputPair],
    pub(super) left_alive_mask: u128,
    pub(super) right_alive_mask: u128,
    pub(super) ctx: &'a CellCtx<'c>,
    pub(super) c2_level_t: &'a TddLevel,
    pub(super) inputs2_scratch: &'a mut Vec<InputPair>,
    pub(super) node_idx: &'a mut [u32],
    pub(super) left: &'a L,
    pub(super) right: &'a R,
    /// The level's ONE work-clock gate. Every cell charges the pairs it walks
    /// into it, and the row loop adds one unit per cell; it is flushed when the
    /// level ends. Per-cell gates cannot do this — a gate narrower than its
    /// stride charges nothing at all, which is what a one-sided cell always is.
    pub(super) gate: &'a mut crate::engine::PollGate,
}

/// Per-row / per-cell action of the shared row loop ([`run_level_rows`]).
///
/// All hooks are `#[inline(always)]` in impls, so each instantiation
/// monomorphizes to what the hand-written per-route loop produced and the hooks
/// three of the four routes leave at their empty defaults vanish entirely.
pub(super) trait CellAction<L: ChildLookup, R: ChildLookup> {
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
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError>;

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
pub(super) fn run_level_rows<const DENSE: bool, L, R, A>(
    eng: &Engine,
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
    let lim = eng.limits();
    // Amortized wall-deadline/cancel poll: one TLS read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = crate::engine::PollGate::new(super::super::budget::DENSE_CELL_POLL_STRIDE);
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
            debug_assert!(
                c1_level_t.nodes[i].is_internal() || c1_level_t.nodes[i].b == u32::MAX,
                "expected internal node at internal vtree position: t_base={} i={i} k1={k1} node_a={:#x} node_b={:#x}",
                ctx.t_base,
                c1_level_t.nodes[i].a,
                c1_level_t.nodes[i].b
            );
        }

        let inputs1 =
            c1_level_t.pairs_view_decoded(i, inputs1_scratch, ctx.left_view, ctx.right_view);
        // Empty pairs means dead (ZERO-containing) node — skip this row.
        if inputs1.is_empty() {
            continue;
        }

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
            action.cell(
                eng,
                CellArgs {
                    j,
                    i,
                    row_base,
                    inputs1,
                    left_alive_mask,
                    right_alive_mask,
                    ctx,
                    c2_level_t,
                    left,
                    right,
                    inputs2_scratch: &mut *inputs2_scratch,
                    node_idx: &mut *node_idx,
                    gate: &mut poll,
                },
            )?;
        }
        // One `tick_by(k2)` per row instead of `tick()` per cell: the ticker only
        // meters accumulated work, so the same total is booked either way. A row
        // wider than the stride now polls once rather than once per stride's worth
        // of cells — the poll is an idempotent deadline read, so firing once per
        // crossing is equivalent, and a row that skipped the `j` loop (empty or
        // dead) books nothing, exactly as before (`k2 == 0` books nothing either).
        lim.poll(&mut poll, k2 as u64)?;
    }
    // The level's residual: what the gate holds is under one stride by
    // construction, and on a level narrower than a stride it is everything.
    lim.flush_poll(&mut poll)?;
    action.finish(k1, ctx, node_idx);
    Ok(())
}

/// Materializing action: each surviving cell becomes a product node in the
/// dense `k1 × k2` output slab. Both dense routes are this action; they differ
/// only in `ASSERT_INTERNAL`, which route B can afford and route A cannot — a
/// marginal-child level's c1 rows may be marginal-encoded.
struct Emit<'a, const ASSERT_INTERNAL: bool> {
    level: &'a mut TddLevel,
}

impl<const A: bool, L: ChildLookup, R: ChildLookup> CellAction<L, R> for Emit<'_, A> {
    const ASSERT_INTERNAL: bool = A;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
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
            &mut EmitSink {
                level: &mut *self.level,
            },
            a.gate,
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
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_level_rows_marg(
    eng: &Engine,
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
        eng,
        k1,
        c1_level_t,
        c2_level_t,
        cell_ctx,
        inputs1_scratch,
        inputs2_scratch,
        node_idx,
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
    fn grid_row(&self, _i: usize) -> usize {
        0
    }

    #[inline(always)]
    fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        let lim = eng.limits();
        let row_pos = a.row_base + a.j;
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
            &mut EmitSink {
                level: &mut *self.level,
            },
            a.gate,
        )?;
        let nid = a.node_idx[row_pos];
        if nid != DEAD {
            lim.try_push(
                self.product_list,
                ProductEntry {
                    c1_idx: C1NodeIdx(a.i as u32),
                    c2_idx: C2NodeIdx(a.j as u32),
                    prod_idx: ProdNodeIdx(nid),
                },
            )?;
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
    eng: &Engine,
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
        eng,
        k1,
        c1_level_t,
        c2_level_t,
        cell_ctx,
        inputs1_scratch,
        inputs2_scratch,
        node_idx,
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
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_level_rows_plain<const DENSE: bool, L: ChildLookup, R: ChildLookup>(
    eng: &Engine,
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
    let mut action = Emit::<true> { level };
    run_level_rows::<DENSE, _, _, _>(
        eng,
        k1,
        c1_level_t,
        c2_level_t,
        cell_ctx,
        inputs1_scratch,
        inputs2_scratch,
        node_idx,
        left_lookup,
        right_lookup,
        &mut action,
    )
}
