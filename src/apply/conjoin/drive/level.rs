//! One level of the bottom-up conjunction: routing a level to the sparse,
//! dense or streaming build and running it.
//!
//! The driver in [`super`] decides the order levels are visited in and what
//! happens at the boundaries; everything here is what a single level costs.

use crate::apply::conjoin::*;
use crate::engine::Engine;

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build both children's product lists, scatter-filter-dedup over the
/// live products, and record the result.
pub(super) fn run_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
    is_marg_target: bool,
) -> Result<(), ApplyError> {
    let LevelShape {
        t, left, right, t_idx, left_idx, right_idx,
        k1_left, k2_left, k1_right, k2_right, ..
    } = shape;
    // Ensure children have product lists for the scatter pipeline.
    run.ensure_product_list_for_child(eng, left_idx, k1_left, k2_left)?;
    run.ensure_product_list_for_child(eng, right_idx, k1_right, k2_right)?;

    // Disjoint borrows of three product lists (left, right, output).
    let [pl_left, pl_right, pl_output] = run.product_lists
        .get_disjoint_mut([left_idx, right_idx, t_idx])
        .expect("left_idx, right_idx, t_idx must be distinct");
    apply_sparse_level(
        eng,
        t, left, right, c1, c2,
        &mut run.levels, &run.c1_widths, &run.c2_widths,
        pl_left,
        pl_right,
        pl_output,
        vtree.node(VtreeIdx(left_idx as u32)).is_leaf(),
        vtree.node(VtreeIdx(right_idx as u32)).is_leaf(),
        is_marg_target,
    )?;
    // Release oversized bucket Vecs to avoid retaining peak allocations.
    release_sparse_ws_if_large(eng);
    finish_sparse_output(
        &mut run.live_counts, &mut run.has_pl,
        &mut run.levels[t_idx], t_idx,
    );
    Ok(())
}

/// Give both children a dense grid and bump-allocate this level's own,
/// returning the base the cell build writes into.
///
/// The sparse-marg route allocates only a reused `k2`-row scratch instead of
/// the dense `k1 * k2` slab: the row driver processes one structural row at a
/// time into it and records the surviving cells in the output product list, so
/// the slab is never materialized and the grandparent densifies the level
/// lazily. That route's base is the scratch, not a slab base, which is why it
/// is returned rather than read back off the grid descriptor.
fn materialize_children_and_grid(
    eng: &Engine,
    run: &mut ApplyRun,
    shape: LevelShape,
    use_sparse_marg: bool,
) -> Result<GridBase, ApplyError> {
    let LevelShape { t_idx, left_idx, right_idx, k1, k2, k1_left, k2_left, k1_right, k2_right, .. } = shape;
    // ── Dense path: ensure children have grids ───────────────────
    //
    // Only when the arena bumps: a child processed by the sparse pipeline has
    // no grid, so materialize one. On this path the child is known ungridded —
    // there is nothing to scan — so the identity fast path is the only way to
    // build its product list.
    if run.arena.is_bump() {
        if run.arena.is_sparse(left_idx) {
            run.materialize_dense_child(eng, left_idx, k1_left, k2_left)?;
        }
        if run.arena.is_sparse(right_idx) {
            run.materialize_dense_child(eng, right_idx, k1_right, k2_right)?;
        }
    }

    // Claim this level's own space, marking it `Dense` up front so a consumer
    // that peeks before the emit loop finishes (a debug-assert path, say) still
    // reads a consistent base.
    //
    // Sparse-marg path: claim only a single reused k2-row scratch instead of
    // the dense k1*k2 slab. `run_level_rows_marg_sparse` processes one
    // structural row at a time into this scratch, records the surviving cells
    // into the output product_list, then frees the scratch — the dense slab is
    // never materialized. The level is tagged ungridded here; the grandparent
    // densifies it lazily. That route only exists when the arena bumps, so a
    // pre-planned layout always takes the dense branch and its `alloc` is the
    // lookup of a base decided at setup.
    let cells = if use_sparse_marg { k2 } else { k1 * k2 };
    let base = run.arena.alloc(eng, t_idx, cells)?;
    if use_sparse_marg {
        run.arena.set_sparse(t_idx);
    } else {
        run.arena.set_dense(t_idx, base);
    }
    Ok(base)
}


/// Pre-size the level's two arenas and arm its emit-growth policy.
///
/// Both reserves are sized from an EXACT upper bound and then capped at
/// [`LEVEL_RESERVE_CAP_BYTES`]: most levels are low-survival, so an uncapped
/// reserve would routinely grab orders of magnitude more than the level ends up
/// using and charge every byte of it. A level that outgrows the cap keeps
/// growing through the ordinary fallible push path, and `finalize_level`'s
/// `shrink_arrays` hands the unused tail back. Both are fallible: under a tight
/// budget even the baseline reservation may not fit.
#[allow(clippy::too_many_arguments)]
fn open_level_arenas(
    lim: &crate::engine::Limits,
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    level: &mut TddLevel,
    route: Route,
    k1: usize,
    k2: usize,
) -> Result<(), ApplyError> {
    // One node per live cell — compaction only removes dead ones — so `k1 * k2`
    // is exact. Never below `max(k1, k2)`, the seed this replaced.
    let nodes_reserve = k1
        .saturating_mul(k2)
        .min(LEVEL_RESERVE_NODES_CAP)
        .max(k1.max(k2));
    lim.reserve(&mut level.nodes, nodes_reserve)?;

    // The bound is a *growth policy*, not a correctness step, and the per-push
    // budget checks in the emit apply either way — so a route that never emits
    // into `level.pairs` can skip it. Called exactly once per level on every
    // route, so a previous level's near-cap decision cannot leak into this one.
    //
    // `Route::Stream { marg_children: false }` reserves even though its fold
    // emits no pair either. That is how it has always run; dropping the
    // reserve there changes what the soft budget sees mid-apply, so it is a
    // measured change rather than part of naming the routes.
    let emits_pairs = !matches!(
        route,
        Route::SparseMarg | Route::Stream { marg_children: true }
    );
    let stream_marginal = matches!(route, Route::Stream { .. });
    if emits_pairs {
        // THE per-level emit-pair bound: every product pair emits at most
        // once, so `|c1.pairs| × |c2.pairs|` bounds this level's emit. Used
        // twice — once to pick the growth mode, once to size the pairs
        // arena — computed once so the two can never disagree.
        let emit_pair_bound = (c1.level(t).pairs.len() as u128)
            .saturating_mul(c2.level(t).pairs.len() as u128);
        lim.begin_level((!stream_marginal).then_some(emit_pair_bound));
        // Seed `level.pairs` at that bound instead of letting it double from
        // empty on every level; the emit's own `try_push_pair_into` choke
        // point, under the growth mode just armed, carries a level that
        // outgrows the cap.
        let pairs_reserve =
            emit_pair_bound.min(LEVEL_RESERVE_PAIRS_CAP as u128) as usize;
        let pre_pairs_cap = level.pairs.capacity();
        lim.reserve(&mut level.pairs, pairs_reserve)?;
        // Output-pair meter: this bulk seed is real arena capacity the
        // emit walk will not charge again. See `ApplyLimits::pairs_in_flight`.
        lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    } else {
        lim.begin_level(None);
    }
    Ok(())
}

/// Build every cell of this level, on whichever of the two row-loop routes the
/// level's marginal-child pattern selects.
///
/// Route A (at least one marginal child) runs the shared cell kernel with
/// `MargLookup` sides; Route B assumes no marginal child and uses positional
/// dense lookups. Both collapse to a streaming fold instead of materializing
/// product nodes when the level is a streaming marginalize target.
#[allow(clippy::too_many_arguments)]
fn run_row_loop(
    eng: &Engine,
    route: Route,
    k1: usize,
    t: VtreeIdx,
    left_idx: usize,
    right_idx: usize,
    c1: &Tdd,
    c2: &Tdd,
    vtree: &crate::vtree::Vtree,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    stream_cache: &StreamCache,
    stream_state: &mut Option<StreamLevelState>,
    level: &mut TddLevel,
    left_level: &TddLevel,
    right_level: &TddLevel,
    ws: Option<&crate::diagram::WeightStore>,
) -> Result<(), ApplyError> {
    // Both operand borrows are taken before `level` — a `&mut` into the OUTPUT
    // levels, a separate allocation — is used, then lent on as plain refs.
    let c1_level_t: &TddLevel = c1.level(t);
    let c2_level_t: &TddLevel = c2.level(t);

    // Marginal sides are read through `MargLookup`, which decodes a count
    // payload or degrades to a dense grid read; structural sides are read
    // positionally, which inlines to the original `get_unchecked` index.
    let left_marg = child_lookup::MargLookup::new(&cell_ctx.sides.left);
    let right_marg = child_lookup::MargLookup::new(&cell_ctx.sides.right);
    let left_dense = child_lookup::DenseLookup {
        base: cell_ctx.sides.left.base, k2: cell_ctx.sides.left.k2 };
    let right_dense = child_lookup::DenseLookup {
        base: cell_ctx.sides.right.base, k2: cell_ctx.sides.right.k2 };

    macro_rules! stream_rows {
        ($l:expr, $r:expr) => {
            run_level_rows_stream_count(
                eng,
                k1,
                c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch,
                node_idx,
                $l, $r,
                stream_state.as_mut().expect("Route::Stream implies an open stream column"),
                left_idx, right_idx, vtree, left_level, right_level,
                stream_cache, ws,
            )?
        };
    }
    macro_rules! plain_rows {
        ($dense:literal) => {
            run_level_rows_plain::<$dense, _, _>(
                eng,
                k1,
                c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch,
                level, node_idx,
                &left_dense, &right_dense,
            )?
        };
    }

    match route {
        // A streaming marginalize target collapses to Σ left × right per cell:
        // nothing downstream survives, so build each alive cell's scalar from
        // the surviving refs and never materialize a product node. Which side
        // carries counts is what picks the lookups.
        Route::Stream { marg_children: true } => stream_rows!(&left_marg, &right_marg),
        Route::Stream { marg_children: false } => stream_rows!(&left_dense, &right_dense),
        // A materializing level with at least one marginal child. It runs the
        // SAME per-cell kernel as the structural routes, with `MargLookup`
        // sides; isolating it here is what lets those routes assume no child
        // is marginal. Refs come out in the encoding the structural path
        // produces, so this level still relies on the end-of-apply tagger.
        // Both-marginal levels route here too — pure Σ left × right with no
        // structural product — and carrying them here keeps them off the
        // inline tagger that would corrupt them.
        Route::MargChild => run_level_rows_marg(
            eng,
            k1,
            c1_level_t, c2_level_t, cell_ctx,
            inputs1_scratch, inputs2_scratch,
            level, node_idx,
        )?,
        // The four level-invariant guards all hold, so the simplified emit
        // runs: no streaming, no NxM masks, no pass-through side.
        Route::PlainDense => plain_rows!(true),
        Route::Dense => plain_rows!(false),
        Route::Sparse | Route::SparseMarg => {
            unreachable!("the sparse routes build their level before the row loop")
        }
    }
    Ok(())
}

/// Build this level's NxM dead-pair liveness masks, which need the children's
/// grids materialized and so cannot be computed with the rest of the marg plan.
///
/// # Errors
///
/// Propagates a refused reservation for the mask buffers.
fn build_level_nxm_masks(
    eng: &Engine,
    run: &mut ApplyRun,
    c2: &Tdd,
    shape: LevelShape,
    plan: &MargPlan,
    bases: Sides<GridBase>,
) -> Result<(), ApplyError> {
    let LevelShape { t, right_idx, k2, k1_left, k2_left, k2_right, .. } = shape;
    let c2_level = c2.level(t);
    build_side_masks::<false>(eng, c2_level, k2, plan.sides.left,
        k1_left, k2_left, bases.left.idx(), run.arena.slab(), &mut run.nxm_masks.left)?;
    build_side_masks::<true>(eng, c2_level, k2, plan.sides.right,
        run.c1_widths[right_idx], k2_right, bases.right.idx(), run.arena.slab(), &mut run.nxm_masks.right)
}

/// The run buffers [`finish_sparse_marg_level`] writes, borrowed field by
/// field: the output level is already split out of the same `ApplyRun`.
struct SparseMargScratch<'a> {
    inputs1: &'a mut Vec<InputPair>,
    inputs2: &'a mut Vec<InputPair>,
    arena: &'a mut GridArena,
    product_list: &'a mut Vec<ProductEntry>,
    live_counts: &'a mut LiveCounts,
    has_pl: &'a mut [bool],
}

/// Build a [`Route::SparseMarg`] level and close it out.
///
/// Runs the shared emit kernel, but writes into the reused `k2`-row scratch —
/// `cell_ctx.t_base` IS the row base, and the driver is called with `i = 0`, so
/// a grid position is just `row_base + j` — and records each surviving cell in
/// the output product list instead of a dense slab. The scratch goes back
/// immediately: the level is tagged sparse, its product list is the
/// authoritative representation, and the grandparent densifies it lazily.
///
/// This route returns before `finalize_level`, so it does that tail's
/// inline-emit marking itself. Exactly one side is the marginal pass-through.
///
/// # Errors
///
/// Propagates a refused reservation from the row driver.
#[allow(clippy::too_many_arguments)]
fn finish_sparse_marg_level(
    eng: &Engine,
    c1: &Tdd,
    c2: &Tdd,
    shape: LevelShape,
    t_base: GridBase,
    cell_ctx: &CellCtx<'_>,
    level: &mut TddLevel,
    scratch: SparseMargScratch<'_>,
    left_passthrough: bool,
    right_passthrough: bool,
) -> Result<(), ApplyError> {
    let LevelShape { t, t_idx, k1, k2, .. } = shape;
    let SparseMargScratch {
        inputs1, inputs2, arena, product_list, live_counts, has_pl,
    } = scratch;
    run_level_rows_marg_sparse(
        eng,
        k1,
        c1.level(t), c2.level(t), cell_ctx,
        inputs1, inputs2,
        level, arena.slab_mut(),
        product_list,
    )?;
    arena.free(t_base, k2);
    finish_sparse_output(live_counts, has_pl, level, t_idx);
    mark_passthrough_inlined(level, left_passthrough, right_passthrough);
    Ok(())
}


/// Gather one level's per-cell context: grid geometry, pass-through carriers,
/// the NxM liveness masks, and the resolved c2 column table.
///
/// `c2_cols` resolves every c2 column ONCE for the level, so `process_cell`
/// indexes the table instead of re-deriving column j's slice on every row. On
/// identity-mask levels the descriptors are zero-copy borrows of c2's own
/// storage; on marg-mask levels they point into a decode arena the table owns
/// and budget-charges. `None` — marginal-encoded c2, or the budget refusing the
/// arena — falls back to the per-cell derivation, never worse than doing it per
/// cell.
fn build_cell_ctx<'a>(
    shape: LevelShape,
    plan: &MargPlan,
    t_base: usize,
    bases: Sides<GridBase>,
    masks: &'a crate::apply::conjoin::liveness::NxmMaskScratch,
    c2_cols: Option<&'a C2Columns>,
) -> CellCtx<'a> {
    let side = |plan: SidePlan, base: usize, k2: usize, masks: &'a liveness::NxmSideMasks|
        ChildPlan { plan, base, k2: k2 as u32, live_cols: &masks.live_cols, reach: &masks.reach };
    CellCtx {
        t_base,
        k2: shape.k2,
        nxm: plan.nxm,
        sides: Sides {
            left: side(plan.sides.left, bases.left.idx(), shape.k2_left, &masks.left),
            right: side(plan.sides.right, bases.right.idx(), shape.k2_right, &masks.right),
        },
        c2_cols,
    }
}

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
    route: Route,
    plan: &MargPlan,
    vtree: &Arc<crate::vtree::Vtree>,
    marginalize_targets: MargTargets<'_>,
    mut ws: Option<&mut crate::diagram::WeightStore>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let LevelShape {
        t, t_idx, left_idx, right_idx,
        k1, k2, ..
    } = shape;
    let MargPlan { sides, nxm } = *plan;
    let (left_passthrough, right_passthrough) =
        (sides.left.is_passthrough(), sides.right.is_passthrough());
    let use_sparse_marg = route == Route::SparseMarg;
    // Materialize any sparse child grid and bump-allocate this level's own.
    // The route is already known, which is what lets this skip `ensure_grid`
    // for a sparse child on a level that will take the plain-dense emit.
    let t_base = materialize_children_and_grid(eng, run, shape, use_sparse_marg)?;

    // Child grid geometry: product `(a, b)` sits at `base + a * k2_child + b`.
    // The DEAD-fill is interleaved with the product construction, one row at a
    // time before that row's cells are computed, which keeps the active row in
    // L1 during `process_cell` instead of polluting the cache with a single
    // bulk fill of the whole `k1 * k2` grid.
    let bases = Sides {
        left: run.arena.materialized(left_idx).expect("the left child's grid is materialized"),
        right: run.arena.materialized(right_idx).expect("the right child's grid is materialized"),
    };

    // Only the grid-reading NxM liveness masks are deferred this far: they need
    // the materialized child grids, and `nxm` implies a route that has them.
    if nxm {
        build_level_nxm_masks(eng, run, c2, shape, plan, bases)?;
    }

    let mut stream_state: Option<StreamLevelState> = build_stream_state(
        eng,
        t_idx, left_idx, right_idx, k1, k2,
        marginalize_targets, vtree, &mut run.levels,
        &mut run.stream_cache,
        ws.as_deref_mut(),
    )?;

    // `t` and its two vtree children are three distinct tree nodes, so these
    // name three disjoint level slots and split apart in one step. That is
    // what lets the streaming row loops read the child count columns IN PLACE
    // while the output level is exclusively borrowed; snapshotting them
    // instead doubled a wide marginal child's storage at exactly the moment
    // streaming exists to relieve. The split's borrow must end before the
    // per-level tail retakes `levels`.
    let [level, left_level, right_level] = run.levels
        .get_disjoint_mut([t_idx, left_idx, right_idx])
        .expect("a vtree node and its two children are distinct level indices");
    let (left_level, right_level) = (&*left_level, &*right_level);


    let c2_cols = C2Columns::build(eng, c2.level(t), k2, sides.left.view, sides.right.view);
    let cell_ctx = build_cell_ctx(shape, plan, t_base.idx(), bases, &run.nxm_masks, c2_cols.as_ref());

    open_level_arenas(lim, c1, c2, t, level, route, k1, k2)?;

    if use_sparse_marg {
        return finish_sparse_marg_level(
            eng, c1, c2, shape, t_base, &cell_ctx, level,
            SparseMargScratch {
                inputs1: &mut run.inputs1_scratch,
                inputs2: &mut run.inputs2_scratch,
                arena: &mut run.arena,
                product_list: &mut run.product_lists[t_idx],
                live_counts: &mut run.live_counts,
                has_pl: &mut run.has_pl,
            },
            left_passthrough, right_passthrough,
        );
    }

    run_row_loop(
        eng, route, k1, t, left_idx, right_idx, c1, c2, vtree, &cell_ctx,
        &mut run.inputs1_scratch, &mut run.inputs2_scratch, run.arena.slab_mut(),
        &run.stream_cache,
        &mut stream_state, level, left_level, right_level, ws.as_deref(),
    )?;


    // Per-level tail: stream commit, live_counts, grid tag, shrink,
    // pass-through flags. See `finalize_level`.
    finalize_level(
        eng,
        &mut stream_state,
        t, t_idx,
        t_base,
        left_passthrough, right_passthrough,
        vtree, &mut run.levels, &mut run.arena, &mut run.live_counts,
        ws,
    );
    Ok(())
}
