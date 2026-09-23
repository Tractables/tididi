//! One level of the bottom-up conjunction: routing a level to the sparse,
//! dense or streaming build and running it.
//!
//! The driver in [`super`] decides the order levels are visited in and what
//! happens at the boundaries; everything here is what a single level costs.

use crate::apply::conjoin::*;
use crate::Engine;
use super::Sweep;

/// One level's build decision: its shape, the route chosen for it, and the
/// marginal plan the route was chosen on.
#[derive(Clone, Copy)]
pub(super) struct LevelBuild {
    pub(super) shape: LevelShape,
    pub(super) route: Route,
    pub(super) plan: MarginalPlan,
}

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build both children's product lists, scatter-filter-dedup over the
/// live products, and record the result.
pub(super) fn run_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    shape: LevelShape,
) -> Result<(), OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    // Ensure children have product lists for the scatter pipeline.
    run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;

    apply_sparse_level(
        eng,
        shape, f, g,
        run.levels,
        run.products.lists(li, ri, ti),
        run.thresholds,
    )?;
    run.products.finish_sparse(&mut run.levels[ti], ti);
    Ok(())
}

/// Give both children a dense grid and bump-allocate this level's own,
/// returning the base the cell build writes into.
///
/// The sparse-marginal route allocates only a reused `right_width`-row scratch instead of
/// the dense `left_width * right_width` slab: the row driver processes one structural row at a
/// time into it and records the surviving cells in the output product list, so
/// the slab is never materialized and the grandparent densifies the level
/// lazily. That route's base is the scratch, not a slab base, which is why it
/// is returned rather than read back off the grid descriptor.
fn materialize_children_and_grid(
    eng: &Engine,
    run: &mut ApplyRun,
    shape: LevelShape,
    use_sparse_marginal: bool,
) -> Result<GridBase, OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    // ── Dense path: ensure children have grids ───────────────────
    //
    // Only when the arena bumps: a child processed by the sparse pipeline has
    // no grid, so materialize one. On this path the child is known ungridded —
    // there is nothing to scan — so the identity fast path is the only way to
    // build its product list.
    if run.products.arena.is_bump() {
        if run.products.arena.is_sparse(li) {
            run.materialize_dense_child(eng, li, fw.left, gw.left)?;
        }
        if run.products.arena.is_sparse(ri) {
            run.materialize_dense_child(eng, ri, fw.right, gw.right)?;
        }
    }

    // Dense output owns an f-by-g slab. Sparse marginal output reuses one
    // g-width row, records live products separately, and remains ungridded
    // until its parent needs a dense view. Only a bump arena uses that route;
    // a preplanned arena already owns the full slab.
    let cells = if use_sparse_marginal { gw.here } else { fw.here * gw.here };
    let base = run.products.arena.alloc(eng, ti, cells)?;
    if use_sparse_marginal {
        run.products.arena.set_sparse(ti);
    } else {
        run.products.arena.set_dense(ti, base);
    }
    Ok(base)
}


/// Pre-size the level's two arenas and arm its emit-growth policy.
///
/// Both reserves are sized from an exact upper bound and then capped at
/// [`LEVEL_RESERVE_CAP_BYTES`]: most levels are low-survival, so an uncapped
/// reserve would routinely grab orders of magnitude more than the level ends up
/// using and charge every byte of it. A level that outgrows the cap keeps
/// growing through the ordinary fallible push path, and `finalize_level`'s
/// `shrink_arrays` hands the unused tail back. Both are fallible: under a tight
/// budget even the capped reservation may not fit.
fn open_level_arenas(
    lim: &crate::limits::Limits,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    level: &mut TddLevel,
    route: Route,
) -> Result<(), OperationError> {
    let (t, left_width, right_width) = (shape.t, shape.f.here, shape.g.here);
    // One node per live cell, and compaction only removes dead ones, so
    // `left_width * right_width` is an exact bound.
    let nodes_reserve = left_width
        .saturating_mul(right_width)
        .min(LEVEL_RESERVE_NODES_CAP)
        .max(left_width.max(right_width));
    lim.reserve(&mut level.nodes, nodes_reserve)?;

    // The bound is a *growth policy*, not a correctness step, and the per-push
    // budget checks in the emit apply either way — so a route that never emits
    // into `level.pairs` can skip it. Called exactly once per level on every
    // route, so a previous level's near-cap decision cannot leak into this one.
    //
    // `Route::Stream { marginal_children: false }` reserves even though its fold
    // emits no pair either: dropping the reserve there would change what the
    // soft budget sees mid-apply, which is a separate decision from naming the
    // routes.
    let emits_pairs = !matches!(
        route,
        Route::SparseMarg | Route::Stream { marginal_children: true }
    );
    let stream_marginal = matches!(route, Route::Stream { .. });
    if emits_pairs {
        // The one per-level emit-pair bound: every product pair emits at most
        // once, so `|f.pairs| × |g.pairs|` bounds this level's emit. Used
        // twice — once to pick the growth mode, once to size the pairs
        // arena — computed once so the two can never disagree.
        let emit_pair_bound = (f.level(t).pairs.len() as u128)
            .saturating_mul(g.level(t).pairs.len() as u128);
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
        // emit walk will not charge again.
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
/// `MarginalLookup` sides; Route B assumes no marginal child and uses positional
/// dense lookups. Both collapse to a streaming fold instead of materializing
/// product nodes when the level is a streaming marginalization target.
fn run_row_loop(
    eng: &Engine,
    route: Route,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    env: StreamEnv<'_>,
    stream_state: &mut Option<StreamLevelState>,
) -> Result<(), OperationError> {
    let cell_ctx = rows.ctx;
    // Marginal sides are read through `MarginalLookup`, which decodes a count
    // payload or degrades to a dense grid read; structural sides are read
    // positionally.
    let left_marginal = child_lookup::MarginalLookup::new(&cell_ctx.sides.left);
    let right_marginal = child_lookup::MarginalLookup::new(&cell_ctx.sides.right);
    let left_dense = child_lookup::DenseLookup {
        base: cell_ctx.sides.left.base, stride: cell_ctx.sides.left.stride };
    let right_dense = child_lookup::DenseLookup {
        base: cell_ctx.sides.right.base, stride: cell_ctx.sides.right.stride };

    macro_rules! stream_rows {
        ($l:expr, $r:expr) => {
            run_level_rows_stream_count(
                eng,
                rows, scratch,
                $l, $r,
                stream_state.as_mut().expect("Route::Stream implies an open stream column"),
                env,
            )?
        };
    }
    macro_rules! plain_rows {
        ($dense:literal) => {
            run_level_rows_plain::<$dense, _, _>(
                eng,
                rows, scratch,
                level,
                &left_dense, &right_dense,
            )?
        };
    }

    match route {
        // A streaming marginalization target collapses to Σ left × right per cell:
        // nothing downstream survives, so build each alive cell's scalar from
        // the surviving refs and never materialize a product node. Which side
        // carries counts is what picks the lookups.
        Route::Stream { marginal_children: true } => stream_rows!(&left_marginal, &right_marginal),
        Route::Stream { marginal_children: false } => stream_rows!(&left_dense, &right_dense),
        // A materializing level with at least one marginal child. It runs the
        // same per-cell kernel as the structural routes, with `MarginalLookup`
        // sides; isolating it here is what lets those routes assume no child
        // is marginal. Refs come out in the encoding the structural path
        // produces, so this level still relies on the end-of-apply tagger.
        // Both-marginal levels route here too — pure Σ left × right with no
        // structural product — and carrying them here keeps them off the
        // inline tagger that would corrupt them.
        Route::MarginalChild => run_level_rows_marginal(eng, rows, scratch, level)?,
        // The four level-invariant guards all hold, so the simplified emit
        // runs: no streaming, no dead-pair masks, no pass-through side.
        Route::PlainDense => plain_rows!(true),
        Route::Dense => plain_rows!(false),
        Route::Sparse | Route::SparseMarg => {
            unreachable!("the sparse routes build their level before the row loop")
        }
    }
    Ok(())
}

/// Build this level's dead-pair liveness masks, which need the children's
/// grids materialized and so cannot be computed with the rest of the marginal plan.
///
/// # Errors
///
/// Propagates a refused reservation for the mask buffers.
fn build_level_prefilter_masks(
    eng: &Engine,
    run: &mut ApplyRun,
    g: &Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
    bases: Sides<GridBase>,
) -> Result<(), OperationError> {
    let LevelShape { t, f: fw, g: gw, .. } = shape;
    let right_level = g.level(t);
    build_side_masks::<false>(
        eng, right_level, gw.here,
        ChildGrid { plan: plan.sides.left, f_width: fw.left, g_width: gw.left, base: bases.left.idx() },
        run.products.arena.slab(), &mut run.prefilter_masks.left,
    )?;
    build_side_masks::<true>(
        eng, right_level, gw.here,
        ChildGrid { plan: plan.sides.right, f_width: fw.right, g_width: gw.right, base: bases.right.idx() },
        run.products.arena.slab(), &mut run.prefilter_masks.right,
    )
}

/// The run buffers [`finish_sparse_marginal_level`] writes, borrowed field by
/// field: the output level is already split out of the same `ApplyRun`.
struct SparseMargScratch<'a> {
    inputs1: &'a mut Vec<ChildPair>,
    inputs2: &'a mut Vec<ChildPair>,
    products: &'a mut super::super::products::Products,
}

/// Build a [`Route::SparseMarg`] level and close it out.
///
/// Runs the shared emit kernel, but writes into the reused `right_width`-row scratch —
/// `cell_ctx.output_grid_base` serves directly as the row base, and the driver is called with `i = 0`, so
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
fn finish_sparse_marginal_level(
    eng: &Engine,
    shape: LevelShape,
    rows: RowLoop<'_>,
    output_grid_base: GridBase,
    level: &mut TddLevel,
    scratch: SparseMargScratch<'_>,
    passthrough: Sides<bool>,
) -> Result<(), OperationError> {
    let LevelShape { t, g: gw, .. } = shape;
    let SparseMargScratch {
        inputs1, inputs2, products,
    } = scratch;
    let (node_idx, product_list) = products.row_buffers(t.idx());
    run_level_rows_marginal_sparse(
        eng,
        rows,
        RowScratch { inputs1, inputs2, node_idx },
        level,
        product_list,
    )?;
    products.arena.free(output_grid_base, gw.here);
    products.finish_sparse(level, t.idx());
    mark_passthrough_inlined(level, passthrough);
    Ok(())
}


/// Gather one level's per-cell context: grid geometry, pass-through carriers,
/// the dead-pair liveness masks, and the resolved g column table.
///
/// `right_cols` resolves every g column once for the level, so `process_cell`
/// indexes the table instead of re-deriving column j's slice on every row. On
/// identity-mask levels the descriptors are zero-copy borrows of g's own
/// storage; on marginal-mask levels they point into a decode arena the table owns
/// and budget-charges. `None` — marginal-encoded g, or the budget refusing the
/// arena — falls back to the per-cell derivation, never worse than doing it per
/// cell.
fn build_cell_ctx<'a>(
    shape: LevelShape,
    plan: &MarginalPlan,
    output_grid_base: usize,
    bases: Sides<GridBase>,
    masks: &'a crate::apply::conjoin::liveness::PrefilterMaskScratch,
    right_cols: Option<&'a RightColumns>,
) -> CellCtx<'a> {
    let side = |plan: SidePlan, base: usize, stride: usize, masks: &'a liveness::PrefilterSideMasks|
        ChildPlan { plan, base, stride: stride as u32, live_cols: &masks.live_cols, reach: &masks.reach };
    CellCtx {
        output_grid_base,
        right_width: shape.g.here,
        both_multi_pair: plan.both_multi_pair,
        sides: Sides {
            left: side(plan.sides.left, bases.left.idx(), shape.g.left, &masks.left),
            right: side(plan.sides.right, bases.right.idx(), shape.g.right, &masks.right),
        },
        right_cols,
    }
}

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
pub(super) fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    level: LevelBuild,
    sweep: &mut Sweep<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let LevelBuild { shape, route, plan } = level;
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let MarginalPlan { sides, both_multi_pair } = plan;
    let passthrough = Sides { left: sides.left.is_passthrough(), right: sides.right.is_passthrough() };
    let use_sparse_marginal = route == Route::SparseMarg;
    // Materialize any sparse child grid and bump-allocate this level's own.
    // The route is already known, which is what lets this skip `ensure_grid`
    // for a sparse child on a level that will take the plain-dense emit.
    let output_grid_base = materialize_children_and_grid(eng, run, shape, use_sparse_marginal)?;

    // Child grid geometry: product `(a, b)` sits at `base + a * k2_child + b`.
    // The `NO_PRODUCT`-fill is interleaved with the product construction, one row at a
    // time before that row's cells are computed, which keeps the active row in
    // L1 during `process_cell` instead of polluting the cache with a single
    // bulk fill of the whole f-by-g grid.
    let bases = Sides {
        left: run.products.arena.materialized(li).expect("the left child's grid is materialized"),
        right: run.products.arena.materialized(ri).expect("the right child's grid is materialized"),
    };

    // Only the grid-reading dead-pair liveness masks are deferred this far: they need
    // the materialized child grids, and `both_multi_pair` implies a route that has them.
    if both_multi_pair {
        build_level_prefilter_masks(eng, run, g, shape, &plan, bases)?;
    }

    let mut stream_state: Option<StreamLevelState> =
        build_stream_state(eng, shape, run.levels, run.stream_cache, sweep)?;

    // `t` and its two vtree children are three distinct tree nodes, so these
    // are three disjoint level slots: the streaming row loops read the child
    // columns in place while the output level is exclusively borrowed. The
    // split's borrow must end before the per-level tail retakes `levels`.
    let [level, left_level, right_level] = run.levels
        .get_disjoint_mut([ti, li, ri])
        .expect("a vtree node and its two children are distinct level indices");
    let (left_level, right_level) = (&*left_level, &*right_level);


    let right_cols = RightColumns::build(eng, g.level(t), gw.here, sides.left.view, sides.right.view);
    let cell_ctx = build_cell_ctx(shape, &plan, output_grid_base.idx(), bases, run.prefilter_masks, right_cols.as_ref());

    open_level_arenas(lim, f, g, shape, level, route)?;

    if use_sparse_marginal {
        return finish_sparse_marginal_level(
            eng, shape,
            RowLoop {
                f_level: f.level(t), g_level: g.level(t),
                children: Sides { left: left_level, right: right_level },
                ctx: &cell_ctx, f_width: fw.here,
            },
            output_grid_base, level,
            SparseMargScratch {
                inputs1: run.inputs1_scratch,
                inputs2: run.inputs2_scratch,
                products: run.products,
            },
            passthrough,
        );
    }

    run_row_loop(
        eng, route,
        RowLoop {
            f_level: f.level(t), g_level: g.level(t),
            children: Sides { left: left_level, right: right_level },
            ctx: &cell_ctx, f_width: fw.here,
        },
        RowScratch {
            inputs1: run.inputs1_scratch,
            inputs2: run.inputs2_scratch,
            node_idx: run.products.arena.slab_mut(),
        },
        level,
        StreamEnv {
            left_idx: li,
            right_idx: ri,
            vtree: sweep.vtree,
            cache: run.stream_cache,
            ws: sweep.ws.as_deref(),
        },
        &mut stream_state,
    )?;

    // Per-level tail: stream commit, live_counts, grid tag, shrink,
    // pass-through flags. See `finalize_level`.
    finalize_level(eng, &mut stream_state, shape, output_grid_base, passthrough, run, sweep);
    Ok(())
}
