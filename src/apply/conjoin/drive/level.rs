//! One level of the bottom-up conjunction: routing a level to the sparse,
//! dense or streaming build and running it.
//!
//! The driver in [`super`] decides the order levels are visited in and what
//! happens at the boundaries; everything here is what a single level costs.

use crate::apply::conjoin::*;
use crate::Engine;
use super::Sweep;
use crate::diagram::ChildSide;
use crate::apply::conjoin::sparse::stream::{
    candidate_bound, choose_pivot, count as stream_count, Built, StreamInput, StreamWidths,
};

/// One level's build decision: its shape, the route chosen for it, and the
/// marginal plan the route was chosen on.
#[derive(Clone, Copy)]
pub(super) struct LevelBuild {
    pub(super) shape: LevelShape,
    pub(super) route: Route,
    pub(super) plan: MarginalPlan,
}

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build the joined children's product lists, scatter-filter-dedup over
/// the live products, and record the result.
///
/// A pass-through side in `plan` is carried rather than joined, so it needs
/// no product list, and the level's fields on that side are marked inline
/// for the end-of-apply tagger, as [`Route::SparseMarg`] marks them.
pub(super) fn run_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
) -> Result<(), OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let passthrough = Passthrough::of(plan.sides);
    let carried = |side| passthrough.is_some_and(|p| p.side == side);
    // Ensure the joined children have product lists for the scatter pipeline.
    if !carried(ChildSide::Left) {
        run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    }
    if !carried(ChildSide::Right) {
        run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;
    }

    apply_sparse_level(
        eng,
        shape, f, g,
        run.levels,
        run.products.lists(li, ri, ti),
        run.thresholds,
        passthrough,
    )?;
    run.products.finish_sparse(&mut run.levels[ti], ti);
    mark_passthrough_inlined(
        &mut run.levels[ti],
        Sides { left: carried(ChildSide::Left), right: carried(ChildSide::Right) },
    );
    Ok(())
}

/// Whether [`count_sparse_root`] may take a level the sparse route was chosen
/// for: the sweep wants only the count, the level is the root both operands
/// output at, f and g have one node there, neither child is carried, and no
/// weight, filter or target changes what the one product's count is.
pub(super) fn counts_root(
    sweep: &Sweep<'_, '_>,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
) -> bool {
    let t = shape.t;
    sweep.count_root
        && t == sweep.vtree.root()
        && f.output.vtree == t
        && g.output.vtree == t
        && shape.f.here == 1
        && shape.g.here == 1
        && Passthrough::of(plan.sides).is_none()
        && sweep.ws.is_none()
        && sweep.filter.is_none()
        && !sweep.targets.contains(t.idx())
}

/// Count the root level instead of building it: the model count of the
/// conjunction, which is the fold of the one root product's pairs against
/// the count columns of the two children.
///
/// The children's columns are folded first, over the levels built so far,
/// as a model count of the finished diagram would fold them; the scatter
/// then folds each candidate in where [`run_sparse_level`] would store it.
pub(super) fn count_sparse_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
) -> Result<num_bigint::BigUint, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, Retention, ValueDomain};
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;

    let lim = eng.limits();
    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    let levels: &[TddLevel] = run.levels;
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput { vtree, levels, store: &() };
    let mut gate = lim.gate();
    for child in [left, right] {
        IntFold::ensure(eng, child, input, &mut computed, &marginal, Retention::Frontier, |w| gate.poll(w))?;
    }
    gate.flush()?;

    let mut fold = CandidateFold::new(
        lim,
        IntFold::child_view(li, vtree, &levels[li], &computed, &()),
        IntFold::child_view(ri, vtree, &levels[ri], &computed, &()),
    )?;
    let lists = run.products.lists(li, ri, ti);
    count_sparse_level(
        eng, shape, f, g, levels,
        Sides { left: lists.left, right: lists.right },
        run.thresholds, None, &mut fold,
    )?;
    Ok(fold.finish())
}

/// Whether the sweep holds back the level at `t` unbuilt, for the root count
/// to stream: the sweep wants only the count of a root both operands output
/// at with one node, `t` is a child of that root, and neither operand is
/// constant-true over it; `t`, its children and its sibling are internal,
/// no weight, filter or quantified subtree touches any of them, and none of
/// the levels the stream reads built (`t`'s children and its sibling) is a
/// target: a target at `t` only decides how `t` is built if the stream
/// declines. Neither operand may be marginal at any of them, now or at
/// entry (an identity fast path moves a marginal level out of its operand),
/// nor carry counts in its references from the root or `t`: the stream reads
/// those references as node indices. Whether the root then streams `t` is
/// priced there, once the levels it reads are built
/// ([`count_streamed_root`]).
pub(super) fn holds_back(
    sweep: &Sweep<'_, '_>,
    run: &ApplyRun,
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
) -> bool {
    let vtree = sweep.vtree;
    let root = vtree.root();
    if !sweep.count_root
        || sweep.ws.is_some()
        || sweep.filter.is_some()
        || !sweep.quantified.is_empty()
        || vtree.node(t).parent() != Some(root)
        || f.output.vtree != root
        || g.output.vtree != root
        || run.f_widths[root.idx()] != 1
        || run.g_widths[root.idx()] != 1
    {
        return false;
    }
    let (root_left, root_right) = vtree.children(root);
    let sibling = if root_left == t { root_right } else { root_left };
    let internal = [t, left, right, sibling].iter().all(|&x| !vtree.node(x).is_leaf());
    let read_built = [left, right, sibling].iter().all(|&x| !sweep.targets.contains(x.idx()));
    let untouched = [root, t, left, right, sibling].iter().all(|&x| {
        !run.entry_marginality.either(x.idx())
            && !f.levels[x.idx()].is_marginal()
            && !g.levels[x.idx()].is_marginal()
    });
    let valued = |x: VtreeIdx| {
        [f, g].iter().any(|d| {
            let level = &d.levels[x.idx()];
            level.has_value_refs(ChildSide::Left) || level.has_value_refs(ChildSide::Right)
        })
    };
    let identity = |widths: &[usize], id: &[bool]| widths[t.idx()] == 1 && id[left.idx()] && id[right.idx()];
    internal
        && read_built
        && untouched
        && !valued(root)
        && !valued(t)
        && !identity(run.f_widths, run.f_identity)
        && !identity(run.g_widths, run.g_identity)
}

/// The widths a streamed count reads, one operand's.
fn stream_widths(widths: &[usize], c: VtreeIdx, o: VtreeIdx, cl: VtreeIdx, cr: VtreeIdx) -> StreamWidths {
    StreamWidths { c: widths[c.idx()], o: widths[o.idx()], cl: widths[cl.idx()], cr: widths[cr.idx()] }
}

/// Which of the held root children the root count would stream: the one
/// whose build would find the more candidates, by [`candidate_bound`].
pub(super) fn pick_streamed(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    held: &[(VtreeIdx, VtreeIdx, VtreeIdx)],
) -> Result<usize, OperationError> {
    if held.len() < 2 {
        return Ok(0);
    }
    let mut best = (0, 0u128);
    for (i, &(t, left, right)) in held.iter().enumerate() {
        for child in [left, right] {
            run.ensure_product_list_for_child(eng, child.idx(), run.f_widths[child.idx()], run.g_widths[child.idx()])?;
        }
        // `o` is not read by the bound; `t` stands in for it.
        let bound = candidate_bound(
            eng.limits(), f.level(t), g.level(t),
            run.products.list(left.idx()), run.products.list(right.idx()),
            stream_widths(run.f_widths, t, t, left, right),
            stream_widths(run.g_widths, t, t, left, right),
        )?;
        if i == 0 || bound > best.1 {
            best = (i, bound);
        }
    }
    Ok(best.0)
}

/// Count the root without building its held child `c`, when pricing says
/// the stream does not outweigh the build; `None` leaves `c` for the sweep
/// to build.
///
/// Every level the stream reads is built by now: `c`'s children and the
/// root's other child. Their counts are folded as a model count of the
/// finished diagram would fold them, and must all fit `u64`, which is also
/// declined otherwise, as is a read level the sweep left marginal.
pub(super) fn count_streamed_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    sweep: &Sweep<'_, '_>,
    root: (VtreeIdx, VtreeIdx, VtreeIdx),
    c: (VtreeIdx, VtreeIdx, VtreeIdx),
) -> Result<Option<num_bigint::BigUint>, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, Retention, ValueDomain};
    let vtree = sweep.vtree;
    let (t, root_left, root_right) = root;
    let (c_t, cl, cr) = c;
    let c_left = c_t == root_left;
    let o = if c_left { root_right } else { root_left };
    if [o, cl, cr].iter().any(|x| run.levels[x.idx()].is_marginal()) {
        return Ok(None);
    }
    for x in [o, cl, cr] {
        run.ensure_product_list_for_child(eng, x.idx(), run.f_widths[x.idx()], run.g_widths[x.idx()])?;
    }
    let lim = eng.limits();
    // Priced on the index sizes alone, before any count is folded: a root
    // that builds `c` after all folds its own columns.
    let read = StreamLevels { root: (t, c_left), c, o };
    let Some(pivot) = choose_pivot(lim, &read.input(run, f, g, [&[]; 3]))? else {
        return Ok(None);
    };

    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    let levels: &[TddLevel] = run.levels;
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput { vtree, levels, store: &() };
    let mut gate = lim.gate();
    for x in [o, cl, cr] {
        IntFold::ensure(eng, x, input, &mut computed, &marginal, Retention::Frontier, |w| gate.poll(w))?;
    }
    gate.flush()?;
    let column = |x: VtreeIdx| {
        let side = IntFold::child_view(x.idx(), vtree, &levels[x.idx()], &computed, &());
        (!side.view.is_marginal() && side.col.all_u64()).then(|| side.col.fast_slice())
    };
    let (Some(col_o), Some(col_cl), Some(col_cr)) = (column(o), column(cl), column(cr)) else {
        return Ok(None);
    };
    stream_count(eng, &read.input(run, f, g, [col_o, col_cl, col_cr]), pivot).map(Some)
}

/// The levels a streamed root count reads: the root with whether `c` is its
/// left child, `c` with its children, and the root's other child.
struct StreamLevels {
    root: (VtreeIdx, bool),
    c: (VtreeIdx, VtreeIdx, VtreeIdx),
    o: VtreeIdx,
}

impl StreamLevels {
    /// The stream's input, with `counts` for `o`, `cl` and `cr` (empty for
    /// pricing, which reads none).
    fn input<'a>(&self, run: &'a ApplyRun, f: &'a Tdd, g: &'a Tdd, counts: [&'a [u128]; 3]) -> StreamInput<'a> {
        let ((t, c_left), (c_t, cl, cr), o) = (self.root, self.c, self.o);
        let built = |x: VtreeIdx, counts| Built { products: run.products.list(x.idx()), counts };
        StreamInput {
            f_root: f.level(t),
            g_root: g.level(t),
            f_c: f.level(c_t),
            g_c: g.level(c_t),
            c_left,
            o: built(o, counts[0]),
            cl: built(cl, counts[1]),
            cr: built(cr, counts[2]),
            f_widths: stream_widths(run.f_widths, c_t, o, cl, cr),
            g_widths: stream_widths(run.g_widths, c_t, o, cl, cr),
        }
    }
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
///
/// A streaming route folds counts and writes neither arena, so it reserves
/// nothing and arms no growth mode. Called exactly once per level on every
/// route, so a previous level's near-cap decision cannot leak into this one.
pub(super) fn open_level_arenas(
    lim: &crate::limits::Limits,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    level: &mut TddLevel,
    route: Route,
) -> Result<(), OperationError> {
    let emits_pairs = matches!(
        route,
        Route::MarginalChild | Route::PlainDense | Route::Dense | Route::SparseMarg
    );
    if !emits_pairs {
        debug_assert!(matches!(route, Route::Stream { .. }), "the sparse route opens no arena here");
        lim.begin_level(None);
        return Ok(());
    }
    let (t, left_width, right_width) = (shape.t, shape.f.here, shape.g.here);
    // One node per live cell, and compaction only removes dead ones, so
    // `left_width * right_width` is an exact bound.
    let nodes_reserve = left_width
        .saturating_mul(right_width)
        .min(LEVEL_RESERVE_NODES_CAP)
        .max(left_width.max(right_width));
    lim.reserve(&mut level.nodes, nodes_reserve)?;

    // The one per-level emit-pair bound: every product pair emits at most
    // once, so `|f.pairs| × |g.pairs|` bounds this level's emit. Used
    // twice — once to pick the growth mode, once to size the pairs
    // arena — computed once so the two can never disagree.
    let emit_pair_bound = (f.level(t).pairs.len() as u128)
        .saturating_mul(g.level(t).pairs.len() as u128);
    lim.begin_level(Some(emit_pair_bound));
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
    Ok(())
}

/// Build every cell of this level on the row loop its route names.
///
/// A streaming marginalization target folds each cell to a value instead of
/// materializing product nodes; [`Route::MarginalChild`] runs the shared cell
/// kernel with `MarginalLookup` sides; [`Route::Dense`] and
/// [`Route::PlainDense`] assume no marginal child and read both child grids
/// positionally.
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
            run_level_rows_plain::<$dense>(eng, rows, scratch, level)?
        };
    }

    match route {
        // A streaming marginalization target collapses to Σ left × right per cell:
        // nothing downstream survives, so build each alive cell's scalar from
        // the surviving refs and never materialize a product node. Which side
        // carries counts is what picks the lookups: a marginal side is read
        // through `MarginalLookup`, which decodes a count payload or degrades
        // to a dense grid read; structural sides are read positionally.
        Route::Stream { marginal_children: true } => {
            let left = child_lookup::MarginalLookup::new(&cell_ctx.sides.left);
            let right = child_lookup::MarginalLookup::new(&cell_ctx.sides.right);
            stream_rows!(&left, &right)
        }
        Route::Stream { marginal_children: false } => {
            let left = child_lookup::DenseLookup { base: cell_ctx.sides.left.base, stride: cell_ctx.sides.left.stride };
            let right = child_lookup::DenseLookup { base: cell_ctx.sides.right.base, stride: cell_ctx.sides.right.stride };
            stream_rows!(&left, &right)
        }
        // A materializing level with at least one marginal child or
        // pass-through side. It runs the same per-cell kernel as the
        // structural routes, with `MarginalLookup` sides; isolating it here is
        // what lets those routes assume no child is marginal and no side is
        // carried through. Refs come out in the encoding the structural path
        // produces, so this level still relies on the end-of-apply tagger.
        // Both-marginal levels route here too — pure Σ left × right with no
        // structural product — and carrying them here keeps them off the
        // inline tagger that would corrupt them.
        Route::MarginalChild => run_level_rows_marginal(eng, rows, scratch, level)?,
        // The four level-invariant guards all hold, so the simplified emit
        // runs: no streaming, no dead-pair masks, no pass-through side.
        Route::PlainDense => plain_rows!(true),
        // Dead-pair masks only: the positional lookups carry nothing through.
        Route::Dense => {
            debug_assert!(
                !cell_ctx.sides.left.plan.is_passthrough() && !cell_ctx.sides.right.plan.is_passthrough(),
                "a pass-through side takes the marginal-child build"
            );
            plain_rows!(false)
        }
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
    f_pairs: &'a mut Vec<ChildPair>,
    g_pairs: &'a mut Vec<ChildPair>,
    products: &'a mut super::super::products::Products,
}

/// Build a [`Route::SparseMarg`] level and close it out.
///
/// Runs the shared emit kernel, but writes into the reused `right_width`-row scratch —
/// `cell_ctx.output_grid_base` serves directly as the row base and the action's
/// `grid_row` is 0, so a grid position is just `row_base + j` — and records each surviving cell in
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
        f_pairs, g_pairs, products,
    } = scratch;
    let (node_idx, product_list) = products.row_buffers(t.idx());
    run_level_rows_marginal_sparse(
        eng,
        rows,
        RowScratch { f_pairs, g_pairs, node_idx },
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
    sweep: &mut Sweep<'_, '_>,
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

    // Child grid geometry: product `(a, b)` sits at `base + a * g_child_width + b`.
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


    let grouped = both_multi_pair && !passthrough.left && !passthrough.right;
    let right_cols = RightColumns::build(eng, g.level(t), gw.here, sides.left.view, sides.right.view, grouped);
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
                f_pairs: run.f_pairs_scratch,
                g_pairs: run.g_pairs_scratch,
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
            f_pairs: run.f_pairs_scratch,
            g_pairs: run.g_pairs_scratch,
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

#[cfg(test)]
#[path = "tests/level.rs"]
mod tests;
