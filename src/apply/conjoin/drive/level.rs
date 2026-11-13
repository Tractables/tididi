//! One level of the bottom-up conjunction, built on the route the driver's
//! `build_level` chose: the sparse level ([`run_sparse_level`]), the dense,
//! marginal-child and streaming levels ([`build_level_dense`]), and the root
//! a count folds instead of building ([`count_sparse_root`]).

use super::Sweep;
use crate::Engine;
use crate::apply::conjoin::cell::GROUPED_MIN_PAIRS;
use crate::apply::conjoin::grid_arena::grid_cells;
use crate::apply::conjoin::price::{built_pairs, census, emit_pair_bound, price_sparse_level};
use crate::apply::conjoin::*;
use crate::diagram::ChildSide;

/// One level's build decision: its shape, the route chosen for it, the
/// marginal plan the route was chosen on, and whether the per-level tail
/// settles its pairs on the output-pair meter ([`finalize_level`]).
#[derive(Clone, Copy)]
pub(super) struct LevelBuild {
    pub(super) shape: LevelShape,
    pub(super) route: Route,
    pub(super) plan: MarginalPlan,
    pub(super) settles: bool,
}

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build the joined children's product lists, scatter-filter-dedup over
/// the live products, and record the result.
///
/// A pass-through side in `plan` is carried rather than joined, so it needs
/// no product list, and the level's fields on that side are marked inline
/// for the end-of-apply tagger, as [`Route::SparseMarg`] marks them. A
/// complete child is joined by arithmetic ([`complete_side`]), and needs no
/// product list either.
pub(super) fn run_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
    filtered: bool,
) -> Result<(), OperationError> {
    let LevelShape {
        t,
        left,
        right,
        g: gw,
        ..
    } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let passthrough = Passthrough::of(plan.sides);
    let carried = |side| passthrough.is_some_and(|p| p.side == side);
    let complete = complete_side(run, &f.vtree, shape, passthrough, filtered);
    let unlisted = |side| carried(side) || complete.is_some_and(|c| c.side == side);
    // Ensure the joined children have product lists for the scatter pipeline.
    if !unlisted(ChildSide::Left) {
        run.ensure_product_list_for_child(eng, li)?;
    }
    if !unlisted(ChildSide::Right) {
        run.ensure_product_list_for_child(eng, ri)?;
    }

    // Priced with its children's lists in hand, before the scatter claims
    // anything: a level that cannot fit is refused here, named.
    let priced = price_sparse_level(
        eng.limits(),
        run,
        (f, g),
        shape,
        (passthrough, complete),
        filtered,
    )?;

    let identity = Sides {
        left: Operands {
            f: run.f_identity[li],
            g: run.g_identity[li],
        },
        right: Operands {
            f: run.f_identity[ri],
            g: run.g_identity[ri],
        },
    };
    let (lists, cells) = run.products.lists_with_cells(
        li,
        ri,
        ti,
        Sides {
            left: gw.left,
            right: gw.right,
        },
        identity,
        filtered,
    );
    let marginal = run.entry_marginality.any() || run.built_marginal;
    let (made, held) = apply_sparse_level(
        eng,
        shape,
        (f, g),
        (run.levels, marginal),
        (lists, cells),
        run.thresholds,
        (passthrough, complete),
    )?;
    if held {
        run.lists.outer_held.push(ti);
    }
    // A sweep that gives its operands back when refused gives f the nodes
    // and arena the level took from it.
    if made == Made::F && run.restoring {
        run.lists.lent.push(ti);
    }
    if made != Made::Written {
        run.lists.as_operand.push((ti, 1));
        crate::apply::conjoin::note_as_operand(ti, true);
    }
    debug_assert!(
        priced.is_none_or(|p| p.admits(built_pairs(&run.levels[ti]))),
        "level {ti} holds {} pairs where its count found {priced:?}",
        built_pairs(&run.levels[ti]),
    );
    run.products.finish_sparse(&mut run.levels[ti], ti);
    mark_passthrough_inlined(
        &mut run.levels[ti],
        Sides {
            left: carried(ChildSide::Left),
            right: carried(ChildSide::Right),
        },
    );
    Ok(())
}

/// The child a sparse level joins by arithmetic rather than through its
/// product list ([`Complete`]): an internal child that is complete and has no
/// list yet, whose list would name every cell of its grid where the level
/// meets only the cells its operands' pairs name. None under a filter, which
/// may have dropped a cell the arithmetic names; beside a pass-through side
/// or a leaf child, which take the leaf arm's inner side themselves; or where
/// an operand is constant-true over the child, whose list is no wider than
/// the other operand's level there. Of two such children, the one with the
/// wider grid is read by arithmetic and the other listed.
pub(super) fn complete_side(
    run: &ApplyRun,
    vtree: &crate::vtree::Vtree,
    shape: LevelShape,
    passthrough: Option<Passthrough>,
    filtered: bool,
) -> Option<Complete> {
    let LevelShape {
        left,
        right,
        f: fw,
        g: gw,
        ..
    } = shape;
    if filtered
        || passthrough.is_some()
        || vtree.node(left).is_leaf()
        || vtree.node(right).is_leaf()
        || complete_sides_listed()
    {
        return None;
    }
    let open = |c: VtreeIdx| {
        let i = c.idx();
        run.products.is_complete(i)
            && !run.products.has_list(i)
            && !run.f_identity[i]
            && !run.g_identity[i]
    };
    let cells = |f_width: usize, g_width: usize| f_width as u128 * g_width as u128;
    let side = match (open(left), open(right)) {
        (true, true) if cells(fw.left, gw.left) >= cells(fw.right, gw.right) => ChildSide::Left,
        (true, false) => ChildSide::Left,
        (_, true) => ChildSide::Right,
        (false, false) => return None,
    };
    let g_width = match side {
        ChildSide::Left => gw.left,
        ChildSide::Right => gw.right,
    };
    note_complete_side(side);
    Some(Complete { side, g_width })
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
/// as a model count of the finished diagram would fold them, from the
/// operands' kept counts where a level was moved untouched
/// ([`seed_kept_counts`]); the scatter then folds each candidate in where
/// [`run_sparse_level`] would store it.
pub(super) fn count_sparse_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    sweep: &Sweep<'_, '_>,
) -> Result<num_bigint::BigUint, OperationError> {
    let vtree = sweep.vtree;
    use crate::value::{IntFold, Retention, ValueDomain};
    let LevelShape { t, left, right, .. } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    run.ensure_product_list_for_child(eng, li)?;
    run.ensure_product_list_for_child(eng, ri)?;

    let lim = eng.limits();
    let computed = fold_child_counts(eng, run, vtree, Some(sweep), &[left, right], Retention::Frontier)?;
    let levels: &[TddLevel] = run.levels;
    let mut fold = CandidateFold::new(
        lim,
        IntFold::child_view(li, vtree, &levels[li], &computed, &()),
        IntFold::child_view(ri, vtree, &levels[ri], &computed, &()),
    )?;
    let lists = run.products.lists(li, ri, ti);
    count_sparse_level(
        eng,
        shape,
        (f, g),
        levels,
        Sides {
            left: lists.left,
            right: lists.right,
        },
        None,
        &mut fold,
    )?;
    Ok(fold.finish())
}

/// Fold `roots`' integer count columns over `run.levels`, keeping what
/// `retention` keeps, seeded from the operands' kept counts
/// ([`seed_kept_counts`]) when `seed` is given.
fn fold_child_counts(
    eng: &Engine,
    run: &ApplyRun,
    vtree: &crate::vtree::Vtree,
    seed: Option<&Sweep<'_, '_>>,
    roots: &[VtreeIdx],
    retention: crate::value::Retention,
) -> Result<Vec<Option<crate::value::CountVec>>, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, ValueDomain};
    let lim = eng.limits();
    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    if let Some(sweep) = seed {
        seed_kept_counts(eng, run, sweep, &mut computed, roots)?;
    }
    let levels: &[TddLevel] = run.levels;
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput {
        vtree,
        levels,
        store: &(),
    };
    let mut gate = lim.gate();
    for &root in roots {
        IntFold::ensure(
            eng,
            root,
            input,
            &mut computed,
            &marginal,
            retention,
            |w| gate.poll(w),
        )?;
    }
    gate.flush()?;
    Ok(computed)
}

/// Seed `computed` with the operands' kept counts ([`Sweep::kept`]) for the
/// levels under `roots` that an identity fast path moved into the output
/// untouched, so the count's fold stops at them instead of walking their
/// subtrees again.
///
/// Such a level brings its whole subtree along, the other operand being
/// constant-true under it, so each of its nodes counts in the output what it
/// counted in the operand. A level the relabelling route moved is not one: a
/// subtree under it is a product. Only the topmost level of each moved
/// subtree is seeded, and only where it and every level under it are
/// structural and no target, so the output's levels there are the operand's
/// as it counted them.
fn seed_kept_counts(
    eng: &Engine,
    run: &ApplyRun,
    sweep: &Sweep<'_, '_>,
    computed: &mut [Option<crate::value::CountVec>],
    roots: &[VtreeIdx],
) -> Result<(), OperationError> {
    if sweep.kept.f.is_none() && sweep.kept.g.is_none() {
        return Ok(());
    }
    let vtree = sweep.vtree;
    // The operand each level was moved from untouched: 1 for `f`, 2 for `g`.
    let moved = |t: usize| run.lists.identity_carrier(t);
    let mut seeded = 0;
    for &(ti, from_f) in &run.lists.carried {
        let t = VtreeIdx(ti as u32);
        let side = moved(ti);
        let topmost = vtree
            .node(t)
            .parent()
            .is_none_or(|p| moved(p.idx()) != side);
        let kept = if from_f {
            sweep.kept.f.as_ref()
        } else {
            sweep.kept.g.as_ref()
        };
        let Some(column) = kept.and_then(|counts| counts.column(t)) else {
            continue;
        };
        if side == 0 || !topmost || computed[ti].is_some() {
            continue;
        }
        let under_a_root =
            std::iter::successors(Some(t), |&x| vtree.node(x).parent()).any(|x| roots.contains(&x));
        let as_counted = vtree
            .subtree(t)
            .all(|x| !run.levels[x.idx()].is_marginal() && !sweep.targets.contains(x.idx()));
        if under_a_root && as_counted && column.len() == run.levels[ti].slot_count() {
            computed[ti] = Some(column.try_clone_on(eng)?);
            seeded += 1;
        }
    }
    super::super::note_kept_counts(seeded);
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
    let LevelShape {
        t,
        left,
        right,
        f: fw,
        g: gw,
    } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    // ── Dense path: ensure children have grids ───────────────────
    //
    // Only when the arena bumps: a child processed by the sparse pipeline has
    // no grid, so materialize one. On this path the child is known ungridded —
    // there is nothing to scan — so the identity fast path is the only way to
    // build its product list.
    if run.products.arena.is_bump() {
        if run.products.arena.is_sparse(li) {
            run.materialize_dense_child(eng, li)?;
        }
        if run.products.arena.is_sparse(ri) {
            run.materialize_dense_child(eng, ri)?;
        }
    }

    // Dense output owns an f-by-g slab. Sparse marginal output reuses one
    // g-width row, records live products separately, and remains ungridded
    // until its parent needs a dense view. Only a bump arena uses that route;
    // a preplanned arena already owns the full slab.
    let cells = if use_sparse_marginal {
        gw.here
    } else {
        grid_cells(fw.here, gw.here)?
    };
    let base = run.products.arena.claim(eng, ti, cells)?;
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
/// nothing and arms no growth mode. Every level [`build_level_dense`] builds
/// calls it before it emits, so a previous level's near-cap decision cannot
/// leak into this one.
#[inline(always)]
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
        debug_assert!(
            matches!(route, Route::Stream { .. }),
            "the sparse route opens no arena here"
        );
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
    lim.reserve(level.nodes.stored_mut(), nodes_reserve)?;

    // The one per-level emit-pair bound ([`emit_pair_bound`]). A bound of
    // the arenas alone, `af·ag`, was zero against a level of one-pair
    // nodes, which seeded nothing and left the arena to double from empty.
    // Used twice — once to pick the growth mode, once to size the pairs
    // arena — computed once so the two can never disagree.
    let pair_bound = emit_pair_bound(f.level(t), g.level(t));
    lim.begin_level(Some(pair_bound));
    // Seed `level.pairs` at that bound instead of letting it double from
    // empty on every level; the emit's own `try_push_pair_into` choke
    // point, under the growth mode just armed, carries a level that
    // outgrows the cap.
    let cap = super::super::seed_pairs_forced().unwrap_or(LEVEL_RESERVE_PAIRS_CAP);
    let pairs_reserve = pair_bound.min(cap as u128) as usize;
    let pre_pairs_cap = level.pairs.capacity();
    lim.reserve(level.pairs.stored_mut(), pairs_reserve)?;
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
/// positionally. `relay` says a level of one-pair cells may be written as
/// a description ([`run_level_rows_single`]).
fn run_row_loop(
    eng: &Engine,
    (route, lookups, single_pair, pairs, relay): (Route, PlainLookups, bool, bool, bool),
    words: Option<(&mut WordScratch, Sides<usize>)>,
    (rows, scratch): (RowLoop<'_>, RowScratch<'_>),
    level: &mut TddLevel,
    env: StreamEnv<'_>,
    stream_state: &mut Option<StreamLevelState>,
) -> Result<(), OperationError> {
    let cell_ctx = rows.ctx;
    if pairs {
        debug_assert!(
            route == Route::Dense,
            "only a masked level is read by pairs"
        );
        return run_level_rows_pairs(eng, rows, scratch, level, lookups);
    }
    if let Some((words, f_rows)) = words {
        if let Route::Stream { marginal_children } = route {
            let state = stream_state
                .as_mut()
                .expect("Route::Stream implies an open stream column");
            return run_level_rows_stream_words(
                eng,
                rows,
                scratch,
                (words, f_rows, marginal_children),
                state,
                env,
            );
        }
        debug_assert!(
            matches!(route, Route::PlainDense | Route::Dense),
            "only the plain and streaming routes read cells by g-pair vectors"
        );
        return run_level_rows_words(eng, rows, scratch, words, level, lookups, f_rows);
    }
    if lookups != PlainLookups::Grid {
        debug_assert!(
            matches!(route, Route::PlainDense | Route::Dense),
            "only the plain routes read a side by arithmetic"
        );
        return run_level_rows_complete(eng, rows, scratch, level, lookups);
    }

    macro_rules! stream_rows {
        ($l:expr, $r:expr) => {
            run_level_rows_stream_count(
                eng,
                rows,
                scratch,
                $l,
                $r,
                stream_state
                    .as_mut()
                    .expect("Route::Stream implies an open stream column"),
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
        Route::Stream {
            marginal_children: true,
        } => {
            let left = child_lookup::MarginalLookup::new(&cell_ctx.sides.left);
            let right = child_lookup::MarginalLookup::new(&cell_ctx.sides.right);
            stream_rows!(&left, &right)
        }
        Route::Stream {
            marginal_children: false,
        } => {
            let left = child_lookup::DenseLookup {
                base: cell_ctx.sides.left.base,
                stride: cell_ctx.sides.left.stride,
            };
            let right = child_lookup::DenseLookup {
                base: cell_ctx.sides.right.base,
                stride: cell_ctx.sides.right.stride,
            };
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
        // runs: no streaming, no dead-pair masks, no pass-through side. Where
        // every cell is one pair against one, its loop reads g's pairs once.
        Route::PlainDense if single_pair && !super::super::single_rows_forced_off() => {
            run_level_rows_single(eng, rows, scratch, level, relay)?
        }
        Route::PlainDense => plain_rows!(true),
        // Dead-pair masks only: the positional lookups carry nothing through.
        Route::Dense => {
            debug_assert!(
                !cell_ctx.sides.left.plan.is_passthrough()
                    && !cell_ctx.sides.right.plan.is_passthrough(),
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
    (lookups, pairs): (PlainLookups, bool),
) -> Result<(), OperationError> {
    let LevelShape {
        t, f: fw, g: gw, ..
    } = shape;
    let right_level = g.level(t);
    build_side_masks::<false>(
        eng,
        (right_level, gw.here),
        ChildGrid {
            plan: plan.sides.left,
            f_width: fw.left,
            g_width: gw.left,
            base: bases.left.idx(),
        },
        (matches!(lookups, PlainLookups::CompleteLeft { .. }), pairs),
        run.products.arena.slab(),
        &mut run.prefilter_masks.left,
    )?;
    build_side_masks::<true>(
        eng,
        (right_level, gw.here),
        ChildGrid {
            plan: plan.sides.right,
            f_width: fw.right,
            g_width: gw.right,
            base: bases.right.idx(),
        },
        (matches!(lookups, PlainLookups::CompleteRight { .. }), pairs),
        run.products.arena.slab(),
        &mut run.prefilter_masks.right,
    )
}

/// The run buffers [`finish_sparse_marginal_level`] writes, borrowed field by
/// field: the output level is already split out of the same `ApplyRun`.
struct SparseMargScratch<'a> {
    f_pairs: &'a mut Vec<ChildPair>,
    g_pairs: &'a mut Vec<ChildPair>,
    products: &'a mut super::super::products::Products,
    /// The vectors of a level whose cells are read by them
    /// (`run_level_rows_marginal_sparse_words`), and f's child widths.
    words: Option<(&'a mut WordScratch, Sides<usize>)>,
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
/// A level whose cells pay to be read by g-pair vectors is read by them.
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
        f_pairs,
        g_pairs,
        products,
        words,
    } = scratch;
    let (node_idx, product_list) = products.row_buffers(t.idx());
    let scratch = RowScratch {
        f_pairs,
        g_pairs,
        node_idx,
    };
    match words {
        Some((words, f_rows)) => run_level_rows_marginal_sparse_words(
            eng,
            rows,
            scratch,
            words,
            f_rows,
            level,
            product_list,
        )?,
        None => run_level_rows_marginal_sparse(eng, rows, scratch, level, product_list)?,
    }
    products.arena.free(output_grid_base, gw.here);
    products.finish_sparse(level, t.idx());
    // The rows were built in order, so the product list numbers the live
    // cells in cell order, as a dense emit would.
    products.note_built(t.idx(), shape.f.here, gw.here, level.nodes().len());
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
/// and budget-charges. `None` — marginal-encoded g, the budget refusing the
/// arena, or a level whose rows read its columns too few times again
/// ([`column_table_pays`]) — falls back to the per-cell derivation.
fn build_cell_ctx<'a>(
    shape: LevelShape,
    plan: &MarginalPlan,
    output_grid_base: usize,
    bases: Sides<GridBase>,
    masks: Option<&'a crate::apply::conjoin::liveness::PrefilterMaskScratch>,
    right_cols: Option<&'a RightColumns>,
) -> CellCtx<'a> {
    // A level that built no masks hands the kernel empty ones, so a read of
    // a mask it did not build is out of bounds rather than stale.
    let side = |plan: SidePlan,
                base: usize,
                stride: usize,
                masks: Option<&'a liveness::PrefilterSideMasks>| ChildPlan {
        plan,
        base,
        stride: stride as u32,
        live_cols: masks.map_or(&[], |m| &m.live_cols),
        reach: masks.map_or(&[], |m| &m.reach),
        reach_cols: masks.map_or(&[], |m| &m.reach_cols),
        row_cols: masks.map_or(&[], |m| &m.row_cols),
        rows: masks.is_some_and(|m| m.rows),
    };
    CellCtx {
        output_grid_base,
        right_width: shape.g.here,
        masked: masks.is_some(),
        sides: Sides {
            left: side(
                plan.sides.left,
                bases.left.idx(),
                shape.g.left,
                masks.map(|m| &m.left),
            ),
            right: side(
                plan.sides.right,
                bases.right.idx(),
                shape.g.right,
                masks.map(|m| &m.right),
            ),
        },
        right_cols,
    }
}

/// The column reads past each column's first that a level's cells must make
/// before the column table ([`RightColumns`]) costs less than resolving a
/// column at every cell that reads it.
const COLUMN_TABLE_MIN_REREADS: usize = 32;

/// Whether a level builds its column table: where its rows read the columns
/// again often enough to pay for it, and wherever a cell could take the
/// grouped walk, which reads the table's runs. That walk needs a row and a
/// column of [`GROUPED_MIN_PAIRS`] pairs, which only an operand level of
/// that many pairs can hold, so a level without the table writes its pairs
/// in the order it would with one.
#[inline(always)]
fn column_table_pays(
    f_level: &TddLevel,
    g_level: &TddLevel,
    f_width: usize,
    g_width: usize,
    grouped: bool,
) -> bool {
    let groups = grouped
        && f_level.pairs.len() >= GROUPED_MIN_PAIRS
        && g_level.pairs.len() >= GROUPED_MIN_PAIRS;
    groups || f_width.saturating_sub(1).saturating_mul(g_width) >= COLUMN_TABLE_MIN_REREADS
}

/// How a level on `route` reads its child sides: by arithmetic on each
/// [complete](crate::apply::conjoin::products::Products::is_complete) side
/// of a plain route, from the grid otherwise. The pair capacity of a level
/// with a complete side is filled in once its arenas are open
/// ([`reserve_complete_level`]).
fn plain_lookups(route: Route, complete: Sides<bool>) -> PlainLookups {
    if grid_lookups_forced() || !matches!(route, Route::PlainDense | Route::Dense) {
        return PlainLookups::Grid;
    }
    match (complete.left, complete.right) {
        (true, true) => PlainLookups::Complete {
            charged: usize::MAX,
        },
        (true, false) => PlainLookups::CompleteLeft {
            charged: usize::MAX,
        },
        (false, true) => PlainLookups::CompleteRight {
            charged: usize::MAX,
        },
        (false, false) => PlainLookups::Grid,
    }
}

/// Reserve the arenas of a level with a complete child side to what its
/// row loop writes where no candidate dies ([`complete_level_size`]), and
/// return the pair capacity the output-pair meter has been charged for (see
/// [`ReservedEmitSink`](crate::apply::conjoin::cell::ReservedEmitSink)).
///
/// With both sides complete that is exactly what the level writes. With one
/// side complete a pair dies only on the other side, and the count bounds
/// the level from above, within one to four times what it wrote on the
/// bit-blasted circuits: the arena never grows. Grown by doubling from the
/// seed of [`open_level_arenas`] instead, the large levels of an evidence
/// walk spent a fifth of its time in the copies and page faults of the
/// growth.
///
/// Under the bounded-growth mode the arena grows as it would on the grid
/// route, and `usize::MAX` says so. The reservations are charged to the
/// budget like any other and refuse with [`OperationError::OverBudget`].
/// Inlined at both of [`build_level_dense`]'s sites, as its count of each
/// operand level is.
#[inline(always)]
fn reserve_complete_level(
    lim: &crate::limits::Limits,
    f_level: &TddLevel,
    g_level: &TddLevel,
    level: &mut TddLevel,
) -> Result<usize, OperationError> {
    if lim.bounded_growth() {
        return Ok(usize::MAX);
    }
    let charged = level.pairs.capacity();
    let (nodes, pairs) = complete_level_size(f_level, g_level)?;
    lim.reserve_exact(level.nodes.stored_mut(), nodes)?;
    lim.reserve_exact(level.pairs.stored_mut(), pairs)?;
    Ok(charged)
}

/// The nodes and the arena pairs of a level whose two child sides are
/// complete, as its row loop writes it.
///
/// No candidate dies on such a level, so every cell of a node of `f` with
/// pairs and a node of `g` with pairs is one node, holding the product of
/// their pair counts; a cell of two one-pair nodes stores its pair inline
/// and every other cell stores its pairs in the arena. Both counts take one
/// pass over the two operand levels' nodes, or none on an implicit one.
/// A count past `usize` refuses with [`OperationError::OverBudget`].
#[inline(always)]
fn complete_level_size(
    f_level: &TddLevel,
    g_level: &TddLevel,
) -> Result<(usize, usize), OperationError> {
    let (f_pairs, f_live, f_single) = census(f_level);
    let (g_pairs, g_live, g_single) = census(g_level);
    let fit = |n: u128| usize::try_from(n).map_err(|_| OperationError::OverBudget);
    Ok((
        fit(f_live * g_live)?,
        fit(f_pairs * g_pairs - f_single * g_single)?,
    ))
}

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
pub(super) fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    level: LevelBuild,
    sweep: &mut Sweep<'_, '_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let LevelBuild { shape, route, plan, settles } = level;
    let LevelShape {
        t,
        left,
        right,
        f: fw,
        g: gw,
    } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let MarginalPlan {
        sides,
        both_multi_pair,
        one_sided_masks,
        single_pair,
    } = plan;
    let passthrough = Sides {
        left: sides.left.is_passthrough(),
        right: sides.right.is_passthrough(),
    };
    let use_sparse_marginal = route == Route::SparseMarg;
    // Materialize any sparse child grid and bump-allocate this level's own.
    // The route is already known, which is what lets this skip `ensure_grid`
    // for a sparse child on a level that will take the plain-dense emit.
    let output_grid_base = materialize_children_and_grid(eng, run, shape, use_sparse_marginal)?;

    // Child grid geometry: product `(a, b)` sits at `base + a * g_child_width + b`.
    let bases = Sides {
        left: run
            .products
            .arena
            .materialized(li)
            .expect("the left child's grid is materialized"),
        right: run
            .products
            .arena
            .materialized(ri)
            .expect("the right child's grid is materialized"),
    };

    let complete = Sides {
        left: run.products.is_complete(li),
        right: run.products.is_complete(ri),
    };
    let mut lookups = plain_lookups(route, complete);
    // Only the grid-reading dead-pair liveness masks are deferred this far: they need
    // the materialized child grids, and either flag implies a route that has them.
    // With both sides complete no mask can clear, and the row loop reads none; on a
    // grid of fewer than `MASK_MIN_CELLS` cells they cost more than they cull. With
    // one multi-pair operand they are built for the row loop over two grids only: a
    // level with a complete side keeps the loop it had.
    let masked = !matches!(lookups, PlainLookups::Complete { .. })
        && if both_multi_pair {
            fw.here.saturating_mul(gw.here) >= liveness::MASK_MIN_CELLS
        } else {
            one_sided_masks && route == Route::Dense && lookups == PlainLookups::Grid
        };
    // A masked level of the dense route whose g level has few pairs in
    // all is read by pairs (`run_level_rows_pairs`).
    let pairs = masked
        && route == Route::Dense
        && !super::super::pair_rows_forced_off()
        && liveness::pairs_fit(
            g.level(t),
            gw.here,
            Sides {
                left: gw.left,
                right: gw.right,
            },
            Sides {
                left: matches!(lookups, PlainLookups::CompleteLeft { .. }),
                right: matches!(lookups, PlainLookups::CompleteRight { .. }),
            },
        );
    let mut stream_state: Option<StreamLevelState> =
        build_stream_state(eng, shape, run.levels, run.stream_cache, sweep)?;

    let grouped = both_multi_pair && !passthrough.left && !passthrough.right;
    // A summed level of two affine operand levels over child sides that are
    // each complete or carried is summed off the description of its pairs,
    // where its child columns keep every product a float; the row loop sums
    // any other. The summing reads each word of the description as a slot of
    // its child column, and sums nothing where a word lies past the column
    // (`run_level_described_stream`): a column holds fewer than 2^30 values,
    // so a word it reads has bit 30 clear and is the slot the row loop reads
    // under either child view.
    if let Some(StreamLevelState::Float(values)) = &mut stream_state
        && sweep.filter.is_none()
        && !crate::diagram::stored_levels_forced()
        && let Some((product, work)) = super::compose::plan_sums(
            lim,
            f.level(t),
            g.level(t),
            shape,
            sides,
            complete,
            grouped,
            run.products.arena.slab(),
            bases,
        )
    {
        let [level, left_level, right_level] = run
            .levels
            .get_disjoint_mut([ti, li, ri])
            .expect("a vtree node and its two children are distinct level indices");
        open_level_arenas(lim, f, g, shape, level, route)?;
        let env = StreamEnv {
            left_idx: li,
            right_idx: ri,
            vtree: sweep.vtree,
            cache: run.stream_cache,
            ws: sweep.ws.as_deref(),
        };
        let children = Sides {
            left: &*left_level,
            right: &*right_level,
        };
        let slab = run.products.arena.slab_vec();
        if run_level_described_stream(
            eng,
            &product,
            children,
            slab,
            output_grid_base.idx(),
            values,
            env,
        )? {
            lim.charge_work(work);
            lim.check_stop()?;
            finalize_level(
                eng,
                stream_state,
                shape,
                output_grid_base,
                passthrough,
                false,
                settles,
                run,
                sweep,
            );
            return Ok(());
        }
    }

    if masked {
        if !both_multi_pair {
            super::super::note_one_sided_masks();
        }
        build_level_prefilter_masks(eng, run, g, shape, &plan, bases, (lookups, pairs))?;
    }
    // A level of two complete sides whose operand levels are affine takes
    // the implicit route. Under bounded-precision weights a level of the
    // marginal-child route takes it too, over child sides that are each
    // complete or carried: its marginal sides name slots of value columns,
    // which the end of the apply leaves as they are. Any other level reads
    // its operands' pairs, an implicit operand's generated from its
    // description.
    let plain = matches!(route, Route::PlainDense | Route::Dense);
    let composed = if use_sparse_marginal
        || stream_state.is_some()
        || sweep.filter.is_some()
        || crate::diagram::stored_levels_forced()
    {
        None
    } else if matches!(lookups, PlainLookups::Complete { .. }) {
        super::compose::plan(lim, f.level(t), g.level(t), shape, grouped)
    } else if route == Route::MarginalChild
        && !grid_lookups_forced()
        && sweep.ws.as_ref().is_some_and(|w| w.is_bounded())
    {
        super::compose::plan_marginal(
            lim,
            f.level(t),
            g.level(t),
            shape,
            sides,
            complete,
            grouped,
            run.products.arena.slab(),
            Sides {
                left: Some(bases.left),
                right: Some(bases.right),
            },
        )
    } else {
        None
    };
    // The arenas the implicit route opened before it found it would have to
    // stop inside the level, or grow the arena in bounded steps, which the
    // row loop does on the written pairs: the capacity the meter was charged
    // for. The planned route runs only with nothing bounding memory, so the
    // column table built after the arenas changes nothing either reads.
    // Both routes end in the one per-level tail below the block.
    let mut opened = None;
    'routes: {
        if let Some(product) = composed {
            let level = &mut run.levels[ti];
            open_level_arenas(lim, f, g, shape, level, route)?;
            // The capacity the reservation of the level's arenas starts
            // from, which the meter is charged from. The description is
            // written without the reservation, which it would drop: its
            // capacities are charged as reserved and not allocated.
            let charged = if lim.bounded_growth() {
                usize::MAX
            } else {
                level.pairs.capacity()
            };
            if plain {
                note_lookups(PlainLookups::Complete { charged });
            }
            let (work, meter, doublings) =
                super::compose::charges((fw.here, gw.here), &product, charged);
            if charged != usize::MAX && lim.cannot_stop_within(work, meter as u64) {
                let size = complete_level_size(f.level(t), g.level(t))?;
                let cells = fw.here * gw.here;
                run.products
                    .arena
                    .fill_to(eng, output_grid_base.idx() + cells)?;
                let slab = &mut run.products.arena.slab_mut()
                    [output_grid_base.idx()..output_grid_base.idx() + cells];
                super::compose::write(
                    eng,
                    product,
                    &mut run.levels[ti],
                    size,
                    slab,
                    work,
                    (meter, doublings),
                )?;
                break 'routes;
            }
            // A level of the marginal-child route reads its sides from their
            // grids and grows its arenas as the row loop writes them.
            opened = Some(if plain {
                Some(reserve_complete_level(lim, f.level(t), g.level(t), level)?)
            } else {
                None
            });
        }

        // `t` and its two vtree children are three distinct tree nodes, so these
        // are three disjoint level slots: the streaming row loops read the child
        // columns in place while the output level is exclusively borrowed. The
        // split's borrow ends with the block, before the per-level tail retakes
        // `levels`.
        let [level, left_level, right_level] = run
            .levels
            .get_disjoint_mut([ti, li, ri])
            .expect("a vtree node and its two children are distinct level indices");
        let (left_level, right_level) = (&*left_level, &*right_level);

        let right_cols = column_table_pays(f.level(t), g.level(t), fw.here, gw.here, grouped)
            .then(|| {
                RightColumns::build(
                    eng,
                    g.level(t),
                    gw.here,
                    sides.left.view,
                    sides.right.view,
                    grouped,
                )
            })
            .flatten();
        let masks = masked.then_some(&*run.prefilter_masks);
        let cell_ctx = build_cell_ctx(
            shape,
            &plan,
            output_grid_base.idx(),
            bases,
            masks,
            right_cols.as_ref(),
        );

        match opened {
            Some(Some(charged)) => lookups = PlainLookups::Complete { charged },
            Some(None) => note_lookups(lookups),
            None => {
                open_level_arenas(lim, f, g, shape, level, route)?;
                if let PlainLookups::Complete { charged }
                | PlainLookups::CompleteLeft { charged }
                | PlainLookups::CompleteRight { charged } = &mut lookups
                {
                    *charged = reserve_complete_level(lim, f.level(t), g.level(t), level)?;
                }
                note_lookups(lookups);
            }
        }

        let rows = RowLoop {
            f_level: f.level(t),
            g_level: g.level(t),
            children: Sides {
                left: left_level,
                right: right_level,
            },
            ctx: &cell_ctx,
            f_width: fw.here,
        };
        let f_rows = Sides {
            left: fw.left,
            right: fw.right,
        };

        if use_sparse_marginal {
            // Read by g-pair vectors where they pay.
            let words = !(passthrough.left && passthrough.right)
                && super::super::word_cells_forced().unwrap_or_else(|| {
                    words_pay_marginal(
                        eng,
                        rows,
                        &mut *run.words,
                        true,
                        f_rows,
                        run.products.arena.slab(),
                    )
                });
            return finish_sparse_marginal_level(
                eng,
                shape,
                rows,
                output_grid_base,
                level,
                SparseMargScratch {
                    f_pairs: run.f_pairs_scratch,
                    g_pairs: run.g_pairs_scratch,
                    products: run.products,
                    words: words.then_some((&mut *run.words, f_rows)),
                },
                passthrough,
            );
        }

        // A level of the plain or streaming routes whose candidates
        // outnumber what the vectors of its child rows cost reads its cells
        // by g-pair vectors (`run_level_rows_words`,
        // `run_level_rows_stream_words`); a streaming level needs a side
        // that is not carried through.
        let words = !pairs
            && match route {
                Route::PlainDense => !single_pair && stream_state.is_none(),
                Route::Dense => stream_state.is_none(),
                Route::Stream { .. } => !(passthrough.left && passthrough.right),
                _ => false,
            }
            && !matches!(lookups, PlainLookups::Complete { .. })
            && super::super::word_cells_forced().unwrap_or_else(|| {
                let slab = run.products.arena.slab();
                match route {
                    Route::Stream { marginal_children } => words_pay_marginal(
                        eng,
                        rows,
                        &mut *run.words,
                        marginal_children,
                        f_rows,
                        slab,
                    ),
                    _ => words_pay(eng, rows, &mut *run.words, lookups, f_rows, slab),
                }
            });
        // Nothing filters the level's nodes or folds them into counts.
        let relay = sweep.filter.is_none() && stream_state.is_none();
        run_row_loop(
            eng,
            (route, lookups, single_pair, pairs, relay),
            words.then_some((&mut *run.words, f_rows)),
            (
                rows,
                RowScratch {
                    f_pairs: run.f_pairs_scratch,
                    g_pairs: run.g_pairs_scratch,
                    node_idx: run.products.arena.slab_vec(),
                },
            ),
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
    }

    // Per-level tail: stream commit, live_counts, grid tag, shrink,
    // pass-through flags, and the close of a level of the plain routes. See
    // `finalize_level`.
    let close = plain && sweep.filter.is_none() && lim.memory_unbounded();
    finalize_level(
        eng,
        stream_state,
        shape,
        output_grid_base,
        passthrough,
        close,
        settles,
        run,
        sweep,
    );
    Ok(())
}

#[cfg(test)]
#[path = "tests/level.rs"]
mod tests;
