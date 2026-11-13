//! The bottom-up driver: one conjunction from entry to finished diagram.
//!
//! Leaf levels are filled first from the constant conjunction table
//! (`apply_leaf_levels`). Each internal level is then built one way, chosen
//! per level before any of its storage is touched:
//!
//! - identity fast path, when one operand is constant-true over the subtree:
//!   the other operand's level is moved into the output;
//! - relabelling, when one operand is one node with one pair: the other
//!   operand's pairs are carried through the children's products, and its
//!   level moved into the output when no product renumbers or dies
//!   (`relabel`);
//! - sparse, when the level's grid, the combinations of its operands' pairs,
//!   or a child grid the dense build would have to materialize, is over the
//!   sparse gate's `min_grid` and the live products are sparse in it:
//!   scatter, filter and dedup over live products only, in the engine-owned
//!   `sparse::SparseWorkspace`, or one `f` node at a time by lookups where
//!   those are priced linear in the level's pairs (`sparse::probe`). A
//!   level with exactly one marginal child that is not a target takes the
//!   sparse-marginal variant;
//! - streaming, when the level is a streaming marginalization target: each
//!   cell folds to a value and no product node is built;
//! - dense, otherwise: the full grid is walked and written to the grid arena,
//!   through `MarginalLookup` sides when a child is marginal.
//!
//! `route::route_level` makes the choice.
//!
//! Two uses of the sweep change what the root does when the sparse route
//! builds it as one product: a count ([`ConjoinMode::Count`]) folds the
//! root's pairs into the count instead of storing them, and a
//! marginalizing conjunction with one target ([`ConjoinMode::Sum`]) has the
//! target's parent, where each operand has one node, add each pair's count
//! into its fused pair instead (`sparse::ChildSum`).
//!
//! A self-conjunction `f ∧ f` returns `f` from `conjoin_on` before the
//! driver runs.

mod compose;
mod level;
mod owed;
mod region;
use level::{
    LevelBuild, build_level_dense, count_sparse_root, count_streamed_root, counts_root, defers,
    pick_streamed, run_sparse_level, sum_sparse_root, sums_root,
};

use super::*;

use super::identity::init_leaf_identity_over;
use super::price::{built_pairs, price_dense_level, refuse_past_cap};
use crate::Engine;
use crate::reduce::prune::settle_loose;

/// What one sweep carries beside its [`ApplyRun`]: the vtree, the
/// marginalization schedule, and the weight store the marginal levels write to.
pub(super) struct Sweep<'a, 'filter> {
    pub(super) vtree: &'a crate::vtree::Vtree,
    /// The levels summed out as they are built.
    pub(super) targets: VtreeMask<'a>,
    /// The subtrees collapsed instead of built. See [`quantify`](super::quantify).
    pub(super) quantified: VtreeMask<'a>,
    pub(super) ws: Option<&'a mut crate::diagram::WeightStore>,
    pub(super) filter: Option<&'a mut (dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool + 'filter)>,
    /// Whether the output is wanted only for its count: then a root level
    /// the sparse route builds as one product is counted instead, into
    /// `counted`, and left empty.
    pub(super) count_root: bool,
    pub(super) counted: Option<num_bigint::BigUint>,
    /// The one target, whose counts its parent sums as its pairs are found
    /// where the sparse route builds the parent as one product
    /// ([`ConjoinMode::Sum`]).
    pub(super) sum_child: Option<VtreeIdx>,
    /// Where the parent summed `sum_child` out: the levels it made marginal,
    /// children before parents.
    pub(super) summed: Option<Vec<VtreeIdx>>,
    /// The level counts each operand kept when the sweep began
    /// ([`Engine::attach_level_counts`](crate::Engine::attach_level_counts)):
    /// a level an identity fast path moves untouched keeps its column.
    pub(super) kept: Operands<Option<LevelCounts>>,
}

/// Walk the vtree bottom-up, building one level at a time.
///
/// At each level the product `f[i] ∧ g[j]` is computed over all node pairs,
/// by dense grid iteration or by the sparse scatter pipeline.
///
/// Each level is routed once — the route decides which of the sparse, dense
/// and streaming builds runs — and the children's grid regions are handed back
/// to the arena as soon as the parent has read them.
///
/// For a count, up to two children of a one-product root are deferred
/// unbuilt to the root: one of them may then be counted without being
/// built ([`stream`](super::sparse::stream)), and the others are built there.
///
/// # Errors
///
/// Propagates the first refusal: a budget or cap the level build hit, or the
/// armed stop, polled at the boundary of every level it builds.
fn sweep_levels(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    sweep: &mut Sweep<'_, '_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let vtree = sweep.vtree;
    // Where this apply has got to, for a caller watching one long conjunction
    // from outside it. The level count is the only thing that costs a walk, so
    // it is taken inside the gate; past that it is one store per level and no
    // clock at all.
    let progress = lim.conjunction_progress_enabled();
    let cone = run.cone;
    let internal = vtree.internal_bottomup_slice();
    if progress {
        lim.conjunction_began(internal.len() as u32);
    }
    // The nodes of the levels built so far; the free levels were taken with
    // their regions before the sweep, and their nodes are counted as if they
    // were built in their places bottom-up.
    let mut output_nodes = 0u64;
    // Children of the root deferred unbuilt, for a count that may stream
    // one of them (see `level::defers`).
    let mut deferred = std::mem::take(&mut run.lists.deferred);
    for (k, &t) in cone.built.iter().enumerate() {
        let (left, right) = vtree.children(t);
        if progress {
            // The level's place among all internal levels, from 1.
            let place = internal.partition_point(|&s| vtree.topo_pos(s) < vtree.topo_pos(t)) + 1;
            lim.conjunction_reached(place as u32);
        }
        // The per-level-boundary cut check: the stop axis, then the output-node
        // cap, tracked across every route independently of sparse-grid density.
        // A free level has no work, so the check at the next level built
        // stands for the checks at the free levels before it.
        lim.level_done(output_nodes + cone.free_before(k))?;

        if deferred.len() < 2 && defers(sweep, run, f, g, (t, left, right)) {
            deferred.push((t, left, right));
            continue;
        }
        if t == vtree.root() && !deferred.is_empty() {
            let c = pick_streamed(eng, run, f, g, &deferred)?;
            for (i, &(h, hl, hr)) in deferred.iter().enumerate() {
                if i != c {
                    build_level(eng, run, f, g, sweep, (h, hl, hr), None)?;
                    output_nodes += run.levels[h.idx()].slot_count() as u64;
                }
            }
            let (c_t, c_l, c_r) = deferred[c];
            if let Some(count) =
                count_streamed_root(eng, run, f, g, sweep, (t, left, right), (c_t, c_l, c_r))?
            {
                sweep.counted = Some(count);
                continue;
            }
            build_level(eng, run, f, g, sweep, (c_t, c_l, c_r), None)?;
            output_nodes += run.levels[c_t.idx()].slot_count() as u64;
        }
        let before = output_nodes + cone.free_before(k);
        build_level(eng, run, f, g, sweep, (t, left, right), Some(before))?;
        output_nodes += run.levels[t.idx()].slot_count() as u64;
    }
    run.lists.deferred = deferred;
    if progress {
        lim.conjunction_reached(internal.len() as u32);
    }
    super::streaming_marginal::store_kept_columns(
        eng,
        run.levels,
        run.stream_cache,
        sweep.ws.as_deref_mut(),
    )?;
    // The root has no following level boundary at which to check its output.
    lim.level_done(output_nodes + cone.free_nodes)
}

/// Build the level at `t` from its children's, by the route it is given, and
/// release the children's grids.
///
/// `before`, where given, is the output nodes the cap check at the next
/// level boundary counts besides this level's: a level of known size that
/// would fail that check is refused before it is built
/// ([`refuse_past_cap`]). A deferred child of the root, built out of its
/// place, is given none.
fn build_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    sweep: &mut Sweep<'_, '_>,
    (t, left, right): (VtreeIdx, VtreeIdx, VtreeIdx),
    before: Option<u64>,
) -> Result<(), OperationError> {
    let vtree = sweep.vtree;
    let shape = run.shape(t, left, right);
    let (left_idx, right_idx) = (left.idx(), right.idx());
    // Before this level's output reserve fires, so the allocator can
    // reuse the children's slabs for it.
    if !run.restoring {
        drop_dead_children(f, g, shape);
    }

    if sweep.quantified.contains(t.idx()) {
        // Every leaf below this level is quantified, so the level is one
        // satisfiability test per cell and no structure at all. The
        // identity fast paths are skipped: what they would build is the
        // structure this route exists not to build.
        super::quantify::build_level_quantified(eng, run, f, g, shape)?;
        run.built_marginal |= run.levels[t.idx()].is_marginal();
        run.reclaim_child_grids(left_idx, right_idx, f.output.vtree);
        return Ok(());
    }

    // A filtered child product cannot be bypassed by an identity copy, nor
    // by a relabelling, which reads no filter.
    let taken = sweep.filter.is_none()
        && (take_level_fast_path(eng, run, f, g, shape)?
            || take_relabel_level(eng, run, f, g, shape, sweep)?);
    debug_assert!(
        taken || !run.cone.free_at(t.idx()),
        "a free level at {} takes the other operand's level",
        t.idx(),
    );

    if !taken {
        // One decision per level, taken before any of the level's storage
        // is touched: the marginal plan and the two gates read only metadata.
        let marginal = run.level_marginal(f, g, shape, sweep.targets);
        let plan = plan_marginal_level(f, g, shape, run, &marginal);
        let mut route = route_level(
            &plan,
            &marginal,
            run.sparse_gate(shape, f.level(t), g.level(t)),
        );
        // A level taken from the sparse-marginal route meters its pairs as
        // that route does, which returns before the per-level tail: what it
        // charged the output-pair meter comes off at the next level that
        // settles (`Limits::level_settled`). Settled, its pairs would stay in
        // flight for the rest of the conjunction and could trip a pair floor
        // the sparse-marginal route does not.
        let mut settles = true;
        if describes_marginal_child(eng, run, (f, g), sweep, shape, (&plan, &marginal), route) {
            settles = route != Route::SparseMarg || taken_levels_settled_forced();
            route = Route::MarginalChild;
        }
        route.validate(f, g, shape, &marginal, run)?;
        note_route(t, route);

        match route {
            Route::Sparse if counts_root(sweep, f, g, shape, &plan) => {
                sweep.counted = Some(count_sparse_root(eng, run, f, g, shape, sweep)?);
            }
            Route::Sparse => match sums_root(sweep, run, shape, &plan) {
                Some(side) => sweep.summed = sum_sparse_root(eng, run, f, g, shape, vtree, side)?,
                None => {
                    run_sparse_level(eng, run, f, g, shape, &plan, sweep.filter.is_some())?;
                    run.products
                        .note_sparse_built(t.idx(), shape.f.here, shape.g.here);
                }
            },
            // Claims no grid of its own: the parent that densifies its
            // product list prices that grid.
            Route::SparseMarg => {
                build_level_dense(eng, run, f, g, LevelBuild { shape, route, plan, settles }, sweep)?
            }
            _ => {
                // Priced before its grid is claimed: a level that cannot fit
                // is refused here, named, rather than grown into the budget.
                let priced = price_dense_level(
                    eng.limits(),
                    run,
                    f,
                    g,
                    shape,
                    route,
                    sweep.filter.is_some(),
                )?;
                let filtered = sweep.filter.is_some();
                if let Some(before) = before {
                    refuse_past_cap(eng.limits(), run, (f, g), shape, (route, filtered), before)?;
                }
                #[cfg(debug_assertions)]
                let predicted = super::price::complete_level(run, (f, g), shape, route, filtered)
                    .map(|level| (level, eng.limits().mark()));
                build_level_dense(eng, run, f, g, LevelBuild { shape, route, plan, settles }, sweep)?;
                #[cfg(debug_assertions)]
                if let Some((level, mark)) = predicted {
                    let built = run.levels[t.idx()].slot_count() as u128;
                    let work = u128::from(eng.limits().work_since(mark));
                    debug_assert!(
                        (level.nodes, level.work) == (built, work),
                        "level {} built {built} nodes for {work} units where {level:?} was read",
                        t.idx(),
                    );
                }
                debug_assert!(
                    priced.is_none_or(|p| p.admits(built_pairs(&run.levels[t.idx()]))),
                    "level {} holds {} pairs where its count found {priced:?}",
                    t.idx(),
                    built_pairs(&run.levels[t.idx()]),
                );
            }
        }
    }

    run.built_marginal |= run.levels[t.idx()].is_marginal();
    if let Some(filter) = sweep.filter.as_deref_mut() {
        run.products.filter_level(eng, shape, filter)?;
    }
    if run.plain && !taken {
        owed::note_operand_level(run, f, g, vtree, t);
    } else if !taken {
        // The live pairs the level's build left are kept by the seat of a
        // plain conjunction only.
        run.levels[t.idx()].forget_left_pairs();
    }
    // Release this level's children's grid regions for a later level to
    // reuse, before the next iteration's own reserve fires.
    run.reclaim_child_grids(left_idx, right_idx, f.output.vtree);
    Ok(())
}

/// Whether a level the sparse routes were chosen for, with a marginal child
/// and not summed out, takes the marginal-child route instead, so that its
/// pairs are written as their description ([`compose::plan_marginal`]):
/// under bounded-precision weights, where the operand levels are affine and
/// each child side is complete or carried. Such a level is every cell of
/// its grid, which the sparse routes would find one product at a time.
fn describes_marginal_child(
    eng: &Engine,
    run: &ApplyRun,
    (f, g): (&Tdd, &Tdd),
    sweep: &Sweep<'_, '_>,
    shape: LevelShape,
    (plan, marginal): (&MarginalPlan, &LevelMarg),
    route: Route,
) -> bool {
    let (t, li, ri) = (shape.t, shape.left.idx(), shape.right.idx());
    if !matches!(route, Route::Sparse | Route::SparseMarg)
        || !(marginal.left_any || marginal.right_any)
        || marginal.is_target
        || !sweep.ws.as_ref().is_some_and(|w| w.is_bounded())
        || sweep.filter.is_some()
        || crate::diagram::stored_levels_forced()
        || grid_lookups_forced()
    {
        return false;
    }
    let complete = Sides {
        left: run.products.is_complete(li),
        right: run.products.is_complete(ri),
    };
    let bases = Sides {
        left: run.products.arena.materialized(li),
        right: run.products.arena.materialized(ri),
    };
    let grouped = plan.both_multi_pair
        && !plan.sides.left.is_passthrough()
        && !plan.sides.right.is_passthrough();
    compose::plan_marginal(
        eng.limits(),
        f.level(t),
        g.level(t),
        shape,
        plan.sides,
        complete,
        grouped,
        run.products.arena.slab(),
        bases,
    )
    .is_some()
}

/// Keep with `out` the operands' kept counts ([`Sweep::kept`]) of the levels
/// an identity fast path moved into it untouched, and of the levels of its
/// free regions ([`Cone`](super::setup::Cone)): such a level brings its
/// whole subtree along, so its nodes count in `out` what they counted in
/// their operand. A level the relabelling route moved keeps none, a subtree
/// under it being a product.
///
/// Infallible: this runs once `out` holds the levels the sweep moved out of
/// the operands, where a refusal could no longer give a restoring caller
/// its operands back whole. The counts are a cache, so where the budget
/// refuses their table `out` keeps none and a count folds its levels.
fn carry_kept_counts(
    eng: &Engine,
    out: &mut Tdd,
    (lists, cone): (&super::setup::SweepLists, super::setup::Cone<'_>),
    kept: &Operands<Option<LevelCounts>>,
) {
    if kept.f.is_none() && kept.g.is_none() {
        return;
    }
    let Ok(mut counts) = LevelCounts::none(eng, out.vtree.num_nodes()) else {
        return;
    };
    // The levels of the free regions are their carrier's as they were,
    // save where both operands were free and the level was built.
    let identity = lists
        .carried
        .iter()
        .filter(|&&(ti, _)| lists.identity_carrier(ti) != 0)
        .copied();
    let free = cone
        .free
        .iter()
        .filter(|t| !(cone.free_in_f(t.idx()) && cone.free_in_g(t.idx())))
        .map(|t| (t.idx(), region::carried_from_f(cone, t.idx())));
    for (ti, from_f) in identity.chain(free) {
        let t = VtreeIdx(ti as u32);
        let side = if from_f {
            kept.f.as_ref()
        } else {
            kept.g.as_ref()
        };
        let Some(column) = side.and_then(|counts| counts.column(t)) else {
            continue;
        };
        let level = &out.levels[ti];
        if !level.is_marginal() && column.len() == level.slot_count() {
            counts.set(t, Arc::clone(column));
        }
    }
    if counts.any() {
        out.levels.keep_counts(counts);
    }
}

/// How the sweep delivers its result and handles refused operands.
pub(crate) enum ConjoinMode<'a> {
    Build,
    Count,
    /// Build, and give the operands back intact on refusal. `free` names
    /// each operand's free levels ([`ApplyRun::cone`]), which only a plain
    /// conjunction may have.
    Restore { free: Operands<VtreeMask<'a>> },
    /// Build, and where the target given has a parent the sparse route
    /// builds as one product, sum the target out as the parent's pairs are
    /// found, instead of building the pairs the marginalization pass would
    /// fuse. The caller's targets must be that level alone.
    Sum(VtreeIdx),
}

/// Conjoin structural, unweighted operands, returning them intact on refusal,
/// with each operand's free levels ([`ApplyRun::cone`]) in `free`.
pub(crate) fn apply_and_restoring(
    eng: &Engine,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    free: Operands<VtreeMask<'_>>,
) -> Result<Tdd, OperationError> {
    debug_assert!(
        f.weights.is_none()
            && g.weights.is_none()
            && !f.has_marginal_level()
            && !g.has_marginal_level()
    );
    apply_and_held(
        eng,
        f,
        g,
        (VtreeMask::default(), VtreeMask::default()),
        None,
        ConjoinMode::Restore { free },
    )
    .map(Conjoined::diagram)
}

/// Give a refused conjunction's operands back what it took: identity levels
/// to their owned operands, latest first, the nodes and arenas of the levels
/// `lists.lent` names to f, which lends only where it is owned, and the free
/// regions ([`region::give_back_regions`]).
fn give_back(
    levels: &mut [TddLevel],
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    lists: &super::setup::SweepLists,
    cone: super::setup::Cone<'_>,
) {
    for &(t, from_f) in lists.carried.iter().rev() {
        match from_f {
            true => f.uncarry(t, &mut levels[t]),
            false => g.uncarry(t, &mut levels[t]),
        }
    }
    if let Some(f) = f.owned() {
        for &t in &lists.lent {
            f.levels.swap_arenas(t, &mut levels[t]);
        }
    }
    region::give_back_regions(levels, cone, f, g);
}

/// What [`apply_and_core`] returns: the conjunction, or only its model count
/// when the root was counted instead of built.
pub(crate) enum Conjoined {
    Built(Tdd),
    Counted(num_bigint::BigUint),
    /// Built, with the root's summed child ([`ConjoinMode::Sum`]) already
    /// marginal and the root's pairs fused: what the marginalization pass
    /// and pair fusion leave of the conjunction [`Self::Built`] carries,
    /// short of the slot prune.
    Summed(Tdd),
}

impl Conjoined {
    pub(super) fn diagram(self) -> Tdd {
        match self {
            Self::Built(out) | Self::Summed(out) => out,
            Self::Counted(_) => unreachable!("only a count request returns a count"),
        }
    }
}

/// Build a conjunction, emitting the levels in `targets` as marginal values,
/// collapsing every subtree `quantified` names instead of building it, and
/// dropping the intermediate products `filter` rejects.
///
/// The sweep consumes operand levels as it proceeds. An error leaves both
/// operands partially drained; retrying requires copies taken before the call.
/// No swap to the narrower operand here: callers of this borrowed path keep
/// per-operand bookkeeping by side, and `conjoin_on` swaps. The result's
/// marginal references are tagged before return. Allocation, cancellation and
/// output-cap failures return [`OperationError`].
///
/// Count mode may return the model count instead of building the root, when
/// it is one product on the sparse route with no weights and no filter.
/// Otherwise the diagram is built and the caller counts it.
pub(crate) fn apply_and_core(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    (targets, quantified): (VtreeMask<'_>, VtreeMask<'_>),
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
    mode: ConjoinMode,
) -> Result<Conjoined, OperationError> {
    apply_and_held(
        eng,
        &mut Held::Owned(f),
        &mut Held::Owned(g),
        (targets, quantified),
        filter,
        mode,
    )
}

/// [`apply_and_core`] of operands each owned or borrowed, without a filter:
/// a borrowed operand is read in place, weights and marginal levels
/// included, and an owned operand's levels are dropped as the sweep reads
/// them ([`apply_and_held`]).
pub(crate) fn apply_and_either(
    eng: &Engine,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    mode: ConjoinMode,
) -> Result<Conjoined, OperationError> {
    apply_and_held(eng, f, g, (targets, quantified), None, mode)
}

/// [`apply_and_core`] of operands each owned or borrowed ([`Held`]). A
/// borrowed operand is on the operands' vtree as it is, with no free level:
/// the caller copies one that is not so first. Its weight store is copied
/// into the result's, which takes every marginal level it has; an owned
/// operand's is moved there.
fn apply_and_held(
    eng: &Engine,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    (targets, quantified): (VtreeMask<'_>, VtreeMask<'_>),
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
    mode: ConjoinMode,
) -> Result<Conjoined, OperationError> {
    let free = match mode {
        ConjoinMode::Restore { free } => free,
        _ => Operands::default(),
    };
    debug_assert!(
        [(&*f, free.f), (&*g, free.g)]
            .into_iter()
            .all(|(op, free)| !op.is_borrowed() || free.is_empty()),
        "a borrowed operand has no free level",
    );
    let count_root = matches!(mode, ConjoinMode::Count);
    let restoring = matches!(mode, ConjoinMode::Restore { .. });
    let sum_child = match mode {
        ConjoinMode::Sum(c) => Some(c),
        _ => None,
    };
    let lim = eng.limits();
    lim.eager_reclaim();

    let _op = lim.begin_operation();

    // The weight store follows the diagram: the operands' stores merge into the
    // result's, and every level this apply marginalizes writes its values there.
    // Their marginal levels are the two disjoint subtrees they were built over,
    // so the merge loses nothing.
    let take_weights = |op: &mut Held<'_>| match op {
        Held::Owned(op) => Ok(op.weights.take()),
        Held::Borrowed(op) => op
            .weights
            .as_ref()
            .map(|ws| ws.try_clone_on(lim))
            .transpose(),
    };
    let ws: Option<crate::diagram::WeightStore> = match (take_weights(f)?, take_weights(g)?) {
        (Some(mut a), Some(b)) => {
            a.absorb(b);
            Some(a)
        }
        (a, b) => a.or(b),
    };
    debug_assert!(
        ws.is_some()
            || !f
                .levels
                .iter()
                .chain(g.levels.iter())
                .any(|l| l.is_weight_marginal()),
        "an operand has weight-marginal levels but neither carries a weight store"
    );
    // Nothing but pairs changes hands: no weights, no summed-out level, no
    // level quantified or filtered, and, as the setup finds, no marginal
    // level.
    let unmarked = ws.is_none() && targets.is_empty() && quantified.is_empty() && filter.is_none();
    // A built conjunction, restoring or not, that sums, quantifies or
    // filters no level and carries no weights takes the subtrees at which
    // both operands are constant true whole from `f`, as the identity fast
    // paths carry them: the setup finds them (`mark_constant_true`). Most levels of two
    // diagrams on a large vtree that each mention a few of its variables
    // are such levels.
    let find_free = unmarked && matches!(mode, ConjoinMode::Build | ConjoinMode::Restore { .. });

    // Early return for zero inputs: `x ∧ 0 = 0`.
    // Avoids allocating the output's levels and arenas for unsatisfiable operands.
    let vtree = Arc::clone(&f.vtree);
    let num_nodes = vtree.num_nodes();
    if f.is_zero() || g.is_zero() {
        lim.check_stop()?;
        let levels = diagram::try_take_levels(eng, num_nodes)?;
        return diagram::Assembly::from_levels(eng, vtree, levels, ws)
            .finish(TddNodeId {
                vtree: f.output.vtree,
                local: ZERO,
            })
            .map(Conjoined::Built);
    }

    let mut scratch = eng.scratch.apply.workspace.checkout(eng);
    let found_free = find_free && free.f.is_empty() && free.g.is_empty() && {
        lim.try_resize(&mut scratch.constant, num_nodes, false)?;
        super::setup::mark_constant_true(
            &vtree,
            Operands { f: &*f, g: &*g },
            &mut scratch.constant[..num_nodes],
        )
    };
    // A conjunction that consumes an owned `f` whose free regions it takes
    // adopts `f`'s level array as the result's: the regions stay where they
    // are, the sweep builds into the workspace's side array, and the finish
    // moves the levels the conjunction visited into `f`'s array. No level
    // outside them is taken from the pool, moved or given back.
    let adopted = found_free && !restoring && !f.is_borrowed();
    let levels = match adopted {
        true => {
            let mut side = std::mem::take(&mut scratch.side);
            if let Some(missing) = num_nodes.checked_sub(side.len()) {
                lim.reserve(&mut side, missing)?;
                side.resize_with(num_nodes, TddLevel::new);
            }
            side.truncate(num_nodes);
            side
        }
        false => diagram::try_take_levels(eng, num_nodes)?,
    };
    let mut assembly = diagram::Assembly::from_levels(eng, Arc::clone(&vtree), levels, ws);
    let (levels, ws) = assembly.parts_mut();
    let kind = crate::apply::conjoin::streaming_marginal::stream_kind(ws.as_ref());
    let mut run = apply_and_setup(
        eng,
        &vtree,
        Operands { f: &*f, g: &*g },
        (targets, free, found_free),
        kind,
        levels,
        &mut scratch,
    )?;
    run.restoring = restoring;
    run.adopted = adopted;
    let plain = unmarked && !run.entry_marginality.any();
    run.plain = plain;
    debug_assert!(
        free.f.is_empty() && free.g.is_empty() || plain,
        "only a plain conjunction reads free levels",
    );
    let cone = run.cone;

    // `g_identity[t]` is true when `g` computes constant-true over subtree
    // `t`, so `f`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    // construction can `mem::swap` them into the output instead of running
    // the per-node inner loop; `f_identity` is the symmetric case, where
    // `g`'s nodes are cloned across. A leaf is identity iff only the One label
    // is referenced by parent pairs; an internal node iff it is width-1 with
    // both children identity, which the sweep accretes as it goes up. The
    // predicate is incomplete; a miss only sends a small grid down the dense
    // path.
    //
    // The levels under a free level are read by no level the sweep builds.
    init_leaf_identity_over(eng, run.g_identity, g, cone.built, cone.tops, cone.leaves)?;
    init_leaf_identity_over(eng, run.f_identity, f, cone.built, cone.tops, cone.leaves)?;

    apply_leaf_levels(eng, &mut run)?;

    // The operands' loose levels as a prune would settle them, read before
    // the sweep drops the operands' levels.
    let operand_loose = match (f.dirty.loose(), g.dirty.loose()) {
        (Some(lf), Some(lg)) if plain => Some(Operands {
            f: settle_loose(eng, f, lf)?,
            g: settle_loose(eng, g, lg)?,
        }),
        _ => None,
    };

    let canon_leaves = super::leaf_seed::seed_output_leaves(
        f,
        g,
        cone.leaves,
        run.levels,
        Operands {
            f: &run.f_identity[..],
            g: &run.g_identity[..],
        },
        ws.as_ref(),
    );
    // Taken here, after the setup and before the regions or the sweep move
    // or drop any operand level: an access that changed a level would have
    // dropped them.
    let kept = Operands {
        f: f.levels.counts().cloned(),
        g: g.levels.counts().cloned(),
    };
    let closed = Operands {
        f: f.levels.is_closed(),
        g: g.levels.is_closed(),
    };
    region::take_regions(&vtree, lim, &mut run, f, g)?;
    let mut sweep = Sweep {
        vtree: &vtree,
        targets,
        quantified,
        ws: ws.as_mut(),
        filter,
        count_root,
        counted: None,
        sum_child,
        summed: None,
        kept,
    };
    if let Err(e) = sweep_levels(eng, &mut run, f, g, &mut sweep) {
        if run.restoring {
            give_back(run.levels, f, g, run.lists, cone);
        }
        return Err(e);
    }
    if let Some(count) = sweep.counted {
        return Ok(Conjoined::Counted(count));
    }
    let summed = sweep.summed.take();
    let kept = Operands {
        f: sweep.kept.f.take(),
        g: sweep.kept.g.take(),
    };
    let fused = if sweep.filter.is_none() && sweep.ws.is_none() {
        owed::fused_built_levels(&vtree, &run, f, g, sweep.targets, sweep.quantified)
    } else {
        Vec::new()
    };

    // A refusal leaves the leaves rewritten before it onto slots of the same
    // values, which the operands read alike.
    if let Err(e) = crate::marginal::canonicalize_weighted_leaf_refs(
        lim,
        &canon_leaves,
        &vtree,
        run.levels,
        ws.as_ref(),
    ) {
        if run.restoring {
            give_back(run.levels, f, g, run.lists, cone);
        }
        return Err(e);
    }

    let out_local = compute_apply_output(f, g, &run, &vtree).unwrap_or(ZERO);
    let out_vtree = f.output.vtree;
    let output = TddNodeId {
        vtree: out_vtree,
        local: out_local,
    };

    // A plain conjunction changed no level an identity fast path carried,
    // and such a level brings its whole subtree along, since the other
    // operand is the identity under it. So the result owes contraction the
    // levels it built and whatever the operands owed; seeding every level
    // made the minimization after a join as long as the diagram. A level the
    // relabelling route moved is its carrier's, but some subtree under it is
    // a product, so it is owed as built.
    //
    // The result's loose levels (`owed::loose_levels`): a carried level
    // where its carrier has it loose, and a level under one the
    // conjunction built where the operands and the sibling's product
    // grid do not prove it tight. The other operand's entry at a carried
    // level is for a level the result did not take: an operand embedded
    // onto a wider scope has every level it gained loose, and a small
    // factor joined to a large diagram gained the large one's levels,
    // which made the prune after the join walk it. Listing every level
    // the conjunction built made that prune walk every one of them.
    let loose = if plain {
        // In level order, for the lookups of the loose levels and the
        // contraction's worklists.
        run.lists.as_operand.sort_unstable();
        run.lists.outer_held.sort_unstable();
        let loose = operand_loose.map(|ops| owed::loose_levels(&vtree, &run, ops));
        owed::split_built(&vtree, &mut run);
        loose
    } else {
        None
    };
    let lists = run.lists;
    // An adopting conjunction moves the levels it visited into `f`'s array,
    // which becomes the result's, and gives the side array back to the
    // workspace with what it then holds of `f` emptied.
    let side = match adopted {
        true => {
            let f = f.owned().expect("an adopting conjunction owns f");
            let mut own = std::mem::take(&mut f.levels).into_vec();
            let levels = assembly.parts_mut().0;
            for t in cone.nodes() {
                std::mem::swap(&mut levels[t], &mut own[t]);
            }
            std::mem::swap(levels, &mut own);
            for t in cone.nodes() {
                own[t] = TddLevel::new();
            }
            Some(own)
        }
        false => None,
    };
    let finished = if plain {
        let mut owed = crate::diagram::Dirty::stacked(&g.dirty, &f.dirty, lists.built.len());
        owed.requeue(
            crate::diagram::Pass::LeafContract,
            lists.twin_free.iter().map(|t| t.0),
        );
        // A level the relabelling route moved is as its carrier's last close
        // left it; where that diagram was closed when the sweep began, the
        // live pairs it keeps are left for the seat's close.
        // The levels it built are also all it changed, and so all the seat
        // closes: a carried level, and a free one, is as the operand's end
        // left it.
        let levels = assembly.parts_mut().0;
        for &(t, from_f) in &lists.carried {
            if (if from_f { closed.f } else { closed.g }) && lists.identity_carrier(t) == 0 {
                levels[t].leave_kept_pairs();
            }
        }
        assembly
            .finish_with_or_return(output, owed, &lists.owed_twins, Some(&lists.built[..]))
            .map(|mut out| {
                if let Some(loose) = loose {
                    out.dirty.set_loose(Some(loose));
                    out.dirty.dedup_above(num_nodes);
                }
                #[cfg(debug_assertions)]
                crate::reduce::contract::debug_assert_no_twins_under(eng, &out, &lists.twin_free);
                carry_kept_counts(eng, &mut out, (lists, cone), &kept);
                out
            })
    } else {
        // Apply emits self-describing marginal refs — bit-30 set is an inline
        // count, bit-30 clear a bare slot; see `INLINE_VALUE_BIT` for why
        // that polarity — so a bit-30-clear ref here is never an already-inline
        // count. The refs are inlined before the seat, which closes the
        // levels as inlined, so that a refusal still finds the assembly to
        // give the operands their levels back from. A plain conjunction has
        // no marginal level, and so no ref to inline; leaving its levels
        // alone keeps them known closed.
        //
        // The subtrees both operands are constant true at are `f`'s levels
        // as `f`'s end left them, with no marginal level and no level above
        // them: no ref there points into a marginal level. The result owes
        // contraction what `f` owed there and the levels the conjunction
        // visited, which are also all the seat closes.
        let levels = assembly.parts_mut().0;
        match crate::diagram::inline_small_marginal_refs_in(lim, &vtree, levels, cone.built, None) {
            Ok(()) if cone.constant => {
                let owed = crate::diagram::Dirty::stacked(&Default::default(), &f.dirty, cone.built.len());
                assembly.finish_with_or_return(output, owed, cone.built, Some(cone.built))
            }
            Ok(()) => assembly.finish_or_return(output),
            Err(e) => Err((e, assembly)),
        }
    };
    let mut out = match finished {
        Ok(out) => out,
        Err((e, mut assembly)) => {
            if restoring {
                // A level the relabelling route moved goes back keeping the
                // count its carrier kept, which is the count left on it.
                for &t in &lists.relabel_moved {
                    assembly.parts_mut().0[t].settle_held_pairs();
                }
                give_back(assembly.parts_mut().0, f, g, lists, cone);
            }
            return Err(e);
        }
    };
    // Recorded where the level's marginal child is still its only one.
    for &t in &fused {
        let (left, right) = vtree.children(t);
        let marginal = |c: VtreeIdx| out.levels[c.idx()].is_marginal();
        if !marginal(t) && marginal(left) != marginal(right) {
            out.dirty.set_fused(eng, t.0);
        }
    }
    #[cfg(debug_assertions)]
    {
        let recorded: Vec<VtreeIdx> = fused
            .iter()
            .copied()
            .filter(|t| out.dirty.is_fused(t.0))
            .collect();
        crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(
            &out,
            Some(&recorded),
            "conjunction",
        );
    }
    if let Some(side) = side {
        scratch.side = side;
    }
    match summed {
        // The marks the marginalization pass leaves, which installs each
        // level on the finished diagram and marks its parent then.
        Some(installed) => {
            for d in installed {
                let parent = vtree.node(d).parent().expect("a summed level has a parent");
                out.invalidate(parent);
            }
            Ok(Conjoined::Summed(out))
        }
        None => Ok(Conjoined::Built(out)),
    }
}
