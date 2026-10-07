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
//! - sparse, when the level's grid, or a child grid the dense build would
//!   have to materialize, is over the sparse gate's `min_grid` and the live
//!   products are sparse in it: scatter, filter and dedup over live products
//!   only, in the engine-owned `sparse::SparseWorkspace`, or one `f` node at
//!   a time by lookups where those are priced linear in the level's pairs
//!   (`sparse::probe`). A level with exactly one marginal child that is not a
//!   target takes the sparse-marginal variant;
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
mod loose;
mod region;
use level::{
    build_level_dense, count_sparse_root, count_streamed_root, counts_root, holds_back, pick_streamed,
    run_sparse_level, sum_sparse_root, sums_root, LevelBuild,
};

use super::*;

use super::identity::init_leaf_identity_over;
use crate::reduce::prune::settle_loose;
use crate::Engine;

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
/// For a count, up to two children of a one-product root are held back
/// unbuilt until the root: one of them may then be counted without being
/// built ([`stream`](super::sparse::stream)), and the others are built there.
///
/// # Errors
///
/// Propagates the first refusal: a budget or cap the level build hit, or the
/// armed stop, polled at the boundary of every level it builds.
fn sweep_levels(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
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
    // Children of the root held back unbuilt, for a count that may stream
    // one of them (see `level::holds_back`).
    let mut held: Vec<(VtreeIdx, VtreeIdx, VtreeIdx)> = Vec::new();
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

        if held.len() < 2 && holds_back(sweep, run, f, g, t, left, right) {
            held.push((t, left, right));
            continue;
        }
        if t == vtree.root() && !held.is_empty() {
            let c = pick_streamed(eng, run, f, g, &held)?;
            for (i, &(h, hl, hr)) in held.iter().enumerate() {
                if i != c {
                    build_level(eng, run, f, g, sweep, h, hl, hr)?;
                    output_nodes += run.levels[h.idx()].slot_count() as u64;
                }
            }
            let (c_t, c_l, c_r) = held[c];
            if let Some(count) = count_streamed_root(eng, run, f, g, sweep, (t, left, right), (c_t, c_l, c_r))? {
                sweep.counted = Some(count);
                continue;
            }
            build_level(eng, run, f, g, sweep, c_t, c_l, c_r)?;
            output_nodes += run.levels[c_t.idx()].slot_count() as u64;
        }
        build_level(eng, run, f, g, sweep, t, left, right)?;
        output_nodes += run.levels[t.idx()].slot_count() as u64;
    }
    if progress {
        lim.conjunction_reached(internal.len() as u32);
    }
    // The root has no following level boundary at which to check its output.
    lim.level_done(output_nodes + cone.free_nodes)
}

/// Build the level at `t` from its children's, by the route it is given, and
/// release the children's grids.
#[expect(clippy::too_many_arguments)]
fn build_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    sweep: &mut Sweep<'_, '_>,
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
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
        run.reclaim_child_grids(left_idx, right_idx);
        return Ok(());
    }

    // A filtered child product cannot be bypassed by an identity copy, nor
    // by a relabelling, which reads no filter.
    let taken = sweep.filter.is_none()
        && (take_level_fast_path(eng, run, f, g, shape)? || take_relabel_level(eng, run, f, g, shape, sweep)?);
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
        let route = route_level(shape, &plan, &marginal, run.sparse_gate(shape));
        route.validate(f, g, shape, &marginal, run)?;

        match route {
            Route::Sparse if counts_root(sweep, f, g, shape, &plan) => {
                sweep.counted = Some(count_sparse_root(eng, run, f, g, shape, vtree)?);
            }
            Route::Sparse => match sums_root(sweep, run, shape, &plan) {
                Some(side) => sweep.summed = sum_sparse_root(eng, run, f, g, shape, vtree, side)?,
                None => {
                    run_sparse_level(eng, run, f, g, shape, &plan, sweep.filter.is_some())?;
                    run.products.note_sparse_built(t.idx(), shape.f.here, shape.g.here);
                }
            },
            _ => build_level_dense(eng, run, f, g, LevelBuild { shape, route, plan }, sweep)?,
        }
    }

    if let Some(filter) = sweep.filter.as_deref_mut() {
        run.products.filter_level(eng, shape, filter)?;
    }
    // Release this level's children's grid regions for a later level to
    // reuse, before the next iteration's own reserve fires.
    run.reclaim_child_grids(left_idx, right_idx);
    Ok(())
}

/// How the sweep delivers its result and handles refused operands.
pub(crate) enum ConjoinMode {
    Build,
    Count,
    Restore,
    /// Build, and where the target given has a parent the sparse route
    /// builds as one product, sum the target out as the parent's pairs are
    /// found, instead of building the pairs the marginalization pass would
    /// fuse. The caller's targets must be that level alone.
    Sum(VtreeIdx),
}

/// Conjoin structural, unweighted operands, returning them intact on refusal,
/// with each operand's free levels ([`ApplyRun::cone`]) in `free`.
pub(crate) fn apply_and_kept(eng: &Engine, f: &mut Tdd, g: &mut Tdd, free: Operands<VtreeMask<'_>>) -> Result<Tdd, OperationError> {
    debug_assert!(f.weights.is_none() && g.weights.is_none() && !f.has_marginal_level() && !g.has_marginal_level());
    apply_and_core(eng, f, g, VtreeMask::default(), VtreeMask::default(), None, ConjoinMode::Restore, free)
        .map(Conjoined::diagram)
}

/// Move identity levels back to their operands, latest first.
fn give_back(levels: &mut [TddLevel], f: &mut Tdd, g: &mut Tdd, kept: &[(usize, bool)]) {
    for &(t, from_f) in kept.iter().rev() {
        let carrier = if from_f { &mut f.levels } else { &mut g.levels };
        std::mem::swap(&mut levels[t], &mut carrier[t]);
    }
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
/// Otherwise the diagram is built and the caller counts it. `free` names
/// each operand's free levels ([`ApplyRun::cone`]), which only a plain
/// conjunction may have.
#[expect(clippy::too_many_arguments)]
pub(crate) fn apply_and_core(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
    mode: ConjoinMode,
    free: Operands<VtreeMask<'_>>,
) -> Result<Conjoined, OperationError> {
    let count_root = matches!(mode, ConjoinMode::Count);
    let keep = matches!(mode, ConjoinMode::Restore);
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
    let ws: Option<crate::diagram::WeightStore> =
        match (f.weights.take(), g.weights.take()) {
            (Some(mut a), Some(b)) => {
                a.absorb(b);
                Some(a)
            }
            (a, b) => a.or(b),
        };
    debug_assert!(
        ws.is_some()
            || !f.levels.iter().chain(g.levels.iter()).any(|l| l.is_weight_marginal()),
        "an operand has weight-marginal levels but neither carries a weight store"
    );
    // Nothing but pairs changes hands: no weights, no summed-out level, no
    // level quantified or filtered, and, as the setup finds, no marginal
    // level.
    let unmarked = ws.is_none() && targets.is_empty() && quantified.is_empty() && filter.is_none();

    // Early return for zero inputs: `x ∧ 0 = 0`.
    // Avoids allocating the output's levels and arenas for unsatisfiable operands.
    let vtree = Arc::clone(&f.vtree);
    let num_nodes = vtree.num_nodes();
    if f.is_zero() || g.is_zero() {
        lim.check_stop()?;
        let levels = diagram::try_take_levels(eng, num_nodes)?;
        return diagram::Assembly::from_levels(eng, vtree, levels, ws)
            .finish(TddNodeId { vtree: f.output.vtree, local: ZERO })
            .map(Conjoined::Built);
    }

    let mut assembly = diagram::Assembly::from_levels(
        eng, Arc::clone(&vtree), diagram::try_take_levels(eng, num_nodes)?, ws,
    );
    let mut scratch = eng.scratch.apply.workspace.checkout(eng);
    let (levels, ws) = assembly.parts_mut();
    let mut run = apply_and_setup(eng, &vtree, f, g, targets, free, ws.is_some(), levels, &mut scratch)?;
    run.restoring = keep;
    let plain = unmarked && !run.entry_marginality.any();
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
        (Some(lf), Some(lg)) if plain => Some(Operands { f: settle_loose(eng, f, lf)?, g: settle_loose(eng, g, lg)? }),
        _ => None,
    };

    let canon_leaves = super::leaf_seed::seed_output_leaves(
        f, g, cone.leaves, run.levels,
        Operands { f: &run.f_identity[..], g: &run.g_identity[..] },
        ws.as_ref(),
    );
    region::take_regions(&vtree, &mut run, f, g);

    let mut sweep = Sweep {
        vtree: &vtree, targets, quantified, ws: ws.as_mut(), filter, count_root, counted: None,
        sum_child, summed: None,
    };
    if let Err(e) = sweep_levels(eng, &mut run, f, g, &mut sweep) {
        if run.restoring {
            give_back(run.levels, f, g, &run.carried);
            region::give_back_regions(run.levels, cone, f, g);
        }
        return Err(e);
    }
    if let Some(count) = sweep.counted {
        return Ok(Conjoined::Counted(count));
    }
    let summed = sweep.summed.take();

    crate::marginal::canonicalize_weighted_leaf_refs(&canon_leaves, &vtree, run.levels, ws.as_ref());

    let out_local = compute_apply_output(f, g, &run, &vtree).unwrap_or(ZERO);
    let out_vtree = f.output.vtree;
    let carried = std::mem::take(&mut run.carried);
    let relabel_moved = std::mem::take(&mut run.relabel_moved);
    let restoring = run.restoring;
    let output = TddNodeId { vtree: out_vtree, local: out_local };

    // A plain conjunction changed no level an identity fast path carried,
    // and such a level brings its whole subtree along, since the other
    // operand is the identity under it. So the result owes contraction the
    // levels it built and whatever the operands owed; seeding every level
    // made the minimization after a join as long as the diagram. A level the
    // relabelling route moved is its carrier's, but some subtree under it is
    // a product, so it is owed as built.
    let finished = if plain {
        // Which operand each level the sweep carried came from: 1 for `f`,
        // 2 for `g`, 0 for a level the conjunction built and for every
        // level outside the cone.
        let mut carrier = vec![0u8; num_nodes];
        for &(t, from_f) in &carried {
            carrier[t] = if from_f { 1 } else { 2 };
        }
        for &t in &relabel_moved {
            carrier[t] = 0;
        }
        // The result's loose levels (`loose::loose_levels`): a carried level
        // where its carrier has it loose, and a level under one the
        // conjunction built where the operands and the sibling's product
        // grid do not prove it tight. The other operand's entry at a carried
        // level is for a level the result did not take: an operand embedded
        // onto a wider scope has every level it gained loose, and a small
        // factor joined to a large diagram gained the large one's levels,
        // which made the prune after the join walk it. Listing every level
        // the conjunction built made that prune walk every one of them.
        let loose = operand_loose
            .map(|ops| loose::loose_levels(&vtree, &run, &carrier, ops));
        // The levels it built are also all it changed, and so all the seat
        // closes: a carried level, and a free one, is as the operand's end
        // left it.
        let mut built: Vec<VtreeIdx> = Vec::with_capacity(cone.built.len());
        built.extend(cone.built.iter().copied().filter(|t| carrier[t.idx()] == 0));
        let owed = crate::diagram::Dirty::stacked(&g.dirty, &f.dirty, built.len());
        assembly.finish_with_or_return(output, owed, &built, Some(&built[..])).map(|mut out| {
            if let Some(loose) = loose {
                out.dirty.set_loose(Some(loose));
                out.dirty.dedup_above(num_nodes);
            }
            out
        })
    } else {
        assembly.finish_or_return(output)
    };
    let mut out = match finished {
        Ok(out) => out,
        Err((e, mut assembly)) => {
            if restoring {
                give_back(assembly.parts_mut().0, f, g, &carried);
                region::give_back_regions(assembly.parts_mut().0, cone, f, g);
            }
            return Err(e);
        }
    };
    // Apply emits self-describing marginal refs — bit-30 set is an inline count,
    // bit-30 clear a bare slot; see `INLINE_VALUE_BIT` for why that polarity —
    // so a bit-30-clear ref here is never an already-inline count. A plain
    // conjunction has no marginal level, and so no ref to inline; leaving its
    // levels alone keeps them known closed.
    if !plain {
        crate::diagram::inline_small_marginal_refs(&mut out, None);
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
