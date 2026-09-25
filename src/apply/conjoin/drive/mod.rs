//! The bottom-up driver: one conjunction from entry to finished diagram.
//!
//! Leaf levels are filled first from the constant conjunction table
//! (`apply_leaf_levels`). Each internal level is then built one way, chosen
//! per level before any of its storage is touched:
//!
//! - identity fast path, when one operand is constant-true over the subtree:
//!   the other operand's level is moved into the output;
//! - sparse, when the level's grid, or a child grid the dense build would
//!   have to materialize, is over the sparse gate's `min_grid` and the live
//!   products are sparse in it: scatter, filter and dedup over live products
//!   only, in the engine-owned `sparse::SparseWorkspace`. A level with
//!   exactly one marginal child that is not a target takes the sparse-marginal
//!   variant;
//! - streaming, when the level is a streaming marginalization target: each
//!   cell folds to a value and no product node is built;
//! - dense, otherwise: the full grid is walked and written to the grid arena,
//!   through `MarginalLookup` sides when a child is marginal.
//!
//! `route::route_level` makes the choice.
//!
//! A self-conjunction `f ∧ f` returns `f` from `conjoin_on` before the
//! driver runs.

mod level;
use level::{build_level_dense, run_sparse_level, LevelBuild};

use super::*;

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
/// # Errors
///
/// Propagates the first refusal: a budget or cap the level build hit, or the
/// armed stop, polled at every level boundary.
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
    if progress {
        lim.conjunction_began(vtree.internal_bottomup().count() as u32);
    }
    let mut level_k: u32 = 0;
    let mut output_nodes = 0u64;
    for (t, left, right) in vtree.internal_bottomup() {
        if progress {
            level_k += 1;
            lim.conjunction_reached(level_k);
        }
        // The per-level-boundary cut check: the stop axis, then the output-node
        // cap, tracked across every route independently of sparse-grid density.
        lim.level_done(output_nodes)?;

        let shape = run.shape(t, left, right);
        let (left_idx, right_idx) = (left.idx(), right.idx());
        // Before this level's output reserve fires, so the allocator can
        // reuse the children's slabs for it.
        drop_dead_children(f, g, shape);

        if sweep.quantified.contains(t.idx()) {
            // Every leaf below this level is quantified, so the level is one
            // satisfiability test per cell and no structure at all. The
            // identity fast paths are skipped: what they would build is the
            // structure this route exists not to build.
            super::quantify::build_level_quantified(eng, run, f, g, shape)?;
            output_nodes += run.levels[t.idx()].slot_count() as u64;
            run.reclaim_child_grids(left_idx, right_idx);
            continue;
        }

        // A filtered child product cannot be bypassed by an identity copy.
        let taken = sweep.filter.is_none() && take_level_fast_path(eng, run, f, g, shape)?;

        if !taken {
            // One decision per level, taken before any of the level's storage
            // is touched: the marginal plan and the two gates read only metadata.
            let marginal = run.level_marginal(f, g, shape, sweep.targets);
            let plan = plan_marginal_level(f, g, shape, run, &marginal);
            let route = route_level(shape, &plan, &marginal, run.sparse_gate(shape));
            route.validate(f, g, shape, &marginal, run)?;

            match route {
                Route::Sparse => run_sparse_level(eng, run, f, g, shape)?,
                _ => build_level_dense(eng, run, f, g, LevelBuild { shape, route, plan }, sweep)?,
            }
        }

        if let Some(filter) = sweep.filter.as_deref_mut() {
            run.products.filter_level(eng, shape, filter)?;
        }
        output_nodes += run.levels[t.idx()].slot_count() as u64;
        // Release this level's children's grid regions for a later level to
        // reuse, before the next iteration's own reserve fires.
        run.reclaim_child_grids(left_idx, right_idx);
    }
    // The root has no following level boundary at which to check its output.
    lim.level_done(output_nodes)
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
pub(crate) fn apply_and_fallible(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
) -> Result<Tdd, OperationError> {
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

    // Early return for zero inputs: `x ∧ 0 = 0`.
    // Avoids allocating the output's levels and arenas for unsatisfiable operands.
    let vtree = Arc::clone(&f.vtree);
    let num_nodes = vtree.num_nodes();
    if f.is_zero() || g.is_zero() {
        lim.check_stop()?;
        let levels = diagram::try_take_levels(eng, num_nodes)?;
        return diagram::Assembly::from_levels(eng, vtree, levels, ws)
            .finish(TddNodeId { vtree: f.output.vtree, local: ZERO });
    }

    let mut assembly = diagram::Assembly::from_levels(
        eng, Arc::clone(&vtree), diagram::try_take_levels(eng, num_nodes)?, ws,
    );
    let mut scratch = eng.scratch.apply.workspace.checkout(lim);
    let (levels, ws) = assembly.parts_mut();
    let mut run = apply_and_setup(eng, f, g, targets, ws.is_some(), levels, &mut scratch)?;

    // `g_identity[t]` is true when `g` computes constant-true over subtree
    // `t`, so `f`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    // construction can `mem::swap` them into the output instead of running
    // the per-node inner loop; `f_identity` is the symmetric case, where
    // `g`'s nodes are cloned across. A leaf is identity iff only the One label
    // is referenced by parent pairs; an internal node iff it is width-1 with
    // both children identity, which the sweep accretes as it goes up. The
    // predicate is incomplete; a miss only sends a small grid down the dense
    // path.
    init_leaf_identity(eng, run.g_identity, g)?;
    init_leaf_identity(eng, run.f_identity, f)?;

    apply_leaf_levels(eng, &vtree, &mut run)?;

    let canon_leaves = super::leaf_seed::seed_output_leaves(
        f, g, &vtree, run.levels,
        Operands { f: &run.f_identity[..], g: &run.g_identity[..] },
        ws.as_ref(),
    );

    sweep_levels(
        eng, &mut run, f, g,
        &mut Sweep { vtree: &vtree, targets, quantified, ws: ws.as_mut(), filter },
    )?;

    crate::marginal::canonicalize_weighted_leaf_refs(&canon_leaves, &vtree, run.levels, ws.as_ref());

    let out_local = compute_apply_output(f, g, &run, &vtree).unwrap_or(ZERO);
    let out_vtree = f.output.vtree;

    let mut out = assembly.finish(TddNodeId { vtree: out_vtree, local: out_local })?;
    // Apply emits self-describing marginal refs — bit-30 set is an inline count,
    // bit-30 clear a bare slot; see `INLINE_VALUE_BIT` for why that polarity —
    // so a bit-30-clear ref here is never an already-inline count.
    crate::diagram::inline_small_marginal_refs(&mut out, None);
    Ok(out)
}
