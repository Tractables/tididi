//! The bottom-up driver: one conjunction from entry to finished diagram.
//!
//! Leaf levels are filled first from the constant conjunction table
//! (`apply_leaf_levels`). Each internal level is then built one way, chosen
//! per level before any of its storage is touched:
//!
//! - identity fast path, when one operand is constant-true over the subtree:
//!   the other operand's level is moved into the output;
//! - sparse, when `left_width * right_width` exceeds the sparse gate's
//!   `min_grid`: scatter, filter and dedup over live products only, in the
//!   engine-owned `sparse::SparseWorkspace`;
//! - dense, otherwise: the full grid is walked and written to the grid arena.
//!
//! A self-conjunction `f ∧ f` returns `f` from `conjoin_owned` before the
//! driver runs.

mod level;
use level::{build_level_dense, run_sparse_level, LevelBuild};

use super::*;

use crate::Engine;

/// What one sweep carries beside its [`ApplyRun`]: the vtree, the
/// marginalization schedule, and the weight store the marginal levels write to.
pub(super) struct Sweep<'a, 'filter> {
    pub(super) vtree: &'a crate::vtree::Vtree,
    pub(super) targets: MarginalTargets<'a>,
    /// The subtrees collapsed instead of built. See [`quantify`](super::quantify).
    pub(super) quantified: QuantifiedSubtrees<'a>,
    pub(super) ws: Option<&'a mut crate::diagram::WeightStore>,
    pub(super) filter: Option<&'a mut (dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool + 'filter)>,
}

/// Build a conjunction, emitting selected levels as marginal values and
/// collapsing every subtree `quantified` names instead of building it.
///
/// The sweep consumes operand levels as it proceeds. An error leaves both
/// operands partially drained; retrying requires copies taken before the call.
/// The result's marginal references are tagged before return.
/// Allocation, cancellation and output-cap failures return [`OperationError`].
pub(crate) fn apply_and_fallible(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    marginalize_targets: MarginalTargets<'_>,
    quantified: QuantifiedSubtrees<'_>,
) -> Result<Tdd, OperationError> {
    // No swap to the narrower operand here: callers of this borrowed path keep
    // per-operand bookkeeping by side. `conjoin_owned` swaps.
    apply_and_filtered(eng, f, g, marginalize_targets, quantified, None)
}

pub(super) fn apply_and_filtered(
    eng: &Engine, f: &mut Tdd, g: &mut Tdd,
    marginalize_targets: MarginalTargets<'_>, quantified: QuantifiedSubtrees<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
) -> Result<Tdd, OperationError> {
    let mut out = apply_and_fallible_inner(eng, f, g, marginalize_targets, quantified, filter)?;
    // Apply emits self-describing marginal refs — bit-30 set is an inline count,
    // bit-30 clear a bare slot; see `INLINE_VALUE_BIT` for why that polarity —
    // so a bit-30-clear ref here is never an already-inline count.
    crate::diagram::inline_small_marginal_refs(&mut out, None);
    Ok(out)
}

/// Drop this level's dead operand children, then try the identity fast paths.
///
/// `Ok(true)` means a fast path built the level and the caller moves on.
fn take_fast_path(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    shape: LevelShape,
) -> Result<bool, OperationError> {
    let (li, ri) = (shape.left.idx(), shape.right.idx());
    // Drop dead operand-child levels before this level's output reserve fires,
    // so the allocator can reuse their slabs for it. Sound because the body
    // reads the children only through the width snapshots taken at setup,
    // never through their arenas; the drop keeps `marginal_counts`, so
    // `is_marginal()` stays accurate.
    drop_dead_operand_level(&mut f.levels[li]);
    drop_dead_operand_level(&mut f.levels[ri]);
    drop_dead_operand_level(&mut g.levels[li]);
    drop_dead_operand_level(&mut g.levels[ri]);

    // Identity fast paths: FP1 (f carrier / g identity), FP2 (symmetric),
    // and the 0-width orphan-marginal case. See `take_level_fast_path` for
    // the full guard logic.
    take_level_fast_path(eng, run, f, g, shape)
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
    let watched = lim.watched();
    if watched {
        lim.merge_began(vtree.internal_bottomup().count() as u32);
    }
    let mut level_k: u32 = 0;
    let mut output_nodes = 0u64;
    for (t, left, right) in vtree.internal_bottomup() {
        if watched {
            level_k += 1;
            lim.merge_reached(level_k);
        }
        // The per-level-boundary cut check: the stop axis, then the output-node
        // cap, tracked across every route independently of sparse-grid density.
        lim.level_done(output_nodes)?;

        let shape = run.shape(t, left, right);
        let (left_idx, right_idx) = (left.idx(), right.idx());

        if sweep.quantified.is_whole(t.idx()) {
            // Every leaf below this level is quantified, so the level is one
            // satisfiability test per cell and no structure at all. The
            // identity fast paths are skipped: what they would build is the
            // structure this route exists not to build.
            super::quantify::drop_dead_children(f, g, shape);
            super::quantify::build_level_quantified(eng, run, f, g, shape)?;
            output_nodes += run.levels[t.idx()].slot_count() as u64;
            run.reclaim_child_grids(left_idx, right_idx);
            continue;
        }

        // Identity fast paths and the operand-child drops that precede them.
        let taken = if sweep.filter.is_some() {
            // A filtered child product cannot be bypassed by an identity copy.
            super::quantify::drop_dead_children(f, g, shape);
            false
        } else { take_fast_path(eng, run, f, g, shape)? };

        if !taken {
            // One decision per level, taken before any of the level's storage
            // is touched: the marginal plan and the two gates read only metadata.
            let plan = plan_marginal_level(f, g, shape, run);
            let marginal = run.level_marginal(f, g, shape, sweep.targets);
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

fn apply_and_fallible_inner(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    marginalize_targets: MarginalTargets<'_>,
    quantified: QuantifiedSubtrees<'_>,
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
    // Avoids allocating levels, level_base, and node_idx for unsatisfiable operands.
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
    let mut scratch = eng.apply().workspace.checkout(lim);
    let (levels, ws) = assembly.parts_mut();
    let mut run = apply_and_setup(eng, f, g, marginalize_targets, ws.is_some(), levels, &mut scratch)?;

    // `right_identity[t]` is true when `g` computes constant-true over subtree
    // `t`, so `f`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    // construction can `mem::swap` them into the output instead of running
    // the per-node inner loop; `left_identity` is the symmetric case, where
    // `g`'s nodes are cloned across. A leaf is identity iff only the One label
    // is referenced by parent pairs; an internal node iff it is width-1 with
    // both children identity, which the sweep accretes as it goes up. The
    // predicate is incomplete; a miss only sends a small grid down the dense
    // path.
    init_leaf_identity(eng, run.right_identity, g, &vtree, num_nodes)?;
    init_leaf_identity(eng, run.left_identity, f, &vtree, num_nodes)?;

    apply_leaf_levels(eng, &vtree, &mut run)?;

    let canon_leaves = super::leaf_seed::seed_output_leaves(
        f, g, &vtree, run.levels,
        Sides { left: &run.left_identity[..], right: &run.right_identity[..] },
        ws.as_ref(),
    );

    sweep_levels(
        eng, &mut run, f, g,
        &mut Sweep { vtree: &vtree, targets: marginalize_targets, quantified, ws: ws.as_mut(), filter },
    )?;

    crate::marginal::canonicalize_apply_leaf_refs(&canon_leaves, &vtree, run.levels, ws.as_ref());

    let out_local = compute_apply_output(f, g, &run, &vtree).unwrap_or(ZERO);
    let out_vtree = f.output.vtree;


    assembly.finish(TddNodeId { vtree: out_vtree, local: out_local })
}
