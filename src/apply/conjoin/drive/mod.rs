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

use crate::engine::Engine;

/// What one sweep carries beside its [`ApplyRun`]: the vtree, the
/// marginalization schedule, and the weight store the marginal levels write to.
pub(super) struct Sweep<'a> {
    pub(super) vtree: &'a crate::vtree::Vtree,
    pub(super) targets: MarginalTargets<'a>,
    pub(super) ws: Option<&'a mut crate::diagram::WeightStore>,
}

/// Conjunction of two diagrams over the same vtree, with optional marginalization.
///
/// `marginalize_targets` set at index `t` requests that the output's level `t`
/// be emitted as a marginal level (Boolean structure replaced by per-node model
/// counts); `None` requests no marginalization.
///
/// Both operands are spent, on `Ok` and on `Err` alike: the sweep moves or
/// drops each level of `f` and `g` as it passes it, so after an `Err` an
/// unknown prefix of both is gone. A caller that may retry keeps a clone taken
/// before the call.
///
/// On return every marginal-side ref in the result carries its slot tag; the
/// tagging is idempotent.
///
/// # Errors
///
/// `OperationError::OverBudget` when a growth step would push scratch plus output
/// past the armed byte budget, or the allocator refuses; `OperationError::OutputCap`
/// on the output-node cap; `OperationError::Stopped` on the armed deadline or a
/// stop decision. The infallible wrapper [`apply_and`] arms nothing and panics
/// on `OverBudget`.
pub(crate) fn apply_and_fallible(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    marginalize_targets: MarginalTargets<'_>,
) -> Result<Tdd, OperationError> {
    // No swap to the narrower operand here: callers of this borrowed path keep
    // per-operand bookkeeping by side. `conjoin_owned` swaps.
    let mut out = apply_and_fallible_inner(eng, f, g, marginalize_targets)?;
    // Apply emits self-describing marginal refs — bit-30 set is an inline count,
    // bit-30 clear a bare slot; see `MARGINAL_OVERFLOW_TAG` for why that polarity —
    // so a bit-30-clear ref here is never an already-inline count.
    crate::diagram::tag_all_marginal_side_slots(&mut out, None);
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
    let taken = take_level_fast_path(eng, run, f, g, shape)?;
    Ok(matches!(taken, FastPathResult::Taken))
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
    sweep: &mut Sweep<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let vtree = sweep.vtree;
    // Where this apply has got to, for a caller conjunction_progress_enabled one long merge from
    // outside it (`budget::merge_position`). The level count is the only thing
    // that costs a walk, so it is taken inside the gate; past that it is one
    // store per level and no clock at all.
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

        // Identity fast paths and the operand-child drops that precede them.
        let taken = take_fast_path(eng, run, f, g, shape)?;

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
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    lim.eager_reclaim();

    let _op = lim.begin_operation();

    // The weight store follows the diagram: the operands' stores merge into the
    // result's, and every level this apply marginalizes writes its values there.
    // Their marginal levels are the two disjoint subtrees they were built over,
    // so the merge loses nothing.
    let mut ws: Option<crate::diagram::WeightStore> =
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
        if lim.should_stop() { return Err(OperationError::Stopped); }
        let levels = diagram::take_levels(eng, num_nodes);
        let mut out = Tdd::from_levels_unchecked(
            vtree,
            levels,
            TddNodeId { vtree: f.output.vtree, local: ZERO },
        );
        out.weights = ws;
        return Ok(out);
    }

    let mut run = apply_and_setup(eng, f, g, &vtree, num_nodes, marginalize_targets, ws.is_some())?;

    // `right_identity[t]` is true when `g` computes constant-true over subtree
    // `t`, so `f`'s nodes pass through unchanged (`x ∧ 1 = x`) and the
    // construction can `mem::swap` them into the output instead of running
    // the per-node inner loop; `left_identity` is the symmetric case, where
    // `g`'s nodes are cloned across. A leaf is identity iff only the One label
    // is referenced by parent pairs; an internal node iff it is width-1 with
    // both children identity, which the sweep accretes as it goes up. The
    // predicate is incomplete; a miss only sends a small grid down the dense
    // path.
    init_leaf_identity(eng, &mut run.right_identity, g, &vtree, num_nodes)?;
    init_leaf_identity(eng, &mut run.left_identity, f, &vtree, num_nodes)?;

    apply_leaf_levels(eng, &vtree, &mut run)?;

    let canon_leaves = super::leaf_seed::seed_output_leaves(
        f, g, &vtree, &mut run.levels,
        Sides { left: &run.left_identity[..], right: &run.right_identity[..] },
        ws.as_ref(),
    );

    sweep_levels(
        eng, &mut run, f, g,
        &mut Sweep { vtree: &vtree, targets: marginalize_targets, ws: ws.as_mut() },
    )?;

    crate::marginal::canonicalize_apply_leaf_refs(&canon_leaves, &vtree, &mut run.levels, ws.as_ref());

    let out_local = compute_apply_output(f, g, &run, &vtree).unwrap_or(ZERO);
    let out_vtree = f.output.vtree;


    let levels = run.finish(eng);

    let output = TddNodeId { vtree: out_vtree, local: out_local };
    let mut out = Tdd::from_levels_unchecked(vtree, levels, output);
    out.weights = ws;
    Ok(out)
}
