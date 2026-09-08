//! The bottom-up driver: one conjunction from entry to finished diagram.

use super::*;
use crate::diagram::NodeIdx;

use crate::engine::Engine;

/// Conjunction of two TDDs over the same vtree, with optional marginalization.
///
/// A level-by-level product construction that yields a fresh canonical TDD.
/// `c1` and `c2` are mutable because per-level scratch / packed encodings may be
/// stripped as their information moves into the output — the underlying TDDs are
/// not semantically modified.
///
/// # Arguments
///
/// - `marginalize_targets`: optional `&[bool]` indexed by `VtreeIdx`. `true` at
///   index `t` requests that the output's level `t` be turned into a *marginal*
///   level (Boolean structure replaced with per-node model counts) during this
///   apply. `None` requests no
///   marginalization.
///
/// # Per-level dispatch
///
/// For each vtree level, the product `k1 × k2` is built in one of several
/// modes, picked locally per level:
///
/// - **Identity fast path** (one operand is constant-true at this subtree):
///   `mem::swap` the other side's level into the output. Zero work, no
///   allocation. Detection propagates bottom-up via the per-operand identity
///   flags seeded by `init_leaf_identity`.
/// - **Self-conjunction** at the top: short-circuit `f ∧ f → f.clone()` and
///   skip the entire traversal.
/// - **Sparse mode** (`k1 * k2 > SPARSE_THRESHOLD`): scatter-filter-dedup over
///   live products only. The
///   reverse-index buckets live in `sparse::SparseWorkspace` (engine-owned).
/// - **Dense mode** (default): iterate the `k1 × k2` grid with 1×1 / N×1 / 1×N
///   / N×M specializations. The dense scratch is a single flat `node_idx`
///   array reused across levels via `Vec::with_capacity` + lazy DEAD-fill.
///
/// Leaf levels are handled separately by `apply_leaf_levels`, which fills the
/// grid from the static `CONJOIN_GRID` (a 3×3 conjunction table).
///
/// # `OverBudget` recovery contract
///
/// Returns `Err(ApplyError::OverBudget)` if any growth step would push cumulative
/// scratch + output past the soft budget held in
/// [`set_apply_budget`]. A caller takes this as the signal to roll back to its
/// pre-apply snapshot and try a case-split. The
/// per-apply in-flight counter (`ApplyLimits::budget_in_flight`) is reset at the top of
/// every call so prior apply growth doesn't leak into this one's budget check.
///
/// Other failure modes (allocator OOM not gated by the budget) also bubble up
/// as `OverBudget` — the infallible wrapper [`apply_and`] panics
/// rather than handle them.
///
/// # End-of-apply slot tagging
///
/// This function is also the tagging wrapper around
/// the apply core: the end-of-apply chokepoint where every
/// persisted marg-side ref in the freshly-built result gets its slot tag
/// (bit 30) set. This runs *after* all intra-apply structural reads (which use
/// raw indices) and *before* the result reaches minimize / canon / a
/// subsequent apply / query — exactly the boundary the strict decode assert in
/// `resolve_marg_ref` audits. Idempotent, so the accumulator's repeated
/// re-tagging across batches is harmless.
///
/// # Operand-state contract
///
/// **On `Err`, `c1` and `c2` are CONSUMED / left in an
/// unspecified state.** The bottom-up loop drains dead operand-child levels in
/// place as it goes (`drop_dead_operand_level`), so on an `Err(OverBudget)` /
/// `Err(Deadline)` an unknown prefix of both operands' levels has already been
/// stolen. Callers MUST NOT reuse `c1`/`c2` after an `Err` — rebuild them (from
/// a clone taken before the call) if a retry is needed. On
/// `Ok`, the operands are likewise spent (their
/// levels moved into the result / recycled); the contract is the same, it just
/// matters most on the error path where a naive caller might try to reuse them.
pub(crate) fn apply_and_fallible(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    marginalize_targets: Option<&[bool]>,
) -> Result<Tdd, ApplyError> {
    // NB: no operand swap-to-narrower here. That optimization lives ONLY in the
    // owned wrappers (`conjoin_owned`), NOT on this
    // shared borrowed path. Order-sensitive callers reach apply through here,
    // and a swap would silently rebind their per-operand bookkeeping to the
    // wrong side. The borrowed/owned asymmetry is intentional.
    let mut out = apply_and_fallible_inner(eng, c1, c2, marginalize_targets, &FullPlan)?;
    // Apply emits self-describing marg refs — bit-30 set is an inline count,
    // bit-30 clear a bare slot; see `MARG_OVERFLOW_TAG` for why that polarity —
    // so a bit-30-clear ref here is never an already-inline count.
    crate::diagram::tag_all_marg_side_slots(&mut out, None);
    Ok(out)
}

/// Spine-bounded variant of [`apply_and_fallible`]: the SAME apply core, run
/// over the restricted level set `restrict.rebuild` and merged back into `c1`'s
/// own level array. See the `restrict` module for what `R` is and why the
/// result is bit-identical to the unrestricted apply.
///
/// Only `restrict::conjoin_batch` calls this; it owns the decline
/// checks that make the restriction sound.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a buffer reservation is refused.
pub(super) fn apply_and_fallible_restricted(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    restrict: &Restrict<'_>,
) -> Result<Tdd, ApplyError> {
    let mut out = apply_and_fallible_inner(eng, c1, c2, None, &RestrictedPlan(restrict))?;
    // Restricted tagger domain: `tag_all_marg_side_slots` only does work at a
    // STRUCTURAL level with at least one MARGINAL child, and every such level
    // is in `R` by construction (that is what `AncClosure(P)` collects). Off
    // `R` neither the level nor its children changed, so the sweep there would
    // re-derive the accumulator's existing tags — restricting it is
    // result-identical, not merely sound.
    crate::diagram::tag_all_marg_side_slots_at(&mut out, None, Some(restrict.rebuild));
    Ok(out)
}


/// Mark the output's marginal vtree LEAVES, which the bottom-up loop never
/// visits as a level of its own, and collect the weight-marginal leaves whose
/// refs still have to be canonicalized once the output's pairs are final.
///
/// A marginalized leaf variable is private to one operand, so the other is the
/// identity there and the parent's marginal-child dispatch carries the refs
/// through. A restricted apply skips the sweep: its output levels are merged
/// back into the accumulator's, which already carries its own marginal leaves.
#[allow(clippy::too_many_arguments)]
pub(super) fn seed_marginal_leaves<P: ApplyPlan>(
    c1: &Tdd,
    c2: &Tdd,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    plan: &P,
    c1_identity: &[bool],
    c2_identity: &[bool],
    ws: Option<&crate::weight_store::WeightStore>,
) -> Vec<usize> {
    // Leaf marginalization: a marginal vtree LEAF is never visited as a `t` by
    // the bottom-up loop, so — unlike a marginal internal child — its OUTPUT level
    // is never marked marginal. Seed it here from the operands. A marginalized
    // leaf var is PRIVATE (summed only once its every clause is compiled), so the
    // other operand is identity at that leaf; the parent's marginal-child dispatch
    // then routes Route A and the passthrough path carries the carrier's inline
    // `ValueRef` refs through verbatim. The output store stays empty (all leaf
    // counts are inline at the parent).
    // In weighted mode the leaf's counts are NOT inline at the parent: the
    // weighted leaf-marg installs a real per-slot column in the (vtree-indexed)
    // `WeightStore` and leaves the parent's bare leaf-label refs to
    // decode as `ValueRef::Slot`. That column is PINNED — immutable, label-ordered,
    // exactly `LEAF_WIDTH` slots, never compacted / erased / appended to by any
    // pass — so the output level reports `LEAF_WIDTH` and this only re-flags it.
    //
    // `canon_leaves` collects the leaves flagged weight-marginal on ONE operand's
    // authority: the other operand was structural there, so its genuine leaf-LABEL
    // refs flow through `CONJOIN_GRID` into the output and may not be canonical
    // (`marginalize::leaf_canon_map`). They are canonicalized once the output's
    // pair lists are final — the bottom-up loop BELOW emits them, so there is
    // nothing to rewrite here yet. When BOTH operands are weight-marginal both
    // sides are already canonical and the grid is closed over each canon class
    // (`{One,Pos}`, `{One,Neg}`, `{One}` are each closed under ∧), so nothing is
    // recorded.
    // Restricted mode skips this sweep entirely. The output level array is
    // merged back into the ACCUMULATOR's, so a marginal accumulator leaf keeps
    // its own level (already marginal) rather than needing to be re-seeded; the
    // batch has no marginal levels at all; and a restricted apply carries no
    // weight store, so `canon_leaves` would stay empty. The only consumer of the
    // seeded flag inside the loop is the "output child is marginal" test, which
    // reads through to `c1.levels[..]` under a restriction (see below).
    let mut canon_leaves: Vec<usize> = Vec::new();
    if plan.output_lives_in_accumulator() {
        return canon_leaves;
    }
    for (leaf, _) in vtree.leaf_bottomup() {
        let li = leaf.idx();
        let c1m = c1.levels[li].is_marginal();
        let c2m = c2.levels[li].is_marginal();
        if c1m || c2m {
            debug_assert!(
                (c1m && c2m) || (c1m && c2_identity[li]) || (c2m && c1_identity[li]),
                "marginal leaf {li} conjoined with a non-identity operand \
                 (var not private?): c1m={c1m} c2m={c2m} \
                 c1_id={} c2_id={}",
                c1_identity[li], c2_identity[li],
            );
            let w1 = c1.levels[li].is_weight_marginal();
            let w2 = c2.levels[li].is_weight_marginal();
            if w1 || w2 {
                // PIN INVARIANT (`marginalize::marginalize_leaf_weighted`): a
                // weight-marginal LEAF's column is an IMMUTABLE, label-ordered,
                // exactly-`LEAF_WIDTH` cache of `WeightStore::leaf_val`. No pass
                // compacts, erases, reorders or appends to it — slot-prune,
                // dup-resolve's twin fold, weighted p-fusion and the subsumption
                // reclaim all decline at leaves — so the output level's slot count
                // is `LEAF_WIDTH`, full stop.
                //
                // Flagging it directly (rather than reading the column's length)
                // is what makes this robust: a `map_or(0, len)` read reports width
                // 0 whenever the global column happens not to be installed for
                // this vtree index, and a width-0 weight-marginal leaf is silently
                // skipped by `marginalize_batch_weighted` and read as an empty
                // column by the streaming child view — dropping the leaf's entire
                // mass with no error anywhere.
                let leaf_slots = crate::diagram::LEAF_WIDTH;
                debug_assert!(
                    ws.is_none_or(|w| w.level(li).is_none_or(|v| v.len() == leaf_slots)),
                    "weight-marginal leaf {li}: WeightStore column is not the \
                     pinned {leaf_slots}-slot leaf_val cache",
                );
                debug_assert!(
                    (!w1 || c1.levels[li].width() == leaf_slots)
                        && (!w2 || c2.levels[li].width() == leaf_slots),
                    "weight-marginal leaf {li}: operand slot carriers \
                     (c1={}, c2={}) disagree with LEAF_WIDTH ({leaf_slots})",
                    c1.levels[li].width(), c2.levels[li].width(),
                );
                levels[li].make_marginal_weighted_with_slots(leaf_slots as u32);
                if w1 != w2 {
                    canon_leaves.push(li);
                }
            } else {
                levels[li].make_marginal(Vec::new(), None);
            }
        }
    }
    canon_leaves
}

/// Clamp an output index that cannot fit the root level's effective width to
/// `ZERO`.
///
/// A conjunction whose product is FALSE leaves the root level with no
/// materialized slot, but the grid branch of the output computation can read a
/// cell the sparse route never wrote and return an index one past the end,
/// which the later passes would use to index their arenas. The width test is
/// the same one those passes index by, so the guard fires exactly where they
/// would fault, and a true result — whose root always has a slot — never
/// reaches it.
fn guard_stale_false(
    out_local: NodeIdx,
    out_vtree: VtreeIdx,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
) -> NodeIdx {
    // Stale-grid FALSE guard. A real (materializing) conjoin whose product is
    // FALSE leaves the output level with zero materialized slots, but
    // `compute_apply_output`'s grid branch reads a stale `node_idx` cell (0,
    // never DEAD-filled in sparse mode) and returns local 0 — an index past the
    // level's slot count. `prune`/`minimize` then index one-past-end of the
    // `remap` arena and panic (prune.rs classic_mark). An `out_local` that
    // cannot fit in the root level's *effective* width is exactly that stale
    // read: the product is FALSE, so emit ZERO (prune early-returns on is_zero,
    // sidestepping the OOB). A valid result always indexes an existing slot
    // (out_local < width), and constant-TRUE keeps width ≥ 1 at every internal
    // level, so this never misfires on a true result.
    //
    // Mirror prune's indexing exactly: `effective_width` (LEAF_WIDTH for leaves,
    // marginal_counts.len() for marginal levels, node count otherwise) — NOT raw
    // `width()` — so the guard fires on the same one-past-end that prune would,
    // including marginal roots under WS_MARGINALIZE and leaf roots.
    {
        let out_ti = out_vtree.idx();
        let eff_width = if vtree.node(VtreeIdx(out_ti as u32)).is_leaf() {
            crate::diagram::LEAF_WIDTH
        } else {
            levels[out_ti].width()
        };
        if out_local != ZERO && (out_local.0 as usize) >= eff_width {
            ZERO
        } else {
            out_local
        }
    }
}


/// Drop this level's dead operand children, then try the identity fast paths.
///
/// `Ok(true)` means a fast path built the level and the caller moves on.
///
/// Restricted mode calls neither half: `c1` is the accumulator and its off-`R`
/// levels ride through into the output verbatim, so their arenas are still live
/// data; and `R` is by construction the set of levels where no fast path fires.
fn try_fast_paths(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
) -> Result<bool, ApplyError> {
    let LevelShape { t, t_idx, left_idx, right_idx, k1, k2, .. } = shape;
    // Drop dead operand-child levels at the START of the iteration: this
    // level's output reserve — a single multi-GB allocation — fires
    // mid-iteration, and freeing the children first is what lets the
    // allocator reuse their slabs for it. Sound because this iteration's
    // body reads the children only through the precomputed `c?_widths`
    // snapshot, never through their arenas. The drop preserves
    // `marginal_counts`, so `is_marginal()` stays accurate.
    // Restricted mode does NOT drop operand child levels: `c1` is the
    // accumulator and its off-`R` levels ride through into the output
    // verbatim (the output array IS c1's, merged at the tail). Generically
    // these drops are free — an FP1'd child was already swapped out of `c1`,
    // so the call sees an empty placeholder — but under a restriction the
    // level is still live data.
    //
    // Nor does it attempt an identity fast path: `R` is by construction the
    // set of levels where none fires (see the `restrict` module), and the
    // OUTPUT-child marginality the guards read lives in `c1.levels[..]`
    // here, not in the fresh `levels[..]`.
    drop_dead_operand_level(&mut c1.levels[left_idx]);
    drop_dead_operand_level(&mut c1.levels[right_idx]);
    drop_dead_operand_level(&mut c2.levels[left_idx]);
    drop_dead_operand_level(&mut c2.levels[right_idx]);

    // Identity fast paths: FP1 (c1 carrier / c2 identity), FP2 (symmetric),
    // and the 0-width orphan-marginal case. See `try_level_fast_paths` for
    // the full guard logic.
    let taken = try_level_fast_paths(eng,
        c1, c2, t,
        k1, k2, t_idx, left_idx, right_idx,
        run.might_use_sparse,
        &mut run.levels, &mut run.c1_identity, &mut run.c2_identity,
        &mut run.live_counts, &mut run.out_nodes_so_far, &mut run.grids, &mut run.node_idx,
    )?;
    Ok(matches!(taken, FastPathResult::Taken))
}

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build both children's product lists, scatter-filter-dedup over the
/// live products, and record the result.
fn run_sparse_level(
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
    ensure_product_list_for_child(
        eng,
        left_idx, k1_left, k2_left,
        &run.c1_identity, &run.c2_identity, &run.grids, &run.node_idx,
        &mut run.product_lists, &mut run.has_pl,
    )?;
    ensure_product_list_for_child(
        eng,
        right_idx, k1_right, k2_right,
        &run.c1_identity, &run.c2_identity, &run.grids, &run.node_idx,
        &mut run.product_lists, &mut run.has_pl,
    )?;

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
        &mut run.live_counts, &mut run.out_nodes_so_far, &mut run.has_pl,
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
) -> Result<usize, ApplyError> {
    let LevelShape { t_idx, left_idx, right_idx, k1, k2, k1_left, k2_left, k1_right, k2_right, .. } = shape;
    // ── Dense path: ensure children have grids ───────────────────
    //
    // Only when might_use_sparse: if a child was processed by the sparse
    // pipeline (no grid), materialize its grid (`ensure_grid`).
    //
    // Note: this branch uses `fill_identity_product_list` directly (not
    // `ensure_product_list_for_child!`) because on the dense path we know
    // the child is sparse — no grid to scan — so the identity fast path
    // is the only way to build the product list.
    // Threaded out of the alloc block below: the k2-row scratch base used by
    // the sparse-marg path (0 when that path is not taken).
    let mut sparse_marg_row_base = 0usize;
    if run.might_use_sparse {
        if run.grids[left_idx].is_sparse() {
            materialize_dense_child(
                eng,
                left_idx, k1_left, k2_left,
                run.c2_identity[left_idx], run.c1_identity[left_idx],
                &mut run.has_pl[left_idx], &mut run.product_lists[left_idx],
                &mut run.grids, &mut run.node_idx, &mut run.grid_end, &mut run.free_regions,
            )?;
        }
        if run.grids[right_idx].is_sparse() {
            materialize_dense_child(
                eng,
                right_idx, k1_right, k2_right,
                run.c2_identity[right_idx], run.c1_identity[right_idx],
                &mut run.has_pl[right_idx], &mut run.product_lists[right_idx],
                &mut run.grids, &mut run.node_idx, &mut run.grid_end, &mut run.free_regions,
            )?;
        }

        // Bump-allocate grid for this level. Kind will be overwritten to
        // DenseStrict at the end of the dense emit loop below; use a
        // placeholder variant (DenseWeak) until then so consumers that
        // peek here (e.g. debug_assert paths) see a consistent base.
        //
        // Sparse-marg path: allocate only a single reused k2-row scratch
        // instead of the dense k1*k2 slab. `run_level_rows_marg_sparse`
        // processes one structural row at a time into this scratch, records
        // the surviving cells into the output product_list, then frees the
        // scratch — the dense slab is never materialized. The level is tagged
        // Sparse here; the grandparent densifies it lazily via ensure_grid.
        let cells = if use_sparse_marg { k2 } else { k1 * k2 };
        let base = grid_alloc(eng, &mut run.node_idx, &mut run.grid_end, &mut run.free_regions, cells)?;
        if use_sparse_marg {
            run.grids[t_idx] = LevelGrid::Sparse;
        } else {
            run.grids[t_idx] = LevelGrid::DenseWeak { base };
        }
        sparse_marg_row_base = base;
    }

    // For the sparse-marg path `grids[t_idx]` is Sparse (no slab base), so the
    // row scratch base is threaded out of the alloc block explicitly.
    Ok(if use_sparse_marg { sparse_marg_row_base } else { run.grids[t_idx].base_unchecked() })
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
    stream_computed: &[Option<CountVec<ApplyBudget>>],
    stream_computed_weights: &[Option<Vec<crate::query::WeightVal>>],
    stream_state: &mut Option<StreamLevelState>,
    level: &mut TddLevel,
    left_level: &TddLevel,
    right_level: &TddLevel,
    ws: Option<&crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    // Both operand borrows are taken before `level` — a `&mut` into the OUTPUT
    // levels, a separate allocation — is used, then lent on as plain refs.
    let c1_level_t: &TddLevel = c1.level(t);
    let c2_level_t: &TddLevel = c2.level(t);

    // Marginal sides are read through `MargLookup`, which decodes a count
    // payload or degrades to a dense grid read; structural sides are read
    // positionally, which inlines to the original `get_unchecked` index.
    let left_marg = child_lookup::MargLookup::left(cell_ctx);
    let right_marg = child_lookup::MargLookup::right(cell_ctx);
    let left_dense = child_lookup::DenseLookup { base: cell_ctx.left_base, k2: cell_ctx.k2_left };
    let right_dense = child_lookup::DenseLookup { base: cell_ctx.right_base, k2: cell_ctx.k2_right };

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
                stream_computed, stream_computed_weights, ws,
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

/// Stand in for the FP1 pass the generic loop would have run at every off-`R`
/// internal child of a rebuilt level (restricted mode only).
///
/// FP1 there carries the accumulator's level through by reference — which
/// restricted mode gets for free, the output array IS the accumulator's — and
/// leaves behind two observable side effects the rebuild above `t` reads:
///
/// * the identity grid `node_idx[base + i] = i` (dense layout), or a `Sparse`
///   tag the parent densifies via `materialize_dense_child` (bump-allocator
///   layout — as in the generic path, which also leaves FP1'd levels ungridded
///   under `might_use_sparse`);
/// * `live_counts[x] = k_carrier`, which the parent's online density check
///   divides by. Seeding it is not optional: a zero live count would send a
///   level to the sparse route the generic path keeps dense.
pub(super) fn seed_restricted_carried_levels(
    run: &mut ApplyRun,
    r: &Restrict<'_>,
    vtree: &crate::vtree::Vtree,
) {
    for &x in r.touched {
        let xi = x.idx();
        if r.in_rebuild[xi] || vtree.node(VtreeIdx(xi as u32)).is_leaf() {
            continue;
        }
        let k1 = run.c1_widths[xi];
        debug_assert_eq!(
            run.c2_widths[xi], 1,
            "spine-bounded merge: off-spine level {xi} is not width-1 in the \
             batch — the batch spine certificate is wrong"
        );
        if run.might_use_sparse {
            bump_live_count(&mut run.live_counts, &mut run.out_nodes_so_far, xi, k1);
        } else {
            let base = run.grids[xi].base_unchecked();
            for idx in 0..k1 {
                run.node_idx[base + idx] = idx as u32;
            }
            run.grids[xi] = LevelGrid::DenseStrict { base };
        }
    }
}

/// Build this level's NxM dead-pair liveness masks, which need the children's
/// grids materialized and so cannot be computed with the rest of the marg plan.
///
/// # Errors
///
/// Propagates a refused reservation for the mask buffers.
#[allow(clippy::too_many_arguments)]
fn build_level_nxm_masks(
    eng: &Engine,
    run: &mut ApplyRun,
    c2: &Tdd,
    shape: LevelShape,
    plan: &MargPlan,
    left_base: usize,
    right_base: usize,
) -> Result<(), ApplyError> {
    let LevelShape { t, right_idx, k2, k1_left, k2_left, k2_right, .. } = shape;
    build_nxm_masks(
        eng,
        c2, t,
        k2,
        k1_left, k2_left, k2_right,
        left_base, right_base, right_idx,
        plan.left_passthrough, plan.right_passthrough,
        plan.left_view, plan.right_view,
        &run.node_idx, &run.c1_widths,
        &mut run.nxm_masks.live_left_cols, &mut run.nxm_masks.reach_c2_left,
        &mut run.nxm_masks.live_right_cols, &mut run.nxm_masks.reach_c2_right,
    )
}

/// The run buffers [`finish_sparse_marg_level`] writes, borrowed field by
/// field: the output level is already split out of the same `ApplyRun`.
struct SparseMargScratch<'a> {
    inputs1: &'a mut Vec<InputPair>,
    inputs2: &'a mut Vec<InputPair>,
    node_idx: &'a mut [u32],
    product_list: &'a mut Vec<ProductEntry>,
    free_regions: &'a mut Vec<(usize, usize)>,
    live_counts: &'a mut [usize],
    out_nodes_so_far: &'a mut u64,
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
    t_base: usize,
    cell_ctx: &CellCtx<'_>,
    level: &mut TddLevel,
    scratch: SparseMargScratch<'_>,
    left_passthrough: bool,
    right_passthrough: bool,
) -> Result<(), ApplyError> {
    let LevelShape { t, t_idx, k1, k2, .. } = shape;
    let SparseMargScratch {
        inputs1, inputs2, node_idx, product_list, free_regions, live_counts,
        out_nodes_so_far, has_pl,
    } = scratch;
    run_level_rows_marg_sparse(
        eng,
        k1,
        c1.level(t), c2.level(t), cell_ctx,
        inputs1, inputs2,
        level, node_idx,
        product_list,
    )?;
    grid_free(free_regions, t_base, k2);
    finish_sparse_output(live_counts, out_nodes_so_far, has_pl, level, t_idx);
    mark_passthrough_inlined(level, left_passthrough, right_passthrough);
    Ok(())
}

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
#[allow(clippy::too_many_arguments)]
fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
    route: Route,
    plan: &MargPlan,
    vtree: &Arc<crate::vtree::Vtree>,
    marginalize_targets: Option<&[bool]>,
    mut ws: Option<&mut crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let LevelShape {
        t, t_idx, left_idx, right_idx,
        k1, k2, k2_left, k2_right, ..
    } = shape;
    let &MargPlan {
        left_pt_c1, right_pt_c1,
        left_passthrough, right_passthrough,
        left_view, right_view,
        nxm,
    } = plan;
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
    let k2_left = k2_left as u32;
    let k2_right = k2_right as u32;
    let left_base = run.grids[left_idx].base_unchecked();
    let right_base = run.grids[right_idx].base_unchecked();

    // Only the grid-reading NxM liveness masks are deferred this far: they need
    // the materialized child grids, and `nxm` implies a route that has them.
    if nxm {
        build_level_nxm_masks(eng, run, c2, shape, plan, left_base, right_base)?;
    }

    let mut stream_state: Option<StreamLevelState> = build_stream_state(
        eng,
        t_idx, left_idx, right_idx, k1, k2,
        marginalize_targets, &vtree, &mut run.levels,
        &mut run.stream_computed,
        &mut run.stream_computed_weights,
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


    // Every c2 column resolved ONCE for the level: `process_cell` then indexes
    // the table instead of re-deriving column j's slice on every row i. On
    // identity-mask levels the descriptors are zero-copy borrows of c2's own
    // storage; on marg-mask levels they point into a decode arena the table
    // owns and budget-charges. `None` — marginal-encoded c2, or the budget
    // refusing the arena — falls back to the per-cell derivation, never worse
    // than doing it per cell.
    let c2_cols = C2Columns::build(eng, c2.level(t), k2, left_view, right_view);
    let cell_ctx = CellCtx {
        t_base, k2, left_base, right_base,
        k2_left, k2_right,
        left_passthrough, right_passthrough,
        left_pt_c1, right_pt_c1,
        nxm, left_view, right_view,
        live_left_cols: &run.nxm_masks.live_left_cols,
        reach_c2_left: &run.nxm_masks.reach_c2_left,
        live_right_cols: &run.nxm_masks.live_right_cols,
        reach_c2_right: &run.nxm_masks.reach_c2_right,
        c2_cols: c2_cols.as_ref(),
    };

    open_level_arenas(lim, c1, c2, t, level, route, k1, k2)?;

    if use_sparse_marg {
        return finish_sparse_marg_level(
            eng, c1, c2, shape, t_base, &cell_ctx, level,
            SparseMargScratch {
                inputs1: &mut run.inputs1_scratch,
                inputs2: &mut run.inputs2_scratch,
                node_idx: &mut run.node_idx,
                product_list: &mut run.product_lists[t_idx],
                free_regions: &mut run.free_regions,
                live_counts: &mut run.live_counts,
                out_nodes_so_far: &mut run.out_nodes_so_far,
                has_pl: &mut run.has_pl,
            },
            left_passthrough, right_passthrough,
        );
    }

    run_row_loop(
        eng, route, k1, t, left_idx, right_idx, c1, c2, &vtree, &cell_ctx,
        &mut run.inputs1_scratch, &mut run.inputs2_scratch, &mut run.node_idx,
        &run.stream_computed, &run.stream_computed_weights,
        &mut stream_state, level, left_level, right_level, ws.as_deref(),
    )?;


    // Per-level tail: stream commit, live_counts, grid tag, shrink,
    // pass-through flags. See `finalize_level`.
    finalize_level(
        eng,
        &mut stream_state,
        t, t_idx,
        t_base,
        run.might_use_sparse,
        left_passthrough, right_passthrough,
        &vtree, &mut run.levels, &mut run.grids, &mut run.live_counts, &mut run.out_nodes_so_far,
        ws,
    );
    Ok(())
}

/// Walk the vtree bottom-up, building one level at a time.
///
/// At each level the product `c1[i] ∧ c2[j]` is computed over all node pairs,
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
#[allow(clippy::too_many_arguments)]
fn sweep_levels<P: ApplyPlan>(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    plan: &P,
    vtree: &Arc<crate::vtree::Vtree>,
    marginalize_targets: Option<&[bool]>,
    mut ws: Option<&mut crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let internal_iter = plan.walk(&vtree);
    // Where this apply has got to, for a caller watching one long merge from
    // outside it (`budget::merge_position`). The level COUNT is the only thing
    // that costs a walk, so it is taken inside the gate; past that it is one
    // store per level and no clock at all.
    let watched = lim.watched();
    if watched {
        lim.merge_began(vtree.internal_bottomup().count() as u32);
    }
    let mut level_k: u32 = 0;
    // The loop body runs inside an immediately-invoked closure so a single seam
    // catches `?` propagation from the many allocation sites below.
    let loop_result: Result<(), ApplyError> = (|| {
    for (t, left, right) in internal_iter {
        if watched {
            level_k += 1;
            lim.merge_reached(level_k);
        }
        // The per-level-boundary cut check: the stop axis, then the output-node
        // cap. `out_nodes_so_far` is maintained in constant time by
        // `bump_live_count` at every build path.
        debug_assert!(
            lim.output_node_cap().is_none()
                || run.out_nodes_so_far == run.live_counts.iter().map(|&c| c as u64).sum::<u64>(),
            "out_nodes_so_far desynced from live_counts sum — a live_counts \
             write bypassed bump_live_count",
        );
        lim.level_done(run.out_nodes_so_far)?;

        let shape = run.shape(t, left, right);
        let LevelShape { left_idx, right_idx, .. } = shape;

        // Identity fast paths and the operand-child drops that precede them.
        // Restricted mode takes neither — see `try_fast_paths`.
        let taken = plan.takes_fast_paths() && try_fast_paths(eng, run, c1, c2, shape)?;

        if !taken {
            // One decision per level, taken before any of the level's storage
            // is touched: the marg plan and the two gates read only metadata.
            let marg_plan = plan_marg_level(
                eng, c1, c2, t, shape.t_idx, left_idx, right_idx,
                &run.levels, &run.c1_identity, &run.c2_identity, run.any_entry_marginal,
            );
            let marg =
                run.level_marg(c1, c2, shape, marginalize_targets, plan.output_lives_in_accumulator());
            let route = route_level(shape, &marg_plan, &marg, run.sparse_gate(shape));
            route.validate(
                c1, c2, shape, &marg,
                &run.c1_identity, &run.c2_identity, &run.c1_widths, &run.c2_widths, &vtree,
            );

            match route {
                Route::Sparse => {
                    run_sparse_level(eng, run, c1, c2, shape, &vtree, marg.is_target)?;
                }
                _ => build_level_dense(
                    eng, run, c1, c2, shape, route, &marg_plan,
                    vtree, marginalize_targets, ws.as_deref_mut(),
                )?,
            }
        }

        // Release this level's children's grid regions for a later level to
        // reuse, before the next iteration's own reserve fires.
        reclaim_child_grids(
            run.might_use_sparse, &mut run.grids, &mut run.free_regions,
            &run.c1_widths, &run.c2_widths, left_idx, right_idx,
        );
    }
    Ok(())
    })();
    loop_result
}

fn apply_and_fallible_inner<P: ApplyPlan>(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    marginalize_targets: Option<&[bool]>,
    plan: &P,
) -> Result<Tdd, ApplyError> {
    let lim = eng.limits();
    lim.eager_reclaim();
    assert!(
        Arc::ptr_eq(&c1.vtree, &c2.vtree),
        "apply_and requires TDDs with the same vtree"
    );
    assert_eq!(
        c1.output.vtree, c2.output.vtree,
        "apply_and requires TDDs with outputs at the same vtree node"
    );

    // Zero the per-operation meters: the in-flight byte counter, so cumulative
    // capacity-grow accounting starts fresh — without this the counter would
    // conflate growth across calls and trip OverBudget spuriously on a small
    // later conjunction.
    lim.begin_operation();

    // Self-conjunction short-circuit: f ∧ f = f. The test is STRUCTURAL
    // equality of every explicit level, not pointer identity, and it declines on
    // any marginal level — see `is_self_conjunction`, where the soundness of
    // both choices is stated.
    if is_self_conjunction(c1, c2) {
        return Ok(c1.clone());
    }

    // The weight store follows the diagram: the operands' stores merge into the
    // result's, and every level this apply freezes writes its values there.
    // Their frozen levels are the two disjoint subtrees they were built over,
    // so the merge loses nothing.
    let mut ws: Option<crate::weight_store::WeightStore> =
        match (c1.weights.take(), c2.weights.take()) {
            (Some(mut a), Some(b)) => {
                a.absorb(b);
                Some(a)
            }
            (a, b) => a.or(b),
        };
    debug_assert!(
        ws.is_some()
            || !c1.levels.iter().chain(c2.levels.iter()).any(|l| l.is_weight_marginal()),
        "an operand has weight-marginal levels but neither carries a weight store"
    );

    // Early return for ZERO inputs: x ∧ ZERO = ZERO.
    // Avoids allocating levels, level_base, and node_idx for unsatisfiable operands.
    let vtree = Arc::clone(&c1.vtree);
    let num_nodes = vtree.num_nodes();
    if c1.is_zero() || c2.is_zero() {
        let levels = diagram::take_levels(eng, num_nodes);
        let mut out = Tdd::with_levels(
            vtree,
            levels,
            TddNodeId { vtree: c1.output.vtree, local: ZERO },
        );
        out.weights = ws;
        return Ok(out);
    }

    let mut run = apply_and_setup(eng, c1, c2, &vtree, num_nodes, marginalize_targets, plan)?;

    plan.seed_identity(eng, &mut run, c1, c2, &vtree, num_nodes)?;

    apply_leaf_levels(
        eng,
        &vtree, &run.c1_widths, &run.c2_widths, &mut run.grids, &mut run.node_idx,
        &mut run.grid_end, &mut run.live_counts, &mut run.out_nodes_so_far, run.might_use_sparse,
        plan.leaf_children(),
    )?;

    plan.seed_carried_levels(&mut run, &vtree);

    let canon_leaves = seed_marginal_leaves(
        c1, c2, &vtree, &mut run.levels, plan, &run.c1_identity, &run.c2_identity, ws.as_ref(),
    );

    sweep_levels(eng, &mut run, c1, c2, plan, &vtree, marginalize_targets, ws.as_mut())?;

    crate::marginal::canonicalize_apply_leaf_refs(&canon_leaves, &vtree, &mut run.levels, ws.as_ref());

    let out_local = compute_apply_output(
        c1, c2, &run.grids, &run.node_idx, &run.c2_widths,
        &run.c1_identity, &run.c2_identity, &run.has_pl, &run.product_lists,
    );
    let out_vtree = c1.output.vtree;

    let out_local = guard_stale_false(out_local, out_vtree, &vtree, &run.levels);

    let levels = run.finish(eng, marginalize_targets.is_some());

    let output = TddNodeId { vtree: out_vtree, local: out_local };
    Ok(plan.finish(c1, vtree, levels, output, ws))
}
