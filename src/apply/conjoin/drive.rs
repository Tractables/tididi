//! The bottom-up driver: one conjunction from entry to finished diagram.

use super::*;

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
    // owned wrappers (`try_apply_and`), NOT on this
    // shared borrowed path. Order-sensitive callers reach apply through here,
    // and a swap would silently rebind their per-operand bookkeeping to the
    // wrong side. The borrowed/owned asymmetry is intentional.
    let mut out = apply_and_fallible_inner(eng, c1, c2, marginalize_targets, None)?;
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
/// Only `restrict::try_apply_and_batch` calls this; it owns the decline
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
    let mut out = apply_and_fallible_inner(eng, c1, c2, None, Some(restrict))?;
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
pub(super) fn seed_marginal_leaves(
    c1: &Tdd,
    c2: &Tdd,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    restrict: Option<&Restrict<'_>>,
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
    // `MargRef` refs through verbatim. The output store stays empty (all leaf
    // counts are inline at the parent).
    // In weighted mode the leaf's counts are NOT inline at the parent: the
    // weighted leaf-marg installs a real per-slot column in the (vtree-indexed)
    // `WeightStore` and leaves the parent's bare leaf-label refs to
    // decode as `MargRef::Slot`. That column is PINNED — immutable, label-ordered,
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
    for (leaf, _) in vtree.leaf_bottomup() {
        if restrict.is_some() {
            break;
        }
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

/// Rewrite the leaf-side refs of every leaf that one operand made
/// weight-marginal onto that leaf's canonical slots.
///
/// It runs after the bottom-up loop because the parent's pairs are only final
/// then, and it shares the walk the marginalize pass uses, so a leaf whose
/// column holds equal values ends up with one representative rather than two
/// slots the contraction would have to recognize as twins.
fn canonicalize_marginal_leaf_refs(
    canon_leaves: &[usize],
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    ws: Option<&crate::weight_store::WeightStore>,
) {
    // EQUAL-VALUE LEAF-REF CANONICALIZATION, apply-side mirror of
    // `marginalize::marginalize_leaf_weighted`'s pass and sharing its ONE walk.
    // Runs here, not at the flag site above: the parent level's pairs are emitted
    // by the bottom-up loop that just finished, so this is the first point at
    // which they are final.
    //
    // Scope is the leaves recorded above — flagged weight-marginal on ONE
    // operand's authority. The structural operand contributes leaf-side refs that
    // never passed through the canon map, and `CONJOIN_GRID` carries them into the
    // output unchanged wherever the marginal side reads `One`. Rewriting them onto
    // the canonical slot of their value class is value-preserving (same column
    // entry) and is what lets the contraction that follows this apply see the
    // parent's `(·, Pos)` / `(·, Neg)` branches as twins.
    {
        use crate::marginal as marg;
        for &li in canon_leaves {
            let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(li as u32)) else { continue };
            let Some(parent) = vtree.node(VtreeIdx(li as u32)).parent() else { continue };
            // A marginal parent folded the leaf's bases into its own aggregate — no
            // leaf-side pairs remain to rewrite (same guard as the marginalize pass).
            if levels[parent.idx()].is_marginal() {
                continue;
            }
            // Exact domain only: `WeightKey::Log` compares `f64` bit patterns, so
            // "equal" there is representation identity, not value identity.
            let Some(w) = ws.as_ref() else { continue };
            let Some(canon) =
                (!w.is_log()).then(|| marg::leaf_canon_map(&marg::leaf_column_vals(w, var)))
            else {
                continue;
            };
            if canon == [0, 1, 2] {
                continue; // no equal-valued slots — the walk would rewrite nothing
            }
            let (pl, _) = vtree.children(parent);
            marg::canonicalize_leaf_refs_at_parent(
                &mut levels[parent.idx()],
                pl.idx() == li,
                &canon,
            );
        }
    }
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
    out_local: LocalNodeIdx,
    out_vtree: VtreeIdx,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
) -> LocalNodeIdx {
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

/// Fault on a marginalize-schedule violation: one operand summed a variable out
/// of level `t` while the other still constrains it.
fn assert_marg_schedule(
    run: &ApplyRun,
    c1: &Tdd,
    c2: &Tdd,
    shape: LevelShape,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] vtree: &crate::vtree::Vtree,
) {
    let LevelShape { t, left, right, left_idx, right_idx, .. } = shape;
    // Marginalize-schedule invariant — ALWAYS ON (the gate is two `is_marginal()`
    // bool reads per level, negligible vs the apply work). If either operand's
    // level t is marginal (mc-mode pair structure replaced with model counts), the
    // identity fast-paths above MUST have consumed it: a marginal level conjoins
    // soundly ONLY with an identity (non-constraining) counterpart, and that case
    // is taken by `try_level_fast_paths` and `continue`s. Reaching this dense path
    // with a marginal level therefore means the OTHER operand still constrains
    // node t — i.e. a variable was summed out of one operand while still live in
    // the other. That is an invalid conjoin; the dense path below would deref
    // `nodes[i]` on an empty Vec (SIGSEGV) or silently miscount. It can only arise
    // from a marginalize-schedule bug, never from a
    // correct run — so fail loudly rather than corrupt the count.
    //
    // The violation is the ASYMMETRIC case: exactly one operand marginalized node
    // t while the OTHER still carries a real (non-identity) function over t's
    // variables. The symmetric both-marginal case (both summed out the same scope)
    // is sound and handled below / by the both-marginal-width-1 fast path; the
    // marginal-vs-identity case is consumed by the identity fast paths above. So we
    // only fault on marginal-vs-constraining.
    let c1_marg = c1.level(t).is_marginal();
    let c2_marg = c2.level(t).is_marginal();
    let c1_identity_at_t = run.c1_identity[left_idx] && run.c1_identity[right_idx];
    let c2_identity_at_t = run.c2_identity[left_idx] && run.c2_identity[right_idx];
    let violation = (c1_marg && !c2_marg && !c2_identity_at_t)
        || (c2_marg && !c1_marg && !c1_identity_at_t);
    if violation {
        // Rich subtree dump (expensive string build + /tmp file) only in debug.
        #[cfg(debug_assertions)]
        debug_assert_marg_schedule(
            c1, c2, t, left, right, vtree,
            shape.k1, shape.k2, left_idx, right_idx,
            &run.c1_widths, &run.c2_widths, &run.c1_identity, &run.c2_identity,
        );
        panic!(
            "apply_and marginalize-schedule violation at vtree node {t:?} \
             (left={left:?} right={right:?}): one operand marginalized this node \
             while the other still constrains it \
             (c1.marg={c1_marg}, c2.marg={c2_marg}, c1_id[L,R]={},{}, c2_id[L,R]={},{}). \
             A variable was summed out of one operand while still live in the other \
             — a marginalize-schedule bug. This conjoin \
             is invalid and would corrupt the model count.",
            run.c1_identity[left_idx],
            run.c1_identity[right_idx],
            run.c2_identity[left_idx],
            run.c2_identity[right_idx],
        );
    }
}

/// Decide dense vs sparse for this level, and run the sparse pipeline when it
/// wins.
///
/// `Ok(None)` means the sparse build produced the level. `Ok(Some(flag))` sends
/// the level to the dense build, with `flag` telling it whether to take the
/// one-marginal-child sparse variant.
fn try_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
    marginalize_targets: Option<&[bool]>,
) -> Result<Option<bool>, ApplyError> {
    let LevelShape {
        t, left, right, t_idx, left_idx, right_idx,
        k1, k2, k1_left, k2_left, k1_right, k2_right,
    } = shape;
    // ── Online density check ─────────────────────────────────────
    //
    // Only when might_use_sparse: decide dense vs sparse based on children's
    // actual product density. Children are already processed (bottom-up order),
    // so live_counts are available.
    // A marginal child's pair ref is a *count payload* (inline count or
    // tagged slot), NOT a structural node index. The sparse reverse index
    // buckets parent nodes by `decode_marg_coord(ref)` (see
    // `build_reverse_index`), which is only a valid per-node key when the
    // payload happens to be per-node-distinct — true for the legacy slot
    // encoding, FALSE under inline, where every equal-count ref decodes to
    // the same value and distinct marginal children collapse into one
    // bucket (dropping multiplicity → model-count corruption, e.g. mc007
    // ×4). The dense path indexes by the real grid position and is immune.
    // So: marginal levels always take the dense path.
    let left_child_marg = run.levels[left_idx].is_marginal()
        || c1.levels[left_idx].is_marginal()
        || c2.levels[left_idx].is_marginal();
    let right_child_marg = run.levels[right_idx].is_marginal()
        || c1.levels[right_idx].is_marginal()
        || c2.levels[right_idx].is_marginal();
    let marg_child = left_child_marg || right_child_marg;
    let use_sparse = run.might_use_sparse && !marg_child && {
        let max_left = (k1_left * k2_left) as u128;
        let max_right = (k1_right * k2_right) as u128;
        let live_l = run.live_counts[left_idx] as u128;
        let live_r = run.live_counts[right_idx] as u128;
        k1 * k2 > run.min_grid
            && max_left > 0 && max_right > 0
            && run.sparsity_factor * live_l * live_r < max_left * max_right
    };

    // Sparse bottom-up build for an *exactly-one*-marginal-child level whose
    // output is STRUCTURAL (the OOM case). The dense path allocates a full
    // k1*k2 slab even though the marginal side never kills a pair (it is a
    // pass-through carrier) — so the live structural sibling alone governs
    // survival and the slab is mostly DEAD. Instead, drive the build from the
    // structural sibling, emit a `product_list` of the surviving cells, and
    // tag this level Sparse; the grandparent densifies lazily via ensure_grid.
    //
    // Gated to the structural case only: the explicit `!is_marg_target`
    // conjunct is what excludes marginalize targets — one-marginal-child
    // TARGETS do exist and are common (measured: thousands of such sites
    // across the marginalize/recovery test suites), so do not assume the
    // conjunct is dead. XOR + !is_marg_target is exactly the structural
    // one-marginal-child level. Both-marginal and target levels fall through
    // to the existing dense Route A unchanged.
    let use_sparse_marg = run.might_use_sparse
        && (left_child_marg ^ right_child_marg)
        && !marginalize_targets.is_some_and(|a| a[t_idx])
        && k1 * k2 > run.min_grid;

    if use_sparse {
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
        let is_marg_target = marginalize_targets.is_some_and(|arr| arr[t_idx]);
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
        finish_sparse_output(&mut run.live_counts, &mut run.out_nodes_so_far, &mut run.has_pl, &mut run.levels[t_idx], t_idx);

        return Ok(None);
    }
    Ok(Some(use_sparse_marg))
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


/// Arm the per-level emit-growth policy and pre-size the pair arena.
#[allow(clippy::too_many_arguments)]
fn arm_emit_growth(
    lim: &crate::engine::Limits,
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    level: &mut TddLevel,
    use_sparse_marg: bool,
    marg_stream_collapse: bool,
    stream_marginal: bool,
) -> Result<(), ApplyError> {
    // Called exactly once per level on every route, so a previous level's
    // near-cap decision can never leak into this one. The mode only matters
    // for the dense emit into `level.pairs`, so the routes that never do one
    // — sparse-marg and stream-collapse — pass no bound. Skipping is sound:
    // the mode is a growth policy, not a correctness step, and the per-push
    // budget checks in the emit still apply.
    if !use_sparse_marg && !marg_stream_collapse {
        // THE per-level emit-pair bound: every product pair emits at most
        // once, so `|c1.pairs| × |c2.pairs|` bounds this level's emit. Used
        // twice — once to pick the growth mode, once to size the pairs
        // arena — computed once so the two can never disagree.
        let emit_pair_bound = (c1.level(t).pairs.len() as u128)
            .saturating_mul(c2.level(t).pairs.len() as u128);
        lim.begin_level((!stream_marginal).then_some(emit_pair_bound));
        // Seed `level.pairs` at that bound instead of letting it double from
        // empty on every level. Same cap/rationale as the caller's nodes
        // reserve: low-survival levels would over-allocate wildly past it, so
        // the reserve stops there and the emit's own `try_push_pair_into`
        // choke point (still under the growth mode just armed) carries any
        // level that outgrows it; `shrink_arrays` reclaims the tail at
        // `finalize_level`. Only the routes that DO emit into `level.pairs`
        // reserve — the sparse-marg and stream-collapse routes never touch it.
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
    k1: usize,
    t: VtreeIdx,
    t_idx: usize,
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
    marg_child_dispatch: bool,
    marg_stream_collapse: bool,
    left_marg_now: bool,
    right_marg_now: bool,
    stream_state: &mut Option<StreamLevelState>,
    level: &mut TddLevel,
    left_level: &TddLevel,
    right_level: &TddLevel,
    ws: Option<&crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    if marg_child_dispatch {
        // Route A: Dedicated marginal-parent build. A level with at least one
        // marginal child is built by a forward-order grid loop running the
        // SHARED cell kernel (`process_cell` with MargLookup sides) — the
        // exact per-cell logic the general path uses. Isolating marginal
        // levels here lets the general product-grid path (Route B) assume no
        // child is marginal.
        //
        // Refs are emitted in the exact encoding the general path produces, so
        // this level still relies on the end-of-apply tagger
        // (`tag_all_marg_side_slots`) for its final marginal-side counts —
        // identical to the general path.
        //
        // Both c1.level(t) and c2.level(t) are immutable borrows into their
        // respective levels vecs. `level` is a mutable borrow into the OUTPUT
        // `levels[t_idx]`, which is a separate allocation from c1 and c2.
        // We pre-take both operand borrows here while `level` is not yet live,
        // then re-lend them to the called function as plain `&TddLevel` refs.
        let c1_level_t: &TddLevel = c1.level(t);
        let c2_level_t: &TddLevel = c2.level(t);
        if marg_stream_collapse {
            // Streaming target → collapse at source, skipping product-node
            // materialization. The gate guarantees `stream_state` is Some;
            // the walker picks the integer or weighted fold from it.
            let left_marg = child_lookup::MargLookup::left(&cell_ctx);
            let right_marg = child_lookup::MargLookup::right(&cell_ctx);
            run_level_rows_stream_count(
                eng,
                k1,
                c1_level_t, c2_level_t, &cell_ctx,
                inputs1_scratch, inputs2_scratch,
                node_idx,
                &left_marg, &right_marg,
                stream_state.as_mut().expect("the streaming gate guarantees a state"),
                left_idx, right_idx, vtree, left_level, right_level,
                &stream_computed, &stream_computed_weights, ws.as_deref(),
            )?;
        } else {
            // Non-streaming marginal-child level (`stream_state` None —
            // includes gate-off, which `stream_marginal_eligible` maps to
            // "don't stream"): plain materializing build. A one-marginal-child
            // MARGINALIZE TARGET always streams, so it takes the collapse
            // walker above and never this arm.
            run_level_rows_marg(
                eng,
                k1,
                c1_level_t, c2_level_t, &cell_ctx,
                inputs1_scratch, inputs2_scratch,
                level, node_idx,
            )?;
        }
    } else {
    // No-marginal-leakage guard (tier-0, every build incl. release). Every
    // parent of a marginal child level is dispatched to Route A above, so
    // Route B must never see a *non-empty* marginal child. The general path's
    // marg-ref decode has been deleted on the strength of this invariant; the
    // assert stays always-on so a future leak aborts loudly instead of
    // silently miscounting.
    //
    // EXEMPTION — a `width()==0` marginal child is benign. The deleted decode
    // only mattered for a width>0 marginal child: the parent carries inline
    // MargRefs into its cells that the general grid path can't interpret. A
    // 0-width marginal level has NO cells, so the parent has NO refs into it
    // (`t` is itself 0-width over that child) and the row-loop — which indexes
    // the OUTPUT child grids, never the operand's marginal store — does no
    // work for it. It is semantically identical to a 0-width *non*-marginal
    // child. Such levels are a legitimate transient state minted by the apply
    // streaming commit (`commit_stream_state`) and leaf-marginalization, and
    // can disagree across two pool members (one went through `restrict`,
    // the other didn't) — which is exactly the `WS_FAST_REDUCE` conjoin that
    // used to trip this assert.
    let marg_wide = |lvl: &TddLevel| lvl.is_marginal() && lvl.width() > 0;
    cheap_assert!(
        !left_marg_now && !right_marg_now
            && !marg_wide(&c1.levels[left_idx]) && !marg_wide(&c1.levels[right_idx])
            && !marg_wide(&c2.levels[left_idx]) && !marg_wide(&c2.levels[right_idx]),
        "general product-grid path reached with a non-empty marginal child \
         (t_idx={t_idx} l={left_idx} r={right_idx}): the dedicated \
         marginal-parent dispatch was bypassed"
    );
    // Route B: Plain (non-marginal-child) row-loop.
    // Pre-take the operand borrows before `level` (a `&mut` into the OUTPUT
    // `levels`) is active. Both row drivers below read them immutably.
    let c1_level_t: &TddLevel = c1.level(t);
    let c2_level_t: &TddLevel = c2.level(t);
    // Branch hoist: when all four level-invariant guards hold, dispatch to the
    // simplified dense path (no streaming, no nxm, no passthrough). Otherwise
    // fall through to the general path.
    let plain_dense = stream_state.is_none()
        && !cell_ctx.nxm
        && !cell_ctx.left_passthrough
        && !cell_ctx.right_passthrough;
    // Dense child lookups. A DenseLookup inlines to the original
    // `get_unchecked` positional index.
    let left_dense = child_lookup::DenseLookup { base: cell_ctx.left_base, k2: cell_ctx.k2_left };
    let right_dense = child_lookup::DenseLookup { base: cell_ctx.right_base, k2: cell_ctx.k2_right };
    macro_rules! run_plain {
        ($dense:literal, $l:expr, $r:expr) => {
            run_level_rows_plain::<$dense, _, _>(
                eng,
                k1,
                c1_level_t, c2_level_t, &cell_ctx,
                inputs1_scratch, inputs2_scratch,
                level, node_idx,
                $l, $r,
            )?
        };
    }
    if plain_dense {
        run_plain!(true, &left_dense, &right_dense);
    } else if stream_state.is_some() {
        // Streaming-marginal level with no marginal child (leaf children at
        // the lowest levels): collapse to Σ left × right per cell WITHOUT
        // building+truncating a product node — the Route B analogue of
        // `marg_stream_collapse`; the walker picks the integer or weighted
        // fold from `stream_state`. The general (non-plain-dense) arm
        // always sees two dense-served children, and streaming forces this
        // arm (`plain_dense` requires `stream_state` None), so the dense
        // lookups are correct. The streaming gate lives in
        // `stream_marginal_eligible`, so `stream_state` alone decides here.
        run_level_rows_stream_count(
            eng,
            k1,
            c1_level_t, c2_level_t, &cell_ctx,
            inputs1_scratch, inputs2_scratch,
            node_idx,
            &left_dense, &right_dense,
            stream_state.as_mut().expect("the streaming gate guarantees a state"),
            left_idx, right_idx, vtree, left_level, right_level,
            &stream_computed, &stream_computed_weights, ws.as_deref(),
        )?;
    } else {
        run_plain!(false, &left_dense, &right_dense);
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
fn seed_restricted_fastpath_levels(
    run: &mut ApplyRun,
    restrict: Option<&Restrict<'_>>,
    vtree: &crate::vtree::Vtree,
) {
    let Some(r) = restrict else { return };
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

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
#[allow(clippy::too_many_arguments)]
fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    c1: &mut Tdd,
    c2: &mut Tdd,
    shape: LevelShape,
    use_sparse_marg: bool,
    vtree: &Arc<crate::vtree::Vtree>,
    marginalize_targets: Option<&[bool]>,
    restrict: Option<&Restrict<'_>>,
    mut ws: Option<&mut crate::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let LevelShape {
        t, t_idx, left_idx, right_idx,
        k1, k2, k1_left, k2_left, k2_right, ..
    } = shape;
    // ── Route prediction (before materializing children) ──────────────
    //
    // Compute the marg-plan flags now: they read only level metadata, no
    // child grid (the grid-reading NxM masks are deferred to
    // `build_nxm_masks` below, which runs only when `nxm` — i.e. the
    // general path, where the grids are materialized). Knowing the route
    // here is what lets the dense path skip `ensure_grid` for a sparse
    // child when the level will take the plain-dense emit.
    let MargPlan {
        left_pt_c1, right_pt_c1,
        left_passthrough, right_passthrough,
        left_mask, right_mask,
        nxm,
    } = plan_marg_level(
        eng,
        c1, c2, t,
        t_idx, left_idx, right_idx,
        &run.levels, &run.c1_identity, &run.c2_identity,
        run.any_entry_marginal,
    );
    // Materialize any sparse child grid and bump-allocate this level's own.
    let t_base = materialize_children_and_grid(eng, run, shape, use_sparse_marg)?;

    // Non-identity internal: DEAD-fill is interleaved with product construction
    // below (per-row fill before each row's cells are computed). This keeps the
    // active row in L1 cache during process_cell, avoiding the cache pollution
    // from a single bulk fill of the entire k1*k2 grid.
    // Child grid dimensions: used to compute flat grid positions.
    // Grid position for child product (a,b) = child_base + a * k2_child + b
    let k2_left = k2_left as u32;
    let k2_right = k2_right as u32;
    let left_base = run.grids[left_idx].base_unchecked();
    let right_base = run.grids[right_idx].base_unchecked();

    // Marg flags (`left_pt_*`, passthrough, masks, `nxm`) were computed
    // above, before child materialization, to decide the route. Only the
    // grid-reading NxM liveness masks are deferred to here — they need the
    // materialized child grids, and `nxm` implies the general (non-sparse-
    // served) path, so both grids exist.
    if nxm {
        build_nxm_masks(
            eng,
            c2, t,
            k2,
            k1_left, k2_left as usize, k2_right as usize,
            left_base, right_base, right_idx,
            left_passthrough, right_passthrough,
            left_mask, right_mask,
            &run.node_idx, &run.c1_widths,
            &mut run.nxm_masks.live_left_cols, &mut run.nxm_masks.reach_c2_left,
            &mut run.nxm_masks.live_right_cols, &mut run.nxm_masks.reach_c2_right,
        )?;
    }

    // Streaming-eligibility gate (see `stream_marginal_eligible`), read by
    // the emit-growth mode decision and threaded into the row loop below.
    let stream_marginal = stream_marginal_eligible(marginalize_targets, t_idx);

    let mut stream_state: Option<StreamLevelState> = build_stream_state(
        eng,
        t_idx, left_idx, right_idx, k1, k2,
        marginalize_targets, &vtree, &mut run.levels,
        &mut run.stream_computed,
        &mut run.stream_computed_weights,
        ws.as_deref_mut(),
    )?;

    // ── Dedicated marginal-child dispatch ──
    // Decide HERE (post-cascade, where child marginality is final; pre-`level`
    // borrow, where we can still read `levels[child]`) whether this level has
    // EXACTLY ONE marginal child. If so, and the gate is on, the cell-build
    // row-loop below branches to the dedicated no-grid path instead of the
    // general product grid. Setup (widths, bases, node_idx, stream_state) and
    // finalize (dedup/permute, shrink, pack) are SHARED — only the inner cell
    // build differs. `^` (xor) = exactly one; both-marginal (deep marginal)
    // and neither (pure structural) fall through to the general body.
    // Under a restriction the OUTPUT level of an off-`R` child is the
    // accumulator's own level (it is never copied into the fresh `levels`
    // array — the two are merged at the tail), so read through to
    // `c1.levels[..]`. Levels IN `R` are structural in the accumulator by
    // construction, so the extra disjunct is inert for them.
    let (left_marg_now, right_marg_now) = if restrict.is_some() {
        (run.levels[left_idx].is_marginal() || c1.levels[left_idx].is_marginal(),
         run.levels[right_idx].is_marginal() || c1.levels[right_idx].is_marginal())
    } else {
        (run.levels[left_idx].is_marginal(), run.levels[right_idx].is_marginal())
    };
    // Dispatch on AT LEAST ONE marginal child (OR, not XOR): the dedicated
    // path tags whichever side(s) are marginal and carries them verbatim, so
    // a both-marginal level (pure Σ left_count × right_count, no structural
    // product) is handled identically — and routing it here keeps it off the
    // legacy inline tagger that otherwise corrupts it under inline-emit.
    let marg_child_dispatch = left_marg_now || right_marg_now;

    // A streaming marginalize target collapses to Σ left × right per cell —
    // no downstream structure survives. Build each alive cell's scalar
    // directly from the surviving (lc, rc) refs, never materializing a
    // product node (see `run_level_rows_stream_count`). Covers EVERY
    // Route A streaming shape (any marginal-child pattern, integer and
    // weighted — D2 stage 1): which side is marginal is carried by
    // `stream_state` itself, and `MargLookup` degrades to a dense grid
    // read on a non-pass-through side. The streaming gate lives in
    // `stream_marginal_eligible` (`stream_state` is None when off), so
    // `stream_state.is_some()` alone decides here (D2 stage 2).
    let marg_stream_collapse = marg_child_dispatch && stream_state.is_some();

    // `t` and its two vtree children are three distinct nodes of a tree, so
    // `t_idx`, `left_idx` and `right_idx` name three disjoint level slots —
    // split them apart in one step. That is what lets the streaming row
    // loops read the child count columns IN PLACE while the output level is
    // exclusively borrowed; snapshotting them instead doubled a wide
    // marginal child's storage (an 8 GiB single alloc at 536M slots) at
    // exactly the moment streaming exists to relieve.
    //
    // The split's borrow of `levels` must end before the per-level tail
    // retakes it — every use of the three below is inside the row loop.
    let [level, left_level, right_level] = run.levels
        .get_disjoint_mut([t_idx, left_idx, right_idx])
        .expect("a vtree node and its two children are distinct level indices");
    let (left_level, right_level) = (&*left_level, &*right_level);

    // Pre-reserve capacity for OR nodes. `k1*k2` is the EXACT upper bound
    // (one node per live cell; compaction only removes dead ones), so
    // reserving it once replaces the Vec-doubling ladder the old
    // `max(k1, k2)` seed left behind. Capped at `LEVEL_RESERVE_CAP_BYTES`
    // because most levels are low-survival — beyond the cap the over-
    // allocation would dwarf the real node count, and growth past it
    // continues through the ordinary fallible `try_push` path. `finalize_level`
    // calls `shrink_arrays`, which hands the unused tail straight back.
    // Never below the old seed. Fallible: under tight budgets, even this
    // baseline reservation may exceed the remaining VAS.
    let nodes_reserve = k1
        .saturating_mul(k2)
        .min(LEVEL_RESERVE_NODES_CAP)
        .max(k1.max(k2));
    lim.reserve(&mut level.nodes, nodes_reserve)?;

    // ── Cell-build shared context ────────────────────────────────────
    //
    // Built here (before the route dispatch) so it can be shared by both
    // row-loop routes without re-stating its 15 fields in each site.  All
    // inputs are available at this point (widths/bases computed above,
    // masks/flags from `plan_marg_level`).
    //
    // Route A calls run_level_rows_marg (forward order, marg path).
    // Route B calls run_level_rows_plain (forward order, plain path).
    //
    // Resolve every c2 column ONCE for this level; `process_cell` (all
    // routes) then indexes the table instead of re-deriving column j's
    // slice on every row i. On identity-mask levels the descriptors are
    // zero-copy borrows of c2's own storage; on marg-mask levels they
    // point into a decode arena the table owns and budget-charges. `None`
    // (marginal-encoded c2, or the budget rejecting the arena) falls back
    // to the per-cell derivation — never worse than doing it per cell.
    let c2_cols = C2Columns::build(eng, c2.level(t), k2, left_mask, right_mask);
    let cell_ctx = CellCtx {
        t_base, k2, left_base, right_base,
        k2_left, k2_right,
        left_passthrough, right_passthrough,
        left_pt_c1, right_pt_c1,
        nxm, left_mask, right_mask,
        live_left_cols: &run.nxm_masks.live_left_cols,
        reach_c2_left: &run.nxm_masks.reach_c2_left,
        live_right_cols: &run.nxm_masks.live_right_cols,
        reach_c2_right: &run.nxm_masks.reach_c2_right,
        c2_cols: c2_cols.as_ref(),
    };

    arm_emit_growth(lim, c1, c2, t, level, use_sparse_marg, marg_stream_collapse, stream_marginal)?;

    // ── Cell-build row loop ──────────────────────────────────────────
    // Branch on the dedicated marginal-child path. See `cell_ctx` above.
    if use_sparse_marg {
        // Sparse bottom-up build (exactly-one-marginal-child, structural
        // output). Runs the shared emit kernel (`process_cell` + MargLookup),
        // but writes into the k2-row scratch (cell_ctx.t_base == row_base,
        // called with i=0 so grid_pos == row_base + j) and records each
        // surviving cell into the output product_list instead of a dense slab.
        let c1_level_t: &TddLevel = c1.level(t);
        let c2_level_t: &TddLevel = c2.level(t);
        run_level_rows_marg_sparse(
            eng,
            k1,
            c1_level_t, c2_level_t, &cell_ctx,
            &mut run.inputs1_scratch, &mut run.inputs2_scratch,
            level, &mut run.node_idx,
            &mut run.product_lists[t_idx],
        )?;
        // Reclaim the transient k2-row scratch; the level is Sparse (its
        // product_list is the authoritative representation, densified lazily
        // by the grandparent's ensure_grid) and never reads the dense slab.
        grid_free(&mut run.free_regions, t_base, k2);
        // Reuse the split's `level` rather than reborrowing `levels[t_idx]`:
        // the child halves of the split are still in scope here.
        finish_sparse_output(&mut run.live_counts, &mut run.out_nodes_so_far, &mut run.has_pl, level, t_idx);
        // This route returns before `finalize_level`, so run its inline-emit
        // marking here. Exactly one side is the marginal pass-through (XOR gate).
        mark_passthrough_inlined(level, left_passthrough, right_passthrough);
        return Ok(());
    }
    run_row_loop(
        eng, k1, t, t_idx, left_idx, right_idx, c1, c2, &vtree, &cell_ctx,
        &mut run.inputs1_scratch, &mut run.inputs2_scratch, &mut run.node_idx,
        &run.stream_computed, &run.stream_computed_weights,
        marg_child_dispatch, marg_stream_collapse, left_marg_now, right_marg_now,
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

fn apply_and_fallible_inner(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    marginalize_targets: Option<&[bool]>,
    restrict: Option<&Restrict<'_>>,
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

    let mut run = apply_and_setup(eng, c1, c2, &vtree, num_nodes, marginalize_targets, restrict)?;

    // ── Identity tracking ─────────────────────────────────────────────────
    //
    // `c2_identity[t]` is true when c2 computes constant-true at subtree t,
    // meaning c1's nodes pass through unchanged (x ∧ 1 = x). This lets the
    // product construction skip the expensive per-node inner loop at identity
    // levels — just copy c1's nodes directly (via `mem::swap` on c1).
    // `c1_identity[t]` is the symmetric case; at those levels c2's nodes are
    // cloned to the output (c2 is immutable, can't swap). This is critical for
    // `conjoin_children(left, right)`, where c1 = the left child's TDD has
    // identity at right-subtree levels and vice versa.
    //
    // A leaf is identity iff only the One label (local index 0) is referenced
    // by parent pairs. An internal node is identity iff width == 1 AND both
    // children are identity. With implicit leaf representation, leaf identity
    // is precomputed by scanning parent pairs for non-One references.
    if let Some(r) = restrict {
        // Restricted mode derives both identity vectors from the spine
        // certificate instead of scanning every leaf's parent pairs twice:
        //
        // * `c2_identity[t] = !on_spine[t]` — the batch is width-1
        //   constant-true off its spine by construction, which is exactly the
        //   fixpoint the generic path's FP1 accretes at every off-spine level.
        // * `c1_identity` all-false — it is read ONLY by FP2's guard (the other
        //   readers are the declined sparse routes and `plan_marg_level`'s
        //   `left_pt_c2`, whose `c2_ref` conjunct is false because the batch
        //   has no marginal levels and no `marg_inlined_*` flags). Restricted
        //   mode never takes a fast path, so an all-false vector just means
        //   every level in `R` is rebuilt — and a rebuild against a width-1
        //   constant-true operand reproduces the carried level.
        //
        // Only `touched` entries are ever read, so only they are written; the
        // pooled buffers keep whatever stale values they had elsewhere.
        lim.try_resize(&mut run.c2_identity, num_nodes, false)?;
        lim.try_resize(&mut run.c1_identity, num_nodes, false)?;
        for &t in r.touched {
            run.c2_identity[t.idx()] = !r.on_spine[t.idx()];
            run.c1_identity[t.idx()] = false;
        }
    } else {
        init_leaf_identity(eng, &mut run.c2_identity, c2, &vtree, num_nodes)?;
        init_leaf_identity(eng, &mut run.c1_identity, c1, &vtree, num_nodes)?;
    }

    // `c{1,2}_identity` is lazily accreted, so it can read false for a child
    // that is structurally identity, sending that child to the dense-grid
    // fallback instead of pass-through. Completing the predicate was measured
    // and did not pay: the misses are rare and land on tiny grids.

    // ── Node-index monotonicity tracking ─────────────────────────────────
    //
    // Each producer tags its level with a `LevelGrid` variant. Only the
    // `DenseStrict` variant (set by dense-internal emit and identity
    // shortcuts) guarantees row-major strict monotonicity of live cells;
    // leaves (`CONJOIN_GRID`) and scatter-materialised grids (`DenseWeak`)
    // do not. Nothing consumes that guarantee today — pair lists are unordered
    // sets and twin contraction is order-independent (see the NOTE in
    // types.rs), so no emit site needs to produce a particular order. The
    // variant tagging is retained as cheap producer metadata.

    // ── Bottom-up product construction ─────────────────────────────────
    //
    // At each vtree level, compute c1[i] ∧ c2[j] for all (i,j) node pairs.
    // Leaf levels: static CONJOIN_GRID lookup (no stored nodes, no iteration).
    // Internal levels: either dense grid iteration or sparse scatter pipeline,
    // chosen online based on children's product density.

    apply_leaf_levels(
        eng,
        &vtree, &run.c1_widths, &run.c2_widths, &mut run.grids, &mut run.node_idx,
        &mut run.grid_end, &mut run.live_counts, &mut run.out_nodes_so_far, run.might_use_sparse,
        restrict.map(|r| r.leaf_children),
    )?;

    seed_restricted_fastpath_levels(&mut run, restrict, &vtree);

    let canon_leaves = seed_marginal_leaves(
        c1, c2, &vtree, &mut run.levels, restrict, &run.c1_identity, &run.c2_identity, ws.as_ref(),
    );

    let internal_iter = match restrict {
        // Restricted mode walks `R` in `topo_pos` order — the generic
        // `internal_bottomup()` order with the fast-path levels removed.
        Some(r) => LevelWalk::Restricted(r.rebuild.iter(), &vtree),
        // The tuned lazy bottom-up iterator: no allocation, no reorder.
        None => LevelWalk::Depth(vtree.internal_bottomup()),
    };
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
        let taken = restrict.is_none() && try_fast_paths(eng, &mut run, c1, c2, shape)?;

        if !taken {
            assert_marg_schedule(&run, c1, c2, shape, &vtree);
            // Dense vs sparse, decided on the children's measured density. `None`
            // means the sparse pipeline built the level; otherwise the flag says
            // whether the dense build takes its one-marginal-child sparse variant.
            match try_sparse_level(eng, &mut run, c1, c2, shape, &vtree, marginalize_targets)? {
                None => {}
                Some(use_sparse_marg) => build_level_dense(
                    eng, &mut run, c1, c2, shape, use_sparse_marg,
                    &vtree, marginalize_targets, restrict, ws.as_mut(),
                )?,
            }
        }

        // Release this level's children's grid regions for a later level to
        // reuse, before the next iteration's own reserve fires.
        reclaim_child_grids(run.might_use_sparse, &mut run.grids, &mut run.free_regions, &run.c1_widths, &run.c2_widths, left_idx, right_idx);
    }
    Ok(())
    })();
    loop_result?;

    canonicalize_marginal_leaf_refs(&canon_leaves, &vtree, &mut run.levels, ws.as_ref());

    let out_local = compute_apply_output(
        c1, c2, &run.grids, &run.node_idx, &run.c2_widths,
        &run.c1_identity, &run.c2_identity, &run.has_pl, &run.product_lists,
    );
    let out_vtree = c1.output.vtree;

    let out_local = guard_stale_false(out_local, out_vtree, &vtree, &run.levels);

    let mut levels = run.finish(eng, marginalize_targets.is_some());

    let output = TddNodeId { vtree: out_vtree, local: out_local };
    let Some(r) = restrict else {
        let mut out = Tdd::with_levels(vtree, levels, output);
        out.weights = ws;
        return Ok(out);
    };

    // ── Restricted tail: merge `R` back into the accumulator's array ─────
    //
    // Every level OFF `R` rode through untouched — it is still the
    // accumulator's own level, byte-for-byte, in the accumulator's own
    // allocation. Moving the `|R|` rebuilt levels across is the whole
    // "output" step; the fresh array goes back to the pool with only empty
    // levels in it (the leaf-marginal seeding sweep, the one other writer,
    // is skipped under a restriction).
    //
    // SWAP rather than assign: the accumulator's superseded level at `t` goes
    // back into the fresh array, which is what heads to the level pool a few
    // lines later. Its `nodes`/`pairs` arenas are then reused by the next
    // merge's rebuild instead of being freed here and reallocated there.
    for &t in r.rebuild {
        let ti = t.idx();
        std::mem::swap(&mut c1.levels[ti], &mut levels[ti]);
    }
    std::mem::swap(&mut c1.levels, &mut levels);

    // Contract seed: `R`, not every internal level. `R` is ancestor-closed
    // (`S` is; `AncClosure(P)` is by construction), so its complement is
    // DESCENDANT-closed — an off-`R` level's own pairs, its parent's pairs and
    // its whole subtree are bit-identical to the accumulator's. A contraction
    // sweep seeded at an off-`R` level would therefore re-run the accumulator's
    // own last sweep on the same bytes and fire nothing. Whatever the
    // accumulator still owed is carried over rather than dropped
    // (`with_levels_dirty`'s obligation 2). Same argument, same shape, as
    // `conjoin_clause::try_apply_and_clause`'s seed.
    let mut dirty_contract = std::mem::take(&mut c1.dirty.contract);
    let mut dirty_leaf_contract = std::mem::take(&mut c1.dirty.leaf_contract);
    dirty_contract.reserve(r.rebuild.len());
    dirty_leaf_contract.reserve(r.rebuild.len());
    for &t in r.rebuild {
        dirty_contract.push(t.0);
        dirty_leaf_contract.push(t.0);
    }
    let mut out = Tdd::with_levels_dirty(vtree, levels, output, dirty_contract, dirty_leaf_contract);
    out.weights = ws;
    Ok(out)
}
