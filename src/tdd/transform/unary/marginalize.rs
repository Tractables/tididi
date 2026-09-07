//! Weight context + marginalization primitives for mc-mode compilation.
//!
//! Lives in `tdd` (the marginalization primitives it defines are tdd-facing);
//! driven by the compile orchestrator. The marginalize-scheduler scope state
//! read here (`NO_MARGINALIZE_VARS_SCOPED`, `EXTRA_DEFER_NODES_SCOPED`) is
//! owned here too and installed by compile via the RAII guards below — the
//! same state-owned-where-it's-read pattern as `tdd::transform::unary::project`'s
//! `ScopedProjectLeaves`. No production edges into the downstream driver's
//! compile module remain.

use rustc_hash::FxHashMap;

use crate::tdd::counts::{
    ensure_fold_walk, unwrap_infallible, ColumnRetention, Count, CountRead, CountVec, IntFold,
    RecoveryPanic, WeightFold, STREAM_OVERFLOW,
};
use crate::tdd::transform::pairwise::conjoin::{reduce_poll_stride, ApplyError, PollTicker};
use crate::tdd::types::{BigSide, LeafLabel, MargRef, Tdd, TddLevel, assert_can_make_marginal};
use crate::tdd::marg_slots::{count_key_at, CountKey};
use crate::tdd::query::semiring::WeightVal;
use crate::tdd::weight_store::WeightStore;
use crate::vtree::{Literal, VarId, Vtree, VtreeIdx, VtreeNode};

// ── Marginalize-scheduler scope state (installed by compile, read here) ──────

// Per-call exclusion set for marginalization. When `Some(set)`, vtree nodes
// whose subtree contains any var in `set` are kept off the marginalize_at
// schedule so their pair structure stays available for downstream
// `apply_and` (e.g. conjoining `build_amo` / `build_biclique` TDDs after
// the main compile in the `--amo-rewrite` / `--biclique-rewrite` paths).
// Set via `ScopedNoMarginalize::new` (RAII guard, panic-safe); read in
// `compute_marginalize_at`. All non-excluded levels still marginalize
// normally, preserving most of `compile_cnf_mc`'s memory wins.
thread_local! {
    /// Variables to leave un-marginalized during the current compile (`Some`
    /// while a [`ScopedNoMarginalize`] guard is live); read in
    /// `compute_marginalize_at`.
    #[doc(hidden)]
    pub static NO_MARGINALIZE_VARS_SCOPED: std::cell::RefCell<Option<std::collections::HashSet<crate::vtree::VarId>>>
        = const { std::cell::RefCell::new(None) };
}

/// RAII guard: installs a `no_marginalize_vars` set in the thread-local
/// exclusion slot for the duration of its scope, restoring the previous
/// value on drop (so nested/re-entrant compiles are safe).
#[doc(hidden)]
pub struct ScopedNoMarginalize {
    _prev: Option<std::collections::HashSet<crate::vtree::VarId>>,
}

impl ScopedNoMarginalize {
    /// Install `vars` as the no-marginalize exclusion set; the previous set is
    /// restored when the returned guard drops.
    pub fn new(vars: std::collections::HashSet<crate::vtree::VarId>) -> Self {
        let prev = NO_MARGINALIZE_VARS_SCOPED.with(|c| c.borrow_mut().replace(vars));
        Self { _prev: prev }
    }
}

impl Drop for ScopedNoMarginalize {
    fn drop(&mut self) {
        let prev = self._prev.take();
        NO_MARGINALIZE_VARS_SCOPED.with(|c| *c.borrow_mut() = prev);
    }
}

// Companion to compile's `EXTRA_TDDS_AT_SCOPED` for the mc (marginalizing)
// joint compile: the vtree nodes at which an extra biconditional TDD is
// conjoined. Read in `compute_marginalize_at`, which lifts the forget point of
// EVERY leaf in each such node's subtree to that node's step, so the
// accumulator carries no marginal levels when the explicit biconditional is
// conjoined (`apply_and` can't combine a marginal level with an explicit
// operand — it panics indexing the marginal level's smaller node array). This
// is what makes `EXTRA_TDDS_AT` sound with marginalization ON: keep each
// conjoin node's subtree explicit until the conjoin lands, then marginalize.
// Deferring only the biconditional's own support is NOT enough — any other
// already-forgotten var in the subtree breaks the conjoin. None means no
// deferral (the common case). Set via `ScopedDeferNodes` (RAII guard).
thread_local! {
    pub(crate) static EXTRA_DEFER_NODES_SCOPED: std::cell::RefCell<Option<Vec<VtreeIdx>>>
        = const { std::cell::RefCell::new(None) };
}

/// RAII guard: installs the `extra_defer_nodes` list (vtree nodes where an extra
/// TDD is conjoined) for one `compile_component` call. Read in
/// `compute_marginalize_at` to keep each such node's subtree explicit until the
/// conjoin step. Restores the previous slot on drop.
#[doc(hidden)]
pub struct ScopedDeferNodes {
    _prev: Option<Vec<VtreeIdx>>,
}

impl ScopedDeferNodes {
    /// Install `nodes` as the extra-defer node list; the previous list is
    /// restored when the returned guard drops.
    pub fn new(nodes: Vec<VtreeIdx>) -> Self {
        let prev = EXTRA_DEFER_NODES_SCOPED.with(|c| c.borrow_mut().replace(nodes));
        Self { _prev: prev }
    }

    /// ADD `nodes` to whatever list is already installed, rather than replacing
    /// it. Deferral is a union: two independent producers of injected TDDs (an
    /// outer caller's biconditionals and an inner compile's bag clusters) each
    /// need their own conjoin nodes kept explicit, and honouring only the
    /// innermost list would let the outer producer's support be forgotten early.
    /// Restores the previous list verbatim on drop, exactly like [`Self::new`].
    pub fn extending(nodes: Vec<VtreeIdx>) -> Self {
        let mut merged =
            EXTRA_DEFER_NODES_SCOPED.with(|c| c.borrow().clone()).unwrap_or_default();
        merged.extend(nodes);
        let prev = EXTRA_DEFER_NODES_SCOPED.with(|c| c.borrow_mut().replace(merged));
        Self { _prev: prev }
    }
}

impl Drop for ScopedDeferNodes {
    fn drop(&mut self) {
        let prev = self._prev.take();
        EXTRA_DEFER_NODES_SCOPED.with(|c| *c.borrow_mut() = prev);
    }
}


/// Per-clause last-mention precompute for the streaming scheduler.
///
/// Returns `completes_at[i]` = list of sub-vtrees S ⊂ `subtree(current_node)`
/// whose last-mention (highest clause-position in this X-batch where any
/// var in `V_S` appears) is exactly `clause_lits[i]`. `clause_lits[i]` is the
/// literal slice of the i-th clause in this X-batch. Filtered to S that the
/// cross-step schedule has already deemed freezable at step `current_node`
/// (`cross_step_targets[s] == true`), since only those should be streamed
/// here — others wait for their own step's batch.
///
/// Cost: O(Σ `clause_length` + `num_vtree_nodes`) per X-batch. Only called when
/// `marginalize_targets` is `Some` (i.e., `marginalize_at[current_node]` is
/// non-empty), so empty-schedule batches pay nothing.
#[doc(hidden)]
pub fn compute_intra_batch_completions(
    clause_lits: &[&[Literal]],
    vtree: &Vtree,
    _current_node: VtreeIdx,
    cross_step_targets: &[bool],
) -> Vec<Vec<VtreeIdx>> {
    let n = vtree.num_nodes();
    let num_vars = vtree.num_vars() as usize;

    // last_clause_pos[v] = largest i where clause_lits[i] mentions v, else None.
    let mut last_clause_pos: Vec<Option<u32>> = vec![None; num_vars];
    for (i, lits) in clause_lits.iter().enumerate() {
        for lit in *lits {
            let v = lit.var.idx();
            if v < num_vars {
                last_clause_pos[v] = Some(i as u32);
            }
        }
    }

    // Bottom-up max over the vtree. Sub-vtrees outside subtree(current_node)
    // have no in-batch mentions (clauses scoped to current_node have all vars
    // in V_{current_node}), so their completion stays None.
    let mut completion: Vec<Option<u32>> = vec![None; n];
    for &t in vtree.bottomup_topo() {
        match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                let v = var.idx();
                if v < num_vars {
                    completion[t.idx()] = last_clause_pos[v];
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                let cl = completion[left.idx()];
                let cr = completion[right.idx()];
                completion[t.idx()] = match (cl, cr) {
                    (None, None) => None,
                    (Some(a), None) | (None, Some(a)) => Some(a),
                    (Some(a), Some(b)) => Some(a.max(b)),
                };
            }
        }
    }

    let mut completes_at: Vec<Vec<VtreeIdx>> = vec![Vec::new(); clause_lits.len()];
    for node_idx in 0..n {
        if !cross_step_targets[node_idx] { continue; }
        if let Some(c) = completion[node_idx] {
            completes_at[c as usize].push(VtreeIdx(node_idx as u32));
        }
    }

    completes_at
}

/// Precompute which vtree levels become frozen after each compilation step.
///
/// Returns `marginalize_at[t]`: vtree nodes whose levels can be marginalized
/// when step `t` completes. Same logic as `ooc_analysis.rs` steps 1–3.
#[doc(hidden)]
pub fn compute_marginalize_at(
    clause_lits: &[&[Literal]],
    vtree: &Vtree,
    clauses_at: &[Vec<usize>],
) -> Vec<Vec<VtreeIdx>> {
    let n = vtree.num_nodes();
    let num_vars = vtree.num_vars() as usize;

    let topo_pos_of = |idx: VtreeIdx| -> u32 { vtree.topo_pos(idx) };
    let topo_at = |pos: u32| -> VtreeIdx { vtree.bottomup_topo()[pos as usize] };

    // Step 1: for each variable, find the highest-scoped clause mentioning it.
    let mut last_scope_pos: Vec<u32> = vec![0; num_vars];
    for (scope_idx, clause_indices) in clauses_at.iter().enumerate() {
        if clause_indices.is_empty() {
            continue;
        }
        let scope_pos = topo_pos_of(VtreeIdx(scope_idx as u32));
        for &ci in clause_indices {
            for lit in clause_lits[ci] {
                let v = lit.var.idx();
                if v < num_vars {
                    last_scope_pos[v] = last_scope_pos[v].max(scope_pos);
                }
            }
        }
    }

    // Fold in `EXTRA_DEFER_NODES_SCOPED` (the mc joint compile): each extra
    // biconditional TDD is conjoined at its bucket node `T` by
    // `conjoin_extra_tdds_at`. For a node carrying extras, `run_*_step` defers
    // the step-`T` marginalization until *after* the conjoin (see step.rs), so
    // the conjoin sees `T`'s accumulator with all of subtree(T) still explicit —
    // PROVIDED no subtree(T) var was already forgotten at a descendant step. We
    // guarantee that here: lift every leaf under each conjoin node `T` so its
    // forget point is no earlier than `topo_pos(T)` (T's own step). Top-down pass
    // (parents before children, via reversed bottom-up topo) propagates each
    // conjoin node's deferral down to all its subtree leaves; the existing Step-2
    // bottom-up pass then carries the lifted leaf positions back up to internals.
    // None/empty in the common (non-joint) case. This works even for a root-
    // bucketed bico (whole-vtree support): all vars stay explicit to the root,
    // where the conjoin precedes the final sum-out.
    EXTRA_DEFER_NODES_SCOPED.with(|c| {
        if let Some(ref defer_nodes) = *c.borrow() {
            if defer_nodes.is_empty() {
                return;
            }
            let mut defer_to = vec![0u32; n];
            for &t in defer_nodes {
                defer_to[t.idx()] = topo_pos_of(t);
            }
            let mut deferral = vec![0u32; n];
            for &t in vtree.bottomup_topo().iter().rev() {
                let base = match vtree.node(t).parent() {
                    Some(p) => deferral[p.idx()],
                    None => 0,
                };
                deferral[t.idx()] = base.max(defer_to[t.idx()]);
            }
            for &t in vtree.bottomup_topo() {
                if let VtreeNode::Leaf { var, .. } = vtree.node(t) {
                    let v = var.idx();
                    if v < num_vars {
                        last_scope_pos[v] = last_scope_pos[v].max(deferral[t.idx()]);
                    }
                }
            }
        }
    });

    // Step 2: compute completion_pos bottom-up.
    let mut completion_pos: Vec<u32> = vec![0; n];
    for &t in vtree.bottomup_topo() {
        match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                let v = var.idx();
                if v < num_vars && last_scope_pos[v] > 0 {
                    completion_pos[t.idx()] = last_scope_pos[v];
                } else {
                    completion_pos[t.idx()] = topo_pos_of(t);
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                completion_pos[t.idx()] = completion_pos[left.idx()]
                    .max(completion_pos[right.idx()]);
            }
        }
    }

    // Step 3: build marginalize_at groups.
    //
    // Freeze d at step completion_pos[d] — the topo position of the highest-
    // scoped clause mentioning any variable in subtree(d). After that step,
    // no future batch references subtree(d), so d can be marginalized.
    //
    // The original R172 schedule was briefly reverted to completion_pos[parent(d)]
    // after `mc2025_track1_181` crashed in --mc mode. Investigation showed the
    // schedule was correct: the actual bugs were (1) `reset_levels` failing to
    // clear `marginal_counts` so pooled levels leaked stale marginal state into
    // fresh TDDs, and (2) `apply_and`'s c1_identity flag not being propagated
    // when the c2-identity fast-path fired first, breaking the marginal-schedule
    // invariant downstream. Both fixed — see types.rs::reset_levels and
    // apply_inner.rs fast-path-1/2 propagation.
    let mut marginalize_at: Vec<Vec<VtreeIdx>> = vec![Vec::new(); n];

    // Honor the `ScopedNoMarginalize` thread-local: bottom-up propagate a
    // "subtree contains excluded var" bit; vtree nodes flagged true are
    // omitted from the schedule so their pair structure stays available
    // for downstream `apply_and` calls (the biclique-rewrite / amo-rewrite
    // paths build native auxiliary TDDs over the scope vars and conjoin
    // them after main compile). Empty `contains_excluded` means no
    // exclusion set was installed (the common case).
    let contains_excluded: Vec<bool> = NO_MARGINALIZE_VARS_SCOPED.with(|c| {
        let exclude = c.borrow();
        if let Some(ref set) = *exclude {
            let mut v = vec![false; n];
            for &t in vtree.bottomup_topo() {
                match vtree.node(t) {
                    VtreeNode::Leaf { var, .. } => {
                        v[t.idx()] = set.contains(var);
                    }
                    VtreeNode::Internal { left, right, .. } => {
                        v[t.idx()] = v[left.idx()] || v[right.idx()];
                    }
                }
            }
            v
        } else {
            Vec::new()
        }
    });
    let exclusion_active = !contains_excluded.is_empty();

    for node_idx in 0..vtree.num_nodes() {
        if vtree.node(VtreeIdx(node_idx as u32)).parent().is_some() {
            if exclusion_active && contains_excluded[node_idx] {
                continue;
            }
            let save_pos = completion_pos[node_idx];
            let save_vtree_idx = topo_at(save_pos);
            marginalize_at[save_vtree_idx.idx()].push(VtreeIdx(node_idx as u32));
        }
    }

    // Sort each group in bottom-up topo order so children are processed first.
    for group in &mut marginalize_at {
        if group.len() > 1 {
            group.sort_by_key(|&idx| topo_pos_of(idx));
        }
    }

    marginalize_at
}

/// Convert frozen levels in a TDD to marginal mode (store only per-node model counts).
///
/// `targets` must be sorted in bottom-up topo order so that when we compute
/// counts for level d, its children are already marginal (with stored counts)
/// or are leaves (fixed counts).
///
/// # Errors
///
/// Returns `Err(ApplyError::Deadline)` if the caller's wall passed while the
/// batch was running and the post-apply poll is armed. The targets processed
/// before the cut keep their marginal stores and the end-sweep tagger has run
/// over them, so the diagram left behind is exactly the one a batch over that
/// prefix would have produced — well-formed, readable, and count-preserving.
pub fn marginalize_batch(
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), ApplyError> {
    if targets.is_empty() {
        return Ok(());
    }
    // Invariant: the integer batch never runs during a weighted compile — its
    // `get_marginal_count` reads integer `marginal_counts`, which a weight-marginal
    // level (MARG_WEIGHTED, values in the external WeightStore) does not have. Every
    // batch site must route through `marginalize_batch_weighted_if_active` first.
    // Asserted here so a future bypass fails loudly at the entry point instead of via
    // the cryptic `unreachable!()` deep in `get_marginal_count`.
    debug_assert!(
        !weight_ctx_active(),
        "integer marginalize_batch invoked under an active weighted context — \
         route via marginalize_batch_weighted_if_active"
    );
    // Snapshot which levels are ALREADY marginal at batch entry. The end-sweep
    // tagger (`tag_all_marg_side_slots`, below) must resolve the bare-coord refs
    // into children that became marginal *in this batch* but must NOT touch
    // children marginalized by a prior batch/apply — those already carry inline
    // counts (emitted by the prior batch's end-sweep). The per-level
    // `marg_inlined_left/right` marker tries to encode this but is clobbered
    // when levels merge/rebuild; this snapshot is the reliable discriminator
    // and cannot be lost (it's not stored on the level).
    let was_marginal: Option<Vec<bool>> =
        Some(tdd.levels.iter().map(|l| l.is_marginal()).collect());

    // Compute per-node model counts bottom-up for all levels in the TDD that
    // need marginalization. We store counts in a flat structure indexed by
    // vtree_idx → Vec<u128>, computed on demand.
    //
    // Process targets in bottom-up order: when we reach level d, its children
    // (left_d, right_d) are either:
    //   - Already marginal (counts stored in marginal_counts)
    //   - Leaf levels (fixed: One=2, Pos=1, Neg=1)
    //   - Not yet processed (still explicit) — compute from their pairs

    // We need a temporary counts buffer for levels that are still explicit
    // at the time we need their child counts. Build counts lazily. One
    // `CountVec` per level replaces the old parallel `computed_counts`/
    // `computed_big` pair — see `tididi/src/tdd/counts.rs`.
    // (`CountVec` is deliberately not `Clone` — clones go through the guarded
    // `try_clone` — so the None-filled buffer can't use `vec![None; n]`.)
    let mut computed: Vec<Option<CountVec<RecoveryPanic>>> =
        (0..vtree.num_nodes()).map(|_| None).collect();

    // The batch's ONE preemption point, amortized. This walk is where a leaf
    // compile forgets its variables and on a near-root step it is minutes of
    // folding with no return to the caller, so without it the grant is observed
    // only at the step seam past it. Metered in nodes of the target level — the
    // unit the fold, the dedup and the parent remap all scale with — and
    // disarmed (every compile outside the DPLL-canopy stage) the whole thing is
    // an add and a relaxed load of a `false` per target.
    let mut poll = PollTicker::reduce(reduce_poll_stride());

    for &d in targets {
        let di = d.idx();
        // Cut BETWEEN targets, never inside one: each iteration makes exactly one
        // level marginal and rewrites its parent's refs, so the prefix already
        // done is a complete batch of its own once the end-sweep tagger below has
        // run over it — which is why the error arm runs the tagger rather than
        // returning straight out.
        if let Err(e) = poll.tick_by(tdd.levels[di].width() as u64 + 1) {
            crate::tdd::types::tag_all_marg_side_slots(tdd, was_marginal.as_deref());
            return Err(e);
        }
        if tdd.levels[di].is_marginal() || tdd.levels[di].width() == 0 {
            continue;
        }
        if vtree.node(VtreeIdx(di as u32)).is_leaf() {
            continue;
        }

        let (left, right) = vtree.children(d);
        let li = left.idx();
        let ri = right.idx();
        let width = tdd.levels[di].width();

        // Ensure child counts are available.
        ensure_counts(tdd, left, vtree, &mut computed);
        ensure_counts(tdd, right, vtree, &mut computed);

        // Compute counts for level d.
        let mut counts = CountVec::<RecoveryPanic>::with_width(width);

        for (i, _pairs) in tdd.levels[di].internal_inputs_iter() {
            // Per-node fold Σ left×right via `compute_marginal_node_int`
            // ([`IntFold::fold`] with this context's readers). `_pairs` is
            // re-derived inside via `pairs_iter_of_idx(i)` — identical to this
            // iterator's yield for internal inputs.
            let c = compute_marginal_node_int(tdd, &tdd.levels[di], i, li, ri, &computed);
            counts.set_i(i, c);
        }

        // Park the counts where the cascade below can reach them (uncompacted,
        // indexed by node index), then take them back for the dedup that installs
        // `d`'s store. Moved, not copied: the store has a single owner across the
        // whole window, so a width-sized duplicate would be pure peak memory —
        // same handoff as the weighted twin's `computed_weights[di]`.
        computed[di] = Some(counts);

        // Cascade children FIRST (bottom-up), then make `d` marginal. This
        // satisfies the `assert_can_make_marginal` precondition: by the time
        // we marginalize `d`, both children are already marginal (or leaves).
        let (l_child, r_child) = vtree.children(d);
        cascade_marginalize(tdd, vtree, l_child, &mut computed);
        cascade_marginalize(tdd, vtree, r_child, &mut computed);

        assert_can_make_marginal(&tdd.levels, vtree, d);
        // Store is born C3: no duplicate count values; enforced here, not by a
        // later canon pass. Dedup the store before make_marginal so the level
        // is C3 from birth. `new_counts`/`new_big` are the compacted store handed
        // to make_marginal; `remap[old_slot] = new_slot` lets us redirect parent
        // refs. Parent-ref remap runs BEFORE make_marginal (the parent level's
        // pair refs are bare slot indices equal to node indices while the child
        // is still explicit; both spellings are identical here).
        let counts = computed[di].take().expect("counts just computed for di");
        let (fast, big) = counts.into_parts();
        let (new_counts, new_big, marg_remap) =
            dedup_fresh_store(fast, big);
        // Remap parent refs before the level becomes marginal. Only meaningful
        // when the vtree-parent is still explicit — a marginal parent has no pair
        // lists for the redirect step.
        if let Some(parent_vi) = vtree.node(d).parent() {
            tdd.mark_contract_dirty(parent_vi);
            if !tdd.levels[parent_vi.idx()].is_marginal() {
                let (pl, _) = vtree.children(parent_vi);
                let t1_is_left = pl == d;
                remap_parent_refs_pretag(
                    tdd, d, parent_vi, t1_is_left, &marg_remap,
                );
            }
        }
        // `d` is now marginal: store is C3 from birth.
        tdd.levels[di].make_marginal(new_counts, new_big);
        // `d` now subsumes its children — free their now-dead stores (O(1)).
        free_subsumed_marginal_children(tdd, vtree, d, None);
        // `computed[di]` is already empty (taken above). Nothing re-fills it:
        // once `d` is marginal every reader — the parent fold via
        // `read_marginal_count`/`ensure_counts` — takes the `is_marginal()`
        // branch and reads `tdd.levels[di].marginal_counts` (the compacted
        // store). `computed` accumulates across all batch levels, so holding
        // the uncompacted store past this point would raise the batch's
        // cumulative peak, not just a transient one.
    }

    // This standalone marginalize path does not go through `apply_and_fallible`,
    // so the end-of-apply tagger never runs on it. canon's redirect tags the
    // refs it rewrites, but its no-duplicate early-return leaves untouched
    // boundary refs raw. Tag every persisted marg-side slot ref once here, at
    // the batch chokepoint (idempotent with canon's per-target tagging), so the
    // 0=inline decode invariant holds before any downstream reader.
    // Refs into the just-marginalized child are bare grid `node_idx` slots —
    // this path runs after apply returns, so they were never read-time-tagged.
    // Declare NOT self-describing → `emit_or_tag` resolves them via counts even
    // under no-reexpand (the inline-decode bug fix, marg-canon #63).
    crate::tdd::types::tag_all_marg_side_slots(tdd, was_marginal.as_deref());

    // Leaf (single-variable) marginalization. The main loop above skips leaf
    // targets — a leaf has no pairs/store of its own. Sum them out by inlining
    // their fixed 0/1/2 model count into the parent's leaf-side refs. This MUST
    // run after `tag_all_marg_side_slots`: the tagger keys off the batch-entry
    // `was_marginal` snapshot, so a leaf flipped marginal earlier would have its
    // side re-resolved as bare slots (misreading our inline refs). Here the leaf
    // is still structural while the tagger walks (its side skipped), then we write
    // fully self-describing inline refs ourselves.
    for &d in targets {
        if vtree.node(d).is_leaf() {
            marginalize_leaf_inline(tdd, d, vtree);
        }
    }
    Ok(())
}

/// Sum out a single-variable vtree LEAF by inlining its fixed model count
/// directly into the parent's leaf-side refs.
///
/// A leaf's marginal count is fixed by its label (One→2, Pos/Neg→1, Zero→0), so
/// it always fits `MargRef::Inline` — no slot store is needed. The leaf's store
/// stays empty; `make_marginal(vec![], None)` only flips the `is_marginal()`
/// reader/apply signal (every reader then routes through the marginal branch and
/// decodes the inline refs). Rewriting Pos and Neg to the byte-identical
/// `Inline(1)` is the size win: the parent's `(·,x)` and `(·,¬x)` branches become
/// structurally equal, so the standard contraction / p-fusion passes merge the
/// now-twin parent nodes — we only seed `mark_contract_dirty`, no new machinery.
///
/// No-op when the parent is already marginal: the leaf was then folded into the
/// parent's store via the leaf-fixed-count fold (`get_marginal_count` leaf
/// branch), so there are no pairs left to rewrite.
#[doc(hidden)]
pub fn marginalize_leaf_inline(tdd: &mut Tdd, leaf: VtreeIdx, vtree: &Vtree) {
    debug_assert!(vtree.node(leaf).is_leaf());
    if tdd.levels[leaf.idx()].is_marginal() {
        return;
    }
    // Projection opt-out: a projected (PMC / ∃-quantified) compile cofactors
    // leaves via `condition_leaf` (reading Pos/Neg labels) inside `project_var`.
    // Leaf-marg inlines a leaf's fixed count into its parent and drops the leaf's
    // Boolean structure, so the cofactor walk reads a corrupted diagram and the
    // projected count comes out wrong. Keep leaves structural whenever a caller
    // has installed a projected set — the parent's ordinary internal marginalize
    // still sums the leaf via its fixed label, exactly as before leaf-marg. The
    // size win is forgone only on this minority path; plain `--mc` (the MCC
    // target) never has projection active, so leaf-marg stays on there.
    if crate::tdd::transform::unary::project::caller_projection_active() {
        return;
    }
    // Inlining a leaf's count (bit-30 ref) is leaf-marg's entire mechanism: Pos/Neg
    // both → Inline(1) makes the parent's branches twins for contraction. It needs
    // the inline budget to hold the max leaf count (One→2). In production
    // `marg_inline_max` is the full 30-bit range so this always holds; only a
    // test-lowered budget (<2) fails it, and there we leave the leaf structural
    // (exact, no size win) rather than synthesize a slot store the bare-ref decode
    // path doesn't integrate correctly.
    if crate::tdd::types::marg_inline_max() < 2 {
        return;
    }
    if let Some(parent_vi) = vtree.node(leaf).parent() {
        let pi = parent_vi.idx();
        if !tdd.levels[pi].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            let leaf_is_left = pl == leaf;
            inline_leaf_refs_at_parent(tdd, parent_vi, leaf_is_left);
            tdd.mark_contract_dirty(parent_vi);
            if leaf_is_left {
                tdd.levels[pi].set_marg_inlined_left(true);
            } else {
                tdd.levels[pi].set_marg_inlined_right(true);
            }
        }
    }
    // Flip the reader/apply signal; the store stays empty (all counts are inline
    // at the parent).
    tdd.levels[leaf.idx()].make_marginal(Vec::new(), None);
}

/// Rewrite every leaf-side ref of `parent_v`'s nodes from a `LeafLabel` index
/// (One/Pos/Neg) into a `MargRef::Inline(count)` (2/1/1). Mirrors
/// `remap_parent_refs_pretag`, but maps leaf labels to inline counts instead of
/// remapping slot indices. Bit 30 (the inline tag) is disjoint from
/// `LEAF_BIT/MULTI_BIT` (bit 31), so the rewritten refs keep their inline/multi
/// node encoding.
fn inline_leaf_refs_at_parent(tdd: &mut Tdd, parent_v: VtreeIdx, leaf_is_left: bool) {
    let to_inline = |raw: u32| -> u32 {
        if raw & (1 << 31) != 0 {
            return raw; // ZERO sentinel (count 0) — already self-describing
        }
        // Idempotent: a ref that already carries the inline tag (bit 30) is an
        // Inline(count) we wrote on a prior pass — leave it untouched. Without
        // this guard a re-entry (parent revisited while its leaf-side refs are
        // already inline) would feed a bit-30 value into `LeafLabel::from_idx`,
        // whose `_ => unreachable!` panics (the leaf labels are only 0/1/2).
        if raw & crate::tdd::types::MARG_OVERFLOW_TAG != 0 {
            return raw;
        }
        let count: u128 = match raw {
            0 => 2,        // One
            1 | 2 => 1,    // Pos / Neg
            3 => 0,        // Zero (sentinel index — defensive; not normally stored)
            other => panic!("inline_leaf_refs_at_parent: unexpected leaf-side ref {other}"),
        };
        MargRef::inline_raw(count).expect("leaf count 0/1/2 always fits inline")
    };
    let plevel = &mut tdd.levels[parent_v.idx()];
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if leaf_is_left {
                node.a = to_inline(node.a);
            } else {
                node.b = to_inline(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                if leaf_is_left {
                    p.left = crate::tdd::types::LocalNodeIdx(to_inline(p.left.idx() as u32));
                } else {
                    p.right = crate::tdd::types::LocalNodeIdx(to_inline(p.right.idx() as u32));
                }
            }
        }
    }
}

/// The pinned column of a weight-marginal vtree LEAF: [`WeightStore::leaf_val`]
/// for One/Pos/Neg in [`LeafLabel::from_idx`] slot order (0 = One = w⁺+w⁻,
/// 1 = Pos = w⁺, 2 = Neg = w⁻).
///
/// THE definition of that column. `marginalize_leaf_weighted` installs what this
/// returns; the apply-side canon pass, the streaming child view and
/// [`debug_check_leaf_columns_pinned`] all re-derive it here rather than each
/// spelling the triple out, so "what the column holds" cannot drift between them.
pub(crate) fn leaf_column_vals(ws: &WeightStore, var: VarId) -> Vec<WeightVal> {
    (0..crate::tdd::types::LEAF_WIDTH)
        .map(|i| ws.leaf_val(var, LeafLabel::from_idx(i)))
        .collect()
}

/// Canonical-slot map for a weight-marginal leaf's pinned column: `canon[r]` is
/// the SMALLEST slot whose value equals `vals[r]`, so every ref that names a
/// value names it by one agreed slot.
///
/// Equality is `weight_key` — the crate's one value-identity choke point, and in
/// the exact-rational domain it is exact equality (`ExactSmall`/`Exact` partition
/// the value space, so equal numbers always produce equal keys). Callers must
/// restrict this to the exact domain: a `WeightKey::Log` compares `f64` bit patterns,
/// which is a *representation* identity, not the value identity this map claims.
///
/// The shapes it can take, given `vals = [w⁺+w⁻, w⁺, w⁻]`:
///   * `w⁺ = w⁻` → `[0, 1, 1]` (Neg → Pos) — the common case, and the one that
///     recovers the integer arm's twin bonus;
///   * `w⁻ = 0`  → `[0, 0, 2]` (Pos → One);
///   * `w⁺ = 0`  → `[0, 1, 0]` (Neg → One);
///   * otherwise → the identity `[0, 1, 2]`, and the caller skips the walk.
/// (`w⁺ = w⁻ = 0` collapses all three onto slot 0, which the same rule produces.)
pub(crate) fn leaf_canon_map(vals: &[WeightVal]) -> [u32; 3] {
    use crate::tdd::query::semiring::weight_key;
    debug_assert_eq!(
        vals.len(),
        crate::tdd::types::LEAF_WIDTH,
        "leaf_canon_map: not a pinned leaf column"
    );
    let keys = [weight_key(&vals[0]), weight_key(&vals[1]), weight_key(&vals[2])];
    let mut canon = [0u32, 1, 2];
    // `s < r` and equality is transitive, so the first earlier slot carrying
    // `vals[r]` IS the minimum one (an even earlier match would have matched `s`
    // too, and `s` was taken as the first).
    for r in 1..crate::tdd::types::LEAF_WIDTH {
        for s in 0..r {
            if keys[s] == keys[r] {
                canon[r] = s as u32;
                break;
            }
        }
    }
    canon
}

/// The pinned column's slot for a VALUE: the SMALLEST slot `s < LEAF_WIDTH` of
/// level `level_idx`'s column whose value equals `want`, or `None` when the level
/// carries no column or no slot holds that value.
///
/// The ONE value-search over a pinned leaf column. A leaf column can never grow
/// (THE PIN INVARIANT on [`marginalize_leaf_weighted`]), so "is this value
/// representable at this leaf?" IS this lookup — which is what both mint-free
/// leaf folds ask: `minimize::contract::dup_resolve::scale_weight_leaf_by_lookup`
/// (is `k·slot` in the column?) and
/// `minimize::contract::p_fusion::resolve_leaf_fusion_refs_by_lookup` (is a (P)
/// group's SUM in the column?), plus the p-fusion census that sizes the second.
///
/// ASCENDING order is a SOUNDNESS requirement, not a style choice. The slot
/// returned here becomes a leaf-side ref, and every leaf-side ref must name the
/// CANONICAL (smallest) slot of its value class ([`leaf_canon_map`]) or pin check
/// #4 in [`debug_check_leaf_columns_pinned`] fires — scanning from 0 and taking
/// the first hit is exactly that minimum. The scan is also bounded at
/// `LEAF_WIDTH` rather than the slice length, so a column that somehow grew past
/// the pin can never hand back a ref no remap window is sized for.
///
/// Equality is `weight_key`, so callers must restrict this to the exact domain for the
/// same reason [`leaf_canon_map`] does: a `WeightKey::Log` compares `f64` bit
/// patterns, and a "hit" there would be a rounding coincidence rather than a
/// value identity.
pub(crate) fn find_leaf_slot_by_value(
    ws: &WeightStore,
    level_idx: usize,
    want: &WeightVal,
) -> Option<u32> {
    use crate::tdd::query::semiring::weight_key;
    let col = ws.level(level_idx)?;
    let want = weight_key(want);
    col.iter()
        .take(crate::tdd::types::LEAF_WIDTH)
        .position(|v| weight_key(v) == want)
        .map(|s| s as u32)
}

/// Rewrite every leaf-side ref of `plevel`'s nodes onto the canonical slot of an
/// equal-value class in a weight-marginal LEAF's pinned column (`canon` from
/// [`leaf_canon_map`]). The weighted analogue of `inline_leaf_refs_at_parent`'s
/// twin bonus, and the ONE implementation of that walk — the leaf-marg pass and
/// conjoin's leaf-marg propagation both call it.
///
/// Value-preserving by construction: a ref is only ever moved onto a slot holding
/// the SAME value, so every reader (`read_marginal_weight`, the streaming child
/// view, `validate::marg`) resolves it to the number it resolved to before.
/// What changes is structure — `(·, Pos)` and `(·, Neg)` become byte-identical
/// when w⁺ = w⁻, so the parent's nodes become twins and contraction collapses
/// them. That is sound only because a marginalized leaf's variable is PRIVATE (no
/// further conjunction can case-split on it), the same premise the integer arm's
/// Pos/Neg → `Inline(1)` rewrite rests on.
///
/// The column itself is NEVER touched — this walk moves refs of ONE `Tdd` only,
/// which is exactly what the compile-global pin permits (see THE PIN INVARIANT on
/// [`marginalize_leaf_weighted`]).
pub(crate) fn canonicalize_leaf_refs_at_parent(
    plevel: &mut TddLevel,
    leaf_is_left: bool,
    canon: &[u32; 3],
) {
    debug_assert!(
        *canon != [0, 1, 2],
        "canonicalize_leaf_refs_at_parent: identity map — the caller must skip \
         the walk rather than pay a level scan that rewrites nothing"
    );
    let to_canon = |raw: u32| -> u32 {
        if raw & (1 << 31) != 0 {
            return raw; // ZERO sentinel — carries no slot
        }
        // Weighted leaf sides never carry an inline (bit-30) ref: a weighted
        // `MargRef::Inline(gidx)` indexes the `WeightStore`'s global intern table,
        // which is rebuilt at every component graft, so nothing mints one into a
        // pair list (`dup_resolve::scale_weight_ref` refuses, and the leaf column
        // exists precisely so leaf refs stay bare slots).
        debug_assert!(
            raw & crate::tdd::types::MARG_OVERFLOW_TAG == 0,
            "canonicalize_leaf_refs_at_parent: inline ref {raw} on a weighted leaf side"
        );
        if raw & crate::tdd::types::MARG_OVERFLOW_TAG != 0 {
            return raw;
        }
        debug_assert!(
            (raw as usize) < crate::tdd::types::LEAF_WIDTH,
            "canonicalize_leaf_refs_at_parent: leaf-side ref {raw} outside the \
             pinned label range"
        );
        // Out of range means the pin is already broken; leave the ref alone so
        // `debug_check_leaf_columns_pinned` check #2 reports it at its own site
        // rather than this one panicking on an index.
        canon.get(raw as usize).copied().unwrap_or(raw)
    };
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if leaf_is_left {
                node.a = to_canon(node.a);
            } else {
                node.b = to_canon(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                if leaf_is_left {
                    p.left = crate::tdd::types::LocalNodeIdx(to_canon(p.left.idx() as u32));
                } else {
                    p.right = crate::tdd::types::LocalNodeIdx(to_canon(p.right.idx() as u32));
                }
            }
        }
    }
}

/// Walk down from a level whose parent is marginal, converting non-leaf,
/// still-explicit descendants to marginal mode using counts cached by
/// `ensure_counts` during the enclosing `marginalize_batch` call.
pub(crate) fn cascade_marginalize(
    tdd: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    computed: &mut [Option<CountVec<RecoveryPanic>>],
) {
    let ti = t.idx();
    if vtree.node(VtreeIdx(ti as u32)).is_leaf() || tdd.levels[ti].is_marginal() {
        return;
    }
    // Recurse children FIRST so they are marginal (or leaves) by the time
    // we marginalize `t` — satisfies the new `assert_can_make_marginal`
    // precondition.
    let (l_child, r_child) = vtree.children(t);
    cascade_marginalize(tdd, vtree, l_child, computed);
    cascade_marginalize(tdd, vtree, r_child, computed);

    let Some(cv) = computed[ti].take() else {
        // No cached counts: ancestor's count computation didn't visit here.
        // The cells of this level are structurally unreachable from the
        // implicit target's pair lists, so they will never be queried.
        return;
    };
    let (counts, big) = cv.into_parts();
    assert_can_make_marginal(&tdd.levels, vtree, t);
    // Store is born C3: no duplicate count values; enforced here, not by a
    // later canon pass. Dedup before make_marginal so the level is C3 from birth.
    let (new_counts, new_big, marg_remap) =
        dedup_fresh_store(counts, big);
    // See `marginalize_batch`: seed the parent of the newly-marginal level so
    // the narrow minimize re-contracts it. Within a marginalizing subtree the
    // parent often becomes marginal too (then contract skips it harmlessly);
    // the load-bearing seed is the boundary parent that stays explicit.
    if let Some(parent_vi) = vtree.node(t).parent() {
        tdd.mark_contract_dirty(parent_vi);
        // Remap parent refs through marg_remap. The parent-pair-list redirect
        // is technically wasted when the parent is about to be cleared by the
        // caller's make_marginal, but t's own count vec compaction IS preserved
        // and feeds into the ancestor's later count-vec construction.
        if !tdd.levels[parent_vi.idx()].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            let t1_is_left = pl == t;
            remap_parent_refs_pretag(
                tdd, t, parent_vi, t1_is_left, &marg_remap,
            );
        }
    }
    // Store is C3 from birth.
    tdd.levels[ti].make_marginal(new_counts, new_big);
    // `t` now subsumes its children — free their now-dead stores (O(1)).
    free_subsumed_marginal_children(tdd, vtree, t, None);
}

/// Global marginal-closure pass: marginalize **every** structural level whose
/// two children are both marginal, to fixpoint.
///
/// A rotation that brings two marginal children together leaves the new parent
/// level *structural* — `restructure_after_*_rotation` only shuffles `NodeIdx`
/// references, it never collapses a node to counts. But a node whose **both**
/// children are fully summed out (marginal) is itself fully summed out and MUST
/// be in marginal form for the diagram to stay canonical and count correctly
/// (skipping this "cascade up" is exactly the bug behind the original
/// count-unsafe parent-of-marginal rotation).
/// This pass closes every such cluster across the whole diagram at once.
///
/// After a re-search sweep performs many rotations, marginal clusters can appear
/// anywhere (a relaxed parent-of-marginal rotation leaves a structural parent over
/// two marginal children — count-unsafe until closed). Rather than hook each
/// committed rotation, run this once after the sweep: it scans all levels, collects
/// the bottom layer of structural-over-two-marginal levels, marginalizes them via
/// [`marginalize_batch`], and repeats until no level qualifies (a freshly-marginal
/// level can complete a cluster one level up).
///
/// It is a **no-op** when the diagram is already in canonical marginal form (the
/// normal bottom-up marginalize leaves no unclosed clusters), so it is safe to run
/// unconditionally — it only does work when rotations created clusters. Returns the
/// number of levels marginalized.
///
/// # Errors
///
/// Passes through [`marginalize_batch`]'s `Err(ApplyError::Deadline)`. The
/// clusters closed before the cut stay closed; the rest are still structural
/// levels over two marginal children, which is the state this pass exists to
/// finish and a caller that resumes will find waiting for it.
pub fn marginalize_closure(tdd: &mut Tdd, vtree: &Vtree) -> Result<usize, ApplyError> {
    let n = vtree.num_nodes();
    let mut total = 0usize;
    loop {
        let mut targets: Vec<VtreeIdx> = Vec::new();
        for i in 0..n {
            if vtree.node(VtreeIdx(i as u32)).is_leaf() || tdd.levels[i].is_marginal() {
                continue;
            }
            let t = VtreeIdx(i as u32);
            let (l, r) = vtree.children(t);
            if tdd.levels[l.idx()].is_marginal() && tdd.levels[r.idx()].is_marginal() {
                targets.push(t);
            }
        }
        if targets.is_empty() {
            break;
        }
        // bottom-up topo order = ascending index after reindex_bottomup.
        targets.sort_by_key(|t| t.idx());
        total += targets.len();
        // Route through the one integer-vs-weighted batch dispatch. Under an
        // active weighted context the targets are WEIGHT-marginal (no integer
        // `marginal_counts`), so the integer `marginalize_batch` would hit the
        // `unreachable!()` in `get_marginal_count`. `_if_active` runs the weighted
        // batch and returns true; for an integer compile it returns false and we
        // fall through to the integer batch unchanged. This is the same dispatch
        // every other batch site uses (see step.rs), so all `marginalize_closure`
        // callers (the rotation cluster pass, segment conjoin) are covered here.
        if !marginalize_batch_weighted_if_active(tdd, &targets, vtree) {
            marginalize_batch(tdd, &targets, vtree)?;
        }
    }
    Ok(total)
}

/// Free the dead per-node store of `parent`'s already-marginal children at the
/// moment `parent` itself becomes marginal.
///
/// Once `parent` is marginal it holds the aggregate that summed out its whole
/// subtree and (being marginal) carries no pair lists referencing anything
/// below it. A vtree node has exactly one parent, so each marginal child is now
/// unreachable from the root: its per-node data is dead weight that nothing will
/// ever read again. We free it here — exactly once, at the marginalization of
/// `parent`, touching only the two children — so the invariant "a marginal level
/// under a marginal parent carries no data" holds with O(1) work and no sweep.
///
/// Count/weight-preserving by construction: a marginal level's data IS its
/// subtree's value and it has no pairs to descend through, so `model_count` /
/// `weighted_output_value` stop at `parent` and never touch the freed children.
/// Handles both representations — integer (`marginal_counts`) and weighted (the
/// external `WeightStore` slot, cleared via `ws` when present; a level's slot
/// carrier `retired_marg_width` is zeroed either way so `width()` reports 0.
fn free_subsumed_marginal_children(
    tdd: &mut Tdd,
    vtree: &Vtree,
    parent: VtreeIdx,
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(parent).is_leaf() {
        return;
    }
    let (l, r) = vtree.children(parent);
    for c in [l.idx(), r.idx()] {
        let lvl = &mut tdd.levels[c];
        if !lvl.is_marginal() {
            // A structural child under a marginal parent never arises here:
            // marginalization makes a level marginal only after both children
            // are marginal-or-leaf (`assert_can_make_marginal`). Skip leaves.
            continue;
        }
        // Integer-marginal child: empty the count store but keep `Some` so the
        // level stays marginal/terminal; drop any big-overflow side-vec.
        if let Some(v) = lvl.marginal_counts.as_mut() {
            if !v.is_empty() {
                *v = Vec::new();
            }
            lvl.marginal_counts_big = None;
        }
        // Weight-marginal child: zero the slot carrier (→ width 0) and drop the
        // external store. `is_weight_marginal()` stays true (flag untouched).
        //
        // EXCEPT a vtree LEAF — the PIN INVARIANT (see `marginalize_leaf_weighted`).
        // A weight-marginal leaf's column is not this `Tdd`'s data to free: it is
        // the compile-GLOBAL, label-ordered 3-slot cache of `WeightStore::leaf_val`,
        // keyed by vtree index and shared by every `Tdd` of this compile (fresh
        // clause TDDs whose leaf level is still STRUCTURAL hold bare leaf-LABEL
        // refs that alias its slots by position). Erasing it here leaves those
        // holders reading an empty column — a panic on `&vals[slot]`, or a silent
        // width-0 mass drop through the `map_or(0, len)` readers. The dead-data
        // reclaim this function exists for simply does not apply: the column is a
        // cache of three constants, O(1) and re-derivable, not a per-node store
        // that grows with the diagram.
        if lvl.is_weight_marginal() && lvl.retired_marg_width != 0 && !vtree.node(VtreeIdx(c as u32)).is_leaf() {
            lvl.retired_marg_width = 0;
            if let Some(ws) = ws.as_deref_mut() {
                ws.set_level(c, Vec::new());
            }
        }
    }
}

/// Invariant scanner (tests): every marginal level whose parent is also marginal
/// must carry NO per-node data — the reclaim-on-marginalize above frees it.
/// Returns the levels that violate this (still hold integer counts, a
/// big-overflow vec, or a non-zero weighted slot carrier). A non-empty result is
/// dead memory we failed to free. Cheap O(num_nodes) scan, no data read.
/// `pub` (not `#[cfg(test)]`) so the downstream compiler crate's regression
/// tests can call it across the crate boundary (public-release P3a).
#[doc(hidden)]
pub fn subsumed_marginal_data_violations(tdd: &Tdd, vtree: &Vtree) -> Vec<VtreeIdx> {
    let mut bad = Vec::new();
    for i in 0..vtree.num_nodes() {
        if !tdd.levels[i].is_marginal() {
            continue;
        }
        let Some(parent) = vtree.node(VtreeIdx(i as u32)).parent() else {
            continue;
        };
        if !tdd.levels[parent.idx()].is_marginal() {
            continue;
        }
        let lvl = &tdd.levels[i];
        let has_int = lvl.marginal_counts.as_ref().is_some_and(|c| !c.is_empty());
        let has_big = lvl
            .marginal_counts_big
            .as_ref()
            .is_some_and(|b| !b.is_empty());
        // Weight-marginal LEAF exemption (the PIN INVARIANT — see
        // `marginalize_leaf_weighted` and `free_subsumed_marginal_children`): a
        // subsumed leaf KEEPS its 3-slot column, because the column is a
        // compile-global cache of `WeightStore::leaf_val` shared by every `Tdd`,
        // not per-`Tdd` data this scanner is meant to police.
        let has_wt = lvl.is_weight_marginal()
            && lvl.retired_marg_width != 0
            && !vtree.node(VtreeIdx(i as u32)).is_leaf();
        if has_int || has_big || has_wt {
            bad.push(VtreeIdx(i as u32));
        }
    }
    bad
}

/// Ensure counts are available for a given level (compute from pairs if still
/// explicit). The shared [`ensure_fold_walk`] with this context's readers
/// wired into [`IntFold::fold`] via [`compute_marginal_node_int`]. Early-outs
/// (walk guard): memoization cache hit, counts already inlined on the TDD
/// level itself (`is_marginal`), or a leaf (counts come from the formula on
/// demand inside the reader).
///
/// [`ColumnRetention::All`] is mandatory here and takes no caller knob: the
/// sole caller is [`marginalize_batch`], which needs EVERY walked level's
/// column — `cascade_marginalize` `take`s each one to install it as that
/// level's marginal store, and the buffer is shared across all batch targets.
fn ensure_counts(
    tdd: &Tdd,
    level_idx: VtreeIdx,
    vtree: &Vtree,
    computed: &mut [Option<CountVec<RecoveryPanic>>],
) {
    unwrap_infallible(ensure_fold_walk::<IntFold, RecoveryPanic, _, _>(
        level_idx.idx(),
        vtree,
        &tdd.levels,
        computed,
        &Count::Fast(0),
        &|i| tdd.levels[i].is_marginal(),
        &|lvl, i, l_i, r_i, computed| {
            compute_marginal_node_int(tdd, &tdd.levels[lvl], i, l_i, r_i, computed)
        },
        ColumnRetention::All,
    ));
}

/// Resolve one child ref to a count read on a finished `Tdd` — the lazy
/// single-reader replacement for the old `get_marginal_count` (fast value
/// with sentinel) + `get_marginal_count_big` (separate overflow fetch) pair.
/// A `Big` read hands back the borrowed `BigUint` directly.
///
/// Marginal level: self-describing decode under the bit-30-clear==slot
/// polarity. Bit 30 alone disambiguates:
///   bit-30 SET   → inline count value (strip the tag; ≤ 2^30−1, so never an
///                  overflow sentinel).
///   bit-30 CLEAR → bare slot index into `ic` (a pre-tag mid-batch ref is a
///                  bare node index, which IS its slot index).
/// The flag-gated decode this polarity replaced was a miscount waiting to
/// happen: a parent level rebuilt from fresh scratch (e.g. by the
/// clause-specialized apply) can lose its `marg_inlined_*` marker while its
/// pairs still carry bit-30 inline refs.
#[inline]
fn read_marginal_count<'a>(
    tdd: &'a Tdd,
    level_idx: usize,
    node_idx: usize,
    computed: &'a [Option<CountVec<RecoveryPanic>>],
) -> CountRead<'a> {
    if let Some(ic) = &tdd.levels[level_idx].marginal_counts {
        let raw = node_idx as u32;
        if raw & (1 << 31) != 0 {
            return CountRead::Fast(0); // ZERO sentinel — never decode (mirrors emit_or_tag)
        }
        return match MargRef::from_raw(raw) {
            MargRef::Inline(v) => CountRead::Fast(v as u128),
            // A marginal LEAF keeps an empty store under the inline path (all
            // counts live inline at the parent), so a bare slot ref here is a
            // leaf-label index with a fixed count — decode it directly rather
            // than indexing the (empty) store. Reached by paths that leave a
            // leaf-side ref bare (e.g. projection) instead of inlining it.
            MargRef::Slot(s) if tdd.vtree.node(VtreeIdx(level_idx as u32)).is_leaf() => {
                CountRead::Fast(match LeafLabel::from_idx(s as usize) {
                    LeafLabel::Zero => 0,
                    LeafLabel::One => 2,
                    LeafLabel::Pos | LeafLabel::Neg => 1,
                })
            }
            MargRef::Slot(s) => {
                let v = ic[s as usize];
                if v != STREAM_OVERFLOW {
                    return CountRead::Fast(v);
                }
                if let Some(bv) = tdd.levels[level_idx]
                    .marginal_counts_big
                    .as_ref()
                    .and_then(|ib| ib.get(s as usize))
                {
                    return CountRead::Big(bv);
                }
                // Belt-and-braces fallback mirroring the old
                // get_marginal_count_big chain.
                if let Some(bv) = computed[level_idx].as_ref().and_then(|cv| cv.big_val(node_idx)) {
                    return CountRead::Big(bv);
                }
                unreachable!(
                    "big count not available for level {} node {}",
                    level_idx, node_idx
                );
            }
        };
    }
    // Check pre-computed buffer (non-marginal level: plain index).
    if let Some(counts) = &computed[level_idx] {
        return counts.get(node_idx);
    }
    // Leaf level: fixed counts.
    if tdd.vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
        return CountRead::Fast(match LeafLabel::from_idx(node_idx) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        });
    }
    unreachable!("counts not available for level {}", level_idx);
}

/// Fold one marginal node: `Σ over pairs (left_count × right_count)`.
/// The finished-`Tdd` reader adapter over [`IntFold::fold`] — the one shared
/// two-pass integer discipline (u128 fast pass; exact `BigUint` re-pass with
/// mixed-magnitude branching on overflow; exact-max promotion owned by
/// `Count::from_u128`).
fn compute_marginal_node_int(
    tdd: &Tdd,
    level: &crate::tdd::types::TddLevel,
    i: usize,
    li: usize,
    ri: usize,
    computed: &[Option<CountVec<RecoveryPanic>>],
) -> Count {
    IntFold::fold(
        level.pairs_iter_of_idx(i),
        |k| read_marginal_count(tdd, li, k, computed),
        |k| read_marginal_count(tdd, ri, k, computed),
    )
}

/// Weighted analogue of [`read_marginal_count`]: resolve a child node's exact
/// semiring value for the `--weighted` marginalization cascade. Reads, in order:
///   0. **LEAF levels resolve by LABEL**, never through the `WeightStore` column
///      — the weighted mirror of [`read_marginal_count`]'s fixed-count leaf arm.
///      A leaf-side ref is a bare `LeafLabel` index in BOTH representations: a
///      structural leaf's implicit {One, Pos, Neg} nodes, and a weight-marginal
///      leaf's pinned 3-slot column (installed in exactly that order by
///      [`marginalize_leaf_weighted`]). Routing through the column instead would
///      key on the compile-GLOBAL store rather than on THIS `Tdd`'s marginality:
///      a structural leaf level of a fresh clause TDD would then decode its
///      genuine label refs against whatever column the global store happens to
///      hold for that vtree index. Label resolution is correct for both, and is
///      the reading the pin invariant exists to keep exact.
///   1. the external [`WeightStore`] for a level already weight-marginalized
///      (this batch or a prior one) — marg-side refs are bare slots in weighted
///      mode; a `MargRef::Inline` here is a global interned-value ref
///      (`TIDIDI_WEIGHTED_INLINE`), not a per-level slot;
///   2. the per-batch `computed_weights` buffer for a level computed earlier in
///      this batch but not yet stored to the `WeightStore`.
/// Returns `Cow`: store-slot and per-batch reads borrow (no clone); only the
/// ZERO sentinel and leaf bases materialize an owned value.
fn read_marginal_weight<'a>(
    tdd: &Tdd,
    level_idx: usize,
    node_idx: usize,
    ws: &'a WeightStore,
    computed_weights: &'a [Option<Vec<WeightVal>>],
) -> std::borrow::Cow<'a, WeightVal> {
    if let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(VtreeIdx(level_idx as u32)) {
        let raw = node_idx as u32;
        if raw & (1 << 31) != 0 {
            // ZERO sentinel — mirrors read_marginal_count. Leaf levels only ever
            // carry Pos/Neg/One, but the bit is tested before every decode.
            return std::borrow::Cow::Owned(ws.wzero());
        }
        let label_idx = match MargRef::from_raw(raw) {
            // A weighted `Inline` is a GLOBAL interned-value index, never a label
            // — never minted at a leaf, but resolved rather than misread.
            MargRef::Inline(g) => return std::borrow::Cow::Borrowed(ws.interned_value(g)),
            MargRef::Slot(s) => s as usize,
        };
        let v = ws.leaf_val(var, LeafLabel::from_idx(label_idx));
        debug_assert!(
            leaf_column_slot_agrees(ws, level_idx, label_idx, &v),
            "weight-marginal leaf {level_idx}: pinned column disagrees with leaf_val \
             at label slot {label_idx}"
        );
        return std::borrow::Cow::Owned(v);
    }
    // INTERNAL level: per-Tdd flag FIRST, mirroring the leaf arm above and the
    // integer twin `read_marginal_count` (whose store lives inside the level, so
    // it is per-Tdd by construction). The compile-global store can hold a column
    // at this index installed by ANOTHER live Tdd (a sibling accumulator) while
    // THIS Tdd's level is still structural — its node indices are NOT slots of
    // that foreign column. At a leaf the label/slot aliasing makes such a read
    // value-correct anyway (the pin); an internal level has no such backstop, so
    // the store read is gated on this Tdd's own marginality and a structural
    // level falls through to the per-batch computed buffer (the WEIGHTED STORE
    // MIRRORS THE READ Tdd invariant — asserted in `ensure_weights`' walk guard).
    if tdd.levels[level_idx].is_weight_marginal() {
        if let Some(vals) = ws.level(level_idx) {
            let raw = node_idx as u32;
            if raw & (1 << 31) != 0 {
                // ZERO sentinel — mirrors read_marginal_count
                return std::borrow::Cow::Owned(ws.wzero());
            }
            let slot = match MargRef::from_raw(raw) {
                MargRef::Inline(g) => return std::borrow::Cow::Borrowed(ws.interned_value(g)),
                MargRef::Slot(s) => s as usize,
            };
            return std::borrow::Cow::Borrowed(&vals[slot]);
        }
    }
    if let Some(w) = &computed_weights[level_idx] {
        return std::borrow::Cow::Borrowed(&w[node_idx]);
    }
    // No trailing leaf arm: the leaf branch is the FIRST test above, so this point
    // is only reached on an internal level (single source of truth for leaf reads).
    unreachable!("weighted value not available for level {}", level_idx);
}

/// Debug-only companion to the leaf branch of [`read_marginal_weight`]: when the
/// pinned leaf column IS installed, its slot must equal the label's `leaf_val`.
/// A mismatch means some pass compacted, reordered, or appended to the column —
/// exactly what the pin invariant forbids. Absent / short columns are not an
/// error here (a leaf level may simply not be weight-marginal yet).
#[cfg(debug_assertions)]
fn leaf_column_slot_agrees(
    ws: &WeightStore,
    level_idx: usize,
    label_idx: usize,
    expect: &WeightVal,
) -> bool {
    use crate::tdd::query::semiring::weight_key;
    let Some(col) = ws.level(level_idx) else { return true };
    if col.len() != crate::tdd::types::LEAF_WIDTH {
        // Any other length means a pass compacted / erased / appended to the
        // column (the zero-slot subsumed state included — it is no longer
        // written). Report it so the debug build catches the regression.
        return false;
    }
    weight_key(&col[label_idx]) == weight_key(expect)
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn leaf_column_slot_agrees(
    _ws: &WeightStore,
    _level_idx: usize,
    _label_idx: usize,
    _expect: &WeightVal,
) -> bool {
    true
}

/// Weighted analogue of [`compute_marginal_node_int`]: the finished-`Tdd`
/// reader adapter over [`WeightFold::fold`] — one clean pass over the exact
/// semiring (rationals don't overflow).
fn compute_marginal_node_weight(
    tdd: &Tdd,
    level: &crate::tdd::types::TddLevel,
    i: usize,
    li: usize,
    ri: usize,
    ws: &WeightStore,
    computed_weights: &[Option<Vec<WeightVal>>],
) -> WeightVal {
    WeightFold::fold(
        level.pairs_iter_of_idx(i),
        |k| read_marginal_weight(tdd, li, k, ws, computed_weights),
        |k| read_marginal_weight(tdd, ri, k, ws, computed_weights),
        ws.wzero(),
    )
}

/// Weighted analogue of [`ensure_counts`] — the same shared
/// [`ensure_fold_walk`] with [`WeightFold`]. Early-outs (walk guard): cached
/// in buffer, already weight-marginal (`ws.is_set` — this quadrant's
/// marginality predicate), or a leaf (base values come from the semiring on
/// demand in `read_marginal_weight`). The scratch column reserves through
/// `RecoveryPanic` (fallible-allocation parity, A1) — a pathological width
/// raises the controlled recovery-split panic instead of an allocator abort.
///
/// `retain` is the caller's column-lifetime policy, and this is the one ensure
/// wrapper whose callers genuinely differ: [`marginalize_batch_weighted`]
/// needs [`ColumnRetention::All`] (its cascade `take`s every level's column),
/// while [`weighted_output_value`] reads ONLY the walk root and passes
/// [`ColumnRetention::Frontier`].
fn ensure_weights(
    tdd: &Tdd,
    level_idx: VtreeIdx,
    vtree: &Vtree,
    ws: &WeightStore,
    computed_weights: &mut [Option<Vec<WeightVal>>],
    retain: ColumnRetention,
) {
    unwrap_infallible(ensure_fold_walk::<WeightFold, RecoveryPanic, _, _>(
        level_idx.idx(),
        vtree,
        &tdd.levels,
        computed_weights,
        &ws.wzero(),
        &|i| {
            let set = ws.is_set(i);
            // WEIGHTED STORE MIRRORS THE READ Tdd: within this walk's reach, a
            // global column at an INTERNAL index must belong to THIS Tdd — a
            // disagreement means a foreign (sibling-accumulator) column would
            // make the walk skip a structural level and the parent fold then
            // decode this Tdd's node indices as foreign slots. Leaves are
            // exempt: the pinned label-aliased column is legitimately global
            // while another Tdd still reads the leaf structurally (and the
            // guard's value is inconsequential there — leaf bases resolve by
            // label in `read_marginal_weight` either way).
            debug_assert!(
                vtree.node(VtreeIdx(i as u32)).is_leaf() || set == tdd.levels[i].is_weight_marginal(),
                "ensure_weights walk at vtree index {i}: global WeightStore column \
                 and this Tdd's marginality disagree — the column belongs to \
                 another live Tdd (WEIGHTED STORE MIRRORS THE READ Tdd)"
            );
            set
        },
        &|lvl, i, l_i, r_i, cw| {
            compute_marginal_node_weight(tdd, &tdd.levels[lvl], i, l_i, r_i, ws, cw)
        },
        retain,
    ));
}

/// Weighted analogue of [`cascade_marginalize`]: walk down from a level whose
/// parent is being marginalized, converting still-explicit non-leaf descendants
/// to weight-marginal mode using values cached by `ensure_weights`. Unlike the
/// integer path there is NO dedup / parent-ref remap / inline tagging — the
/// weighted store is full-width and marg-side refs stay bare node-index slots
/// (slot index == node index), read directly by `read_marginal_weight`.
fn cascade_marginalize_weighted(
    tdd: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    ws: &mut WeightStore,
    computed_weights: &mut [Option<Vec<WeightVal>>],
) {
    let ti = t.idx();
    if vtree.node(VtreeIdx(ti as u32)).is_leaf() || tdd.levels[ti].is_marginal() {
        return;
    }
    let (l_child, r_child) = vtree.children(t);
    cascade_marginalize_weighted(tdd, vtree, l_child, ws, computed_weights);
    cascade_marginalize_weighted(tdd, vtree, r_child, ws, computed_weights);

    let Some(weights) = computed_weights[ti].take() else {
        // No cached values: cells structurally unreachable from the target's
        // pair lists, never queried (mirrors cascade_marginalize).
        return;
    };
    assert_can_make_marginal(&tdd.levels, vtree, t);
    if let Some(parent_vi) = vtree.node(t).parent() {
        tdd.mark_contract_dirty(parent_vi);
    }
    tdd.levels[ti].make_marginal_weighted();
    ws.set_level(ti, weights);
    // `t` now subsumes its children — free their now-dead weighted stores (O(1)).
    free_subsumed_marginal_children(tdd, vtree, t, Some(ws));
}

/// Read the overall weighted value at the output node after a `--weighted`
/// marginalizing compile — the weighted analogue of `model_count_hybrid`'s root
/// read. The output level is often left EXPLICIT by the compile (only its
/// descendants are marginalized), so this folds it (and any still-explicit
/// descendants) from the `WeightStore` / leaf bases on demand; if the output
/// level is itself weight-marginal, it reads the stored value directly.
///
/// # Panics
///
/// Panics if the output level is weight-marginal but its stored value is
/// absent from `ws`.
#[doc(hidden)]
pub fn weighted_output_value(tdd: &Tdd, vtree: &Vtree, ws: &WeightStore) -> WeightVal {
    // UNSAT / constant-false output: the ZERO sentinel carries no level slot
    // (`output.local` is the ZERO idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return ws.wzero();
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return ws.level(out_t).expect("output level weight-marginalized")[out_i].clone();
    }
    // All-backbone / single-residual-var output: when preprocessing forces every
    // variable, the driver promotes one var to live and the compile collapses the
    // output to a LEAF level. `ensure_weights` early-returns on leaf levels (their
    // bases come from the semiring on demand), so `computed[out_t]` would stay
    // `None` and the unwrap below would panic. Fold the leaf base directly —
    // mirrors `read_marginal_weight`'s leaf branch and `model_count_hybrid`'s
    // leaf-seeding on the integer path. (One = w_pos+w_neg, Pos = w_pos, Neg = w_neg.)
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return ws.leaf_val(var, LeafLabel::from_idx(out_i));
    }
    let mut computed: Vec<Option<Vec<WeightVal>>> = vec![None; vtree.num_nodes()];
    // Root-only read: the single value below is the ONLY thing taken from
    // `computed`, so the walk releases each child column as its parent's
    // completes ([`ColumnRetention::Frontier`]) — peak is the walk frontier,
    // not one `Vec<WeightVal>` per level of the whole diagram. `out_t` is the
    // walk root, so its column is the one the walk never frees.
    ensure_weights(tdd, tdd.output.vtree, vtree, ws, &mut computed, ColumnRetention::Frontier);
    computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone()
}

// ── Thread-local weighted-marginalization context ─────────────────────────────
//
// A `--weighted` compile reuses the integer `--mc` marginalizing compile core
// verbatim; the only difference is that marginal levels carry exact `BigRational`
// semiring values (in a `WeightStore`) instead of integer counts. Threading a
// `WeightStore` through every compile-core signature would perturb the hot integer
// path's codegen, so instead the store lives in this thread-local, installed by
// the weighted entry point for the duration of one compile. The compile core is
// single-threaded per CNF (no rayon/spawn in `compile_cnf_inner`), so a
// thread-local is safe.
//
// When no context is installed (`weight_ctx_active()` is false) every weighted
// branch below is skipped and the integer path runs byte-identically.

thread_local! {
    static WEIGHT_CTX: std::cell::RefCell<Option<WeightStore>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a fresh weighted-marginalization context on this thread for a
/// weighted compile. `num_levels` should be `vtree.num_nodes()`; the store
/// auto-grows if v-split later adds vtree nodes.
#[doc(hidden)]
pub fn init_weight_ctx(
    semiring: crate::tdd::query::semiring::RationalSemiring,
    num_levels: usize,
    precision: crate::tdd::weight_store::Precision,
) {
    WEIGHT_CTX.with(|c| *c.borrow_mut() = Some(WeightStore::new(num_levels, semiring, precision)));
}

/// True while a weighted compile is in progress on this thread. Every
/// integer-count-reading marginal-side op (`p_fusion`, slot prune, count-table
/// invariant probes) is gated off on this so it never reads a `MARG_WEIGHTED`
/// level's `None` counts.
#[inline]
#[doc(hidden)]
pub fn weight_ctx_active() -> bool {
    WEIGHT_CTX.with(|c| c.borrow().is_some())
}

/// SINGLE SOURCE OF TRUTH for "may weighted (P) pair fusion run right now".
/// Every consumer — the `p_fusion` gate and the post-marginalize sweep in the
/// compile driver — asks this and nothing else.
///
/// True iff ALL of:
///   * a weighted compile is in progress on this thread (`weight_ctx_active`);
///   * the store's value domain is EXACT (`BigRational`). Fusion sums a group's
///     values, and the identity it relies on — `Σᵢ W(x)·W(mᵢ) = W(x)·Σᵢ W(mᵢ)` —
///     needs exact distributivity in ℚ. It holds for SIGNED measures (slot
///     reprs are pairwise disjoint, so finite additivity applies regardless of
///     sign), but NOT in the bounded-precision `WeightVal::Log` domain, where
///     repeated signed `add_assign` is order-dependent and cancellation-prone.
///     Log domain therefore stays on the old skip-entirely behavior.
///
/// Default is therefore ON for Exact-domain weighted compiles — integer-mode
/// parity; a Log-domain store is excluded by the domain test.
#[inline]
#[doc(hidden)]
pub fn weighted_fusion_active() -> bool {
    WEIGHT_CTX.with(|c| match c.borrow().as_ref() {
        Some(ws) => !ws.is_log(),
        None => false,
    })
}

/// Remove and return the weighted context after a compile (the caller reads the
/// output node's value from the returned store). Clears the thread-local.
#[doc(hidden)]
pub fn take_weight_ctx() -> Option<WeightStore> {
    WEIGHT_CTX.with(|c| c.borrow_mut().take())
}

/// Run `f` with shared access to the installed weighted store. Panics if no
/// weighted context is active (callers gate on `weight_ctx_active()`).
pub(crate) fn with_weight_ctx<R>(f: impl FnOnce(&WeightStore) -> R) -> R {
    WEIGHT_CTX.with(|c| {
        let g = c.borrow();
        f(g.as_ref().expect("weight ctx active"))
    })
}

/// Run `f` with mutable access to the installed weighted store. Panics if no
/// weighted context is active (callers gate on `weight_ctx_active()`).
pub(crate) fn with_weight_ctx_mut<R>(f: impl FnOnce(&mut WeightStore) -> R) -> R {
    WEIGHT_CTX.with(|c| {
        let mut g = c.borrow_mut();
        f(g.as_mut().expect("weight ctx active"))
    })
}

/// Clone the semiring of the installed weighted context, if any. Used by the
/// multi-component graft path to derive per-component (locally-reindexed)
/// semirings and to build the merged store's global semiring.
#[doc(hidden)]
pub fn weight_ctx_semiring_clone() -> Option<crate::tdd::query::semiring::RationalSemiring> {
    WEIGHT_CTX.with(|c| c.borrow().as_ref().map(|ws| ws.semiring.clone()))
}

/// Install a prebuilt weighted store as this thread's context (replacing any
/// current one). Used to install the merged multi-component store after graft.
#[doc(hidden)]
pub fn set_weight_ctx(ws: WeightStore) {
    WEIGHT_CTX.with(|c| *c.borrow_mut() = Some(ws));
}

/// Weighted-mode marginalize entry used at the integer `marginalize_batch` call
/// sites: if a weight ctx is installed, marginalize `targets` into it (skipping
/// the integer count machinery AND the caller's count-reading p-fusion/slot-prune
/// follow-ups) and return `true`; otherwise return `false` so the caller runs the
/// integer path unchanged.
///
/// # Panics
///
/// Panics if a weight context is detected active but its store is unexpectedly
/// absent (an internal invariant violation).
#[doc(hidden)]
pub fn marginalize_batch_weighted_if_active(
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
) -> bool {
    if !weight_ctx_active() {
        return false;
    }
    WEIGHT_CTX.with(|c| {
        let mut guard = c.borrow_mut();
        let ws = guard.as_mut().expect("weight ctx active");
        marginalize_batch_weighted(tdd, targets, vtree, ws);
    });
    true
}

/// Weighted analogue of [`marginalize_batch`]: marginalize the scheduled
/// `targets` carrying exact semiring values into `ws` instead of integer counts.
/// Faithful mirror of the integer driver's target loop minus the overflow /
/// dedup / parent-ref remap / inline-tag machinery (none needed for the
/// full-width weighted store). Used by the `--weighted` marginalizing compile.
///
/// # Panics
///
/// Panics if the internal per-target weight computation is inconsistent (a
/// just-computed weight slot is unexpectedly empty).
pub fn marginalize_batch_weighted(
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
    ws: &mut WeightStore,
) {
    if targets.is_empty() {
        return;
    }
    let mut computed_weights: Vec<Option<Vec<WeightVal>>> = vec![None; vtree.num_nodes()];
    for &d in targets {
        let di = d.idx();
        if tdd.levels[di].is_marginal() || tdd.levels[di].width() == 0 {
            continue;
        }
        if vtree.node(VtreeIdx(di as u32)).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(d);
        let li = left.idx();
        let ri = right.idx();
        let width = tdd.levels[di].width();
        // `ColumnRetention::All`: `cascade_marginalize_weighted` below `take`s
        // the column of EVERY level in both walked subtrees to install it as
        // that level's weighted store — frontier release would free exactly
        // those. (The buffer is also shared across batch targets.)
        ensure_weights(tdd, left, vtree, ws, &mut computed_weights, ColumnRetention::All);
        ensure_weights(tdd, right, vtree, ws, &mut computed_weights, ColumnRetention::All);
        let mut weights = vec![ws.wzero(); width];
        for (i, _pairs) in tdd.levels[di].internal_inputs_iter() {
            weights[i] =
                compute_marginal_node_weight(tdd, &tdd.levels[di], i, li, ri, ws, &computed_weights);
        }
        computed_weights[di] = Some(weights);
        let (l_child, r_child) = vtree.children(d);
        cascade_marginalize_weighted(tdd, vtree, l_child, ws, &mut computed_weights);
        cascade_marginalize_weighted(tdd, vtree, r_child, ws, &mut computed_weights);
        assert_can_make_marginal(&tdd.levels, vtree, d);
        let vals = computed_weights[di].take().expect("weights just computed for di");
        tdd.levels[di].make_marginal_weighted();
        ws.set_level(di, vals);
        // `d` now subsumes its children — free their now-dead weighted stores.
        free_subsumed_marginal_children(tdd, vtree, d, Some(ws));
    }

    // Leaf (single-variable) marginalization — the weighted mirror of the integer
    // batch's closing leaf pass. The loop above skips leaf targets (a leaf has no
    // pairs or store of its own), so sum them out here.
    //
    // ORDERING (why the END, like the integer arm): the integer pass MUST run last
    // because `tag_all_marg_side_slots` keys off the batch-entry `was_marginal`
    // snapshot and would re-resolve a prematurely-marginal leaf's side as bare
    // slots. The weighted batch has no tagger and no such snapshot at all —
    // marg-side refs stay bare node-index slots end to end — so that hazard does
    // not exist here and both orders are exact (the column installed below is
    // value-for-value what `read_marginal_weight`'s leaf branch computes). Running
    // last is still the right order: a leaf marginalized before its internal parent
    // in the same batch would have its column installed and then immediately freed
    // again by the parent's `free_subsumed_marginal_children` — same end state,
    // strictly more work.
    for &d in targets {
        if vtree.node(d).is_leaf() {
            marginalize_leaf_weighted(tdd, d, vtree, ws);
        }
    }
}

/// Weighted analogue of [`marginalize_leaf_inline`]: sum out a single-variable
/// vtree LEAF carrying exact semiring values.
///
/// The representation deliberately differs from the integer arm. The integer path
/// rewrites the parent's leaf-side refs into self-describing `MargRef::Inline`
/// counts (One→2, Pos/Neg→1) and leaves the leaf store empty; under weights that
/// is unavailable, because a weighted `MargRef::Inline(gidx)` indexes the
/// `WeightStore`'s GLOBAL intern table, which is rebuilt (and its indices
/// invalidated) at every component graft — an interned ref persisted in a pair
/// list would dangle (same reason `dup_resolve::scale_weight_ref` refuses to mint
/// one).
///
/// Instead we install a real 3-slot weighted store on the leaf level, in
/// [`LeafLabel::from_idx`] order (0 = One, 1 = Pos, 2 = Neg). A parent's leaf-side
/// refs are already bare leaf-LABEL indices, and a bare marg-side ref IS its slot
/// index, so they decode as the correct `MargRef::Slot` with NO parent-ref rewrite.
/// The values come from [`WeightStore::leaf_val`] — the one place every weighted
/// leaf read resolves its bases (One = w⁺+w⁻, Pos = w⁺, Neg = w⁻) — so a parent
/// marginalized later reads exactly what it would have read with the leaf still
/// structural. A Zero leaf-side ref never reaches the slot decode: Zero is a
/// sentinel with bit 31 set (`Tdd::is_zero`; leaf levels only ever carry
/// Pos/Neg/One), and every weighted reader tests that bit before decoding.
///
/// The integer arm's Pos/Neg→`Inline(1)` twin-merge bonus IS attempted here — but
/// only where it is a *value-preserving* rewrite. The integer arm may merge Pos
/// and Neg unconditionally because both leaf counts are 1; under weights the two
/// slots may hold different numbers, so the merge is licensed exactly when they
/// hold the SAME number. [`leaf_canon_map`] computes that equal-value partition of
/// the pinned column and [`canonicalize_leaf_refs_at_parent`] moves each leaf-side
/// ref onto its class's canonical (smallest) slot — Neg→Pos when w⁺ = w⁻ (the
/// common case, and the one that restores the twin cascade), Pos→One when w⁻ = 0,
/// Neg→One when w⁺ = 0, nothing at all when the three values are distinct. Only
/// refs move; the column is untouched, so the pin below still holds. The walk is
/// restricted to the exact-rational domain, where `weight_key`
/// equality IS value equality.
/// `mark_contract_dirty` is seeded for a STRUCTURAL parent, so contraction gets to
/// act on the new marginal boundary — and after canonicalization it has real work:
/// the parent's `(·, Pos)` / `(·, Neg)` branches are now byte-identical twins.
/// (Weighted p-fusion DOES run at leaf boundaries, but folds by SUM-LOOKUP only:
/// a redex group whose summed value already sits in the pinned column collapses
/// to one pair naming that slot — `(·,Pos) + (·,Neg) = w⁺+w⁻ = the One slot`, by
/// definition and for every weight table — and a group whose sum is not in the
/// column is left exactly as it was. Minting a 4th slot at a leaf stays
/// forbidden; see `p_fusion::resolve_leaf_fusion_refs_by_lookup`.)
///
/// # THE PIN INVARIANT
///
/// **A weight-marginal LEAF level's column is an immutable, label-ordered,
/// exactly-`LEAF_WIDTH` cache of [`WeightStore::leaf_val`]. No pass may compact,
/// erase, reorder, or append to it, ever.** The column is compile-GLOBAL (one
/// `WeightStore`, keyed by vtree index) while a parent-ref rewrite can only reach
/// ONE `Tdd`, so any mutation desynchronises every other holder — including fresh
/// clause TDDs whose leaf level is still structural and hold genuine leaf-LABEL
/// refs. Enforced at:
///   * `minimize::slot_prune::prune_marg_slots_generic` — both walks skip
///     weight-marginal leaves (no compaction, no dead-store clear);
///   * `minimize::contract::dup_resolve::try_scale_child` — a C2 twin-fold into a
///     weight-marginal leaf LOOKS the scaled value up among the column's own
///     three slots and takes that slot if it is there, declining otherwise. It
///     never mints, and never writes the column;
///   * `minimize::contract::p_fusion::resolve_leaf_fusion_refs_by_lookup` — the
///     weighted arm folds a LEAF boundary by SUM-LOOKUP only: the (P) group's
///     summed value is folded onto the column slot that already holds it (found
///     via [`find_leaf_slot_by_value`], so the ref is canonical), and the plan is
///     DROPPED when no slot holds it. It never mints, never writes the column,
///     and never bumps the level's width;
///   * [`free_subsumed_marginal_children`] / [`subsumed_marginal_data_violations`]
///     — leaves are exempt from the subsumed-data reclaim and its scanner;
///   * [`read_marginal_weight`] — leaf refs resolve by LABEL, never through the
///     column;
///   * `conjoin`'s leaf-marg propagation — flags the output level `LEAF_WIDTH`
///     directly rather than reading the column's length;
///   * [`canonicalize_leaf_refs_at_parent`] — the equal-value ref rewrite (this
///     function, and conjoin's leaf-marg propagation) moves REFS of one `Tdd`
///     between slots that already hold the same value; it reads the column and
///     writes nothing to it.
/// Checked centrally by [`debug_check_leaf_columns_pinned`] at slot-prune entry.
///
/// WHERE THE EXACT REGIME LIVES. Weighted p-fusion — the growth-direction
/// breaker — is inactive whenever the store is in the bounded LOG domain
/// (`weighted_fusion_active` requires the exact domain), so a leaf mint is
/// reachable only from an exact-domain weighted compile. Do not read "the log
/// domain is fine" as "the bug is unreachable" — exact-domain compiles are
/// production.
#[doc(hidden)]
pub fn marginalize_leaf_weighted(
    tdd: &mut Tdd,
    leaf: VtreeIdx,
    vtree: &Vtree,
    ws: &mut WeightStore,
) {
    debug_assert!(vtree.node(leaf).is_leaf());
    let li = leaf.idx();
    if tdd.levels[li].is_marginal() {
        return;
    }
    // Same projection opt-out as `marginalize_leaf_inline`: a projected (PMC /
    // ∃-quantified) compile cofactors leaves via `condition_leaf` inside
    // `project_var`, and `assert_conditionable` fail-fasts on a marginal leaf
    // level. Keep leaves structural whenever a caller has installed a projected
    // set — the parent's ordinary internal marginalize still sums the leaf via its
    // semiring bases, exactly as before leaf-marg.
    if crate::tdd::transform::unary::project::caller_projection_active() {
        return;
    }
    let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(li as u32)) else { return };
    let parent = vtree.node(leaf).parent();
    // Slot i ≡ `LeafLabel::from_idx(i)`, which is what makes the parent's existing
    // bare leaf-label refs valid slot refs without a rewrite. If that order ever
    // changes, every parent ref into a weight-marginal leaf silently reads the
    // wrong base.
    debug_assert!(matches!(
        (LeafLabel::from_idx(0), LeafLabel::from_idx(1), LeafLabel::from_idx(2)),
        (LeafLabel::One, LeafLabel::Pos, LeafLabel::Neg)
    ));
    let vals: Vec<WeightVal> = leaf_column_vals(ws, var);
    // SUBSUMED LEAF (parent already marginal) gets the SAME full column — the pin
    // invariant admits no second leaf state. An earlier revision installed ZERO
    // slots here, on the theory that the parent's aggregate already folded the
    // bases in so a column would be dead per-`Tdd` data. It is not per-`Tdd` data:
    // the column is compile-GLOBAL (keyed by vtree index, one `WeightStore` for the
    // whole compile), and every OTHER holder of this leaf — a fresh clause TDD
    // whose leaf level is still structural, a sibling partial product — decodes its
    // bare leaf-LABEL refs against it. A zero-slot column made those reads panic on
    // `&vals[slot]` or, through the `map_or(0, len)` width readers, silently drop
    // the leaf's whole mass. Three cached constants cost nothing to keep.
    // The only thing subsumption still changes is contract seeding: a marginal
    // parent is not a fusion boundary, so it is not marked dirty.
    if let Some(parent_vi) = parent {
        if !tdd.levels[parent_vi.idx()].is_marginal() {
            // EQUAL-VALUE REF CANONICALIZATION — the weighted form of the integer
            // arm's Pos/Neg → `Inline(1)` twin bonus (`marginalize_leaf_inline`).
            // Same guard as there: a MARGINAL parent has already folded this leaf's
            // bases into its own aggregate, so there are no leaf-side pairs left to
            // rewrite. Exact domain only — `leaf_canon_map`'s `weight_key` equality
            // is value equality there, whereas a `WeightKey::Log` compares `f64`
            // bit patterns and would merge refs on a rounding coincidence.
            if !ws.is_log() {
                let canon = leaf_canon_map(&vals);
                if canon != [0, 1, 2] {
                    let (pl, _) = vtree.children(parent_vi);
                    canonicalize_leaf_refs_at_parent(
                        &mut tdd.levels[parent_vi.idx()],
                        pl == leaf,
                        &canon,
                    );
                }
            }
            tdd.mark_contract_dirty(parent_vi);
        }
    }
    tdd.levels[li].make_marginal_weighted_with_slots(vals.len() as u32);
    ws.set_level(li, vals);
}

/// PIN-INVARIANT CHECK (debug builds only): every weight-marginal vtree LEAF must
/// advertise exactly `LEAF_WIDTH` slots, and — when its compile-global column is
/// installed — that column must equal the `leaf_val` triple in `LeafLabel` order.
///
/// This is the one invariant that makes bare leaf-LABEL refs and `MargRef::Slot`
/// refs interchangeable at a leaf, which is what lets `marginalize_leaf_weighted`
/// flip a leaf marginal without rewriting a single parent ref. Every pass that
/// could break it (slot-prune compaction, dup-resolve twin-fold minting, weighted
/// p-fusion allocation, subsumption reclaim) declines to touch leaves; this check
/// is placed on a hot, frequently-run path (slot-prune entry) so a regression in
/// any of them surfaces immediately instead of as a silently low weighted count.
///
/// Four things are checked, in the order a breakage shows up:
///   1. the level advertises `LEAF_WIDTH` slots (catches a `retired_marg_width`
///      bump — how weighted p-fusion's `allocate_fusion_slots_weighted` records a
///      minted slot);
///   2. no parent ref into the leaf names a slot ≥ `LEAF_WIDTH` (catches a minted
///      REF that outlived the width, and is the check that fails closest to the
///      real damage: a `Slot(3)` ref is decoded by every label-first reader —
///      `query::count`, `query::sat`, `validate`, `dup_resolve` — as
///      `LeafLabel::from_idx(3)`, the never-satisfied ZERO sentinel, so the
///      models under it vanish with no error anywhere, and `prune_unreachable`
///      indexes the NEIGHBOURING level's remap window with it);
///   3. the installed column equals the `leaf_val` triple in label order
///      (catches compaction / erasure / reordering);
///   4. every bare leaf-side ref is the CANONICAL slot of its value class
///      ([`leaf_canon_map`]) — catches a site that CREATES a leaf-side ref and
///      skips the canon pass. That is a size regression rather than a wrong
///      count (a non-canonical ref still resolves to the right value), so it has
///      no other symptom: without this check the twin cascade would just quietly
///      stop firing on the affected leaves. Exact domain only, and only once the
///      compile-global column is installed — the canon partition is undefined
///      otherwise.
///
/// No-op in release and whenever no weight context is installed.
#[cfg(debug_assertions)]
pub(crate) fn debug_check_leaf_columns_pinned(tdd: &Tdd) {
    use crate::tdd::query::semiring::weight_key;
    use crate::tdd::marg_slots::ChildSide;
    use crate::tdd::marg_slots::{referenced_marg_slots, RefSlotScratch};
    if !weight_ctx_active() {
        return;
    }
    let mut scratch = RefSlotScratch::default();
    with_weight_ctx(|ws| {
        for i in 0..tdd.levels.len() {
            let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(VtreeIdx(i as u32)) else { continue };
            if !tdd.levels[i].is_weight_marginal() {
                continue;
            }
            debug_assert_eq!(
                tdd.levels[i].width(),
                crate::tdd::types::LEAF_WIDTH,
                "pin invariant: weight-marginal leaf level {i} must advertise \
                 LEAF_WIDTH slots"
            );
            // No parent ref may name a slot outside the pinned label range.
            if let Some(parent) = tdd.vtree.node(VtreeIdx(i as u32)).parent() {
                let side = match tdd.vtree.node(parent) {
                    VtreeNode::Internal { left, .. } if left.idx() == i => ChildSide::Left,
                    _ => ChildSide::Right,
                };
                let refs = referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut scratch);
                debug_assert!(
                    refs.last().is_none_or(|&s| (s as usize) < crate::tdd::types::LEAF_WIDTH),
                    "pin invariant: weight-marginal leaf level {i} is referenced at slot \
                     {:?} — outside the label range, so every label-first reader decodes \
                     it as the ZERO sentinel and drops that branch's mass",
                    refs.last()
                );
                // #4 — canonicality. Every surviving ref must already name the
                // smallest slot of its value class; anything else means some site
                // minted a leaf-side ref without running
                // `canonicalize_leaf_refs_at_parent`.
                if !ws.is_log() {
                    if let Some(col) = ws.level(i) {
                        let canon = leaf_canon_map(col);
                        for &s in refs {
                            // Out-of-range refs are check #2's report, not ours.
                            debug_assert!(
                                (s as usize) >= crate::tdd::types::LEAF_WIDTH
                                    || canon[s as usize] == s,
                                "pin invariant: weight-marginal leaf level {i} is referenced \
                                 at NON-CANONICAL slot {s} (canonical slot for that value is \
                                 {}) — a leaf-side ref was created without the equal-value \
                                 canon pass, so the twin cascade cannot fire there",
                                canon[s as usize]
                            );
                        }
                    }
                }
            }
            let Some(col) = ws.level(i) else { continue };
            debug_assert_eq!(
                col.len(),
                crate::tdd::types::LEAF_WIDTH,
                "pin invariant: weight-marginal leaf level {i} column was \
                 compacted/erased/appended to"
            );
            for k in 0..col.len() {
                debug_assert!(
                    weight_key(&col[k]) == weight_key(&ws.leaf_val(var, LeafLabel::from_idx(k))),
                    "pin invariant: weight-marginal leaf level {i} column slot {k} \
                     is not the label-ordered leaf_val cache"
                );
            }
        }
    });
}

#[cfg(not(debug_assertions))]
#[inline(always)]
pub(crate) fn debug_check_leaf_columns_pinned(_tdd: &Tdd) {}

// ── Born-C3 marginalize helpers (dedup_fresh_store + parent-ref remap) ───────
//
// C3 — no two slots at a marginal level share a model count — is established
// **at birth** for compile_marginalize-path stores (cascade_marginalize,
// marginalize_batch, above) by these two helpers: `dedup_fresh_store` merges
// duplicate-count slots before the store is installed, and
// `remap_parent_refs_pretag` redirects the parent level's marg-side refs onto
// the surviving canonical slots. (The apply streaming-emit path establishes C3
// later, at post-tagger slot-prune in `minimize/slot_prune.rs`.)

/// Compact a freshly-built marginal store so that **each count value occupies
/// at most one slot** (invariant C3), returning the deduped store and a
/// slot-index remap table: `remap[old] = new` (identity where no dedup occurred,
/// canonical-slot index otherwise).
///
/// The caller must remap every parent-side ref that indexes into the old store
/// using `remap[old_slot]`.  Refs are NOT remapped here — this function only
/// touches the store itself.
///
/// # Store is born C3 (for compile_marginalize-path callers): no duplicate
/// count values; enforced here. Apply-emit-born stores do NOT call this at
/// emit time — their C3 is established later by `prune_marg_slots`.
///
/// Duplicate slots are merged to the FIRST occurrence of each value. The returned vecs may
/// be shorter than the inputs when duplicates were found; if no duplicates
/// exist they are returned unchanged.
///
/// The fast count column is compacted **in place**: callers hand it over by
/// move (see `marginalize_batch` / `cascade_marginalize`, which `take` the
/// level's `CountVec` and pass `into_parts()`), so no second full-length store
/// is ever resident beside this one at the peak. The sparse overflow table is
/// rekeyed into a fresh [`BigSide`] instead — its keys are slot indices, and a
/// survivor's index changes — which costs at most the surviving overflow
/// entries, never a width-sized buffer. Same mechanism and same soundness
/// argument as `IntFold::compact_store` (`minimize/slot_prune.rs`), which
/// compacts an already-installed store; both take their value-dedup key from
/// the shared `count_key_at`, so the Small/Big split is decided in one place.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    big: Option<BigSide>,
) -> (Vec<u128>, Option<BigSide>, Vec<u32>) {
    let n = counts.len();
    // Written on every path below (mint or merge), for every `i`.
    let mut remap: Vec<u32> = vec![0; n];
    let mut count_to_canonical: FxHashMap<CountKey, u32> = FxHashMap::default();
    let mut new_len = 0usize;

    // SOUNDNESS (why a move can't clobber a slot still to be read): dedup never
    // grows the store — distinct values ≤ slots — so the write cursor `new_len`
    // is at or behind the read cursor `i` at every step (`new_len` advances at
    // most once per `i`). The key at `i` is read BEFORE the move, and every
    // later read is at a strictly larger index than any write done so far.
    //
    // `count_to_canonical` maps a value to the COMPACTED index of its first
    // slot, so the remap is final as it is written — no second composition pass
    // over a `compact_idx` table, and no reading of the destroyed layout.
    //
    // The overflow table is NOT touched in here: it is keyed by slot, so it is
    // rekeyed in one drain after `remap` is complete (below).
    for i in 0..n {
        let key = count_key_at(&counts, big.as_ref(), i);
        // A hit means `i` holds a value an earlier surviving slot already
        // carries (C3 merge); a miss mints the next compacted slot.
        if let Some(&hit) = count_to_canonical.get(&key) {
            remap[i] = hit;
            continue;
        }
        count_to_canonical.insert(key, new_len as u32);
        counts[new_len] = counts[i];
        remap[i] = new_len as u32;
        new_len += 1;
    }

    if new_len == n {
        // No duplicates: every slot minted its own, so the store is already C3
        // and `remap` is the identity — hand both back untouched, overflow
        // table included (rekeying it would be the identity too).
        return (counts, big, remap);
    }

    // Rekey the overflow table: consume it in one ascending drain and re-file
    // each value under its slot's compacted index. A slot that merged away maps
    // onto its canonical's index and writes an EQUAL value over it (equality is
    // what made them merge), so the result is the same either way — and a
    // merged-away `BigUint` is dropped as the drain passes it. Values move;
    // nothing here clones.
    let new_big = big.map(|b| {
        b.into_iter().map(|(slot, v)| (remap[slot as usize], v)).collect::<BigSide>()
    });

    counts.truncate(new_len);
    // Slack ceiling: the fast column is compacted IN PLACE, so the capacity
    // observed here is the PRE-compaction one. Shrinking at 2× therefore
    // reclaims exactly when the store more than halved — the effective ceiling
    // on the slack this level's store keeps for its lifetime, matching
    // `IntFold::compact_store`. The rebuilt overflow table needs no such policy:
    // its slack is bounded by the surviving overflow set, not by the width.
    if counts.capacity() > 64 && counts.capacity() > 2 * counts.len() {
        counts.shrink_to_fit();
    }

    (counts, new_big, remap)
}

/// Remap parent-level marg-side refs into a child level using a slot remap
/// table built by [`dedup_fresh_store`].
///
/// At the **pre-tagger** construction sites (`marginalize_batch`, `cascade_marginalize`)
/// every parent ref into the child is a bare slot index (bit-30 clear, never an
/// inline count). `remap[old_slot] = new_slot` was returned by `dedup_fresh_store`.
///
/// Also redirects `tdd.output.local` when the TDD root lives at the marginal
/// level (rare but defensive).
///
/// # Store is born C3: no duplicate count values; enforced here, not by a
/// later canon pass.
pub(crate) fn remap_parent_refs_pretag(
    tdd: &mut Tdd,
    child_v: VtreeIdx,
    parent_v: VtreeIdx,
    t1_is_left: bool,
    remap: &[u32],
) {
    use crate::tdd::types::{LocalNodeIdx, MARG_VALUE_MASK, decode_marg_coord};

    if remap.iter().enumerate().all(|(i, &r)| r == i as u32) {
        // Identity remap — nothing to do.
        return;
    }

    // Bare slot remap: bit-30 clear = bare slot; mask strips high bits.
    let remap_ref = |raw: u32| -> u32 {
        if raw & (1 << 31) != 0 {
            return raw; // ZERO sentinel
        }
        // Pre-tagger: no inline refs exist yet; all marg-side refs are bare slots.
        MargRef::slot_raw(remap[decode_marg_coord(raw, MARG_VALUE_MASK) as usize])
    };

    let plevel = &mut tdd.levels[parent_v.idx()];
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if t1_is_left {
                node.a = remap_ref(node.a);
            } else {
                node.b = remap_ref(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                let f = if t1_is_left { &mut p.left } else { &mut p.right };
                *f = LocalNodeIdx(remap_ref(f.idx() as u32));
            }
        }
    }

    // Output update when TDD root is at this marginal level (rare but defensive).
    if tdd.output.vtree == child_v {
        let old = tdd.output.local.idx() as u32;
        tdd.output.local = LocalNodeIdx(remap[old as usize]);
    }
}

#[cfg(test)]
#[path = "marginalize_marginal_alloc_guard_tests.rs"]
mod marginal_alloc_guard_tests;

#[cfg(test)]
#[path = "marginalize_deadline_tests.rs"]
mod marginalize_deadline_tests;
