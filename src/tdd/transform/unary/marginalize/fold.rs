//! The bottom-up count/weight fold that freezes scheduled levels.

use crate::tdd::counts::{ColumnRetention, CountVec, RecoveryPanic};
use crate::tdd::limits::{reduce_poll_stride, ApplyError, PollTicker};
use crate::tdd::query::WeightVal;
use crate::tdd::types::{assert_can_make_marginal, Tdd};
use crate::tdd::weight_store::WeightStore;
use crate::vtree::{Vtree, VtreeIdx};

use super::leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};
use super::store::{
    compute_marginal_node_int, compute_marginal_node_weight, dedup_fresh_store, ensure_counts,
    ensure_weights, free_subsumed_marginal_children, remap_parent_refs_pretag,
};

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
pub(crate) fn marginalize_batch(
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), ApplyError> {
    if targets.is_empty() {
        return Ok(());
    }
    // Invariant: the integer batch never runs during a weighted compile — its
    // `read_marginal_count` reads integer `marginal_counts`, which a weight-marginal
    // level (MARG_WEIGHTED, values in the external WeightStore) does not have. Every
    // batch site must route through `marginalize_batch_weighted_if_active` first.
    // Asserted here so a future bypass fails loudly at the entry point instead of via
    // the cryptic `unreachable!()` deep in `read_marginal_count`.
    debug_assert!(
        tdd.weights.is_none(),
        "the integer batch cannot run on a diagram carrying a weight store"
    );
    // Snapshot which levels are ALREADY marginal at batch entry. The end-sweep
    // tagger (`tag_all_marg_side_slots`, below) must resolve the bare-coord refs
    // into children that became marginal *in this batch* but must NOT touch
    // children marginalized by a prior batch/apply — those already carry inline
    // counts (emitted by the prior batch's end-sweep). The per-level
    // `marg_inlined_left/right` marker tries to encode this but is clobbered
    // when levels merge/rebuild; this snapshot is the reliable discriminator
    // and cannot be lost (it's not stored on the level).
    // The `Option` is the shape `tag_all_marg_side_slots` takes — `None` there
    // means "no snapshot, treat every marginal level as new". This batch always
    // has a snapshot, so it is always `Some`.
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
    // unit the fold, the dedup and the parent remap all scale with — and with no
    // stop axis installed the whole thing is an add and three cell loads per
    // target.
    let mut poll = PollTicker::reduce(reduce_poll_stride());

    for &d in targets {
        let di = d.idx();
        // Cut BETWEEN targets, never inside one: each iteration makes exactly one
        // level marginal and rewrites its parent's refs, so the prefix already
        // done is a complete batch of its own once the end-sweep tagger below has
        // run over it — which is why the error arm runs the tagger rather than
        // returning straight out.
        if let Err(e) = poll.tick_by(tdd.levels[di].width() as u64 + 1) {
            tag_batch_marg_sides(tdd, was_marginal.as_deref());
            return Err(e);
        }
        marginalize_one_level(tdd, d, vtree, &mut computed);
    }

    tag_batch_marg_sides(tdd, was_marginal.as_deref());
    inline_leaf_targets(tdd, targets, vtree);
    Ok(())
}

/// Freeze one internal level: fold its per-node counts, cascade the levels
/// beneath it, dedup the fresh store and redirect the parent's refs onto it.
/// A no-op on a leaf, an empty level, or one that is already marginal.
fn marginalize_one_level(
    tdd: &mut Tdd,
    d: VtreeIdx,
    vtree: &Vtree,
    computed: &mut [Option<CountVec<RecoveryPanic>>],
) {
    let di = d.idx();
    if tdd.levels[di].is_marginal() || tdd.levels[di].width() == 0 {
        return;
    }
    if vtree.node(VtreeIdx(di as u32)).is_leaf() {
        return;
    }

    let (left, right) = vtree.children(d);
    let li = left.idx();
    let ri = right.idx();
    let width = tdd.levels[di].width();

    // Ensure child counts are available.
    ensure_counts(tdd, left, vtree, computed);
    ensure_counts(tdd, right, vtree, computed);

    // Compute counts for level d.
    let mut counts = CountVec::<RecoveryPanic>::with_width(width);

    for (i, _pairs) in tdd.levels[di].internal_inputs_iter() {
        // Per-node fold Σ left×right via `compute_marginal_node_int`
        // ([`IntFold::fold`] with this context's readers). `_pairs` is
        // re-derived inside via `pairs_iter_of_idx(i)` — identical to this
        // iterator's yield for internal inputs.
        let c = compute_marginal_node_int(tdd, &tdd.levels[di], i, li, ri, computed);
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
    cascade_marginalize(tdd, vtree, l_child, computed);
    cascade_marginalize(tdd, vtree, r_child, computed);

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

/// Make every marg-side slot ref persisted by the batch self-describing, once,
/// at the batch's chokepoint.
///
/// This standalone marginalize path does not go through `apply_and_fallible`,
/// so the end-of-apply tagger never runs on it. canon's redirect tags the
/// refs it rewrites, but its no-duplicate early-return leaves untouched
/// boundary refs raw. Tag every persisted marg-side slot ref once here, at
/// the batch chokepoint (idempotent with canon's per-target tagging), so the
/// 0=inline decode invariant holds before any downstream reader.
/// Refs into the just-marginalized child are bare grid `node_idx` slots —
/// this path runs after apply returns, so they were never read-time-tagged.
/// Declare NOT self-describing → `emit_or_tag` resolves them via counts even
/// under no-reexpand.
fn tag_batch_marg_sides(tdd: &mut Tdd, was_marginal: Option<&[bool]>) {
    crate::tdd::types::tag_all_marg_side_slots(tdd, was_marginal);
}

/// Sum out the batch's single-variable vtree LEAF targets.
///
/// The main loop skips leaf targets — a leaf has no pairs/store of its own. Sum
/// them out by inlining their fixed 0/1/2 model count into the parent's
/// leaf-side refs. This MUST run after [`tag_batch_marg_sides`]: the tagger keys
/// off the batch-entry `was_marginal` snapshot, so a leaf flipped marginal
/// earlier would have its side re-resolved as bare slots (misreading our inline
/// refs). Here the leaf is still structural while the tagger walks (its side
/// skipped), then we write fully self-describing inline refs ourselves.
fn inline_leaf_targets(tdd: &mut Tdd, targets: &[VtreeIdx], vtree: &Vtree) {
    for &d in targets {
        if vtree.node(d).is_leaf() {
            marginalize_leaf_inline(tdd, d, vtree);
        }
    }
}

/// Walk down from a level whose parent is marginal, converting non-leaf,
/// still-explicit descendants to marginal mode using counts cached by
/// `ensure_counts` during the enclosing `marginalize_batch` call.
fn cascade_marginalize(
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

/// Weighted analogue of [`marginalize_batch`]: marginalize the scheduled
/// `targets` carrying exact semiring values into `ws` instead of integer counts.
/// Faithful mirror of the integer driver's target loop minus the overflow /
/// dedup / parent-ref remap / inline-tag machinery (none needed for the
/// full-width weighted store). Used by the weighted marginalizing compile.
///
/// # Panics
///
/// Panics if the internal per-target weight computation is inconsistent (a
/// just-computed weight slot is unexpectedly empty).
pub(crate) fn marginalize_batch_weighted(
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
