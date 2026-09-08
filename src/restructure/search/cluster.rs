//! Mid-compile marginal-clustering rotation pass: re-group two already-marginal
//! levels under one parent so `marginalize_closure` can collapse a whole
//! structural level out of a diagram that is still being built. The module's
//! one entry point is `cluster_marginal_rotations_in_subtree`.

use std::sync::Arc;

use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::diagram::Tdd;
use crate::restructure::relevel::{RestructureScratch, return_scratch, take_scratch};
use crate::reduce::minimize_after_rotation;
use crate::limits::{reduce_poll_stride, ApplyError, PollTicker};

use super::core::*;

// ─── Marginal-clustering rotation pass (mid-compile) ───────────────────────
//
// After step `t` marginalizes some levels in subtree(t), two already-marginal
// levels can sit under different parents. A single rotation can re-group them
// as the two children of one node, which `marginalize_closure` then collapses
// into a marginal count level — removing a whole structural level from the
// in-flight diagram (lowering the working-set peak, not just the final size).
//
// This is the size-driven specialization of the greedy rotation search,
// targeted at exactly the parent-of-marginal (gc=1) case. Count-safety comes
// from the same rotate → restructure → `marginalize_closure` path the general
// search uses (full marginal-context expansion preserves #F; see
// `parent_of_marginal_rotation_preserves_model_count` /
// `fuzz_search_preserves_marginal_count` in `tdd/restructure/relevel.rs`). Confined to
// subtree(t) via `subtree_allow_mask`, so it honors the compile-loop invariant
// that only indices inside the just-processed subtree may change.

/// Local per-level cost cap for a clustering rotation: skip it when the two
/// affected levels together exceed this many input pairs. A clustering rotation
/// multiset-expands the lower level by up to `bound_mult`× before
/// `marginalize_closure` collapses it, so the restructure churn scales with the
/// pre-rotation level size. Above this threshold that work is large while the
/// closure only removes final-size pairs — measured cost with no peak win. This
/// is a *local* gate on the rotated levels, not a whole-diagram cost, and the
/// number is a policy value chosen at the same scale as the minimize twin-scan
/// cap (`C2_SCAN_MAX_NODES`). Below it, rotate freely; above it, skip and mark
/// the pair tried.
const CLUSTER_MAX_LEVEL_PAIRS: usize = 131_072;

/// Collect candidate `(pivot, kind)` rotations that would cluster two marginal
/// levels under one parent. Read-only — does not touch the vtree Arc, so the
/// common "nothing to cluster" case costs no clone.
///
/// LEFT rotation at `v=(A, w)`, `w=(B, C)` produces a lower node `(A, B)`
/// (rotate.rs cascade test): it clusters `A = v.left` and `B = v.right.left`.
/// RIGHT rotation at `v=(w, C)`, `w=(A, B)` produces `(B, C)`: it clusters
/// `B = v.left.right` and `C = v.right`.
fn collect_cluster_candidates(tdd: &Tdd, allow: &[bool]) -> Vec<(VtreeIdx, RotationKind)> {
    let vtree = &*tdd.vtree;
    let mut out = Vec::new();
    for (v, _, _) in vtree.internal_bottomup() {
        if !allow[v.idx()] || tdd.levels[v.idx()].is_marginal() {
            continue;
        }
        let (vl, vr) = vtree.children(v);
        // LEFT: needs v.right internal; clusters v.left and v.right.left.
        if !vtree.node(vr).is_leaf() {
            let (vrl, _) = vtree.children(vr);
            if tdd.levels[vl.idx()].is_marginal() && tdd.levels[vrl.idx()].is_marginal() {
                out.push((v, RotationKind::Left));
            }
        }
        // RIGHT: needs v.left internal; clusters v.left.right and v.right.
        if !vtree.node(vl).is_leaf() {
            let (_, vlr) = vtree.children(vl);
            if tdd.levels[vlr.idx()].is_marginal() && tdd.levels[vr.idx()].is_marginal() {
                out.push((v, RotationKind::Right));
            }
        }
    }
    out
}

/// Pair-count of the levels `marginalize_closure` would collapse if `seed`
/// (the rotation's new lower node `w_idx`) became a structural level over two
/// marginal children. Mirrors `marginalize_closure`'s parent climb without
/// mutating: a level collapses iff it is
/// non-leaf, non-marginal, and both children are (or will become) marginal;
/// each collapse can complete a cluster one level up. The summed pairs are the
/// size the imminent closure will remove — the credit that makes a clustering
/// rotation worth accepting even when its local restructure grew.
fn predict_closure_savings(tdd: &Tdd, vtree: &Vtree, seed: VtreeIdx) -> usize {
    let mut savings = 0usize;
    let mut will_marg: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut cur = Some(seed);
    while let Some(t) = cur {
        let ti = t.idx();
        if vtree.node(VtreeIdx(ti as u32)).is_leaf() || tdd.levels[ti].is_marginal() || will_marg.contains(&ti) {
            // Already (will be) marginal/leaf — no pairs here, but it can be one
            // half of a cluster one level up. Keep climbing.
            cur = vtree.node(VtreeIdx(ti as u32)).parent();
            continue;
        }
        let (l, r) = vtree.children(t);
        let lm = tdd.levels[l.idx()].is_marginal() || will_marg.contains(&l.idx());
        let rm = tdd.levels[r.idx()].is_marginal() || will_marg.contains(&r.idx());
        if lm && rm {
            savings += level_pair_count(&tdd.levels[ti]);
            will_marg.insert(ti);
            cur = vtree.node(VtreeIdx(ti as u32)).parent();
        } else {
            break;
        }
    }
    savings
}

/// Attempt one clustering rotation at `v`. Returns true if accepted (rotation
/// kept and the resulting marginal cluster closed). Mirrors the greedy
/// rotation-probe protocol (rotate/guard/bounded-restructure/accept-or-revert),
/// but (a) only proceeds when the rotation clusters two marginal levels and (b)
/// credits the imminent `marginalize_closure` collapse in the accept test — the
/// win the pair-delta-only accept criterion cannot see.
///
/// `Err(ApplyError::Deadline)` is the closure of an ACCEPTED rotation running out
/// of the caller's wall. The rotation itself is committed and count-preserving;
/// what the cut leaves unfinished is the collapse of the cluster it created,
/// which is a level that is still structural rather than a level that is wrong.
fn try_cluster_rotate(
    tdd: &mut Tdd,
    v: VtreeIdx,
    kind: RotationKind,
    scratch: &mut RestructureScratch,
    bound_mult: usize,
) -> Result<bool, ApplyError> {
    let backup_output = tdd.output;
    let Some(info) = rotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), v, kind) else {
        return Ok(false);
    };
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();

    // v/w-marginal rotations are genuinely unhandled; grandchild-marginal (our
    // target) is count-safe. Same guard the general search uses.
    if any_rotation_level_marginal(tdd, &info) {
        unrotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), &info, kind);
        tdd.output = backup_output;
        return Ok(false);
    }

    // Cheap pre-filter: only a rotation that leaves w_idx over two marginal
    // children can produce a collapse. The pointer rotation hasn't touched the
    // grandchildren's levels, so reading them now is valid.
    let (wl, wr) = tdd.vtree.children(info.w_idx);
    if !(tdd.levels[wl.idx()].is_marginal() && tdd.levels[wr.idx()].is_marginal()) {
        unrotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), &info, kind);
        tdd.output = backup_output;
        return Ok(false);
    }

    // Generous bound: clustering two marginal children full-expands w_idx as a
    // multiset (no Boolean dedup) before the collapse removes it, so the tight
    // `old_pairs` restructure bound would bail on exactly the rotations we want.
    // Cap at bound_mult× (plus a floor) to still reject pathological blow-up.
    let old_pairs = level_pair_count(&tdd.levels[v_idx]) + level_pair_count(&tdd.levels[w_idx]);
    // Local cost cap: the restructure churns ~`old_pairs × bound_mult` pairs, so on
    // multi-million-pair levels it is expensive while (measured) not lowering the
    // realized peak. Skip + revert pointers; the pass marks the pair tried so we
    // never reconsider it (levels only grow during compile).
    if old_pairs > CLUSTER_MAX_LEVEL_PAIRS {
        unrotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), &info, kind);
        tdd.output = backup_output;
        return Ok(false);
    }
    let bound = old_pairs.saturating_mul(bound_mult).max(64);
    let Some((old_v, old_w)) = restructure_kind_bounded(tdd, &info, kind, scratch, bound) else {
        unrotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), &info, kind);
        tdd.output = backup_output;
        return Ok(false);
    };
    minimize_after_rotation(tdd, info.w_idx);

    // Local accept — NO whole-diagram `tdd.size()`. The restructure +
    // `minimize_after_rotation` change only the v/w levels (the multiset
    // expansion of w plus its dedup), so the net-shrink test
    // `restructured - savings < pre_size` reduces algebraically to a purely
    // two-level comparison: `pre_size` appears on both sides and cancels, leaving
    // `(new_vw - old_pairs) < savings` — the v/w pair growth must be repaid by the
    // imminent closure. This is the same size-increment locality the rotation
    // search uses (`size_after_rotation`), credited with the closure. Calling
    // `tdd.size()` here instead was O(total nodes) per rotation — the dominant
    // cost on large diagrams. (Approximate if `minimize` touches other levels
    // under marginal expansion, but this only decides keep-vs-revert; the rotation
    // and closure are count-preserving either way.)
    let new_vw = level_pair_count(&tdd.levels[v_idx]) + level_pair_count(&tdd.levels[w_idx]);
    let vt = Arc::clone(&tdd.vtree);
    let savings = predict_closure_savings(tdd, &vt, info.w_idx);
    let accept = savings > 0 && new_vw.saturating_sub(old_pairs) < savings;

    // Accept iff the imminent collapse makes this a strict net shrink.
    if accept {
        // Rollback is unreachable from here, so the two pre-images are dead:
        // release them before the closure rather than hold a full copy of both
        // pre-rotation levels across the levels it allocates.
        drop(old_v);
        drop(old_w);
        // Use the FULL fixup (pointers + `refresh_filtered_topo`), NOT the
        // pointers-only variant: the rotation can flip a parent/child relation
        // between two vtree indices (e.g. node 17 ceases to be a child of 18 and
        // becomes its parent). `internal_bottomup` walks `internal_topo`, which
        // the pointers-only fixup leaves STALE — so the next `apply_and` would
        // sweep a parent level before its (now-)child, the bottom-up identity
        // propagation never runs at the child before the parent reads it, and the
        // marginalized subtree meets an operand whose identity is unrecognized →
        // dense path derefs a freed marginal level → panic. Every other rotation
        // caller refilters after its sweep; the mid-compile cluster pass conjoins
        // (via the later apply) before any such refilter, so it must refresh here.
        Arc::make_mut(&mut tdd.vtree)
            .fixup_topo_after_rotate(&info, kind);
        let vt2 = Arc::clone(&tdd.vtree);
        crate::marginal::marginalize_closure(tdd, &vt2)?;
        Ok(true)
    } else {
        unrotate_pointers_kind(Arc::make_mut(&mut tdd.vtree), &info, kind);
        tdd.levels[v_idx] = old_v;
        tdd.levels[w_idx] = old_w;
        tdd.output = backup_output;
        Ok(false)
    }
}

/// Mid-compile marginal-clustering rotation pass over subtree(`root`). Returns
/// the number of rotations accepted. Count-preserving; a no-op (and no vtree
/// clone) when subtree(root) has no two-marginal cluster reachable by one
/// rotation. The caller must afterward reseat sibling TDDs onto the (possibly
/// rotated) vtree Arc so they re-share it.
///
/// # Errors
///
/// Returns `Err(ApplyError::Deadline)` if the caller's wall passed while the pass
/// was running and the post-apply poll is armed. The rotations accepted before
/// the cut stay accepted and stay count-preserving; the pass is a size
/// optimization, so what a cut costs is diagram size and never the answer.
pub fn cluster_marginal_rotations_in_subtree(
    tdd: &mut Tdd,
    root: VtreeIdx,
    bound_mult: usize,
    tried: &mut [u8],
) -> Result<usize, ApplyError> {
    let allow = subtree_allow_mask(&tdd.vtree, root);
    // Read-only bail: no candidate ⇒ no vtree clone, no work.
    let n_cands = collect_cluster_candidates(tdd, &allow).len();
    if n_cands == 0 {
        return Ok(0);
    }
    // Detach to a uniquely-owned vtree so the per-rotation `Arc::make_mut`s are
    // no-ops (the refcount-1 probe precondition). Sibling TDDs keep the old
    // shared Arc until the caller reseats them — sound
    // because rotations only change indices inside subtree(root).
    let _ = Arc::make_mut(&mut tdd.vtree);

    // Pooled: this function runs tens of times per leaf compile, and a
    // per-call scratch paid a full teardown (~1.4k frees/leaf) plus re-growth
    // of the same buffers each time. See `rotate::take_scratch`.
    let mut scratch = take_scratch();
    let mut accepted = 0usize;
    // The pass's ONE preemption point, amortized. A sweep re-scans and re-attempts
    // for as long as it makes progress, and one attempt restructures the pivot's
    // two levels as a multiset — tens of calls per leaf compile, none of
    // which returned to the caller's wall. Metered in pairs of the pivot level,
    // the size `try_cluster_rotate`'s churn is bounded by (`bound_mult ×
    // old_pairs`). With no stop axis installed it is an add and three cell loads
    // per candidate.
    let mut poll = PollTicker::reduce(reduce_poll_stride());
    // Each accept strictly shrinks size, so the fixpoint terminates. Re-scan
    // each sweep: a closed cluster can expose a fresh one a level up.
    loop {
        let cands = collect_cluster_candidates(tdd, &allow);
        if cands.is_empty() {
            break;
        }
        let mut progress = false;
        for (v, kind) in cands {
            // Cut BETWEEN attempts: an attempt either commits its rotation and
            // closes the cluster or reverts everything it touched, so the pass is
            // only ever interrupted at a point where the diagram is one some
            // completed attempt left behind. `tried` keeps whatever it recorded —
            // a pivot marked before the cut is one this compile will not
            // reconsider, which is the flag's own best-effort contract.
            if let Err(e) = poll.tick_by(level_pair_count(&tdd.levels[v.idx()]) as u64 + 1) {
                return_scratch(scratch);
                return Err(e);
            }
            // Attempt-once per (pivot, kind). Marginality is monotonic within a
            // compile, so a rejected cluster stays a candidate and — without this
            // guard — would be re-considered (full O(size) restructure + revert)
            // on every later ancestor step that re-enters this subtree, for the
            // same guaranteed reject. One shot per pivot makes total attempts
            // linear in vtree nodes. Best-effort: a rotation relabels indices, so
            // a stale flag can occasionally mis-skip or re-clear a pivot; that
            // only narrows the optimization, never the counts (still exact).
            let bit = if matches!(kind, RotationKind::Left) { 0b01u8 } else { 0b10u8 };
            if tried[v.idx()] & bit != 0 {
                continue;
            }
            // A prior accept this sweep may have collapsed this pivot already.
            if tdd.levels[v.idx()].is_marginal() {
                continue;
            }
            tried[v.idx()] |= bit;
            match try_cluster_rotate(tdd, v, kind, &mut scratch, bound_mult) {
                Ok(true) => {
                    accepted += 1;
                    progress = true;
                }
                Ok(false) => {}
                Err(e) => {
                    return_scratch(scratch);
                    return Err(e);
                }
            }
        }
        if !progress {
            break;
        }
    }
    return_scratch(scratch);
    Ok(accepted)
}

#[cfg(test)]
#[path = "cluster_deadline_tests.rs"]
mod tests;
