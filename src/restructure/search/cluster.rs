//! Mid-compile marginal-clustering rotation pass: re-group two already-marginal
//! levels under one parent so `marginalize_closure` can collapse a whole
//! structural level out of a diagram that is still being built. The module's
//! one entry point is `rotate_marginal_cluster`.

use crate::engine::Engine;
use std::sync::Arc;

use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{Tdd, TddLevel};
use crate::restructure::relevel::{return_scratch, take_scratch};
use crate::engine::PollGate;
use crate::error::ApplyError;

use super::local::RotationObjective;

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
// `fuzz_search_preserves_marginal_count` in `restructure/relevel.rs`). Confined to
// subtree(t) via `subtree_allow_mask`, so it honors the compile-loop invariant
// that only indices inside the just-processed subtree may change.

/// Local per-level cost cap for a clustering rotation: skip it when the two
/// affected levels together exceed this many input pairs. A clustering rotation
/// multiset-expands the lower level by up to `bound_mult`× before
/// `marginalize_closure` collapses it, so the restructure churn scales with the
/// pre-rotation level size. Above this threshold that work is large while the
/// closure only removes final-size pairs, so the churn buys no peak reduction.
/// This is a *local* gate on the rotated levels, not a whole-diagram cost, and the
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
    let mut will_marginal: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut cur = Some(seed);
    while let Some(t) = cur {
        let ti = t.idx();
        if vtree.node(VtreeIdx(ti as u32)).is_leaf() || tdd.levels[ti].is_marginal() || will_marginal.contains(&ti) {
            // Already (will be) marginal/leaf — no pairs here, but it can be one
            // half of a cluster one level up. Keep climbing.
            cur = vtree.node(VtreeIdx(ti as u32)).parent();
            continue;
        }
        let (l, r) = vtree.children(t);
        let lm = tdd.levels[l.idx()].is_marginal() || will_marginal.contains(&l.idx());
        let rm = tdd.levels[r.idx()].is_marginal() || will_marginal.contains(&r.idx());
        if lm && rm {
            savings += level_pair_count(&tdd.levels[ti]);
            will_marginal.insert(ti);
            cur = vtree.node(VtreeIdx(ti as u32)).parent();
        } else {
            break;
        }
    }
    savings
}

/// The clustering pass's [`ProbeRule`]: the shared probe protocol, plus the two
/// things this pass knows that a pair-count delta cannot see.
///
/// It only wants a rotation that leaves `w` over two marginal children, it
/// needs a looser rebuild bound than the general search (clustering
/// full-expands `w` as a multiset before the closure removes it, so the tight
/// bound would bail on exactly the rotations worth making), and the win it is
/// buying is the closure that follows, not the rotation itself.
struct ClusterRule {
    bound_mult: usize,
}

impl RotationObjective for ClusterRule {
    /// The pair growth of the two affected levels — the same measure
    /// `SizeDelta` uses, scored here against the closure credit.
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64 {
        let old = level_pair_count(before.0) + level_pair_count(before.1);
        let new = level_pair_count(after.0) + level_pair_count(after.1);
        new as i64 - old as i64
    }
}

impl ProbeRule for ClusterRule {
    /// Only a rotation that leaves `w` over two marginal children can produce a
    /// collapse — and a pivot whose two levels are already vast costs a
    /// `bound_mult`× multiset churn while the closure only removes final-size
    /// pairs, which measures as cost with no peak win. The pass marks such a
    /// pivot tried, so it is never reconsidered (levels only grow mid-compile).
    fn admits(&mut self, tdd: &Tdd, info: &RotationInfo) -> bool {
        let (wl, wr) = tdd.vtree.children(info.w_idx);
        if !(tdd.levels[wl.idx()].is_marginal() && tdd.levels[wr.idx()].is_marginal()) {
            return false;
        }
        pivot_pairs(tdd, info) <= CLUSTER_MAX_LEVEL_PAIRS
    }

    /// Generous: clustering two marginal children full-expands `w` as a
    /// multiset (no Boolean dedup) before the collapse removes it, so the tight
    /// `old_pairs` bound would bail on the rotations this pass exists for. The
    /// multiple still rejects pathological blow-up.
    fn bound(&mut self, tdd: &Tdd, info: &RotationInfo, _default_bound: usize) -> usize {
        pivot_pairs(tdd, info).saturating_mul(self.bound_mult).max(64)
    }

    /// The pairs `marginalize_closure` is about to remove. Scored as a credit,
    /// the accept test `delta - credit < 0` reads as "the v/w growth must be
    /// repaid by the imminent closure" — a purely two-level comparison, where
    /// calling `tdd.size()` would be O(total nodes) per rotation.
    ///
    /// A rotation with nothing to close is not this pass's business even when
    /// it happens to shrink, so no closure means a credit that declines.
    fn credit(&mut self, tdd: &Tdd, info: &RotationInfo) -> i64 {
        let vt = Arc::clone(&tdd.vtree);
        match predict_closure_savings(tdd, &vt, info.w_idx) {
            0 => i64::MIN / 2,
            savings => savings as i64,
        }
    }

    /// Collapse the cluster the rotation just created. An `Err` here is the
    /// caller's wall passing mid-closure: the rotation is committed and
    /// count-preserving, and what the cut leaves behind is a level that is
    /// still structural, not a level that is wrong.
    fn on_accept(
        &mut self,
        eng: &Engine,
        tdd: &mut Tdd,
        _info: &RotationInfo,
    ) -> Result<(), ApplyError> {
        let vt = Arc::clone(&tdd.vtree);
        crate::marginal::marginalize_closure(eng, tdd, &vt).map(|_| ())
    }
}

/// The pair count of the two levels a rotation rebuilds — what both the local
/// cost cap and the rebuild bound are measured in.
fn pivot_pairs(tdd: &Tdd, info: &RotationInfo) -> usize {
    level_pair_count(&tdd.levels[info.v_idx.idx()])
        + level_pair_count(&tdd.levels[info.w_idx.idx()])
}

/// Mid-compile marginal-clustering rotation pass over subtree(`root`). Returns
/// the number of rotations accepted. Count-preserving; a no-op (and no vtree
/// clone) when subtree(root) has no two-marginal cluster reachable by one
/// rotation. The caller must afterward reseat sibling diagrams onto the (possibly
/// rotated) vtree Arc so they re-share it.
///
/// # Errors
///
/// Returns `Err(ApplyError::Deadline)` if the caller's wall passed while the pass
/// was running and the post-apply poll is armed. The rotations accepted before
/// the cut stay accepted and stay count-preserving; the pass is a size
/// optimization, so what a cut costs is diagram size and never the answer.
pub fn rotate_marginal_cluster(
    eng: &Engine,
    tdd: &mut Tdd,
    root: VtreeIdx,
    bound_mult: usize,
    tried: &mut [u8],
) -> Result<usize, ApplyError> {
    let lim = eng.limits();
    let allow = subtree_allow_mask(&tdd.vtree, root);
    // Read-only bail: no candidate ⇒ no vtree clone, no work.
    let n_cands = collect_cluster_candidates(tdd, &allow).len();
    if n_cands == 0 {
        return Ok(0);
    }
    // Detach to a uniquely-owned vtree so the per-rotation `Arc::make_mut`s are
    // no-ops (the refcount-1 probe precondition). Sibling diagrams keep the old
    // shared Arc until the caller reseats them — sound
    // because rotations only change indices inside subtree(root).
    let _ = Arc::make_mut(&mut tdd.vtree);

    // Pooled: this function runs tens of times per leaf compile, and a
    // per-call scratch paid a full teardown (~1.4k frees/leaf) plus re-growth
    // of the same buffers each time. See `restructure::scratch::take_scratch`.
    let mut scratch = take_scratch(eng);
    let mut rule = ClusterRule { bound_mult };
    let mut accepted = 0usize;
    // The pass's one preemption point, amortized. A sweep re-scans and re-attempts
    // for as long as it makes progress, and one attempt restructures the pivot's
    // two levels as a multiset — tens of calls per leaf compile, none of
    // which returned to the caller's wall. Metered in pairs of the pivot level,
    // the size `rotate_cluster`'s churn is bounded by (`bound_mult ×
    // old_pairs`). With no stop axis installed it is an add and three cell loads
    // per candidate.
    let mut poll = PollGate::new(lim.reduce_poll_stride());
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
            if let Err(e) = lim.poll(&mut poll, level_pair_count(&tdd.levels[v.idx()]) as u64 + 1) {
                return_scratch(eng, scratch);
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
            match probe(eng, tdd, v, kind, &mut rule, &mut scratch, usize::MAX) {
                Ok(true) => {
                    accepted += 1;
                    progress = true;
                }
                Ok(false) => {}
                Err(e) => {
                    return_scratch(eng, scratch);
                    return Err(e);
                }
            }
        }
        if !progress {
            break;
        }
    }
    return_scratch(eng, scratch);
    Ok(accepted)
}

#[cfg(test)]
#[path = "cluster_deadline_tests.rs"]
mod tests;
