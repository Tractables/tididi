//! Mid-compile marginal-clustering rotation pass: re-group two already-marginal
//! levels under one parent so `marginalize_closure` can collapse a whole
//! structural level out of a diagram that is still being built. The module's
//! one entry point is `rotate_marginal_cluster`.

use crate::Engine;

use rustc_hash::FxHashSet;

use crate::vtree::{RotationKind, Vtree, VtreeIdx, VtreeNode};
use crate::vtree::rotate::RotationInfo;
use crate::diagram::Tdd;
use crate::limits::OperationError;

use super::local::{MinimizePairs, RotationObjective};

use super::probe::*;

/// Build an allow-mask for `subtree(root)`: every internal node in
/// `subtree(root)` (root included) is marked true. Used by the mid-compile
/// rotation pass to confine the search to the post-order frontier's
/// fully-compiled region. Including root is safe: the parent level has no
/// compiled data yet, and rotation at root preserves 1-to-1 node
/// correspondence at v_idx (rotation locality).
fn subtree_allow_mask(vtree: &Vtree, root: VtreeIdx) -> Vec<bool> {
    let mut mask = vec![false; vtree.num_nodes()];
    let mut stack: Vec<VtreeIdx> = vec![root];
    while let Some(n) = stack.pop() {
        if let VtreeNode::Internal { left, right, .. } = *vtree.node(n) {
            mask[n.idx()] = true;
            stack.push(left);
            stack.push(right);
        }
    }
    mask
}

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
/// A `Left` rotation at `v=(A, w)`, `w=(B, C)` produces a lower node `(A, B)`
/// (rotate.rs cascade test): it clusters `A = v.left` and `B = v.right.left`.
/// A `Right` rotation at `v=(w, C)`, `w=(A, B)` produces `(B, C)`: it clusters
/// `B = v.left.right` and `C = v.right`.
fn collect_cluster_candidates(tdd: &Tdd, allow: &[bool]) -> Vec<(VtreeIdx, RotationKind)> {
    let vtree = &*tdd.vtree;
    let mut out = Vec::new();
    for (v, _, _) in vtree.internal_bottomup() {
        if !allow[v.idx()] || tdd.levels[v.idx()].is_marginal() {
            continue;
        }
        let (vl, vr) = vtree.children(v);
        // `Left`: needs v.right internal; clusters v.left and v.right.left.
        if !vtree.node(vr).is_leaf() {
            let (vrl, _) = vtree.children(vr);
            if tdd.levels[vl.idx()].is_marginal() && tdd.levels[vrl.idx()].is_marginal() {
                out.push((v, RotationKind::Left));
            }
        }
        // `Right`: needs v.left internal; clusters v.left.right and v.right.
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
    let mut will_marginal: FxHashSet<usize> = FxHashSet::default();
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
            savings += tdd.levels[ti].live_pairs();
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

    /// The pair growth of the two rebuilt levels — the default objective's
    /// measurement, unchanged — against the pairs `marginalize_closure` is
    /// about to remove. The test `delta < credit` reads as "the v/w growth
    /// must be repaid by the imminent closure", a purely two-level comparison,
    /// where calling `tdd.pair_count()` would be O(total nodes) per rotation.
    ///
    /// A rotation with nothing to close is not this pass's business even when
    /// it happens to shrink, so no closure means a credit that declines.
    fn keeps(&mut self, probe: &RotationProbe<'_>, info: &RotationInfo) -> bool {
        let tdd = probe.diagram();
        let credit = match predict_closure_savings(tdd, &tdd.vtree, info.w_idx) {
            0 => i64::MIN / 2,
            savings => savings as i64,
        };
        MinimizePairs.delta(probe) < credit
    }
}

/// The pair count of the two levels a rotation rebuilds — what both the local
/// cost cap and the rebuild bound are measured in.
fn pivot_pairs(tdd: &Tdd, info: &RotationInfo) -> usize {
    tdd.levels[info.v_idx.idx()].live_pairs()
        + tdd.levels[info.w_idx.idx()].live_pairs()
}

impl Engine {
    /// Rotate within `root` to bring already-marginal levels under a common
    /// parent, then sum out that parent when doing so reduces pair count.
    ///
    /// Returns the number of accepted rotations. Each preserves the count.
    /// `bound_mult` limits a
    /// trial's intermediate pairs to that multiple of the two original
    /// levels' pair count, with a minimum allowance of 64 pairs.
    ///
    /// `tried` records attempted directions by vtree index: bit 0 is left and
    /// bit 1 is right. Start with an empty or zero-filled vector; this method
    /// grows it as needed. Reuse it across calls on the same evolving diagram
    /// to avoid retrying rejected candidates; clear it for a different diagram.
    /// Retained entries may skip opportunities after a rotation, without
    /// changing the count.
    ///
    /// The diagram may acquire a new vtree allocation. Subsequent operands must
    /// use [`Tdd::vtree`]. A diagram over the old allocation is not transformed
    /// by this call. With no candidate, the original allocation is retained.
    ///
    /// # Errors
    ///
    /// An invalid `root` returns [`OperationError::LevelNotInVtree`] before
    /// mutation. Allocation refusal or cancellation returns the corresponding
    /// [`OperationError`]; accepted rotations and attempt flags remain in place,
    /// and the diagram remains count-correct. A trial's own storage is charged
    /// to the byte budget and given back when the trial ends, as in
    /// [`Engine::rotation_search`]; the output cap does not apply.
    pub fn rotate_marginal_cluster(
        &self,
        tdd: &mut Tdd,
        root: VtreeIdx,
        bound_mult: usize,
        tried: &mut Vec<u8>,
    ) -> Result<usize, OperationError> {
        tdd.check_level_indices(&[root])?;
        let eng = self;
        let lim = eng.limits();
        let additional = tdd.vtree.num_nodes().saturating_sub(tried.len());
        if additional != 0 {
            lim.reserve(tried, additional)?;
            tried.resize(tdd.vtree.num_nodes(), 0);
        }
        let allow = subtree_allow_mask(&tdd.vtree, root);
        // Read-only bail: no candidate ⇒ no vtree clone, no work.
        let mut cands = collect_cluster_candidates(tdd, &allow);
        if cands.is_empty() {
            return Ok(0);
        }
        // Detach to a uniquely-owned vtree so the per-rotation `Arc::make_mut`s are
        // no-ops (the refcount-1 probe precondition). Sibling diagrams keep the old
        // shared Arc until the caller reseats them — sound
        // because rotations only change indices inside subtree(root).
        let mut search = super::SearchTree::new(tdd);
        let tdd = &mut *search.tdd;

        let mut scratch = eng.restructure().checkout(lim);
        let mut rule = ClusterRule { bound_mult };
        let mut accepted = 0usize;
        // The pass's one preemption point, amortized. A sweep re-scans and re-attempts
        // for as long as it makes progress, and one attempt restructures the pivot's
        // two levels as a multiset — tens of calls per leaf compile, none of
        // which returned to the caller's wall. Metered in pairs of the pivot level,
        // the size `rotate_cluster`'s churn is bounded by (`bound_mult ×
        // old_pairs`). With no stop axis installed it is an add and three cell loads
        // per candidate.
        let mut poll = lim.gate();
        // Each accept strictly shrinks size, so the fixpoint terminates. Re-scan
        // after each sweep: a closed cluster can expose a fresh one a level up.
        loop {
            let mut progress = false;
            for (v, kind) in cands {
                // Poll between attempts. A later closure refusal retains its
                // committed rotation and the attempt flags already recorded.
                poll.poll(tdd.levels[v.idx()].live_pairs() as u64 + 1)?;
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
                if probe(eng, tdd, v, kind, &mut rule, &mut scratch, usize::MAX)? {
                    // Keep the committed vtree even if the follow-up closure fails.
                    search.original = None;
                    crate::marginal::marginalize_closure(eng, tdd)?;
                    accepted += 1;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
            cands = collect_cluster_candidates(tdd, &allow);
            if cands.is_empty() {
                break;
            }
        }
        Ok(accepted)
    }
}

#[cfg(test)]
#[path = "tests/cluster/mod.rs"]
mod tests;
