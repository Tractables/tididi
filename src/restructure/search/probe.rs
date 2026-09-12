//! Rotation-kind dispatch and the per-rotation helpers the mid-compile
//! marginal-clustering pass ([`cluster`](super::cluster)) builds on.
//!
//! The rotate/unrotate/restructure kind wrappers, the per-level pair-count
//! helper, the marginal-level guard, the subtree allow-mask, and the one
//! rotation probe both passes run — they differ in the four decisions
//! [`ProbeRule`] names, not in the protocol.

use std::sync::Arc;

use crate::vtree::{RotationKind, Vtree, VtreeIdx, VtreeNode};
use crate::vtree::rotate::{rotate_pointers, PendingTopo, RotationInfo};
use crate::diagram::{Dirty, Tdd, TddLevel, TddNodeId};
use crate::engine::Engine;
use crate::limits::OperationError;
use crate::restructure::relevel::{restructure_inner_search, RestructureScratch};

use super::local::RotationObjective;

// ─── Shared utilities ─────────────────────────────────────────────────────

/// Build an allow-mask for `subtree(root)`: every internal node in
/// `subtree(root)` (root included) is marked true. Used by the mid-compile
/// rotation pass to confine the search to the post-order frontier's
/// fully-compiled region. Including root is safe: the parent level has no
/// compiled data yet, and rotation at root preserves 1-to-1 node
/// correspondence at v_idx (rotation locality).
pub(super) fn subtree_allow_mask(vtree: &Vtree, root: VtreeIdx) -> Vec<bool> {
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

/// Returns true if a rotation level is marginal in a way that blocks the rotation.
///
/// `v`/`w` have their pairs iterated and rebuilt during restructure, and once a node
/// is marginalized its children no longer exist as levels (collapsed to counts) —
/// so the rotation that would split it is ill-defined; a `v`/`w`-marginal rotation
/// is genuinely unhandled and always blocks.
///
/// `a`,`b`,`c` (grandchildren) are referenced only as bare node indices, so a
/// rotation whose only marginal levels are those is always allowed; the
/// marginal-context expansion in `restructure::relevel` keeps the count exact.
pub(super) fn any_rotation_level_marginal(tdd: &Tdd, info: &RotationInfo) -> bool {
    // `v`/`w` marginal: pairs would be iterated and the marginalized node would
    // have to be decomposed into children that no longer exist. Always blocks.
    if tdd.levels[info.v_idx.idx()].is_marginal()
        || tdd.levels[info.w_idx.idx()].is_marginal()
    {
        return true;
    }
    // a/b/c (grandchild) marginal is the parent-of-marginal case: count-safe via
    // the marginal-context full expansion, so the rotation always proceeds.
    false
}

// ─── The rotation probe ───────────────────────────────────────────────────

/// What a caller of [`probe`] adds to the shared protocol.
///
/// The protocol is fixed — rotate, guard, restructure under a bound,
/// re-minimize, score, keep or restore — and a pass varies it only at these
/// four points. Every method has a default, so an objective that just wants the
/// protocol implements nothing.
pub(super) trait ProbeRule: RotationObjective {
    /// A last gate before the expensive restructure, read on the rotated vtree
    /// with the levels still untouched. `false` reverts the pointers and
    /// declines the probe.
    fn admits(&mut self, _tdd: &Tdd, _info: &RotationInfo) -> bool {
        true
    }

    /// The pair bound the restructure bails past, given the caller's default.
    fn bound(&mut self, _tdd: &Tdd, _info: &RotationInfo, default_bound: usize) -> usize {
        default_bound
    }

    /// What an accepted rotation is worth beyond its own two-level delta — the
    /// win a pass knows is about to follow but the scored levels cannot show.
    /// The probe accepts iff `delta - credit < 0`.
    fn credit(&mut self, _tdd: &Tdd, _info: &RotationInfo) -> i64 {
        0
    }

    /// Run after the rotation is committed. Its `Err` propagates with the
    /// rotation kept: what it leaves unfinished is an optimization, never the
    /// diagram's correctness.
    fn on_accept(
        &mut self,
        _eng: &Engine,
        _tdd: &mut Tdd,
        _info: &RotationInfo,
    ) -> Result<(), OperationError> {
        Ok(())
    }
}

/// Probe the `kind` rotation at pivot `v` and keep it iff `rule` scores it an
/// improvement. Returns whether the rotation was kept; on a decline the diagram
/// is restored bit-for-bit, vtree included.
///
/// The two affected levels are the only ones the restructure and the following
/// re-minimize touch (Rotation Locality), which is what makes both the score and
/// the restore two levels wide.
pub(super) fn probe<R: ProbeRule>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    kind: RotationKind,
    rule: &mut R,
    scratch: &mut RestructureScratch,
    default_bound: usize,
) -> Result<bool, OperationError> {
    let Some(mut trial) = RotationTrial::new(tdd, v, kind) else { return Ok(false) };
    let info = trial.pending.as_ref().unwrap().info();
    if any_rotation_level_marginal(trial.tdd, &info) || !rule.admits(trial.tdd, &info) {
        return Ok(false);
    }
    let bound = rule.bound(trial.tdd, &info, default_bound);
    trial.old_levels = restructure_inner_search(trial.tdd, &info, kind, scratch, bound);
    let Some((old_v, old_w)) = trial.old_levels.as_ref() else { return Ok(false) };
    #[cfg(debug_assertions)]
    crate::test_helpers::check::debug_assert_rotation_locality(eng, trial.tdd, info.w_idx);
    trial.tdd.clear_worklists();
    let delta = rule.delta((old_v, old_w), (&trial.tdd.levels[info.v_idx.idx()], &trial.tdd.levels[info.w_idx.idx()]));
    let credit = rule.credit(trial.tdd, &info);
    if delta < credit {
        trial.commit();
        rule.on_accept(eng, tdd, &info)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Own a candidate's preimage until its topology and levels are committed together.
struct RotationTrial<'a> {
    tdd: &'a mut Tdd,
    pending: Option<PendingTopo>,
    old_levels: Option<(TddLevel, TddLevel)>,
    old_output: TddNodeId,
    old_dirty: Option<Dirty>,
    shared_tree: Option<Arc<Vtree>>,
}

impl<'a> RotationTrial<'a> {
    /// Rotate the pointers while retaining the state needed for a non-allocating rollback.
    fn new(tdd: &'a mut Tdd, v: VtreeIdx, kind: RotationKind) -> Option<Self> {
        let old_dirty = Some(tdd.dirty.clone());
        let old_output = tdd.output;
        let shared_tree = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        let pending = rotate_pointers(Arc::make_mut(&mut tdd.vtree), v, kind);
        if pending.is_none() {
            if let Some(tree) = shared_tree { tdd.vtree = tree; }
            return None;
        }
        Some(Self { tdd, pending, old_levels: None, old_output, old_dirty, shared_tree })
    }

    /// Repair topology and release the preimage before any accepted-rotation callback.
    fn commit(mut self) {
        self.pending.take().unwrap().commit(Arc::make_mut(&mut self.tdd.vtree));
    }
}

impl Drop for RotationTrial<'_> {
    fn drop(&mut self) {
        let Some(pending) = self.pending.take() else { return };
        let info = pending.info();
        pending.revert(Arc::make_mut(&mut self.tdd.vtree));
        if let Some((v, w)) = self.old_levels.take() {
            self.tdd.levels[info.v_idx.idx()] = v;
            self.tdd.levels[info.w_idx.idx()] = w;
        }
        self.tdd.output = self.old_output;
        self.tdd.dirty = self.old_dirty.take().unwrap();
        if let Some(tree) = self.shared_tree.take() { self.tdd.vtree = tree; }
    }
}
