//! Shared rotation trial: rebuild two levels, score, then commit or restore.

use std::sync::Arc;

use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::vtree::rotate::{rotate_pointers, PendingTopo, RotationInfo};
use crate::diagram::{Dirty, Tdd, TddLevel, TddNodeId};
use crate::Engine;
use crate::limits::{Limits, OperationError};
use crate::restructure::relevel::{restructure_inner_search, RestructureScratch};

use super::local::RotationObjective;

/// What a caller of [`probe`] adds to the shared protocol.
///
/// The search supplies admission, size bounds, acceptance credit and an
/// accepted-rotation callback. Defaults use the objective alone.
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
/// Rotation locality restricts rebuilding, scoring and rollback to two levels.
pub(super) fn probe<R: ProbeRule>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    kind: RotationKind,
    rule: &mut R,
    scratch: &mut RestructureScratch,
    default_bound: usize,
) -> Result<bool, OperationError> {
    let Some(mut trial) = RotationTrial::new(tdd, eng.limits(), v, kind) else { return Ok(false) };
    let info = trial.pending.as_ref().unwrap().info();
    // The two rebuilt levels need explicit pairs; marginal grandchildren are
    // allowed because the restructure preserves their contribution multiset.
    if trial.tdd.levels[info.v_idx.idx()].is_marginal()
        || trial.tdd.levels[info.w_idx.idx()].is_marginal()
        || !rule.admits(trial.tdd, &info)
    {
        return Ok(false);
    }
    let bound = rule.bound(trial.tdd, &info, default_bound);
    trial.old_levels = restructure_inner_search(eng.limits(), trial.tdd, &info, kind, scratch, bound)?;
    let Some((old_v, old_w)) = trial.old_levels.as_ref() else { return Ok(false) };
    #[cfg(debug_assertions)]
    crate::test_helpers::check::debug_assert_rotation_locality(eng, trial.tdd, info.w_idx);
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
    /// Where the levels the trial builds and drops give their bytes back.
    lim: &'a Limits,
    pending: Option<PendingTopo>,
    old_levels: Option<(TddLevel, TddLevel)>,
    old_output: TddNodeId,
    shared_tree: Option<Arc<Vtree>>,
    old_dirty: Option<Dirty>,
}

impl<'a> RotationTrial<'a> {
    /// Rotate the pointers while retaining the state needed for a non-allocating rollback.
    fn new(tdd: &'a mut Tdd, lim: &'a Limits, v: VtreeIdx, kind: RotationKind) -> Option<Self> {
        let old_output = tdd.output;
        let shared_tree = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        let pending = rotate_pointers(Arc::make_mut(&mut tdd.vtree), v, kind);
        if pending.is_none() {
            if let Some(tree) = shared_tree { tdd.vtree = tree; }
            return None;
        }
        let old_dirty = Some(std::mem::take(&mut tdd.dirty));
        Some(Self { tdd, lim, pending, old_levels: None, old_output, shared_tree, old_dirty })
    }

    /// Repair topology and release the preimage before any accepted-rotation callback.
    fn commit(mut self) {
        // The preimage is about to be dropped, and the rebuild charged the
        // levels that replaced it, so the operation's in-flight total should end
        // up carrying the difference rather than both.
        if let Some((old_v, old_w)) = self.old_levels.take() {
            self.lim.release_bytes(old_v.arena_capacity_bytes() + old_w.arena_capacity_bytes());
        }
        self.pending.take().unwrap().commit(Arc::make_mut(&mut self.tdd.vtree));
        // Both exits end with the pre-probe obligations still present: `Drop`
        // restores them wholesale on reject, and the accept path puts them back
        // underneath what the rotation itself recorded. Clearing here instead
        // dropped both, which left a later `minimize` skipping levels it still
        // owed work on.
        let carried = self.old_dirty.take().expect("a trial takes the worklists when it is created");
        self.tdd.dirty.merge_under(carried);
    }
}

impl Drop for RotationTrial<'_> {
    fn drop(&mut self) {
        let Some(pending) = self.pending.take() else { return };
        let info = pending.info();
        pending.revert(Arc::make_mut(&mut self.tdd.vtree));
        if let Some((v, w)) = self.old_levels.take() {
            // The rebuilt levels are the ones being dropped here, so their
            // charge is what the trial hands back.
            self.lim.release_bytes(
                self.tdd.levels[info.v_idx.idx()].arena_capacity_bytes()
                    + self.tdd.levels[info.w_idx.idx()].arena_capacity_bytes(),
            );
            self.tdd.levels[info.v_idx.idx()] = v;
            self.tdd.levels[info.w_idx.idx()] = w;
        }
        self.tdd.output = self.old_output;
        self.tdd.dirty = self.old_dirty.take().unwrap();
        if let Some(tree) = self.shared_tree.take() { self.tdd.vtree = tree; }
    }
}
