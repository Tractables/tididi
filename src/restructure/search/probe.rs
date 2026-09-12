//! Rotation-kind dispatch and the per-rotation helpers the mid-compile
//! marginal-clustering pass ([`cluster`](super::cluster)) builds on.
//!
//! The rotate/unrotate/restructure kind wrappers, the per-level pair-count
//! helper, the marginal-level guard, the subtree allow-mask, and the one
//! rotation probe both passes run — they differ in the four decisions
//! [`ProbeRule`] names, not in the protocol.

use std::sync::Arc;

use crate::vtree::{RotationKind, Vtree, VtreeIdx, VtreeNode};
use crate::vtree::rotate::{rotate_pointers, RotationInfo};
use crate::diagram::Tdd;
use crate::engine::Engine;
use crate::limits::ApplyError;
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
/// child/grandchild-only-marginal rotation is the "rotate the parent of a
/// marginalized subtree" case. It is structurally sound and count-safe — the
/// marginal-context full expansion in `restructure::relevel` keeps the full cell/outer
/// multiset instead of sharing/deduping, so `#F` is preserved exactly — so it is
/// always allowed (no flag, no guard). This is what lets the cluster pass rotate
/// through marginalized nodes.
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
    ) -> Result<(), ApplyError> {
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
) -> Result<bool, ApplyError> {
    let saved_output = tdd.output;
    // `None` = the rotation does not apply at this pivot (a rotation child is a
    // leaf): nothing was installed, so there is nothing to undo.
    let Some(pending) = rotate_pointers(Arc::make_mut(&mut tdd.vtree), v, kind) else {
        return Ok(false);
    };
    let info = pending.info();
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();

    // The pointer rotation left every level untouched, so both of these read the
    // pre-rotation levels through the rotated vtree — which is what they want.
    if any_rotation_level_marginal(tdd, &info) || !rule.admits(tdd, &info) {
        pending.revert(Arc::make_mut(&mut tdd.vtree));
        tdd.output = saved_output;
        return Ok(false);
    }

    let bound = rule.bound(tdd, &info, default_bound);
    // `None` = the rebuild ran past the bound; it restored the levels itself, so
    // only the pointers are owed.
    let Some((old_v, old_w)) = restructure_inner_search(tdd, &info, kind, scratch, bound) else {
        pending.revert(Arc::make_mut(&mut tdd.vtree));
        tdd.output = saved_output;
        return Ok(false);
    };
    #[cfg(debug_assertions)]
    crate::check::debug_assert_rotation_locality(eng, tdd, info.w_idx);
    // Rotation locality, argued on the checker above: a diagram canonical
    // before the rotation is canonical after it, so no reduction pass has
    // anything to do and only the worklists the rotation seeded are drained.
    tdd.clear_worklists();

    let delta = rule.delta((&old_v, &old_w), (&tdd.levels[v_idx], &tdd.levels[w_idx]));
    let credit = rule.credit(tdd, &info);
    if delta - credit < 0 {
        // The pre-images are dead the moment the rotation stands: release them
        // before `on_accept`, which may allocate levels of its own.
        drop(old_v);
        drop(old_w);
        // The bottom-up order must be repaired before anything walks the vtree:
        // a rotation can flip a parent/child relation between two indices.
        pending.commit(Arc::make_mut(&mut tdd.vtree));
        rule.on_accept(eng, tdd, &info)?;
        Ok(true)
    } else {
        pending.revert(Arc::make_mut(&mut tdd.vtree));
        tdd.levels[v_idx] = old_v;
        tdd.levels[w_idx] = old_w;
        tdd.output = saved_output;
        Ok(false)
    }
}
