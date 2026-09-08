//! The reduce walk's mid-loop preemption point.
//!
//! `contract_all_twins_topdown` is the most expensive phase of a minimize and it
//! runs BETWEEN two applies of one bottom-up step, so before the poll a caller's
//! wall was observed only where the step ended. These tests pin the three
//! properties the poll is worth having for: it fires when the wall has passed, it
//! stays out of the way when it is disarmed or when there is no wall, and it
//! amortizes — the meter comes due on a stride, not on every popped parent.

use super::*;
use crate::tdd::limits::{
    apply_limits, enable_reduce_deadline_check, reset_reduce_deadline_check_for_test,
    with_reduce_poll_stride,
};
use crate::vtree::Vtree;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A diagram with one contractible twin pair at an explicit level, and its root
/// declared dirty so the walk has a parent to pop. Same shape as
/// `twins_with_marginal_sibling_are_contracted` in `tests.rs`, reduced to what
/// these tests need: at least one iteration of the pop loop.
fn dirty_tdd() -> (Tdd, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let one = LocalNodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let a = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b = levels[v_left.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    levels[vl_left.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![TddNodeData::leaf(LeafLabel::One)];
    levels[v_right.idx()].make_marginal(vec![3u128], None);
    let sib_slot0 = LocalNodeIdx(MargRef::slot_raw(0));
    levels[root.idx()].push_internal_node(&[
        InputPair { left: a, right: sib_slot0 },
        InputPair { left: b, right: sib_slot0 },
    ]);

    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = Tdd::with_levels(vtree, levels, output);
    tag_all_marg_side_slots(&mut tdd, None);
    tdd.scratch.dirty_contract.push(root.0);
    (tdd, v_left)
}

/// Armed, with a wall already in the past, the walk cuts at its first metered
/// pop and reports it through `ApplyError::Deadline` — the same arm the compile
/// already takes when an apply runs out of wall. Stride 1 makes every pop a poll,
/// which is what "the first pop is metered" means.
#[test]
fn armed_expired_wall_cuts_the_contract_walk() {
    let _thr = set_marg_inline_max(0);
    let (mut tdd, _) = dirty_tdd();

    let r = {
        let _lim = apply_limits()
            .deadline(Some(Instant::now() - Duration::from_secs(1)))
            .apply();
        enable_reduce_deadline_check();
        with_reduce_poll_stride(1, || contract_all_twins_topdown(&mut tdd, None))
    };
    reset_reduce_deadline_check_for_test();

    assert!(
        matches!(r, Err(ApplyError::Deadline)),
        "a wall in the past must surface Deadline, not run the walk to completion; got {r:?}",
    );
    // The cut is resumable, not a loss: the popped parent went back on the
    // worklist, so a later minimize finishes the contraction this one abandoned.
    assert!(
        !tdd.scratch.dirty_contract.is_empty(),
        "a cut walk must hand its unprocessed parents back to dirty_contract",
    );
}

/// Armed but with no wall installed, the walk runs to completion: the poll reads
/// the caller's deadline cell, and `None` there is the shield every non-canopy
/// compile path holds.
#[test]
fn armed_without_a_wall_completes() {
    let _thr = set_marg_inline_max(0);
    let (mut tdd, v_left) = dirty_tdd();

    let r = {
        let _lim = apply_limits().deadline(None).apply();
        enable_reduce_deadline_check();
        with_reduce_poll_stride(1, || contract_all_twins_topdown(&mut tdd, None))
    };
    reset_reduce_deadline_check_for_test();

    r.expect("no wall → the walk must complete");
    assert_eq!(
        tdd.levels[v_left.idx()].width(),
        1,
        "the completed walk must still contract the twins",
    );
}

/// Disarmed — every compile outside the DPLL-canopy stage — a wall in the past is
/// invisible to the walk. This is the bit-identical-when-unset property: the
/// knob's gate is checked before the clock is ever read.
#[test]
fn disarmed_ignores_an_expired_wall() {
    let _thr = set_marg_inline_max(0);
    let (mut tdd, v_left) = dirty_tdd();

    reset_reduce_deadline_check_for_test();
    let r = {
        let _lim = apply_limits()
            .deadline(Some(Instant::now() - Duration::from_secs(1)))
            .apply();
        with_reduce_poll_stride(1, || contract_all_twins_topdown(&mut tdd, None))
    };

    r.expect("disarmed → an expired wall must not cut the walk");
    assert_eq!(
        tdd.levels[v_left.idx()].width(),
        1,
        "the disarmed walk must behave exactly as it did before the poll existed",
    );
}

/// The poll is amortized, not per-iteration: with a stride wider than the whole
/// walk's work, an expired wall goes unnoticed and the walk completes. Paired
/// with `armed_expired_wall_cuts_the_contract_walk` (same fixture, same expired
/// wall, stride 1) this pins that the stride — not the arming — is what decides
/// when the clock is read, which is the property the production cadence rests on.
/// The production stride is untouched; the cadence is pinned per-test.
#[test]
fn a_stride_wider_than_the_walk_never_polls() {
    let _thr = set_marg_inline_max(0);
    let (mut tdd, v_left) = dirty_tdd();

    let r = {
        let _lim = apply_limits()
            .deadline(Some(Instant::now() - Duration::from_secs(1)))
            .apply();
        enable_reduce_deadline_check();
        with_reduce_poll_stride(u64::MAX, || contract_all_twins_topdown(&mut tdd, None))
    };
    reset_reduce_deadline_check_for_test();

    r.expect("a stride the walk never reaches must not read the clock at all");
    assert_eq!(tdd.levels[v_left.idx()].width(), 1, "the unpolled walk must still contract");
}
