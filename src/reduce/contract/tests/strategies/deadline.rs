//! The reduce walk's mid-loop preemption point.
//!
//! `contract_all_twins` is the most expensive phase of a minimize and it
//! runs BETWEEN two applies of one bottom-up step, so before the poll a caller's
//! wall was observed only where the step ended. These tests pin the three
//! properties the cut is worth having for: it fires when the wall has passed,
//! it stays out of the way when no wall is installed, and it does not depend
//! on the gate coming due, because `joint_contract_fixpoint` tests
//! cancellation at every round.

use super::*;
use crate::diagram::{ValueRef, NodeIdx};

use crate::Engine;
use crate::vtree::Vtree;
use crate::test_helpers::deadline_probe;
use std::sync::Arc;

/// A diagram with one contractible twin pair at an explicit level, and its root
/// declared dirty so the walk has a parent to pop. Same shape as
/// `twins_with_marginal_sibling_are_contracted` in
/// `reduce/contract/tests/inline_denorm.rs`, reduced to what
/// these tests need: at least one iteration of the pop loop.
fn dirty_tdd() -> (Tdd, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let (vl_left, vl_right) = vtree.children(v_left);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let a = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let b = levels[v_left.idx()].push_internal_node(&[ChildPair::new(one, pos)]);
    levels[vl_left.idx()].nodes = vec![EncodedNode::leaf(LeafLabel::Pos)];
    levels[vl_right.idx()].nodes = vec![EncodedNode::leaf(LeafLabel::One)];
    levels[v_right.idx()].become_marginal(vec![3u128], None);
    let sib_slot0 = NodeIdx(ValueRef::slot_raw(0));
    levels[root.idx()].push_internal_node(&[
        ChildPair::new(a, sib_slot0),
        ChildPair::new(b, sib_slot0),
    ]);

    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    tag_all_marginal_side_slots(&mut tdd, None);
    tdd.seed_contract_worklist([root.0]);
    (tdd, v_left)
}

/// With a wall already in the past, the walk cuts at its first metered pop and
/// reports it through `OperationError::Stopped` — the same arm a caller already
/// takes when an apply runs out of wall. Stride 1 makes every pop a poll, which
/// is what "the first pop is metered" means.
#[test]
fn an_expired_wall_cuts_the_contract_walk() {
    let (mut tdd, _) = dirty_tdd();

    let r = deadline_probe(Some(1), |eng| contract_all_twins(eng, &mut tdd));
    assert!(
        matches!(r, Err(OperationError::Stopped)),
        "a wall in the past must surface Deadline, not run the walk to completion; got {r:?}",
    );
    // The cut is resumable, not a loss: the popped parent went back on the
    // worklist, so a later minimize finishes the contraction this one abandoned.
    assert!(
        !tdd.contract_worklist().is_empty(),
        "a cut walk must hand its unprocessed parents back to dirty_contract",
    );
}

/// With no wall installed the walk runs to completion: the poll reads the
/// caller's deadline cell, and `None` there means no cut — installing a deadline
/// is the only thing that arms the cut.
#[test]
fn no_wall_installed_completes() {
    let (mut tdd, v_left) = dirty_tdd();

    // Not `deadline_probe`: this is the one case whose engine must carry NO
    // wall, which is exactly what the probe installs.
    let r = {
        let eng = Engine::new();
        eng.limits().pin_reduce_poll_stride(Some(1));
        contract_all_twins(&eng, &mut tdd)
    };
    r.expect("no wall → the walk must complete");
    assert_eq!(
        tdd.levels[v_left.idx()].slot_count(),
        1,
        "the completed walk must still contract the twins",
    );
}

/// The amortized poll is not the only cut. `joint_contract_fixpoint` argues its
/// termination from a decreasing measure rather than bounding it by a count, so
/// it tests cancellation once per round; without that, a stride wider than the
/// whole walk's work would leave an expired wall unnoticed and a minimize
/// uninterruptible. Paired with `an_expired_wall_cuts_the_contract_walk` (same
/// fixture, same expired wall, stride 1) this pins that both cuts reach the
/// caller. The production stride is untouched; the cadence is pinned per-test.
#[test]
fn the_fixpoint_round_cuts_even_when_the_gate_never_comes_due() {
    let (mut tdd, _) = dirty_tdd();

    let r = deadline_probe(Some(u64::MAX), |eng| contract_all_twins(eng, &mut tdd));
    assert!(
        matches!(r, Err(OperationError::Stopped)),
        "a round boundary must cut whatever stride the gate carries; got {r:?}",
    );
}
