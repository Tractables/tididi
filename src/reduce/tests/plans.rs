//! What each public reduction plan and content-twin policy leaves behind.

use std::sync::Arc;

use crate::diagram::{NEG_LEAF_IDX, POS_LEAF_IDX, Tdd};
use crate::reduce::{ContentTwinPolicy, ContentTwinSchedule, ReductionPlan};
use crate::test_helpers::check::marginal::check_no_orphan_slots;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses, toy};
use crate::vtree::{Vtree, VtreeIdx};
use crate::Engine;

/// A boundary store of three slots, of which the root node names two.
fn with_an_orphaned_slot() -> Tdd {
    let f = toy(vec![5, 7, 9], &[&[(POS_LEAF_IDX.0, 0), (NEG_LEAF_IDX.0, 2)]]);
    assert!(check_no_orphan_slots(&f).is_err(), "the fixture must start with an orphan");
    f
}

#[test]
fn a_full_reduce_without_a_content_scan_drops_orphaned_slots() {
    let mut f = with_an_orphaned_slot();
    let count = f.model_count().unwrap();
    f.reduce(ReductionPlan::Full(ContentTwinPolicy::Skip)).unwrap();
    check_no_orphan_slots(&f).unwrap();
    assert_eq!(f.model_count().unwrap(), count);
}

#[test]
fn a_prune_drops_orphaned_slots() {
    let mut f = with_an_orphaned_slot();
    let count = f.model_count().unwrap();
    f.reduce(ReductionPlan::Prune).unwrap();
    check_no_orphan_slots(&f).unwrap();
    assert_eq!(f.model_count().unwrap(), count);
}

/// A count-marginal diagram straight out of a conjunction, still owing every
/// pass.
fn edited_marginal_diagram(eng: &Engine) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(8));
    let mut g = eng.clause(&vtree, [1, 5]).unwrap();
    let (left, _) = vtree.children(vtree.root());
    let summed: Vec<VtreeIdx> = vtree.internal_bottomup_slice().iter().copied()
        .filter(|&t| { let mut cur = t; loop {
            if cur == left { break true; }
            match vtree.node(cur).parent() { Some(p) => cur = p, None => break false }
        } })
        .collect();
    eng.marginalize_levels(&mut g, &summed).unwrap();
    let f = eng.and(g, eng.literal(&vtree, 8).unwrap()).unwrap();
    assert!(f.has_marginal_level() && !f.dirty.is_empty());
    f
}

#[test]
fn the_adaptive_policy_reaches_the_fresh_policy_s_diagram() {
    let eng = Engine::new();
    let f = edited_marginal_diagram(&eng);
    let mut fresh = f.clone();
    eng.reduce(&mut fresh, ReductionPlan::Full(ContentTwinPolicy::Fresh)).unwrap();
    assert_canonical(&fresh);
    let mut schedule = ContentTwinSchedule::default();
    let mut adaptive = f;
    eng.reduce(&mut adaptive, ReductionPlan::Full(ContentTwinPolicy::Adaptive(&mut schedule))).unwrap();
    assert_canonical(&adaptive);
    assert_eq!(schedule.next_scan_at_nodes, 0, "a small diagram is scanned on every call");
    assert_same_shape(&fresh, &adaptive, "fresh against adaptive");
}

#[test]
fn the_contract_plan_drains_its_worklist_and_a_full_plan_finishes_the_job() {
    let vtree = Arc::new(Vtree::balanced(5));
    let clauses = [vec![1, 2, 3], vec![-2, 4], vec![3, -5], vec![-1, 5]];
    let mut stepwise = compile_clauses(&vtree, &clauses);
    stepwise.reduce(ReductionPlan::Contract).unwrap();
    assert!(stepwise.contract_worklist().is_empty(), "the contract plan owes contraction nothing");
    stepwise.minimize().unwrap();
    assert_canonical(&stepwise);
    let mut direct = compile_clauses(&vtree, &clauses);
    direct.minimize().unwrap();
    assert_same_shape(&stepwise, &direct, "contract then minimize against minimize");
}
