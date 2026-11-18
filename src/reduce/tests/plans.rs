//! What each public reduction plan and content-twin policy leaves behind.

use std::sync::Arc;

use crate::diagram::{NEG_LEAF_IDX, POS_LEAF_IDX, Tdd};
use crate::reduce::{ContentTwinPolicy, ContentTwinSchedule, ReductionPlan};
use crate::test_helpers::check::marginal::check_no_orphan_slots;
use crate::test_helpers::{
    CnfShape, Lcg, assert_canonical, assert_same_shape, compile_clauses, compile_clauses_on, rand_cnf,
    same_as_stored, toy, vtree_shapes, with_floor,
};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

use super::edited_marginal_diagram;

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

/// Vtree ((x h) (y1 y2)) with h summed out. Scanning between the two
/// conjunctions merges h's two classes before x can separate them, and no
/// rewrite splits them again; without scans both orders meet.
#[test]
fn a_content_scan_between_conjunctions_moves_a_count_diagram_s_fixpoint() {
    let vtree = Arc::new(Vtree::balanced(4));
    let h = vtree.leaf_of(VarId(2)).unwrap();
    let (f1, f2) = (vec![vec![-2, -3], vec![2, 4]], vec![vec![-1, 4], vec![1, -3]]);
    let full = |f: &mut Tdd, scan: bool| {
        let twins = if scan { ContentTwinPolicy::Fresh } else { ContentTwinPolicy::Skip };
        f.reduce(ReductionPlan::Full(twins)).unwrap();
    };
    let summed = |clauses: &[Vec<i32>], scan: bool| {
        let mut f = compile_clauses(&vtree, clauses);
        f.marginalize_levels(&[h]).unwrap();
        full(&mut f, scan);
        f
    };
    let orders = |scan: bool| {
        let at_once = summed(&[f1.clone(), f2.clone()].concat(), scan);
        let mut staged = crate::and(summed(&f1, scan), compile_clauses(&vtree, &f2)).unwrap();
        full(&mut staged, scan);
        assert_eq!(at_once.model_count().unwrap(), staged.model_count().unwrap());
        (at_once, staged)
    };
    let (at_once, staged) = orders(true);
    assert_ne!(at_once.pair_count(), staged.pair_count(), "the scan's place moves the fixpoint");
    let (at_once, staged) = orders(false);
    assert_same_shape(&at_once, &staged, "both orders without scans");
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

/// The close at a plan's end changes how levels hold their pairs, not the
/// diagram: a canonical diagram stays certified through the prune and
/// contract plans, at the floor and at a floor of two pairs.
#[test]
fn the_prune_and_contract_plans_keep_a_canonical_diagram_certified() {
    for floor in [crate::diagram::FLOOR, 2] {
        with_floor(floor, || {
            let vtree = Arc::new(Vtree::balanced(5));
            let mut f = compile_clauses(&vtree, &[vec![1, 2, 3], vec![-2, 4], vec![3, -5], vec![-1, 5]]);
            f.minimize().unwrap();
            assert!(f.levels.is_canonical(f.output), "a minimized diagram is certified");
            for plan in [ReductionPlan::Prune, ReductionPlan::Contract] {
                f.reduce(plan).unwrap();
                assert!(f.levels.is_canonical(f.output), "floor {floor}: a plan's close forgot the certificate");
            }
        });
    }
}

/// The contract plan on unminimized conjunctions, on implicit levels and on
/// stored ones alike ([`same_as_stored`]).
#[test]
fn the_contract_plan_leaves_the_diagram_the_stored_route_leaves() {
    same_as_stored(|| {
        let eng = Engine::new();
        let mut rng = Lcg::new(0x6c0e_77a1);
        let mut out = Vec::new();
        for num_vars in [4u32, 6, 8] {
            for (_name, vtree) in vtree_shapes(num_vars) {
                for _ in 0..6 {
                    let mut operand = || {
                        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 5, width: 3 });
                        compile_clauses_on(&eng, &vtree, &clauses)
                    };
                    let (f, g) = (operand(), operand());
                    let mut h = eng.and(f, g).unwrap();
                    eng.reduce(&mut h, ReductionPlan::Contract).unwrap();
                    out.push(h);
                }
            }
        }
        out
    });
}
