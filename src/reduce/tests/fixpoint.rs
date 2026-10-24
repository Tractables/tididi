//! What a whole-diagram minimize leaves behind: no pending work, and a
//! certificate the next minimize reads instead of running the passes again.

use std::sync::Arc;

use crate::diagram::Pass;
use crate::test_helpers::{assert_canonical, compile_clauses, test_cases, vtree_shapes};
use crate::vtree::Vtree;
use crate::Engine;

#[test]
fn a_minimized_diagram_owes_the_passes_nothing() {
    for (num_vars, clauses) in test_cases() {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let mut f = compile_clauses(&vtree, &clauses);
            f.minimize().unwrap();
            assert_canonical(&f);
            for pass in [Pass::Contract, Pass::LeafContract, Pass::ContentTwin] {
                assert!(f.dirty.levels(pass).is_empty(), "{shape}: {pass:?} still has entries");
            }
        }
    }
}

#[test]
fn a_second_minimize_of_a_certified_diagram_runs_no_pass() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = compile_clauses(&vtree, &[vec![1, 2, 3], vec![-2, 4], vec![3, -5, 6]]);
    eng.minimize(&mut f).unwrap();
    assert!(f.levels.is_canonical(f.output));
    let before = eng.limits().work_units();
    eng.minimize(&mut f).unwrap();
    assert_eq!(eng.limits().work_units(), before, "the prune walk was charged, so it ran");
    assert!(f.levels.is_canonical(f.output));
    assert_canonical(&f);
}

#[test]
fn an_edit_after_minimize_is_reduced_by_the_next_minimize() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3]]);
    eng.minimize(&mut f).unwrap();
    let g = compile_clauses(&vtree, &[vec![3, -4]]);
    let mut h = crate::apply::apply_and(f, g);
    assert!(!h.dirty.is_empty());
    let before = eng.limits().work_units();
    eng.minimize(&mut h).unwrap();
    assert!(eng.limits().work_units() > before);
    assert!(h.dirty.is_empty());
    assert_canonical(&h);
}

/// A contraction's merges push the levels they change onto every worklist,
/// but the sweep takes each to its fixpoint: none is left on the
/// contraction's own list, and contracting every level again merges nothing.
#[test]
fn a_contraction_leaves_none_of_its_merges_on_its_own_worklist() {
    use crate::reduce::contract::contract_all_twins;
    use crate::reduce::ReductionPlan;
    use crate::test_helpers::{rand_cnf, CnfShape, Lcg};
    let eng = Engine::new();
    let mut rng = Lcg::new(17);
    let mut merged = 0;
    for round in 0..24 {
        let num_vars = 6 + (round % 5) as u32;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 12, width: 3 });
        for (shape, vtree) in vtree_shapes(num_vars) {
            let mut f = eng.clause(&vtree, clauses[0].iter().copied()).unwrap();
            for clause in &clauses[1..] {
                let g = eng.clause(&vtree, clause.iter().copied()).unwrap();
                f = eng.and(f, g).unwrap();
                let before = f.levels.node_count();
                eng.reduce(&mut f, ReductionPlan::Contract).unwrap();
                merged += before - f.levels.node_count();
                assert!(f.dirty.levels(Pass::Contract).is_empty(), "{shape}: the contraction left itself work");
                let settled = f.levels.node_count();
                f.seed_contract_worklist(vtree.internal_bottomup().map(|(t, _, _)| t.0));
                contract_all_twins(&eng, &mut f).unwrap();
                assert_eq!(f.levels.node_count(), settled, "{shape}: a level the contraction left had twins");
            }
            let expected = compile_clauses(&vtree, &clauses);
            f.minimize().unwrap();
            assert_canonical(&f);
            assert!(eng.equivalent(&f, &expected).unwrap(), "{shape}: the contracted diagram changed its function");
        }
    }
    assert!(merged > 0, "no conjunction left a twin to merge");
}
