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
