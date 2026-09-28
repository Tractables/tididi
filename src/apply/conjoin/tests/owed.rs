//! A plain conjunction owes contraction the levels it built and what its
//! operands owed, not the levels it carried from an operand at its fixpoint;
//! minimizing from those worklists reaches the canonical diagram.
use std::sync::Arc;

use crate::diagram::Pass;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses, test_cases, vtree_shapes};
use crate::{Engine, Tdd, Vtree};

#[test]
fn operands_under_the_two_children_owe_only_the_root() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![-3, 4]]);
    let g = compile_clauses(&vtree, &[vec![5, -6], vec![7, 8]]);
    let mut out = eng.and(f, g).unwrap();
    for pass in [Pass::Contract, Pass::LeafContract] {
        let mut owed = out.dirty.take(pass);
        owed.sort_unstable();
        owed.dedup();
        assert_eq!(owed, vec![vtree.root().0], "{pass:?}");
    }
}

/// The clauses split in two, each half compiled on `vtree`; with `settled`
/// both halves are minimized, else each is a bare conjunction that still
/// owes its contraction.
fn halves(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>], settled: bool) -> (Tdd, Tdd) {
    let (a, b) = clauses.split_at(clauses.len() / 2);
    let half = |part: &[Vec<i32>]| {
        let mut f = eng.cube(vtree, std::iter::empty::<i32>()).unwrap();
        for c in part {
            f = eng.and(f, eng.clause(vtree, c.iter().copied()).unwrap()).unwrap();
        }
        if settled {
            eng.minimize(&mut f).unwrap();
        }
        f
    };
    (half(a), half(b))
}

#[test]
fn every_shape_minimizes_a_conjunction_to_the_canonical_diagram() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases() {
        for (label, vtree) in vtree_shapes(num_vars) {
            let expected = compile_clauses(&vtree, &clauses);
            for settled in [true, false] {
                let (f, g) = halves(&eng, &vtree, &clauses, settled);
                let mut out = eng.and(f, g).unwrap();
                eng.minimize(&mut out).unwrap();
                assert_canonical(&out);
                assert_same_shape(&out, &expected, &format!("{label}, settled {settled}"));
            }
        }
    }
}

#[test]
fn seeded_conjunctions_minimize_to_the_canonical_diagram() {
    use crate::test_helpers::{CnfShape, Lcg, rand_cnf};
    let eng = Engine::new();
    for seed in 0..60u64 {
        let num_vars = 6 + (seed % 5) as u32;
        let mut rng = Lcg::new(seed);
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 14, width: 3 });
        for (label, vtree) in vtree_shapes(num_vars) {
            let expected = compile_clauses(&vtree, &clauses);
            for settled in [true, false] {
                let (f, g) = halves(&eng, &vtree, &clauses, settled);
                let mut out = eng.and(f, g).unwrap();
                eng.minimize(&mut out).unwrap();
                assert_same_shape(&out, &expected, &format!("seed {seed}, {label}, settled {settled}"));
            }
        }
    }
}
