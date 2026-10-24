//! The row loop of one-pair cells, checked against the row loop of any
//! level it replaces: the same diagram.

use super::*;
use crate::Engine;
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::limits::{LimitConfig, SparseRoute};
use crate::vtree::{VarId, Vtree};

/// Conjoin `f` and `g` on a fresh engine, every level on a dense route.
fn conjoin(f: &Tdd, g: &Tdd) -> Tdd {
    let eng = Engine::new();
    let _dense = eng.limits().scope(LimitConfig::none().with_sparse_route(SparseRoute { sparsity: 64, min_grid: usize::MAX }));
    eng.and(f.clone(), g.clone()).unwrap()
}

/// Conjoined with the loop of one-pair cells and without: the same levels,
/// each node's pairs in the same order. Returns the levels that loop built.
fn check(f: &Tdd, g: &Tdd, what: &str) -> u64 {
    let oracle = single_rows_off(|| conjoin(f, g));
    let before = single_rows_levels();
    let out = conjoin(f, g);
    let levels = single_rows_levels() - before;
    let eng = Engine::new();
    assert!(eng.equivalent(&out, &oracle).unwrap(), "{what}: the loop of one-pair cells built another function");
    assert!(is_self_conjunction(&out, &oracle), "{what}: the loop of one-pair cells built other levels");
    levels
}

/// Random formulas over 18 variables on every vtree shape, the narrowest
/// of mostly units and pairs of literals, whose levels are mostly of
/// one-pair nodes, and random
/// functions over 12 variables on a linear vtree, sparse ones of 1 model
/// in 16 or 64: where both operands' nodes at a level hold one pair, the
/// loop of one-pair cells builds the level the row loop of any level does.
#[test]
fn the_loop_of_one_pair_cells_builds_the_levels_of_the_general_loop() {
    let mut rng = Lcg::new(0x51d6_7a03);
    let (mut levels, mut cases) = (0u64, 0usize);
    for shape in [CnfShape { clauses: 12, width: 2 }, CnfShape { clauses: 20, width: 3 }, CnfShape { clauses: 40, width: 4 }] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..4 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                levels += check(&f, &g, &format!("{name}, {shape:?}, case {case}"));
                cases += 1;
            }
        }
    }
    let vtree = Arc::new(Vtree::linear(12));
    let vars: Vec<VarId> = (1..=12).map(VarId).collect();
    for case in 0..6 {
        let one_in = [16, 64, 256][case % 3];
        let mut operand = || {
            let models: Vec<u64> = (0..1u64 << 12).filter(|_| rng.below(one_in) == 0).collect();
            Tdd::from_models(&vtree, &vars, &models).unwrap()
        };
        let (f, g) = (operand(), operand());
        levels += check(&f, &g, &format!("random functions, case {case}"));
        cases += 1;
    }
    eprintln!("levels built by the loop of one-pair cells {levels} over {cases} cases");
    assert!(levels > 0, "expected levels of one-pair cells, got none over {cases} cases");
}
