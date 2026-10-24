//! Masked rows that read their live columns off the reach columns, checked
//! against the scan of the reach masks they replace: the same diagram.

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

/// Conjoined with reach columns and without: the same levels, each node's
/// pairs in the same order. Returns the rows read off the reach columns.
fn check(f: &Tdd, g: &Tdd, what: &str) -> u64 {
    let oracle = reach_columns_off(|| conjoin(f, g));
    let before = reach_columns_rows();
    let out = conjoin(f, g);
    let rows = reach_columns_rows() - before;
    let eng = Engine::new();
    assert!(eng.equivalent(&out, &oracle).unwrap(), "{what}: the reach columns built another function");
    assert!(is_self_conjunction(&out, &oracle), "{what}: the reach columns built other levels");
    rows
}

/// Random formulas over 18 variables on every vtree shape, and random
/// functions over 14 variables on a linear vtree, whose middle levels have
/// a few hundred nodes of two pairs, sparse ones of 1 model in 16 or 64
/// conjoining to false on most cells: rows read off the reach columns
/// enter the cells the scan enters.
#[test]
fn the_reach_columns_enter_the_cells_the_scan_enters() {
    let mut rng = Lcg::new(0x7e4c_0a19);
    let (mut rows, mut cases) = (0u64, 0usize);
    for shape in [CnfShape { clauses: 30, width: 3 }, CnfShape { clauses: 60, width: 4 }] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..4 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                rows += check(&f, &g, &format!("{name}, {shape:?}, case {case}"));
                cases += 1;
            }
        }
    }
    let vtree = Arc::new(Vtree::linear(14));
    let vars: Vec<VarId> = (1..=14).map(VarId).collect();
    for case in 0..6 {
        let one_in = [4, 16, 64][case % 3];
        let mut operand = || {
            let models: Vec<u64> = (0..1u64 << 14).filter(|_| rng.below(one_in) == 0).collect();
            Tdd::from_models(&vtree, &vars, &models).unwrap()
        };
        let (f, g) = (operand(), operand());
        rows += check(&f, &g, &format!("random functions, case {case}"));
        cases += 1;
    }
    eprintln!("rows read off the reach columns {rows} over {cases} cases");
    assert!(rows > 0, "expected rows read off the reach columns, got none over {cases} cases");
}
