//! Stops inside the levels of the dense routes that charge a row, or a
//! cell, of the row loop in one poll, checked against the row loop's own
//! polls: under a stop at any work, the same outcome, stopped or not, at
//! the same work. The gates come due every few units, inside the rows of
//! small diagrams.

use super::*;
use crate::Engine;
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::vtree::{VarId, Vtree};

/// The units the dense levels' gates come due at.
const STRIDE: u64 = 1 << 6;

/// The stops a case is run under: one at each `k / CUTS` of its work.
const CUTS: u64 = 24;

/// A route a case is checked under: its name, and the conjunction under a
/// stop, as [`conjoin`] answers it.
type Arm<'a> = (&'a str, &'a dyn Fn(Option<u64>) -> (bool, u64));

/// Conjoin `f` and `g` on a fresh engine, every level on a dense route,
/// under a stop at `stop` work units when given: whether it finished, and
/// the work it took.
fn conjoin(f: &Tdd, g: &Tdd, stop: Option<u64>) -> (bool, u64) {
    let eng = Engine::new();
    let rules = StopRules { unconditional: stop.map(StopAt::WorkUnits), after_pairs: None };
    let config = LimitConfig::none()
        .with_sparse_route(SparseRoute { sparsity: 64, min_grid: usize::MAX })
        .with_stop_rules(rules);
    let _scope = eng.limits().scope(config);
    let done = with_dense_poll_stride(STRIDE, || eng.and(f.clone(), g.clone())).is_ok();
    (done, eng.limits().work_units())
}

/// Conjoined with every level that can read its cells by g-pair vectors
/// reading them (once with their rows written off their sorted live
/// candidates), with the levels read by pairs and the rows of one-pair
/// cells and no vectors, and with the routes that pay, against the row
/// loop alone: without a stop and under a stop at each of [`CUTS`] works
/// spread over the conjunction's, the same outcome at the same work.
/// Returns the stops that fell.
fn check(f: &Tdd, g: &Tdd, what: &str) -> usize {
    let oracle = |stop| word_cells_off(|| pair_rows_off(|| single_rows_off(|| conjoin(f, g, stop))));
    let arms: [Arm<'_>; 4] = [
        ("every level read by vectors", &|stop| word_cells_on(|| conjoin(f, g, stop))),
        ("every level read by vectors, rows sorted,", &|stop| word_cells_on(|| word_rows_sorted(|| conjoin(f, g, stop)))),
        ("levels read by pairs and one-pair rows", &|stop| word_cells_off(|| conjoin(f, g, stop))),
        ("the routes that pay", &|stop| conjoin(f, g, stop)),
    ];
    let (done, total) = oracle(None);
    assert!(done, "{what}: the conjunction stopped with no stop");
    for (name, arm) in &arms {
        assert_eq!(arm(None), (true, total), "{what}: {name} took other work");
    }
    let mut stopped = 0;
    for k in 1..CUTS {
        let cut = total * k / CUTS;
        let want = oracle(Some(cut));
        stopped += usize::from(!want.0);
        for (name, arm) in &arms {
            assert_eq!(arm(Some(cut)), want, "{what}: {name} under a stop at {cut} of {total} units");
        }
    }
    stopped
}

/// Random formulas over 16 variables on every vtree shape, the narrowest
/// of units and pairs of literals, whose levels are mostly of one-pair
/// nodes, sparse random functions over 14 variables on a linear vtree and
/// dense ones over 10 variables on every shape: each route that charges a
/// row or a cell in one poll stops where the row loop does.
#[test]
fn a_stop_inside_a_dense_level_falls_where_the_row_loop_stops() {
    let mut rng = Lcg::new(0x570b_e1a5);
    let (words, pairs, single) = (word_cells_levels()[3], pair_rows_levels()[0], single_rows_levels());
    let (mut cases, mut stopped) = (0usize, 0usize);
    for shape in [CnfShape { clauses: 12, width: 2 }, CnfShape { clauses: 30, width: 3 }, CnfShape { clauses: 50, width: 4 }] {
        for (name, vtree) in vtree_shapes(16) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            stopped += check(&f, &g, &format!("{name}, {shape:?}"));
            cases += 1;
        }
    }
    let vtree = Arc::new(Vtree::linear(14));
    let vars: Vec<VarId> = (1..=14).map(VarId).collect();
    for one_in in [4, 16, 64] {
        let mut operand = || {
            let models: Vec<u64> = (0..1u64 << 14).filter(|_| rng.below(one_in) == 0).collect();
            Tdd::from_models(&vtree, &vars, &models).unwrap()
        };
        let (f, g) = (operand(), operand());
        stopped += check(&f, &g, &format!("random functions of 14 variables, 1 in {one_in}"));
        cases += 1;
    }
    let vars: Vec<VarId> = (1..=10).map(VarId).collect();
    for (name, vtree) in vtree_shapes(10) {
        let mut operand = || {
            let models: Vec<u64> = (0..1u64 << 10).filter(|_| rng.below(2) == 0).collect();
            Tdd::from_models(&vtree, &vars, &models).unwrap()
        };
        let (f, g) = (operand(), operand());
        stopped += check(&f, &g, &format!("{name}, dense random functions of 10 variables"));
        cases += 1;
    }
    let (words, pairs, single) = (word_cells_levels()[3] - words, pair_rows_levels()[0] - pairs, single_rows_levels() - single);
    eprintln!("{stopped} stops over {cases} cases; rows read by vectors {words}, levels read by pairs {pairs}, of one-pair rows {single}");
    assert!(stopped > 10 * cases, "expected most cuts to stop, got {stopped} over {cases} cases");
    assert!(words > 0 && pairs > 0 && single > 0, "expected every route, got {words} rows by vectors, {pairs} levels by pairs and {single} of one-pair rows");
}
