//! Levels whose cells are read by g-pair vectors, checked against the row
//! loop that reads every candidate off the child grids: the same diagram,
//! each node's pairs in the same order, and the same work.

use super::*;
use super::pair_rows::subsets;
use crate::Engine;
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::limits::{LimitConfig, SparseRoute};
use crate::vtree::VarId;

/// Conjoin `f` and `g` on a fresh engine, every level on a dense route,
/// and the work it took.
fn conjoin(f: &Tdd, g: &Tdd) -> (Tdd, u64) {
    let eng = Engine::new();
    let _dense = eng.limits().scope(LimitConfig::none().with_sparse_route(SparseRoute { sparsity: 64, min_grid: usize::MAX }));
    let mark = eng.limits().mark();
    let out = eng.and(f.clone(), g.clone()).unwrap();
    (out, eng.limits().work_since(mark))
}

/// Conjoined with every level that can read its cells by vectors reading
/// them, with the levels that pay reading them, and with none: the same
/// levels, each node's pairs in the same order, and the same work.
fn check(f: &Tdd, g: &Tdd, what: &str) {
    let (oracle, work) = word_cells_off(|| conjoin(f, g));
    let eng = Engine::new();
    for (name, (out, out_work)) in [("every level", word_cells_on(|| conjoin(f, g))), ("the levels that pay", conjoin(f, g))] {
        assert!(eng.equivalent(&out, &oracle).unwrap(), "{what}: {name} read by vectors built another function");
        assert!(is_self_conjunction(&out, &oracle), "{what}: {name} read by vectors built other levels");
        assert_eq!(out_work, work, "{what}: {name} read by vectors took other work");
    }
}

/// Random formulas over 18 variables on every vtree shape, dense random
/// functions over 10 and 14 variables, whose levels hold many pairs, and
/// random functions over 12 variables that leave alternate variables under
/// a vtree node to each operand, whose parent level has a complete side:
/// the levels read by vectors build what the row loop builds.
#[test]
fn the_levels_read_by_vectors_build_what_the_row_loop_builds() {
    let mut rng = Lcg::new(0x77d0_c311);
    let before = word_cells_levels();
    let mut cases = 0usize;
    for shape in [CnfShape { clauses: 30, width: 3 }, CnfShape { clauses: 60, width: 4 }] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..3 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                check(&f, &g, &format!("{name}, {shape:?}, case {case}"));
                cases += 1;
            }
        }
    }
    for n in [10u32, 14] {
        let vars: Vec<VarId> = (1..=n).map(VarId).collect();
        for (name, vtree) in vtree_shapes(n) {
            for one_in in [2, 3, 8] {
                let mut operand = || {
                    let models: Vec<u64> = (0..1u64 << n).filter(|_| rng.below(one_in) == 0).collect();
                    Tdd::from_models(&vtree, &vars, &models).unwrap()
                };
                let (f, g) = (operand(), operand());
                check(&f, &g, &format!("{name}, random functions of {n} variables, 1 in {one_in}"));
                cases += 1;
            }
        }
    }
    let n = 12u32;
    let vars: Vec<VarId> = (1..=n).map(VarId).collect();
    for (name, vtree) in vtree_shapes(n) {
        let root = vtree.root();
        for u in vtree.subtree(root).collect::<Vec<_>>() {
            if u == root || vtree.node(u).is_leaf() || vtree.node(u).parent() == Some(root) {
                continue;
            }
            let leaves: Vec<u64> = vtree.subtree(u).filter(|&v| vtree.node(v).is_leaf())
                .map(|v| 1u64 << (vtree.leaf_var(v).0 - 1)).collect();
            let free_f: u64 = leaves.iter().step_by(2).sum();
            let free_g: u64 = leaves.iter().skip(1).step_by(2).sum();
            for (fi, gi) in [(2u64, 3u64), (2, 16)] {
                let mut operand = |one_in: u64, free: u64| {
                    let models: Vec<u64> = (0..1u64 << n).filter(|m| m & free == 0).filter(|_| rng.below(one_in) == 0)
                        .flat_map(|m| subsets(free).map(move |s| m | s)).collect();
                    Tdd::from_models(&vtree, &vars, &models).unwrap()
                };
                let (f, g) = (operand(fi, free_f), operand(gi, free_g));
                check(&f, &g, &format!("{name}, supports split under {u:?}, 1 in {fi} and {gi}"));
                check(&g, &f, &format!("{name}, supports split under {u:?}, 1 in {gi} and {fi}"));
                cases += 2;
            }
        }
    }
    let after = word_cells_levels();
    let [grids, complete, grouped, rows, cells] = std::array::from_fn(|k| after[k] - before[k]);
    eprintln!("levels read by vectors: {grids} of two grids, {complete} with a complete side, {grouped} grouped cells, {rows} rows ({cells} walking their live cells), over {cases} cases");
    assert!(grids > 0, "expected levels of two grids read by vectors, got none over {cases} cases");
    assert!(complete > 0, "expected levels with a complete side read by vectors, got none over {cases} cases");
    assert!(grouped > 0, "expected grouped cells read by vectors, got none over {cases} cases");
    assert!(rows > 0, "expected rows read by vectors, got none over {cases} cases");
    assert!(cells > 0, "expected rows walking their live cells alone, got none over {cases} cases");
}
