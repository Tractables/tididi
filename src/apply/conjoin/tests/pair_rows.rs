//! Masked levels read by pairs, checked against the row loop of any level
//! on the same masks: the same diagram.

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

/// Conjoined with levels read by pairs and without: the same levels, each
/// node's pairs in the same order. Returns the levels read by pairs, and
/// those of them with a complete side.
fn check(f: &Tdd, g: &Tdd, what: &str) -> (u64, u64) {
    let oracle = pair_rows_off(|| conjoin(f, g));
    let (all, whole) = pair_rows_levels();
    let out = conjoin(f, g);
    let eng = Engine::new();
    assert!(eng.equivalent(&out, &oracle).unwrap(), "{what}: the levels read by pairs built another function");
    assert!(is_self_conjunction(&out, &oracle), "{what}: the levels read by pairs built other levels");
    let (all_after, whole_after) = pair_rows_levels();
    (all_after - all, whole_after - whole)
}

/// Random formulas over 18 variables on every vtree shape, alone and
/// against formulas of a few clauses, sparse random functions over 14
/// variables on a linear vtree, dense ones over 10 variables on every
/// shape, and random functions over 14 variables that leave alternate
/// variables under a vtree node to each operand: the levels read by pairs
/// build what the row loop builds.
#[test]
fn the_levels_read_by_pairs_build_what_the_row_loop_builds() {
    let mut rng = Lcg::new(0x5a1d_93e7);
    let (mut levels, mut cases) = ((0u64, 0u64), 0usize);
    let mut add = |(a, b): (u64, u64)| { levels.0 += a; levels.1 += b; };
    for shape in [CnfShape { clauses: 30, width: 3 }, CnfShape { clauses: 60, width: 4 }] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..4 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                add(check(&f, &g, &format!("{name}, {shape:?}, case {case}")));
                cases += 1;
            }
        }
    }
    // Against a formula of a few clauses, whose levels have few pairs and
    // often a child level each of whose products with the other's is live.
    for small in [2, 4, 8] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..2 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, CnfShape { clauses: 40, width: 3 }));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, CnfShape { clauses: small, width: 3 }));
                add(check(&f, &g, &format!("{name}, against {small} clauses, case {case}")));
                add(check(&g, &f, &format!("{name}, {small} clauses against, case {case}")));
                cases += 2;
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
        add(check(&f, &g, &format!("sparse random functions, case {case}")));
        cases += 1;
    }
    let vars: Vec<VarId> = (1..=10).map(VarId).collect();
    for (name, vtree) in vtree_shapes(10) {
        for one_in in [2, 3] {
            let mut operand = || {
                let models: Vec<u64> = (0..1u64 << 10).filter(|_| rng.below(one_in) == 0).collect();
                Tdd::from_models(&vtree, &vars, &models).unwrap()
            };
            let (f, g) = (operand(), operand());
            add(check(&f, &g, &format!("{name}, dense random functions, 1 in {one_in}")));
            cases += 1;
        }
    }
    // Operands each ignoring alternate variables under one vtree node:
    // there every product is live, and its parent level has a complete
    // side beside one that kills.
    let n = 14u32;
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
            for (fi, gi) in [(2u64, 8u64), (3, 32), (2, 64)] {
                let mut operand = |one_in: u64, free: u64| {
                    let models: Vec<u64> = (0..1u64 << n).filter(|m| m & free == 0).filter(|_| rng.below(one_in) == 0)
                        .flat_map(|m| subsets(free).map(move |s| m | s)).collect();
                    Tdd::from_models(&vtree, &vars, &models).unwrap()
                };
                let (f, g) = (operand(fi, free_f), operand(gi, free_g));
                add(check(&f, &g, &format!("{name}, supports split under {u:?}, 1 in {fi} and {gi}")));
                add(check(&g, &f, &format!("{name}, supports split under {u:?}, 1 in {gi} and {fi}")));
                cases += 2;
            }
        }
    }
    let (all, whole) = levels;
    eprintln!("levels read by pairs {all} ({whole} with a complete side), over {cases} cases");
    assert!(all > 0, "expected levels read by pairs, got none over {cases} cases");
    assert!(whole > 0, "expected levels read by pairs with a complete side, got none over {cases} cases");
}

/// The subsets of `free`.
pub(super) fn subsets(free: u64) -> impl Iterator<Item = u64> {
    let mut next = Some(free);
    std::iter::from_fn(move || {
        let s = next?;
        next = (s != 0).then(|| (s - 1) & free);
        Some(s)
    })
}
