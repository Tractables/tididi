//! Levels whose candidates outnumber their operands' pairs several times
//! over: a level of many f nodes of a few pairs each, and a root of one
//! node a side walked by outer key. The walk grows the level's pair arena
//! once, to its projected candidates, instead of doubling it at each fill,
//! and the level is the dense grid's.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};

use crate::apply::conjoin::route::Route;
use crate::apply::conjoin::tests::{ahead_after, levels_ahead, levels_built, seeded_with, without_ahead};
use crate::test_helpers::assert_same_shape;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

/// The blocks `a`, `b`, `c` of the join below.
fn blocks() -> (Vec<VarId>, Vec<VarId>, Vec<VarId>) {
    (block(1, 6), block(7, 6), block(13, 6))
}

/// The vtree `((a b) c)`, every block a balanced subtree.
fn vtree() -> Arc<Vtree> {
    let (a, b, c) = blocks();
    let ab = Vtree::join(&Vtree::balanced_over(&a).unwrap(), &Vtree::balanced_over(&b).unwrap()).unwrap();
    Arc::new(Vtree::join(&ab, &Vtree::balanced_over(&c).unwrap()).unwrap())
}

/// `f(a, b, c) ∧ g(a, b)` on `((a b) c)`, each level's route and arena
/// growths. `f` is `b ≠ (a mod 8 + c) mod 64`, which at `(a b)` is 64 nodes
/// of eight pairs; `g` is `b ≠ a / 8 + 32`, one node there. Each of f's
/// nodes makes 63 candidates at `(a b)`.
fn crossed_join(eng: &Engine, vtree: &Arc<Vtree>, thresholds: SparseThresholds) -> (Tdd, Vec<(VtreeIdx, Route, u64)>) {
    let (a, b, c) = blocks();
    let mut f_rows = Vec::new();
    for z in 0..64u64 {
        for x in 0..64u64 {
            for y in 0..64u64 {
                if y != (x % 8 + z) % 64 {
                    f_rows.push(vec![x, y, z]);
                }
            }
        }
    }
    let g_rows: Vec<Vec<u64>> =
        (0..64u64).flat_map(|x| (0..64u64).filter(move |&y| y != x / 8 + 32).map(move |y| vec![x, y])).collect();
    let (fv, fr) = pack(&[&a, &b, &c], &f_rows);
    let (gv, gr) = pack(&[&a, &b], &g_rows);
    let f = eng.from_models(vtree, &fv, &fr).unwrap();
    let g = eng.from_models(vtree, &gv, &gr).unwrap();
    let ab = vtree.children(vtree.root()).0.idx();
    assert_eq!((f.levels[ab].node_count(), f.levels[ab].pairs.len()), (64, 512));

    // No arena recycled from the builds above, and the operands' clones
    // hold no room past their pairs: each level starts at its seed.
    eng.clear_scratch();
    let _forced = ForcedThresholds::install(thresholds);
    levels_built(|| eng.and(f.clone(), g.clone()).expect("an unarmed engine refuses nothing"))
}

#[test]
fn a_level_of_many_more_candidates_than_pairs_grows_its_arena_once() {
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let vtree = vtree();
    let ab = vtree.children(vtree.root()).0;
    let (from_sparse, levels) = crossed_join(&Engine::new(), &vtree, sparse);
    let (from_dense, _) = crossed_join(&Engine::new(), &vtree, dense);
    assert!(from_sparse.equivalent(&from_dense).unwrap());
    // Eight pairs a node against a seed of f's 512 pairs: doubling would
    // grow the arena three times, the projection grows it once.
    assert_eq!(from_sparse.levels[ab.idx()].live_pairs(), 64 * 63);
    assert!(levels.contains(&(ab, Route::Sparse, 1)), "the level's arena grew more than once: {levels:?}");
}

/// `f = c ≠ a` and `g = c ≠ b` on `((a b) c)`, with each level's route
/// and arena growths. At the root each has one node of 4,032 pairs, one
/// for each value of `c` and of the other block that differ, and each pair
/// of f meets the 62 or 63 pairs of g at its value of `c`: the one product
/// has 254,016 pairs.
fn root_join(eng: &Engine, vtree: &Arc<Vtree>, thresholds: SparseThresholds) -> (Tdd, Vec<(VtreeIdx, Route, u64)>) {
    let (a, b, c) = blocks();
    let mut f_rows = Vec::new();
    let mut g_rows = Vec::new();
    for x in 0..64u64 {
        for y in 0..64u64 {
            for z in 0..64u64 {
                if z != x {
                    f_rows.push(vec![x, y, z]);
                }
                if z != y {
                    g_rows.push(vec![x, y, z]);
                }
            }
        }
    }
    let (fv, fr) = pack(&[&a, &b, &c], &f_rows);
    let (gv, gr) = pack(&[&a, &b, &c], &g_rows);
    let f = eng.from_models(vtree, &fv, &fr).unwrap();
    let g = eng.from_models(vtree, &gv, &gr).unwrap();
    let root = vtree.root().idx();
    assert_eq!((f.levels[root].pairs.len(), g.levels[root].pairs.len()), (4032, 4032));

    eng.clear_scratch();
    let _forced = ForcedThresholds::install(thresholds);
    levels_built(|| eng.and(f.clone(), g.clone()).expect("an unarmed engine refuses nothing"))
}

#[test]
fn a_root_of_many_more_candidates_than_pairs_grows_its_arena_once() {
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let vtree = vtree();
    let root = vtree.root();
    let (from_sparse, levels) = root_join(&Engine::new(), &vtree, sparse);
    let (from_dense, _) = root_join(&Engine::new(), &vtree, dense);
    assert!(from_sparse.equivalent(&from_dense).unwrap());
    // About 63 candidates an f pair against a seed of 4,032 pairs: doubling
    // would grow the arena six times, a projection held to eight times the
    // arena's capacity twice.
    assert_eq!(from_sparse.levels[root.idx()].live_pairs(), 254_016);
    assert!(levels.contains(&(root, Route::Sparse, 1)), "the root's arena grew more than once: {levels:?}");
}

/// `f ∧ g` on a fresh engine under `thresholds`, the arenas left to grow
/// where `grown` holds, under a stop at `floor` output pairs when given:
/// the result and the work units it took.
fn joined(f: &Tdd, g: &Tdd, thresholds: SparseThresholds, grown: bool, floor: Option<u64>) -> (Result<Tdd, crate::OperationError>, u64) {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = Engine::new();
    let _forced = ForcedThresholds::install(thresholds);
    let run = || {
        let rules = StopRules { unconditional: None, after_pairs: floor.map(|p| (p, StopAt::WorkUnits(0))) };
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        eng.and(f.clone(), g.clone())
    };
    let out = if grown { without_ahead(run) } else { run() };
    (out, eng.limits().work_units())
}

/// `f ∧ g` under `thresholds`, its levels sized ahead once one in `share`
/// of their walk is done, against the same conjunction with its arenas
/// left to grow: the same diagram and work, and under a stop at each
/// output-pair floor the same stop at the same work.
fn same_sized_ahead(f: &Tdd, g: &Tdd, thresholds: SparseThresholds, share: usize) {
    let (oracle, oracle_work) = joined(f, g, thresholds, true, None);
    let (out, work) = ahead_after(share, || joined(f, g, thresholds, false, None));
    let (oracle, out) = (oracle.unwrap(), out.unwrap());
    assert_same_shape(&out, &oracle, "sized ahead");
    assert_eq!(work, oracle_work, "the levels sized ahead did other work");
    let total: usize = out.levels.iter().map(|l| l.pairs.len()).sum();
    let mut floor = 1u64;
    while floor <= 2 * total as u64 {
        let (oracle, oracle_work) = joined(f, g, thresholds, true, Some(floor));
        let (stopped, work) = ahead_after(share, || joined(f, g, thresholds, false, Some(floor)));
        assert_eq!(stopped.is_ok(), oracle.is_ok(), "a stop at {floor} pairs fell differently");
        assert_eq!(work, oracle_work, "a stop at {floor} pairs fell at other work");
        floor = floor * 3 / 2 + 1;
    }
}

/// The sparse leaf arm's levels of random conjunctions, sized ahead once
/// half their walk is done, against the same levels left to grow: the
/// meter charged as the arena would have grown, wherever the reservation
/// leaves the arena short or past its growth.
#[test]
fn small_leaf_levels_sized_ahead_build_and_stop_as_left_to_grow() {
    use crate::test_helpers::{rand_conj_over, Lcg};
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let mut rng = Lcg::new(0x5eed_a4ead);
    let before = levels_ahead();
    for (k, vtree) in [Vtree::linear(14), Vtree::random(14, 7), Vtree::balanced(14), Vtree::random(14, 11)].into_iter().enumerate() {
        let vtree = Arc::new(vtree);
        let vars: Vec<u32> = (1..=14).collect();
        for _ in 0..4 {
            let mut f = rand_conj_over(&vtree, &vars, 12, 3, false, &mut rng);
            let mut g = rand_conj_over(&vtree, &vars, 12, 3, false, &mut rng);
            f.minimize().unwrap();
            g.minimize().unwrap();
            same_sized_ahead(&f, &g, sparse, 2 + k);
        }
    }
    let after = levels_ahead();
    assert!(after[1] > before[1], "no leaf level was sized ahead: {after:?}");
}

/// Random formulas over 18 variables on every vtree shape, against
/// formulas of a few clauses, on the dense route with their pair arenas
/// seeded with at most 16 pairs, sized ahead once one in four of a level's
/// rows is written, against the same levels left to grow: the levels read
/// by pairs build, charge and stop as they would.
#[test]
fn small_dense_levels_sized_ahead_build_and_stop_as_left_to_grow() {
    use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let mut rng = Lcg::new(0x5eed_de45);
    let before = levels_ahead();
    for (_, vtree) in vtree_shapes(18) {
        for small in [2, 4, 8] {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, CnfShape { clauses: 40, width: 3 }));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, CnfShape { clauses: small, width: 3 }));
            seeded_with(16, || {
                same_sized_ahead(&f, &g, dense, 4);
                same_sized_ahead(&g, &f, dense, 4);
            });
        }
    }
    let after = levels_ahead();
    assert!(after[0] > before[0], "no dense level was sized ahead: {after:?}");
}
