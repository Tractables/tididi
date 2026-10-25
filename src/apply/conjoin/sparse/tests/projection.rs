//! Levels whose candidates outnumber their operands' pairs several times
//! over: a level of many f nodes of a few pairs each, and a root of one
//! node a side walked by outer key. The walk grows the level's pair arena
//! to its projected candidates instead of doubling it at each fill, and
//! the level is the dense grid's.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};

use crate::apply::conjoin::route::Route;
use crate::apply::conjoin::tests::levels_built;
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
    // grow the arena three times.
    assert_eq!(from_sparse.levels[ab.idx()].live_pairs(), 64 * 63);
    assert!(levels.contains(&(ab, Route::Sparse, 0)), "the level's arena outgrew its projection: {levels:?}");
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
fn a_root_of_many_more_candidates_than_pairs_grows_only_by_projection() {
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let vtree = vtree();
    let root = vtree.root();
    let (from_sparse, levels) = root_join(&Engine::new(), &vtree, sparse);
    let (from_dense, _) = root_join(&Engine::new(), &vtree, dense);
    assert!(from_sparse.equivalent(&from_dense).unwrap());
    // About 63 candidates an f pair against a seed of 4,032 pairs: doubling
    // would grow the arena six times.
    assert_eq!(from_sparse.levels[root.idx()].live_pairs(), 254_016);
    assert!(levels.contains(&(root, Route::Sparse, 0)), "the root's arena outgrew its projection: {levels:?}");
}
