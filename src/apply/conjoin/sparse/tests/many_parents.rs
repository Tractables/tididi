//! A level with an f parent per key, each holding one pair, against one g
//! parent: each f parent hands the emit at most one candidate, and the
//! output must be the dense grid's whether the walk writes the candidates
//! to the level's pair arena or hands them to the emit as one list, its
//! lookups pipelined or not.
use std::sync::Arc;

use num_bigint::BigUint;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};

use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// The blocks `a`, `b`, `c` of the join below.
fn blocks() -> (Vec<VarId>, Vec<VarId>, Vec<VarId>) {
    (block(1, 4), block(5, 3), block(8, 4))
}

/// The vtree `((a b) c)`, every block a balanced subtree.
fn vtree() -> Arc<Vtree> {
    let (a, b, c) = blocks();
    let ab = Vtree::join(
        &Vtree::balanced_over(&a).unwrap(),
        &Vtree::balanced_over(&b).unwrap(),
    )
    .unwrap();
    Arc::new(Vtree::join(&ab, &Vtree::balanced_over(&c).unwrap()).unwrap())
}

/// `f(a, b, c) ∧ g(a, b)` on `((a b) c)`.
/// `f` pairs each key `a` with one `b` and a `c` of its own, so at the level
/// `(a b)` it has one parent per key, each holding one pair, while `g` holds
/// the same `(a, b)` rows for all but two keys in one parent. The level's
/// parents outnumber its candidates.
fn keyed_join(eng: &Engine, vtree: &Arc<Vtree>, thresholds: SparseThresholds) -> Tdd {
    let (a, b, c) = blocks();
    let f_rows: Vec<Vec<u64>> = (0..12u64).map(|k| vec![k, (k * 5) % 8, k]).collect();
    let g_rows: Vec<Vec<u64>> = (0..12u64).filter(|k| k % 5 != 4).map(|k| vec![k, (k * 5) % 8]).collect();
    let (fv, fr) = pack(&[&a, &b, &c], &f_rows);
    let (gv, gr) = pack(&[&a, &b], &g_rows);
    let f = eng.from_models(vtree, &fv, &fr).unwrap();
    let g = eng.from_models(vtree, &gv, &gr).unwrap();

    let _forced = ForcedThresholds::install(thresholds);
    eng.and(f, g).expect("an unarmed engine refuses nothing")
}

#[test]
fn a_level_of_many_parents_agrees_with_the_dense_grid() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let collected = SparseThresholds { collected_pairs: 1, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let vtree = vtree();
    let from_arena = keyed_join(&eng, &vtree, sparse);
    let from_list = keyed_join(&eng, &vtree, collected);
    let from_dense = keyed_join(&eng, &vtree, dense);
    let pipelined = SparseThresholds { pipelined_bytes: 0, ..sparse };
    let from_pipelined = keyed_join(&eng, &vtree, pipelined);
    // Twelve keys, of which two (4 and 9) have no `g` row.
    for out in [&from_arena, &from_list, &from_pipelined, &from_dense] {
        assert_eq!(out.model_count().unwrap(), BigUint::from(10u32));
    }
    assert!(from_arena.equivalent(&from_dense).unwrap());
    assert!(from_list.equivalent(&from_dense).unwrap());
    assert!(from_pipelined.equivalent(&from_dense).unwrap());
}
