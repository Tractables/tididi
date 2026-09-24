//! The flat candidate list: a level with far more f parents than candidates
//! collects them in one list and sorts it by parent, instead of touching a
//! bucket per parent; the output must be the bucketed path's and the dense
//! grid's.
use std::sync::Arc;

use num_bigint::BigUint;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds, BYTES_PER_PAR_ENTRY};

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
/// parents outnumber its candidates, which is the flat list's case.
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
fn flat_candidates_agree_with_buckets_and_the_dense_grid() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let flat = SparseThresholds { flat_parents: 1, ..sparse };
    let buckets = SparseThresholds { flat_parents: usize::MAX, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let vtree = vtree();
    let from_flat = keyed_join(&eng, &vtree, flat);
    let from_buckets = keyed_join(&eng, &vtree, buckets);
    let from_dense = keyed_join(&eng, &vtree, dense);
    // Twelve keys, of which two (4 and 9) have no `g` row.
    for out in [&from_flat, &from_buckets, &from_dense] {
        assert_eq!(out.model_count().unwrap(), BigUint::from(10u32));
    }
    assert!(from_flat.equivalent(&from_buckets).unwrap());
    assert!(from_flat.equivalent(&from_dense).unwrap());
}

#[test]
fn chunked_flat_candidates_match_the_unchunked_list() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let flat = SparseThresholds { flat_parents: 1, ..sparse };
    // A one-entry budget gives every parent with a candidate a chunk of its own.
    let chunked = SparseThresholds { chunk_bytes: BYTES_PER_PAR_ENTRY, ..flat };
    let vtree = vtree();
    let whole = keyed_join(&eng, &vtree, flat);
    let in_chunks = keyed_join(&eng, &vtree, chunked);
    assert_eq!(in_chunks.model_count().unwrap(), BigUint::from(10u32));
    assert!(in_chunks.equivalent(&whole).unwrap());
}
