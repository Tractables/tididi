use super::*;
use crate::{Engine, Tdd, Vtree};
use crate::test_helpers::assert_canonical;
use std::sync::Arc;

#[test]
fn product_filter_agrees_across_dense_and_sparse_routes() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    for a in [[1,2],[-1,3],[2,4],[-3,-4]] {
        for b in [[1,-2],[-1,3],[2,4],[-2,-4]] {
            let f = Tdd::clause(&tree,a).unwrap();
            let g = Tdd::clause(&tree,b).unwrap();
            assert_canonical(&f); assert_canonical(&g);
            for divisor in 2..6 {
                let build = |thresholds| {
                    let _guard = ForcedThresholds::install(thresholds);
                    let mut out = engine.and_filter_products(f.clone(),g.clone(),|t,a,b| {
                        (t.idx()+a.idx()+b.idx())%divisor!=0
                    }).unwrap();
                    engine.minimize(&mut out).unwrap(); assert_canonical(&out);
                    out
                };
                let dense = build(SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION });
                let sparse = build(SparseThresholds { min_grid: 0, sparsity_factor: 0, ..SparseThresholds::PRODUCTION });
                assert!(engine.equivalent(&dense,&sparse).unwrap(),"{a:?} {b:?} {divisor}");
            }
        }
    }
}
