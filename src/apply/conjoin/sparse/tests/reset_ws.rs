use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};
use crate::diagram::Tdd;
use crate::vtree::Vtree;
use crate::Engine;

#[test]
fn clearing_scratch_leaves_active_workspace_independent() {
    let eng = Engine::new();
    {
        let mut outer = eng.sparse().checkout(eng.limits());
        outer.emit_pairs.reserve(256);
        let allocation = outer.emit_pairs.as_ptr();
        eng.clear_scratch();
        let mut inner = eng.sparse().checkout(eng.limits());
        inner.emit_pairs.reserve(128);
        assert_ne!(inner.emit_pairs.as_ptr(), allocation);
        drop(inner);
        assert_eq!(outer.emit_pairs.as_ptr(), allocation);
    }
    assert!(eng.sparse().checkout(eng.limits()).emit_pairs.capacity() >= 256);
    eng.clear_scratch();
    let ws = eng.sparse().checkout(eng.limits());
    assert_eq!(ws.par_buckets.capacity(), 0);
    assert_eq!(ws.emit_pairs.capacity(), 0);
    assert_eq!(ws.rev_c1.entries.capacity(), 0);
}

#[test]
fn unwinding_discards_partially_filled_sparse_workspace() {
    let eng = Engine::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ws = eng.sparse().checkout(eng.limits());
        ws.p2_map.push(42);
        panic!("interrupt scatter");
    }));
    assert!(eng.sparse().checkout(eng.limits()).p2_map.is_empty());
}

/// `f(a, b, c)` and `g(a, b)` on `((a b) c)`, keyed so that every internal
/// level has parents on both sides and the emit pass touches `p2_map`.
fn keyed_operands(eng: &Engine) -> (Tdd, Tdd) {
    let (a, b, c) = (block(1, 4), block(5, 3), block(8, 4));
    let ab = Vtree::join(&Vtree::balanced_over(&a).unwrap(), &Vtree::balanced_over(&b).unwrap()).unwrap();
    let vtree = Arc::new(Vtree::join(&ab, &Vtree::balanced_over(&c).unwrap()).unwrap());
    let f_rows: Vec<Vec<u64>> = (0..12u64).map(|k| vec![k, (k * 5) % 8, k]).collect();
    let g_rows: Vec<Vec<u64>> = (0..12u64).filter(|k| k % 5 != 4).map(|k| vec![k, (k * 5) % 8]).collect();
    let (fv, fr) = pack(&[&a, &b, &c], &f_rows);
    let (gv, gr) = pack(&[&a, &b], &g_rows);
    (eng.from_models(&vtree, &fv, &fr).unwrap(), eng.from_models(&vtree, &gv, &gr).unwrap())
}

#[test]
fn a_level_refused_mid_scatter_leaves_the_workspace_clean_for_the_next_conjunction() {
    let eng = Engine::new();
    let _forced = ForcedThresholds::install(SparseThresholds {
        min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION
    });
    let (f, g) = keyed_operands(&eng);
    let reference = eng.and(f.clone(), g.clone()).unwrap();
    let mut refusals = 0;
    for cut in 0..40 {
        eng.limits().refuse_nth_reserve(cut);
        let result = eng.and(f.clone(), g.clone());
        eng.limits().grant_every_reserve();
        if result.is_err() { refusals += 1; }
        let recovered = eng.and(f.clone(), g.clone()).unwrap();
        assert_eq!(recovered.model_count().unwrap(), reference.model_count().unwrap());
        assert!(recovered.equivalent(&reference).unwrap());
    }
    assert!(refusals > 0);
}
