//! The sparse gate reads the child grids, not only the level's own: an
//! operand's root has one node each side, so its own grid is one cell, while
//! the dense route would first materialize both children's grids — the
//! product of the operands' node counts at each child — to conjoin two
//! pair lists whose result is a few thousand pairs.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::inner_index::{block, pack};
use crate::apply::conjoin::conjoin_owned;
use crate::diagram::Tdd;
use crate::limits::{LimitConfig, MemoryHooks};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// Rows of the block-keyed relation: `2000` values of `a`, each paired with
/// one `b`; the rows of `g` are those of `f` with `b` sent through a
/// different permutation, so the join survives on a fraction of the keys.
fn rows(stride: u64) -> Vec<Vec<u64>> {
    (0..2000u64).map(|k| vec![k, (k * stride) % 2048]).collect()
}

/// `f(a, b) ∧ g(a, b)` on the vtree `(a b)`, both blocks 11 bits wide.
///
/// At each block's root level both operands hold one node per value, so the
/// child grids of the vtree root are `2000 × 2000` cells each while the root
/// level itself is `1 × 1`.
fn root_join(eng: &Engine) -> (Tdd, Tdd, Arc<Vtree>) {
    let (a, b): (Vec<VarId>, Vec<VarId>) = (block(1, 11), block(12, 11));
    let vtree = Arc::new(Vtree::join(
        &Vtree::balanced_over(&a).unwrap(),
        &Vtree::balanced_over(&b).unwrap(),
    ).unwrap());
    let (fv, fr) = pack(&[&a, &b], &rows(7919));
    let (gv, gr) = pack(&[&a, &b], &rows(104729));
    let f = eng.from_models(&vtree, &fv, &fr).unwrap();
    let g = eng.from_models(&vtree, &gv, &gr).unwrap();
    (f, g, vtree)
}

#[test]
fn a_one_node_root_over_wide_children_never_materializes_their_grids() {
    let eng = Engine::new();
    let (f, g, _vtree) = root_join(&eng);
    let child_grid_bytes = 2000u64 * 2000 * std::mem::size_of::<u32>() as u64;

    let largest = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&largest);
    let hooks = MemoryHooks::new(
        move |bytes| { seen.fetch_max(bytes, Ordering::Relaxed); },
        || 0, || None, || {},
    );
    let out = {
        let _installed = eng.limits().scope(LimitConfig::none().with_memory_hooks(hooks));
        conjoin_owned(&eng, f, g, None).expect("an unarmed engine refuses nothing")
    };
    assert!(out.model_count().unwrap() > num_bigint::BigUint::from(0u32), "the join is non-empty");
    let largest = largest.load(Ordering::Relaxed);
    assert!(
        largest < child_grid_bytes,
        "a {largest}-byte reservation: the root densified a {child_grid_bytes}-byte child grid",
    );
}
