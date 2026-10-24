//! The sparse gate reads the child grids and the operands' pairs, not only
//! the level's own grid: an operand's root has one node each side, so its
//! own grid is one cell, while the dense route would first materialize both
//! children's grids — the product of the operands' node counts at each
//! child — and then walk every combination of the two roots' pairs, to
//! conjoin two pair lists whose result is a few thousand pairs.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::inner_index::{block, pack};

use crate::apply::conjoin::route::Route;
use crate::diagram::Tdd;
use crate::limits::{LimitConfig, MemoryHooks, SparseRoute};
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
        eng.and(f, g).expect("an unarmed engine refuses nothing")
    };
    assert!(out.model_count().unwrap() > num_bigint::BigUint::from(0u32), "the join is non-empty");
    let largest = largest.load(Ordering::Relaxed);
    assert!(
        largest < child_grid_bytes,
        "a {largest}-byte reservation: the root densified a {child_grid_bytes}-byte child grid",
    );
}

#[test]
fn the_engine_sparse_route_decides_the_gate() {
    let eng = Engine::new();
    let (f, g, _vtree) = root_join(&eng);
    let child_grid_bytes = 2000u64 * 2000 * std::mem::size_of::<u32>() as u64;

    let conjoin = |route: SparseRoute| {
        let largest = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&largest);
        let hooks = MemoryHooks::new(
            move |bytes| { seen.fetch_max(bytes, Ordering::Relaxed); },
            || 0, || None, || {},
        );
        let _installed = eng.limits().scope(
            LimitConfig::none().with_memory_hooks(hooks).with_sparse_route(route),
        );
        let out = eng.and(f.clone(), g.clone()).expect("an unarmed engine refuses nothing");
        (out, largest.load(Ordering::Relaxed))
    };
    let (sparse, sparse_largest) = conjoin(SparseRoute::DEFAULT);
    let (dense, dense_largest) = conjoin(SparseRoute { min_grid: usize::MAX, ..SparseRoute::DEFAULT });
    assert!(sparse_largest < child_grid_bytes, "the default route densified a child grid");
    assert!(dense_largest >= child_grid_bytes, "a route that never goes sparse kept the child grids sparse");
    assert!(eng.equivalent(&sparse, &dense).unwrap(), "the route changed the conjunction");
    assert_eq!(eng.limits().armed().sparse_route(), SparseRoute::DEFAULT, "the scope restored the route");
}

/// `blocks` rectangles `a ∈ A_i, b ∈ B_i` over the vtree `(a b)`, both
/// blocks `bits` wide: `A_i` and `B_i` the `i`-th of `blocks` equal
/// intervals of the values, moved up by `shift` and wrapped. The root is
/// one node of `blocks` pairs, `(A_i, B_i)`.
fn rectangles(eng: &Engine, vtree: &Arc<Vtree>, bits: u32, blocks: u64, shift: u64) -> Tdd {
    let (a, b) = (block(1, bits), block(1 + bits, bits));
    let (n, width) = (1u64 << bits, (1u64 << bits) / blocks);
    let mut rows = Vec::new();
    for i in 0..blocks {
        for x in 0..width {
            for y in 0..width {
                rows.push(vec![(i * width + x + shift) % n, (i * width + y + shift) % n]);
            }
        }
    }
    let (vars, packed) = pack(&[&a, &b], &rows);
    eng.from_models(vtree, &vars, &packed).unwrap()
}

#[test]
fn a_one_cell_root_over_long_pair_lists_takes_the_sparse_route() {
    // Each root is one node of 32 pairs, so the root's grid is one cell
    // while the dense walk would take 32 × 32 pair combinations. Each of
    // f's intervals meets two of g's at either child, so 64 of the 1,024
    // products are live there: neither child is 64 times sparser than its
    // grid, while together they are 256 times sparser than the root's
    // combinations. With the floor at 512, the root is big by its
    // combinations alone.
    let eng = Engine::new();
    let bits = 10;
    let vtree = Arc::new(Vtree::join(
        &Vtree::balanced_over(&block(1, bits)).unwrap(),
        &Vtree::balanced_over(&block(1 + bits, bits)).unwrap(),
    ).unwrap());
    let f = rectangles(&eng, &vtree, bits, 32, 0);
    let g = rectangles(&eng, &vtree, bits, 32, 16);
    let root = vtree.root();
    assert_eq!((f.level(root).node_count(), g.level(root).node_count()), (1, 1), "one node at each root");
    let conjoin = |route: SparseRoute| {
        let _installed = eng.limits().scope(LimitConfig::none().with_sparse_route(route));
        crate::apply::conjoin::tests::routes_taken(|| eng.and(f.clone(), g.clone()).unwrap())
    };
    let (sparse, routes) = conjoin(SparseRoute { sparsity: 64, min_grid: 512 });
    let root_route = routes.iter().rev().find(|(t, _)| *t == root).map(|&(_, r)| r);
    assert_eq!(root_route, Some(Route::Sparse), "the root's route; every level's: {routes:?}");
    let (dense, _) = conjoin(SparseRoute { min_grid: usize::MAX, ..SparseRoute::DEFAULT });
    assert!(eng.equivalent(&sparse, &dense).unwrap(), "the route changed the conjunction");
}
