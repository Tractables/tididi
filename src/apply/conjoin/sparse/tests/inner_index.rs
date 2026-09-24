//! The general scatter arm's second way of building an outer key's
//! `filtered` index, from g's index keyed by the inner-g child. It is taken
//! when the outer's g keys hold more of the level's g pairs than the wanted
//! inner-g children do, which a g operand free over the outer child makes
//! true at every outer: all of its pairs sit under one key.
use std::sync::Arc;

use num_bigint::BigUint;

use super::{ForcedThresholds, SparseThresholds};

use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// The block of `bits` variables starting at `first`, most significant first.
pub(super) fn block(first: u32, bits: u32) -> Vec<VarId> {
    (first..first + bits).map(VarId).collect()
}

/// Rows of `values` packed for `from_models` over `blocks`, one value per
/// block, in the order the blocks' variables are concatenated.
pub(super) fn pack(blocks: &[&[VarId]], values: &[Vec<u64>]) -> (Vec<VarId>, Vec<u64>) {
    let vars: Vec<VarId> = blocks.iter().flat_map(|b| b.iter().copied()).collect();
    assert!(vars.len() <= 64);
    let rows = values
        .iter()
        .map(|row| {
            let mut packed = 0u64;
            let mut lo = 0;
            for (b, &v) in blocks.iter().zip(row) {
                let bits = b.len();
                for i in 0..bits {
                    packed |= ((v >> (bits - 1 - i)) & 1) << (lo + i);
                }
                lo += bits;
            }
            packed
        })
        .collect();
    (vars, rows)
}

/// The three blocks `o`, `p`, `c` and the vtree `((o p) c)` over them, each
/// block a balanced subtree.
fn blocks_and_vtree() -> ([Vec<VarId>; 3], Arc<Vtree>) {
    let (o, p, c) = (block(1, 4), block(5, 3), block(8, 3));
    let op = Vtree::join(
        &Vtree::balanced_over(&o).unwrap(),
        &Vtree::balanced_over(&p).unwrap(),
    )
    .unwrap();
    let vtree = Arc::new(Vtree::join(&op, &Vtree::balanced_over(&c).unwrap()).unwrap());
    ([o, p, c], vtree)
}

/// A key-to-key join `f(o, p) ∧ g(o, c)`. At the level `(o p)`, `g` is free
/// over `p`, so every `g` pair there sits under the one identity key of that
/// child: an outer key's walk by that key reads all of `g`'s pairs, while
/// its walk by the wanted inner-g children reads the one or two parents of
/// the keys it joins, so every outer key takes the second.
fn fk_join(
    eng: &Engine,
    [o, p, c]: &[Vec<VarId>; 3],
    vtree: &Arc<Vtree>,
    thresholds: SparseThresholds,
) -> Tdd {
    // Fixed rows: keys 0..12 each with one to three fact rows, and a
    // dimension row for every key but two, so some fact rows join nothing.
    let mut fact = Vec::new();
    let mut dim = Vec::new();
    for key in 0..12u64 {
        for i in 0..(key % 3 + 1) {
            fact.push(vec![key, (key * 5 + i * 3) % 8]);
        }
        if key % 5 != 4 {
            dim.push(vec![key, (key * 3) % 7]);
        }
    }
    let (fv, fr) = pack(&[o, p], &fact);
    let (gv, gr) = pack(&[o, c], &dim);
    let f = eng.from_models(vtree, &fv, &fr).unwrap();
    let g = eng.from_models(vtree, &gv, &gr).unwrap();

    let _forced = ForcedThresholds::install(thresholds);
    eng.and(f, g).expect("an unarmed engine refuses nothing")
}

#[test]
fn inner_index_agrees_with_the_dense_grid() {
    let eng = Engine::new();
    let (blocks, vtree) = blocks_and_vtree();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let from_sparse = fk_join(&eng, &blocks, &vtree, sparse);
    let from_dense = fk_join(&eng, &blocks, &vtree, dense);
    // 24 fact rows, of which the two missing keys' rows (keys 4 and 9: two
    // and one) join nothing.
    assert_eq!(from_sparse.model_count().unwrap(), BigUint::from(21u32));
    assert_eq!(from_dense.model_count().unwrap(), BigUint::from(21u32));
    assert!(from_sparse.equivalent(&from_dense).unwrap());
}
