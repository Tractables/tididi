//! The sparse emit writes each f parent's nodes straight into the output
//! level, one way for a parent whose candidates make one product, one for a
//! parent whose candidates are each their own product and a counting sort
//! for the rest, and a level where f and g have one node each takes its
//! pairs from the scatter directly. Before any reduction, every way must
//! build the dense grid's nodes.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};
use crate::apply::conjoin::{conjoin_on, VtreeMask};
use crate::diagram::Tdd;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{assert_canonical, normalized_levels};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// `n` rows of values below `2^bits[i]` in column `i`, from a fixed seed.
fn rows(seed: u64, n: usize, bits: &[u32]) -> Vec<Vec<u64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n).map(|_| bits.iter().map(|&b| next() % (1u64 << b)).collect()).collect()
}

/// The vtrees the join runs on: every way of pairing two of the three
/// blocks under the root, each block a balanced subtree.
fn vtrees(a: &[VarId], b: &[VarId], c: &[VarId]) -> Vec<Arc<Vtree>> {
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    vec![
        Arc::new(join(&join(&bal(a), &bal(b)), &bal(c))),
        Arc::new(join(&bal(a), &join(&bal(b), &bal(c)))),
        Arc::new(join(&join(&bal(a), &bal(c)), &bal(b))),
    ]
}

/// `f ∧ g` under `thresholds`.
fn conjoin(eng: &Engine, vtree: &Arc<Vtree>, f: (&[VarId], &[u64]), g: (&[VarId], &[u64]), thresholds: SparseThresholds) -> Tdd {
    let f = eng.from_models(vtree, f.0, f.1).unwrap();
    let g = eng.from_models(vtree, g.0, g.1).unwrap();
    let _forced = ForcedThresholds::install(thresholds);
    conjoin_on(eng, f, g, VtreeMask::default()).expect("an unarmed engine refuses nothing").0
}

#[test]
fn every_emit_builds_the_dense_grids_nodes() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let flat = SparseThresholds { flat_parents: 1, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let (a, b, c) = (block(1, 3), block(4, 3), block(7, 3));
    for seed in 0..12u64 {
        // A join on `b` (a path) and a join on every block (an intersection).
        let path_f = rows(seed, 10 + seed as usize * 3, &[3, 3]);
        let path_g = rows(seed + 100, 12 + seed as usize * 2, &[3, 3]);
        let both_f = rows(seed + 200, 40, &[3, 3, 3]);
        let mut both_g = rows(seed + 300, 30, &[3, 3, 3]);
        both_g.extend(both_f.iter().step_by(3).cloned());
        let joins = [
            (pack(&[&a, &b], &path_f), pack(&[&b, &c], &path_g)),
            (pack(&[&a, &b, &c], &both_f), pack(&[&a, &b, &c], &both_g)),
        ];
        for vtree in vtrees(&a, &b, &c) {
            for ((fv, fr), (gv, gr)) in &joins {
                let from_dense = conjoin(&eng, &vtree, (fv, fr), (gv, gr), dense);
                for thresholds in [sparse, flat] {
                    let mut from_sparse = conjoin(&eng, &vtree, (fv, fr), (gv, gr), thresholds);
                    assert_eq!(
                        normalized_levels(&from_sparse),
                        normalized_levels(&from_dense),
                        "seed {seed}: the sparse emit's nodes differ from the dense grid's"
                    );
                    eng.reduce(&mut from_sparse, ReductionPlan::default()).unwrap();
                    assert_canonical(&from_sparse);
                    assert!(from_sparse.equivalent(&from_dense).unwrap());
                }
            }
        }
    }
}
