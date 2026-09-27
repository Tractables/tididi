//! The listing arm's emit tests a bit per `filtered` bucket before it reads
//! the bucket. On levels wider than one word of those bits, with rounds
//! whose buckets fall in either word and differ from one round to the next,
//! the sparse route must build the dense grid's nodes: its conjunction,
//! projection, marginal and count are the dense route's, on every vtree,
//! both collections and both operand orders.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};
use crate::test_helpers::{assert_canonical, assert_same_shape};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// `n` rows of values below `2^bits` in each of `cols` columns, from a fixed
/// seed.
fn rows(seed: u64, n: usize, cols: usize, bits: u32) -> Vec<Vec<u64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n).map(|_| (0..cols).map(|_| next() % (1u64 << bits)).collect()).collect()
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

#[test]
fn the_bit_per_bucket_keeps_every_candidate_of_a_wide_level() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let flat = SparseThresholds { flat_parents: 1, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    // 7-bit blocks: up to 128 children a side, two words of bits.
    let (a, b, c) = (block(1, 7), block(8, 7), block(15, 7));
    for (seed, n) in [(3u64, 300usize), (4, 900)] {
        let (f2, g2) = (rows(seed, n, 2, 7), rows(seed + 100, n, 2, 7));
        let f3 = rows(seed + 200, n, 3, 7);
        // Most rows share a few values in one column, so some rounds touch
        // many buckets in both words and the next few.
        let skew: Vec<Vec<u64>> = rows(seed + 400, n, 2, 7).into_iter()
            .map(|r| vec![r[0] % 5, r[1]]).collect();
        let high: Vec<Vec<u64>> = rows(seed + 500, n, 2, 7).into_iter()
            .map(|r| vec![r[0], 64 + r[1] % 64]).collect();
        let joins = [
            (pack(&[&a, &b], &f2), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b], &skew), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b], &high), pack(&[&b, &c], &skew)),
            (pack(&[&a, &b, &c], &f3), pack(&[&b, &c], &g2)),
            (pack(&[&a, &c], &skew), pack(&[&a, &b, &c], &f3)),
        ];
        for vtree in vtrees(&a, &b, &c) {
            let (_, right) = vtree.children(vtree.root());
            for ((fv, fr), (gv, gr)) in &joins {
                let f = eng.from_models(&vtree, fv, fr).unwrap();
                let g = eng.from_models(&vtree, gv, gr).unwrap();
                for (l, r) in [(&f, &g), (&g, &f)] {
                    let run = |thresholds: SparseThresholds| {
                        let _thresholds = ForcedThresholds::install(thresholds);
                        let and = eng.and(l.clone(), r.clone()).unwrap();
                        let exists = eng.and_exists(l.clone(), r.clone(), &b).unwrap();
                        let marginal = eng.and_marginalizing(l.clone(), r.clone(), &[right]).unwrap();
                        let count = eng.and_model_count(l.clone(), r.clone(), &[]).unwrap();
                        (and, exists, marginal, count)
                    };
                    let (and_d, exists_d, marginal_d, count_d) = run(dense);
                    for thresholds in [sparse, flat] {
                        let (and, exists, marginal, count) = run(thresholds);
                        // The dense grid's nodes, before any reduction.
                        assert_same_shape(&and, &and_d, "and");
                        assert_eq!(count, count_d, "and_model_count, {thresholds:?}");
                        assert_eq!(count, eng.model_count(&and).unwrap(), "and_model_count against the built and");
                        assert_eq!(eng.model_count(&marginal).unwrap(), eng.model_count(&marginal_d).unwrap());
                        // `assert_canonical` checks the marginal form where
                        // a level is marginal, so it serves all three.
                        for (mut built, dense_built) in [(and, &and_d), (exists, &exists_d), (marginal, &marginal_d)] {
                            eng.minimize(&mut built).unwrap();
                            assert_canonical(&built);
                            let mut dense_min = dense_built.clone();
                            eng.minimize(&mut dense_min).unwrap();
                            assert_same_shape(&built, &dense_min, "minimized against the dense route");
                        }
                    }
                }
            }
        }
    }
}
