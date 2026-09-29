//! The scatter's prefetches: on a level whose inner products span more
//! than the gate's size, each walk asks the cache for the inner runs and
//! the parents' buckets it reads some f pairs ahead, and the emit asks for a
//! bucket some parents ahead. Prefetches are hints, so with the gate pinned
//! open every result must be the one the walk without them returns, and the
//! dense grid's: on random joins, on skewed ones whose few keys carry most
//! of the rows, and on a join with more parents than candidates, collected
//! into buckets, flat, and in chunks of one entry, for the conjunction, the
//! conjunction with a block quantified and the counted conjunction, whose
//! root either sums its candidates per inner child or, with counts past
//! `u64`, folds them one by one.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{prefetches_run, ForcedPrefetch, ForcedThresholds, SparseThresholds, BYTES_PER_PAR_ENTRY};
use crate::test_helpers::{assert_canonical, assert_same_shape};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// `n` rows over `cols` columns of values below `2^bits`, from a fixed
/// seed; with `skew`, the first column takes one of four values in three
/// rows of four, so a few keys carry most of the rows.
fn rows(seed: u64, n: usize, cols: usize, bits: u32, skew: bool) -> Vec<Vec<u64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|_| {
            let mut row: Vec<u64> = (0..cols).map(|_| next() % (1u64 << bits)).collect();
            if skew && next() % 4 != 0 {
                row[0] = next() % 4;
            }
            row
        })
        .collect()
}

/// The vtrees the joins run on: the three blocks paired every way under the
/// root, each block a balanced subtree.
fn vtrees(a: &[VarId], b: &[VarId], c: &[VarId]) -> Vec<Arc<Vtree>> {
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let (x, y, z) = (bal(a), bal(b), bal(c));
    vec![
        Arc::new(join(&join(&x, &y), &z)),
        Arc::new(join(&x, &join(&y, &z))),
        Arc::new(join(&join(&x, &z), &y)),
    ]
}

#[test]
fn a_walk_that_prefetches_returns_what_the_walk_without_returns() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let routes = [
        SparseThresholds { flat_parents: usize::MAX, ..sparse },
        SparseThresholds { flat_parents: 1, ..sparse },
        SparseThresholds { flat_parents: 1, chunk_bytes: BYTES_PER_PAR_ENTRY, ..sparse },
        SparseThresholds { flat_parents: usize::MAX, chunk_bytes: BYTES_PER_PAR_ENTRY, ..sparse },
    ];
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let (a, b, c) = (block(1, 7), block(8, 7), block(15, 7));
    let before = prefetches_run();
    for (seed, n, skew) in [(0u64, 300usize, false), (1, 1200, false), (2, 2500, true)] {
        let f3 = rows(seed + 200, n, 3, 7, skew);
        let mut g3 = rows(seed + 300, n / 2, 3, 7, skew);
        g3.extend(f3.iter().step_by(2).cloned());
        let (f2, g2) = (rows(seed, n, 2, 7, skew), rows(seed + 100, n, 2, 7, skew));
        // Every key `x` pairs with one `y` and a `z` of its own, and `g`
        // holds the same `(x, y)` rows for most keys: more f parents than
        // candidates at the level over `x` and `y`.
        let keyed: Vec<Vec<u64>> = (0..n as u64).map(|k| vec![k % 128, (k * 5) % 128, (k * 7) % 128]).collect();
        let keyed_g: Vec<Vec<u64>> = keyed.iter().filter(|r| r[0] % 5 != 4).map(|r| r[..2].to_vec()).collect();
        let joins = [
            (pack(&[&a, &b], &f2), pack(&[&a, &c], &g2)),
            (pack(&[&a, &b], &f2), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b, &c], &f3), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b, &c], &f3), pack(&[&a, &b, &c], &g3)),
            (pack(&[&a, &b, &c], &keyed), pack(&[&a, &b], &keyed_g)),
        ];
        for vtree in vtrees(&a, &b, &c) {
            for ((fv, fr), (gv, gr)) in &joins {
                let f = eng.from_models(&vtree, fv, fr).unwrap();
                let g = eng.from_models(&vtree, gv, gr).unwrap();
                assert_canonical(&f);
                assert_canonical(&g);
                let (built, exists) = {
                    let _forced = ForcedThresholds::install(dense);
                    let built = eng.and(f.clone(), g.clone()).unwrap();
                    let mut exists = eng.and_exists(f.clone(), g.clone(), &b).unwrap();
                    eng.minimize(&mut exists).unwrap();
                    (built, exists)
                };
                assert_canonical(&built);
                assert_canonical(&exists);
                let expected = eng.model_count(&built).unwrap();
                for route in routes {
                    let _forced = ForcedThresholds::install(route);
                    for gate in [None, Some(false), Some(true)] {
                        let _pinned = ForcedPrefetch::install(gate);
                        for (l, r) in [(&f, &g), (&g, &f)] {
                            let what = format!("seed {seed}, {route:?}, gate {gate:?}");
                            let out = eng.and(l.clone(), r.clone()).unwrap();
                            assert_canonical(&out);
                            assert_same_shape(&out, &built, &what);
                            let mut out = eng.and_exists(l.clone(), r.clone(), &b).unwrap();
                            eng.minimize(&mut out).unwrap();
                            assert_canonical(&out);
                            assert_same_shape(&out, &exists, &what);
                            let counted = eng.and_model_count(l.clone(), r.clone(), &[]).unwrap();
                            assert_eq!(counted, expected, "{what}");
                        }
                    }
                }
            }
        }
    }
    // Seventy variables beside `a` that no relation reads: each child count
    // on that side is past `u64`, so the counted root folds its candidates
    // one by one rather than summing them per inner child.
    let free = block(22, 70);
    let side = Vtree::join(&Vtree::balanced_over(&a).unwrap(), &Vtree::balanced_over(&free).unwrap()).unwrap();
    let (y, z) = (Vtree::balanced_over(&b).unwrap(), Vtree::balanced_over(&c).unwrap());
    let wide = [
        Arc::new(Vtree::join(&Vtree::join(&side, &y).unwrap(), &z).unwrap()),
        Arc::new(Vtree::join(&side, &Vtree::join(&y, &z).unwrap()).unwrap()),
    ];
    for (seed, n, skew) in [(4u64, 400usize, false), (5, 1500, true)] {
        let (f2, g2) = (rows(seed, n, 2, 7, skew), rows(seed + 100, n, 2, 7, skew));
        let f3 = rows(seed + 200, n, 3, 7, skew);
        let joins = [
            (pack(&[&a, &b], &f2), pack(&[&b, &c], &g2)),
            (pack(&[&a, &b, &c], &f3), pack(&[&a, &c], &g2)),
        ];
        for vtree in &wide {
            for ((fv, fr), (gv, gr)) in &joins {
                let f = eng.from_models(vtree, fv, fr).unwrap();
                let g = eng.from_models(vtree, gv, gr).unwrap();
                assert_canonical(&f);
                assert_canonical(&g);
                let expected = {
                    let _forced = ForcedThresholds::install(dense);
                    eng.model_count(&eng.and(f.clone(), g.clone()).unwrap()).unwrap()
                };
                let _forced = ForcedThresholds::install(routes[0]);
                for gate in [Some(false), Some(true)] {
                    let _pinned = ForcedPrefetch::install(gate);
                    for (l, r) in [(&f, &g), (&g, &f)] {
                        let counted = eng.and_model_count(l.clone(), r.clone(), &[]).unwrap();
                        assert_eq!(counted, expected, "seed {seed}, gate {gate:?}, wide");
                    }
                }
            }
        }
    }
    // Every prefetching walk ran with the gate open: a listing walk
    // collecting flat and one collecting into buckets, a counted fold's
    // walk, a walk writing a one-product level directly, and the emit over
    // buckets.
    let after = prefetches_run();
    for site in 0..5 {
        assert!(after[site] > before[site], "prefetch site {site} never ran: {before:?} -> {after:?}");
    }
}
