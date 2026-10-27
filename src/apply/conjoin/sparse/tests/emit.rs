//! The sparse emit writes each f parent's nodes straight into the output
//! level, one way for a parent whose candidates make one product, one for a
//! parent whose candidates are each their own product and a counting sort
//! for the rest, and a level where f and g have one node each takes its
//! pairs from the scatter directly. Before any reduction, every way must
//! build the dense grid's nodes.
use std::sync::Arc;

use super::inner_index::{block, pack};
use super::{random_rows, vtrees, ForcedThresholds, SparseThresholds};
use crate::apply::conjoin::{conjoin_on, VtreeMask};
use crate::diagram::Tdd;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{assert_canonical, normalized_levels};
use crate::vtree::{VarId, Vtree};
use crate::Engine;

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
    let pipelined = SparseThresholds { pipelined_bytes: 0, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let (a, b, c) = (block(1, 3), block(4, 3), block(7, 3));
    for seed in 0..12u64 {
        // A join on `b` (a path) and a join on every block (an intersection).
        let path_f = random_rows(seed, 10 + seed as usize * 3, &[3, 3], false);
        let path_g = random_rows(seed + 100, 12 + seed as usize * 2, &[3, 3], false);
        let both_f = random_rows(seed + 200, 40, &[3, 3, 3], false);
        let mut both_g = random_rows(seed + 300, 30, &[3, 3, 3], false);
        both_g.extend(both_f.iter().step_by(3).cloned());
        let joins = [
            (pack(&[&a, &b], &path_f), pack(&[&b, &c], &path_g)),
            (pack(&[&a, &b, &c], &both_f), pack(&[&a, &b, &c], &both_g)),
        ];
        for vtree in vtrees(&a, &b, &c) {
            for ((fv, fr), (gv, gr)) in &joins {
                let from_dense = conjoin(&eng, &vtree, (fv, fr), (gv, gr), dense);
                for thresholds in [sparse, pipelined] {
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

/// Gathering a node's pairs a pass a product leaves each product's pairs
/// in its range in candidate order, and a product of one pair in its single,
/// as a stable sort of the candidates by product would.
#[test]
fn gathering_by_product_is_a_stable_sort() {
    use crate::apply::conjoin::sparse::scatter::gather_by_product;
    use crate::diagram::{ChildPair, EncodedChildRef};
    let mut rng = crate::test_helpers::Lcg::new(11);
    for (products, alone) in (2..=4u32).flat_map(|p| [(p, true), (p, false)]) {
        for n in [64usize, 65, 100, 301] {
            // Product 0 holds one candidate where `alone`, the others the
            // rest at random, each at least one.
            let lone = rng.below(n as u64) as usize;
            let rest = if alone { products - 1 } else { products };
            let first = products - rest;
            let mut owners: Vec<u32> = (0..n)
                .map(|i| if alone && i == lone { 0 } else { first + rng.below(u64::from(rest)) as u32 })
                .collect();
            for k in first..products {
                if !owners.contains(&k) {
                    owners[(lone + 1 + k as usize) % n] = k;
                }
            }
            let pairs: Vec<ChildPair> = (0..n as u32)
                .map(|i| ChildPair::new(EncodedChildRef::from_raw(i), EncodedChildRef::from_raw(i * 7 + 1)))
                .collect();
            let mut cursors: Vec<u32> = (0..products).map(|k| owners.iter().filter(|&&o| o == k).count() as u32).collect();
            let total = super::prefix_offsets(&mut cursors, true).unwrap() as usize;
            let starts = cursors.clone();
            let zero = ChildPair::new(EncodedChildRef::from_raw(0), EncodedChildRef::from_raw(0));
            let mut singles = vec![zero; products as usize];
            let mut into = vec![zero; total + 1];
            gather_by_product(&pairs, &owners, &mut cursors, &mut singles, &mut into);
            for k in 0..products {
                let mine: Vec<ChildPair> = owners.iter().zip(&pairs).filter(|(o, _)| **o == k).map(|(_, p)| *p).collect();
                if starts[k as usize] == u32::MAX {
                    assert_eq!(mine, [singles[k as usize]], "products {products}, n {n}: product {k}'s single");
                    assert_eq!(cursors[k as usize], u32::MAX);
                } else {
                    let (start, end) = (starts[k as usize] as usize, cursors[k as usize] as usize);
                    assert_eq!(&into[start..end], &mine[..], "products {products}, n {n}: product {k}'s range");
                }
            }
        }
    }
}

/// A node of two g parents keyed by comparison makes the products, node
/// words and arena the map's emission makes, with either product of one
/// candidate or both of many.
#[test]
fn two_parents_emit_as_the_map_does() {
    use crate::apply::conjoin::budget::NO_PRODUCT;
    use crate::apply::conjoin::sparse::index::EmitScratch;
    use crate::apply::conjoin::sparse::scatter::{emit_owned, emit_two};
    use crate::diagram::{ChildPair, EncodedChildRef, TddLevel};
    let eng = Engine::new();
    let mut rng = crate::test_helpers::Lcg::new(29);
    let (a, b) = (5u32, 2u32);
    for n in [64usize, 65, 100, 301] {
        for lone in [None, Some(a), Some(b)] {
            // The first candidate is a's; a lone parent holds one candidate.
            let at = 1 + rng.below(n as u64 - 1) as usize;
            let owners: Vec<u32> = (0..n)
                .map(|i| match lone {
                    _ if i == 0 => a,
                    Some(g) if g == a => b,
                    Some(_) => if i == at { b } else { a },
                    None => if i == at || rng.below(2) == 0 { b } else { a },
                })
                .collect();
            let pairs: Vec<ChildPair> = (0..n as u32)
                .map(|i| ChildPair::new(EncodedChildRef::from_raw(i), EncodedChildRef::from_raw(i * 5 + 3)))
                .collect();
            let run = |two: bool| {
                let mut level = TddLevel::new();
                level.pairs.stored_mut().extend_from_slice(&pairs);
                let mut out = Vec::new();
                let mut emit = EmitScratch { p2_map: vec![NO_PRODUCT; 8], ..EmitScratch::default() };
                let mut owners = owners.clone();
                let made = if two {
                    emit_two(&eng, &mut emit, &mut level, &mut out, 3, 0, &mut owners, b, false)
                } else {
                    emit_owned(&eng, &mut emit, &mut level, &mut out, 3, 0, &mut owners, false)
                };
                made.unwrap();
                assert!(emit.p2_map.iter().all(|&p| p == NO_PRODUCT), "the map is left clear");
                let stored = level.stored().expect("an emitted level is stored");
                let products: Vec<(u32, u32, u32)> = out.iter().map(|e| (e.f_idx.0, e.g_idx.0, e.prod_idx.0)).collect();
                (products, stored.nodes().to_vec(), stored.arena().to_vec())
            };
            assert!(run(true) == run(false), "n {n}, lone {lone:?}: the two emits differ");
        }
    }
}
