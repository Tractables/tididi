//! `Engine::and_model_count` counts a one-product root on the fly, from its
//! scatter's candidates, where the sparse route builds it; it must return
//! the count of the conjunction it did not build, on every route, every
//! vtree and every target set, whichever way the scatter collects and
//! whether the child counts fit the grouped `u64` path, spill past `u128`,
//! or are exact big counts.
use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::BigUint;

use super::inner_index::{block, pack};
use super::{ForcedThresholds, SparseThresholds};
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree, VtreeIdx};
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

/// The vtrees the join runs on: each of the three blocks under its own
/// subtree, `a` and `b` beside `free_a` and `free_b` unconstrained variables,
/// paired every way under the root.
fn vtrees(a: &[VarId], b: &[VarId], c: &[VarId], free_a: &[VarId], free_b: &[VarId]) -> Vec<Arc<Vtree>> {
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let side = |x: &[VarId], free: &[VarId]| if free.is_empty() { bal(x) } else { join(&bal(x), &bal(free)) };
    let (x, y, z) = (side(a, free_a), side(b, free_b), bal(c));
    vec![
        Arc::new(join(&join(&x, &y), &z)),
        Arc::new(join(&x, &join(&y, &z))),
        Arc::new(join(&join(&x, &z), &y)),
    ]
}

/// The target sets the counts are taken under: none, each child of the
/// root, and block `a`'s own subtree.
fn target_sets(vtree: &Vtree, a: &[VarId]) -> Vec<Vec<VtreeIdx>> {
    let (left, right) = vtree.children(vtree.root());
    let leaf = |v: VarId| vtree.leaf_of(v).unwrap();
    let block_a = vtree.lca(leaf(a[0]), leaf(a[a.len() - 1]));
    vec![vec![], vec![left], vec![right], vec![block_a], vec![left, block_a]]
}

#[test]
fn a_counted_root_is_the_count_of_the_conjunction() {
    let eng = Engine::new();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let flat = SparseThresholds { flat_parents: 1, ..sparse };
    let dense = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };
    let (a, b, c) = (block(1, 3), block(4, 3), block(7, 3));
    // No free variable: the grouped path. 60 beside `a`: counts near the
    // top of `u64` where `a`'s subtree is a child of the root, and past it
    // where it is not. 70: that side's counts are big, so each candidate
    // folds exactly.
    let free: [(Vec<VarId>, Vec<VarId>); 3] = [
        (vec![], vec![]),
        (block(10, 60), vec![]),
        (block(10, 70), block(80, 4)),
    ];
    for seed in 0..6u64 {
        let path_f = rows(seed, 12 + seed as usize * 4, 2, 3);
        let path_g = rows(seed + 100, 14 + seed as usize * 3, 2, 3);
        let both_f = rows(seed + 200, 60, 3, 3);
        let mut both_g = rows(seed + 300, 50, 3, 3);
        both_g.extend(both_f.iter().step_by(3).cloned());
        // The joined `(a, b, c)` tuples, by nested loops over the rows.
        let path: BTreeSet<(u64, u64, u64)> = path_f.iter()
            .flat_map(|f| path_g.iter().filter(|g| g[0] == f[1]).map(move |g| (f[0], f[1], g[1])))
            .collect();
        let both_g_set: BTreeSet<&Vec<u64>> = both_g.iter().collect();
        let both: BTreeSet<(u64, u64, u64)> = both_f.iter()
            .filter(|r| both_g_set.contains(r))
            .map(|r| (r[0], r[1], r[2]))
            .collect();
        let joins = [
            (pack(&[&a, &b], &path_f), pack(&[&b, &c], &path_g), path.len() as u64),
            (pack(&[&a, &b, &c], &both_f), pack(&[&a, &b, &c], &both_g), both.len() as u64),
        ];
        for (free_a, free_b) in &free {
            // Every variable a relation leaves free doubles its count; the
            // path join leaves no block free, only the padding variables.
            let scale = BigUint::from(1u32) << (free_a.len() + free_b.len());
            for vtree in vtrees(&a, &b, &c, free_a, free_b) {
                for ((fv, fr), (gv, gr), tuples) in &joins {
                    let expected = BigUint::from(*tuples) * &scale;
                    let f = eng.from_models(&vtree, fv, fr).unwrap();
                    let g = eng.from_models(&vtree, gv, gr).unwrap();
                    assert_canonical(&f);
                    assert_canonical(&g);
                    for targets in target_sets(&vtree, &a) {
                        for thresholds in [sparse, flat, dense] {
                            let _forced = ForcedThresholds::install(thresholds);
                            let counted = eng.and_model_count(f.clone(), g.clone(), &targets).unwrap();
                            assert_eq!(counted, expected, "seed {seed}, targets {targets:?}, {thresholds:?}");
                            // Either operand order, which the entry swaps
                            // by width.
                            let swapped = eng.and_model_count(g.clone(), f.clone(), &targets).unwrap();
                            assert_eq!(swapped, expected, "seed {seed}, targets {targets:?}, swapped");
                        }
                    }
                }
            }
        }
    }
}

/// The shortcuts `and_model_count` shares with the conjunction: a false
/// operand counts zero, and `f ∧ f` counts `f`.
#[test]
fn a_false_operand_and_a_self_conjunction_count_through_the_shortcuts() {
    let eng = Engine::new();
    let (a, b) = (block(1, 3), block(4, 3));
    let vtree = Arc::new(Vtree::join(
        &Vtree::balanced_over(&a).unwrap(),
        &Vtree::balanced_over(&b).unwrap(),
    ).unwrap());
    let (fv, fr) = pack(&[&a, &b], &rows(7, 20, 2, 3));
    let f = eng.from_models(&vtree, &fv, &fr).unwrap();
    assert_canonical(&f);
    let zero = eng.from_models(&vtree, &fv, &[]).unwrap();
    assert_canonical(&zero);
    let expected = eng.model_count(&f).unwrap();
    assert_eq!(eng.and_model_count(f.clone(), f.clone(), &[]).unwrap(), expected);
    assert_eq!(eng.and_model_count(f.clone(), zero.clone(), &[]).unwrap(), BigUint::ZERO);
    assert_eq!(eng.and_model_count(zero, f, &[]).unwrap(), BigUint::ZERO);
}

/// The fold's own arithmetic, on columns near the top of `u64`: a grouped
/// product past `u128`, a running total past it, and an exact big count, all
/// against the same sums in `BigUint`.
#[test]
fn the_candidate_fold_spills_past_u128_exactly() {
    use crate::apply::conjoin::sparse::CandidateFold;
    use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef};
    use crate::value::{CountRef, IntFold, StreamChild};

    let eng = Engine::new();
    let lim = eng.limits();
    let near = u64::MAX as u128;
    let left_counts = [near, near - 1, 3, 1];
    let right_counts = [near, 2, near - 5, 1];
    fn view(counts: &[u128]) -> StreamChild<'_, IntFold> {
        StreamChild { col: CountRef::new(counts, None).certified(), view: ChildDecoder::structural() }
    }

    // Every candidate pushed one by one, in batches that fill.
    let mut fold = CandidateFold::new(lim, view(&left_counts), view(&right_counts)).unwrap();
    assert!(fold.grouped());
    let mut expected = BigUint::ZERO;
    for round in 0..600u32 {
        let (l, r) = ((round % 4) as usize, ((round / 4) % 4) as usize);
        fold.push(ChildPair::new(EncodedChildRef::from_raw(l as u32), EncodedChildRef::from_raw(r as u32)));
        expected += BigUint::from(left_counts[l]) * right_counts[r];
    }
    // Grouped: one inner product against a bucket summing past `u64`, and
    // one against a bucket summed key by key within `u64`.
    fold.prepare_sums(lim, 3, 2).unwrap();
    fold.begin_round();
    let sum = right_counts.iter().sum::<u128>();
    fold.set_sum(1, sum);
    for &r in &right_counts[..2] {
        fold.add_to_sum(2, r as u64);
    }
    for l in 0..4u32 {
        fold.add_grouped(fold.inner_count::<false>(l), 1);
        expected += BigUint::from(left_counts[l as usize]) * sum;
        fold.add_grouped(fold.inner_count::<false>(l), 2);
        expected += BigUint::from(left_counts[l as usize]) * (right_counts[0] + right_counts[1]);
        // A key no sum was written to this round adds nothing.
        fold.add_grouped(fold.inner_count::<false>(l), 0);
        assert_eq!(u128::from(fold.outer_count::<false>(l)), right_counts[l as usize]);
    }
    // An outer count reads back within its round and as 0 outside it.
    fold.set_outer(1, near as u64);
    assert_eq!((fold.outer_or_zero(0), fold.outer_or_zero(1)), (0, near as u64));
    // A new round empties every sum: the old ones add nothing, and a key
    // summed again starts from its first term.
    fold.begin_round();
    assert_eq!(fold.outer_or_zero(1), 0);
    fold.add_grouped(near as u64, 1);
    fold.add_to_sum(2, 5);
    fold.add_grouped(3, 2);
    expected += BigUint::from(15u32);
    // Only an opened key takes terms by `add_to_open`, and an opened key
    // adds its weight, summed over the products that opened it, times its
    // sum.
    fold.begin_round();
    assert!(fold.open_weighted(0, near as u64));
    assert!(!fold.open_weighted(0, near as u64));
    fold.add_to_open(0, 4);
    fold.add_to_open(1, 9);
    fold.add_grouped(2, 0);
    fold.add_grouped(2, 1);
    expected += BigUint::from(8u32);
    fold.add_weighted(0);
    expected += BigUint::from(2 * near) * 4u32;
    // A weight times a sum past `u128` spills.
    for _ in 0..8 {
        fold.open_weighted(1, u64::MAX);
        fold.add_to_open(1, u64::MAX);
    }
    fold.add_weighted(1);
    expected += BigUint::from(8 * near) * (8 * near);
    assert!(expected > BigUint::from(u128::MAX), "the total spills");
    assert_eq!(fold.finish(), expected);
}
