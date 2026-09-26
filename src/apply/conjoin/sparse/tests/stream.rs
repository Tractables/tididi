//! `Engine::and_model_count` may count a one-product root from the children
//! of one of its children, which it then never builds. On every vtree, every
//! pivot and every operand order, and whether that child is streamed, built
//! or declined for its counts, the count must be the conjunction's.
use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::BigUint;

use super::inner_index::{block, pack};
use super::ForcedStream;
use super::super::stream::Operand;
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};

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

/// The choices a test pins: the pricing's, building `c`, and streaming with
/// either pivot.
const CHOICES: [Option<Option<Operand>>; 4] = [None, Some(None), Some(Some(Operand::F)), Some(Some(Operand::G))];

/// Count `f ∧ g` under every choice and both operand orders, checking each
/// against `expected`; returns how many counts reached the choice.
fn count_every_way(eng: &Engine, f: &Tdd, g: &Tdd, targets: &[VtreeIdx], expected: &BigUint, what: &str) -> u32 {
    let before = ForcedStream::asked();
    for choice in CHOICES {
        let _pin = ForcedStream::install(choice);
        let counted = eng.and_model_count(f.clone(), g.clone(), targets).unwrap();
        assert_eq!(&counted, expected, "{what}, {choice:?}");
        let swapped = eng.and_model_count(g.clone(), f.clone(), targets).unwrap();
        assert_eq!(&swapped, expected, "{what}, {choice:?}, swapped");
    }
    ForcedStream::asked() - before
}

/// The 4-cycle `R(a,b), R(b,c), R(c,d), R(d,a)` as a path over the four
/// blocks conjoined with its closing edge, on vtrees that put either child of
/// the root, or both, over several blocks.
#[test]
fn a_streamed_root_counts_a_cycle() {
    let eng = Engine::new();
    let bits = 3;
    let (a, b, c, d) = (block(1, bits), block(4, bits), block(7, bits), block(10, bits));
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let (va, vb, vc, vd) = (bal(&a), bal(&b), bal(&c), bal(&d));
    let vtrees = [
        join(&join(&join(&va, &vb), &vc), &vd),
        join(&vd, &join(&join(&va, &vb), &vc)),
        join(&join(&va, &vb), &join(&vc, &vd)),
        join(&join(&va, &join(&vb, &vc)), &vd),
        join(&join(&vd, &va), &join(&vb, &vc)),
        join(&join(&join(&vb, &vc), &va), &vd),
    ];
    let (mut streamed, mut streamed_under_root) = (0, 0);
    for seed in 0..5u64 {
        let edges: BTreeSet<(u64, u64)> = rows(seed, 18 + 4 * seed as usize, 2, bits)
            .into_iter().map(|r| (r[0], r[1])).collect();
        let path: Vec<Vec<u64>> = edges.iter()
            .flat_map(|&(x, y)| edges.iter().filter(move |e| e.0 == y).map(move |&(_, z)| (x, y, z)))
            .flat_map(|(x, y, z)| edges.iter().filter(move |e| e.0 == z).map(move |&(_, w)| vec![x, y, z, w]))
            .collect();
        let closing: Vec<Vec<u64>> = edges.iter().map(|&(x, y)| vec![y, x]).collect();
        let cycles = path.iter().filter(|p| edges.contains(&(p[3], p[0]))).count();
        let expected = BigUint::from(cycles);
        let (pv, pr) = pack(&[&a, &b, &c, &d], &path);
        let (cv, cr) = pack(&[&a, &d], &closing);
        for (i, vtree) in vtrees.iter().enumerate() {
            let vtree = Arc::new(vtree.clone());
            let f = eng.from_models(&vtree, &pv, &pr).unwrap();
            let g = eng.from_models(&vtree, &cv, &cr).unwrap();
            assert_canonical(&f);
            assert_canonical(&g);
            // A target at the root, as a join that ends in a count passes
            // it, sums what is counted anyway.
            for targets in [vec![], vec![vtree.root()]] {
                let what = format!("seed {seed}, vtree {i}, targets {targets:?}");
                let asked = count_every_way(&eng, &f, &g, &targets, &expected, &what);
                streamed += asked;
                streamed_under_root += if targets.is_empty() { 0 } else { asked };
            }
        }
    }
    assert!(streamed > 0, "no root count was streamed");
    assert!(streamed_under_root > 0, "no root count was streamed under a target at the root");
}

/// Relations that constrain every block on both sides, so neither operand
/// has one node at `c`'s children and each pair's probe marks its row; with
/// free blocks beside the joined ones whose counts reach near the top of
/// `u64`, past it (the stream declines and `c` is built), and under targets.
#[test]
fn a_streamed_root_matches_the_built_count_on_dense_relations() {
    let eng = Engine::new();
    let bits = 2;
    let (a, b, c, d) = (block(1, bits), block(3, bits), block(5, bits), block(7, bits));
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let free_sets: [Vec<VarId>; 3] = [vec![], block(20, 60), block(20, 70)];
    let mut streamed = 0;
    for free in &free_sets {
        let joined = join(&join(&join(&bal(&a), &bal(&b)), &bal(&c)), &bal(&d));
        // The free block beside the joined ones under the root, and beside
        // `d` below it; with no free block, the joined blocks alone.
        let vtrees = if free.is_empty() {
            vec![joined]
        } else {
            vec![
                join(&joined, &bal(free)),
                join(&join(&join(&bal(&a), &bal(&b)), &bal(&c)), &join(&bal(&d), &bal(free))),
            ]
        };
        let scale = BigUint::from(1u32) << free.len();
        for seed in 0..4u64 {
            let f_rows = rows(seed + 40, 90, 4, bits);
            let mut g_rows = rows(seed + 60, 70, 4, bits);
            g_rows.extend(f_rows.iter().step_by(2).cloned());
            let f_set: BTreeSet<&Vec<u64>> = f_rows.iter().collect();
            let g_set: BTreeSet<&Vec<u64>> = g_rows.iter().collect();
            let expected = BigUint::from(f_set.intersection(&g_set).count()) * &scale;
            let (fv, fr) = pack(&[&a, &b, &c, &d], &f_rows);
            let (gv, gr) = pack(&[&a, &b, &c, &d], &g_rows);
            for (i, vtree) in vtrees.iter().enumerate() {
                let vtree = Arc::new(vtree.clone());
                let f = eng.from_models(&vtree, &fv, &fr).unwrap();
                let g = eng.from_models(&vtree, &gv, &gr).unwrap();
                assert_canonical(&f);
                assert_canonical(&g);
                let leaf = |v: VarId| vtree.leaf_of(v).unwrap();
                let block_a = vtree.lca(leaf(a[0]), leaf(a[a.len() - 1]));
                let (left, right) = vtree.children(vtree.root());
                for targets in [vec![], vec![block_a], vec![left], vec![right], vec![vtree.root()]] {
                    let what = format!("free {}, seed {seed}, vtree {i}, targets {targets:?}", free.len());
                    streamed += count_every_way(&eng, &f, &g, &targets, &expected, &what);
                }
            }
        }
    }
    assert!(streamed > 0, "no root count was streamed");
}

/// An operand summed over a block below a child of the root, as a join's
/// earlier steps leave it, conjoined with one that does not read that block:
/// the identity fast path moves the summed level out of its operand, whose
/// references into it are still counts, and neither the pricing nor the
/// stream may read them as nodes. The count is the join's either way.
#[test]
fn a_summed_operand_level_is_not_read_as_nodes() {
    let eng = Engine::new();
    let bits = 2;
    let (a, b, c, d) = (block(1, bits), block(3, bits), block(5, bits), block(7, bits));
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let (va, vb, vc, vd) = (bal(&a), bal(&b), bal(&c), bal(&d));
    let vtrees = [join(&join(&va, &vb), &join(&vc, &vd)), join(&join(&join(&va, &vb), &vc), &vd)];
    for (i, vtree) in vtrees.iter().enumerate() {
        let vtree = Arc::new(vtree.clone());
        let leaf = |v: VarId| vtree.leaf_of(v).unwrap();
        // Sum `b` out of f, or `c`, and leave it out of g.
        for (summed, kept) in [(1, [0, 2, 3]), (2, [0, 1, 3])] {
            let blocks: [&[VarId]; 4] = [&a, &b, &c, &d];
            let at = vtree.lca(leaf(blocks[summed][0]), leaf(blocks[summed][bits as usize - 1]));
            for seed in 0..4u64 {
                let f_rows = rows(seed + 80, 40, 4, bits);
                let g_rows = rows(seed + 90, 30, 3, bits);
                let (fv, fr) = pack(&blocks, &f_rows);
                let (gv, gr) = pack(&kept.map(|k| blocks[k]), &g_rows);
                let mut f = eng.from_models(&vtree, &fv, &fr).unwrap();
                let g = eng.from_models(&vtree, &gv, &gr).unwrap();
                eng.marginalize_levels(&mut f, &[at]).unwrap();
                eng.minimize(&mut f).unwrap();
                assert_canonical(&f);
                assert_canonical(&g);
                let f_set: BTreeSet<&Vec<u64>> = f_rows.iter().collect();
                let g_set: BTreeSet<&Vec<u64>> = g_rows.iter().collect();
                let joined = f_set.iter().filter(|r| g_set.contains(&kept.map(|k| r[k]).to_vec())).count();
                let what = format!("vtree {i}, block {summed} summed, seed {seed}");
                count_every_way(&eng, &f, &g, &[], &BigUint::from(joined), &what);
            }
        }
    }
}
