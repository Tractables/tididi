//! `Engine::and_marginalizing` with one target under the root sums the target
//! out as the sparse route finds the root's pairs (`sparse::sum`), where the
//! two-step path builds every pair, marginalizes the target once the sweep is
//! over and fuses the pairs. The two must leave the same diagram, level for
//! level and pair for pair: the same value slots in the same order, the same
//! references, the same pair order, the same markers and the same
//! re-contraction worklists. That is checked on every vtree shape, with the
//! target on either side of the root, whichever way the scatter collects and
//! walks, for counts that sit inline, in slots, past the sums' one-word path,
//! past `u64` and past `u128`, for roots wide enough to take every grouping
//! pair fusion has, for a root with no pair, and beside a level an operand
//! had already summed out.
use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::BigUint;

use super::inner_index::{block, pack};
use super::{ForcedPrefetch, ForcedThresholds, SparseThresholds};
use crate::apply::conjoin::tests::{summed_roots, two_step};
use crate::diagram::{ChildSide, Tdd};
use crate::test_helpers::{assert_canonical, compile_clauses, rand_cnf, vtree_shapes, CnfShape, Lcg};
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

/// The two diagrams are the same object: output, worklists, and per level
/// the state (every value slot, big ones included), the value-reference
/// markers, the node encodings and each node's pairs in order. The arenas'
/// dead tails are not compared: the two-step path abandons the slots its
/// fusion shrinks and reclaims them only past a threshold.
fn assert_identical(a: &Tdd, b: &Tdd, what: &str) {
    assert_eq!(a.output, b.output, "{what}: output");
    assert_eq!(format!("{:?}", a.dirty), format!("{:?}", b.dirty), "{what}: worklists");
    assert!(a.weights.is_none() && b.weights.is_none(), "{what}: weights");
    assert_eq!(a.levels.len(), b.levels.len(), "{what}: level count");
    for (i, (x, y)) in a.levels.iter().zip(b.levels.iter()).enumerate() {
        assert_eq!(format!("{:?}", x.state), format!("{:?}", y.state), "{what}: level {i} state");
        assert_eq!(x.value_ref_sides, y.value_ref_sides, "{what}: level {i} markers");
        assert_eq!(x.nodes.len(), y.nodes.len(), "{what}: level {i} width");
        assert_eq!(x.ranges, y.ranges, "{what}: level {i} ranges");
        for n in 0..x.nodes.len() {
            assert_eq!(x.nodes[n].kind(), y.nodes[n].kind(), "{what}: level {i} node {n} encoding");
            assert_eq!(x.pairs_of_idx(n), y.pairs_of_idx(n), "{what}: level {i} node {n} pairs");
        }
    }
}

/// What one comparison saw: whether the root summed its child out, and the
/// root's pairs and fused groups in the conjunction built in full.
#[derive(Clone, Copy, Default)]
struct Seen {
    summed: bool,
    pairs: usize,
    groups: usize,
}

/// The pairs of the root of `full` and how many of its kept-side refs `side`
/// leaves shared by two pairs or more: the groups pair fusion fuses.
fn root_groups(full: &Tdd, side: ChildSide) -> (usize, usize) {
    let root = full.vtree.root();
    if full.is_zero() || full.output.vtree != root {
        return (0, 0);
    }
    let level = &full.levels[root.idx()];
    let mut keys: Vec<u32> = (0..level.nodes.len())
        .flat_map(|n| level.pairs_of_idx(n).iter().map(|p| match side {
            ChildSide::Right => p.left.0,
            ChildSide::Left => p.right.0,
        }))
        .collect();
    let pairs = keys.len();
    keys.sort_unstable();
    let groups = keys.chunk_by(|a, b| a == b).filter(|run| run.len() > 1).count();
    (pairs, groups)
}

/// `and_marginalizing(f, g, [c])` summed at the root where it can be and on
/// the two-step path: the same diagram, the count of the conjunction built
/// in full, and canonical once minimized.
fn both_ways(eng: &Engine, f: &Tdd, g: &Tdd, c: VtreeIdx, what: &str) -> Seen {
    let before = summed_roots();
    let summed = eng.and_marginalizing(f.clone(), g.clone(), &[c]).unwrap();
    let did = summed_roots() > before;
    let two = two_step(|| eng.and_marginalizing(f.clone(), g.clone(), &[c]).unwrap());
    assert_eq!(summed_roots(), before + u64::from(did), "{what}: the two-step path summed");
    assert_identical(&summed, &two, what);

    let mut full = eng.and(f.clone(), g.clone()).unwrap();
    let vtree = Arc::clone(&full.vtree);
    let side = if vtree.children(vtree.root()).1 == c { ChildSide::Right } else { ChildSide::Left };
    let (pairs, groups) = root_groups(&full, side);
    full.minimize().unwrap();
    assert_canonical(&full);
    let count = eng.model_count(&full).unwrap();
    assert_eq!(eng.model_count(&summed).unwrap(), count, "{what}: count");
    for mut d in [summed, two] {
        d.minimize().unwrap();
        assert_canonical(&d);
        assert_eq!(eng.model_count(&d).unwrap(), count, "{what}: minimized count");
    }
    Seen { summed: did, pairs, groups }
}

/// The internal children of the root.
fn internal_root_children(vtree: &Vtree) -> Vec<VtreeIdx> {
    let (left, right) = vtree.children(vtree.root());
    [left, right].into_iter().filter(|&c| !vtree.node(c).is_leaf()).collect()
}

/// The thresholds every comparison runs under: every level sparse,
/// collecting into buckets and then flat, emitted in one chunk and then in
/// many, and the engine's own.
fn threshold_sets() -> [SparseThresholds; 4] {
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    [
        sparse,
        SparseThresholds { flat_parents: 1, ..sparse },
        SparseThresholds { chunk_bytes: 1, ..sparse },
        SparseThresholds::PRODUCTION,
    ]
}

/// Every comparison of `f ∧ g`, both operand orders, under every threshold
/// set, the first under both prefetch gates, the target each internal child
/// of the root.
fn compare_all(eng: &Engine, f: &Tdd, g: &Tdd, what: &str, seen: &mut Vec<Seen>) {
    let vtree = Arc::clone(&f.vtree);
    for c in internal_root_children(&vtree) {
        for (t, thresholds) in threshold_sets().into_iter().enumerate() {
            let _forced = ForcedThresholds::install(thresholds);
            let gates: &[Option<bool>] = if t == 0 { &[Some(true), Some(false)] } else { &[None] };
            for &gate in gates {
                let _prefetch = ForcedPrefetch::install(gate);
                for (l, r, order) in [(f, g, "f g"), (g, f, "g f")] {
                    let label = format!("{what}, target {c:?}, thresholds {t}, prefetch {gate:?}, {order}");
                    seen.push(both_ways(eng, l, r, c, &label));
                }
            }
        }
    }
}

#[test]
fn a_summed_root_is_the_fused_root_on_random_formulas() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5eed_5a3d);
    let mut seen = Vec::new();
    for case in 0..24 {
        let n = 6 + (case % 7) as u32;
        let clauses = rand_cnf(&mut rng, n, CnfShape { clauses: 3 * n as usize, width: 3 });
        let mid = clauses.len() / 2;
        for (name, vtree) in vtree_shapes(n) {
            let f = compile_clauses(&vtree, &clauses[..mid]);
            let g = compile_clauses(&vtree, &clauses[mid..]);
            assert_canonical(&f);
            assert_canonical(&g);
            compare_all(&eng, &f, &g, &format!("case {case}, {name}"), &mut seen);
        }
    }
    // A target one node wide in both operands is streamed, not built, so
    // on formulas this small most roots find their target marginal already.
    let summed: Vec<&Seen> = seen.iter().filter(|s| s.summed).collect();
    assert!(summed.len() * 20 > seen.len(), "only {} of {} roots summed", summed.len(), seen.len());
    assert!(summed.iter().any(|s| s.groups > 0), "no summed root had a pair to fuse");
}

/// The vtrees a join runs on: each block under its own subtree, `a` and `b`
/// beside `free_a` and `free_b` unconstrained variables, paired every way
/// under the root.
fn vtrees(a: &[VarId], b: &[VarId], c: &[VarId], free_a: &[VarId], free_b: &[VarId]) -> Vec<Arc<Vtree>> {
    let bal = |x: &[VarId]| Vtree::balanced_over(x).unwrap();
    let join = |l: &Vtree, r: &Vtree| Vtree::join(l, r).unwrap();
    let side = |x: &[VarId], free: &[VarId]| if free.is_empty() { bal(x) } else { join(&bal(x), &bal(free)) };
    let (x, y, z) = (side(a, free_a), side(b, free_b), bal(c));
    vec![
        Arc::new(join(&join(&x, &y), &z)),
        Arc::new(join(&x, &join(&y, &z))),
        Arc::new(join(&join(&x, &z), &y)),
        Arc::new(join(&z, &join(&y, &x))),
    ]
}

/// Joins of relations over three 3-bit blocks, with no variable free, and
/// with 28, 60, 70 and 130 free beside `a` (and 4 beside `b`): the summed
/// counts then sit inline and in slots on either side of the inline limit,
/// near the top of the sums' one-word path and past it, past `u64`, and past
/// `u128`, on whichever side of the root `a`'s subtree lands.
#[test]
fn a_summed_root_is_the_fused_root_on_joins_at_every_count_width() {
    let eng = Engine::new();
    let (a, b, c) = (block(1, 3), block(4, 3), block(7, 3));
    let free: [(Vec<VarId>, Vec<VarId>); 5] = [
        (vec![], vec![]),
        (block(10, 28), vec![]),
        (block(10, 60), vec![]),
        (block(10, 70), block(80, 4)),
        (block(10, 130), block(140, 4)),
    ];
    let mut seen = Vec::new();
    for seed in 0..4u64 {
        let path_f = rows(seed, 12 + seed as usize * 4, 2, 3);
        let path_g = rows(seed + 100, 14 + seed as usize * 3, 2, 3);
        let both_f = rows(seed + 200, 60, 3, 3);
        let mut both_g = rows(seed + 300, 50, 3, 3);
        both_g.extend(both_f.iter().step_by(3).cloned());
        let wide_g = rows(seed + 400, 40, 2, 3);
        let joins = [
            (pack(&[&a, &b], &path_f), pack(&[&b, &c], &path_g)),
            (pack(&[&a, &b, &c], &both_f), pack(&[&a, &b, &c], &both_g)),
            (pack(&[&a, &b, &c], &both_f), pack(&[&a, &c], &wide_g)),
        ];
        for (free_a, free_b) in &free {
            for vtree in vtrees(&a, &b, &c, free_a, free_b) {
                for (j, ((fv, fr), (gv, gr))) in joins.iter().enumerate() {
                    let f = eng.from_models(&vtree, fv, fr).unwrap();
                    let g = eng.from_models(&vtree, gv, gr).unwrap();
                    assert_canonical(&f);
                    assert_canonical(&g);
                    let what = format!("seed {seed}, free {}+{}, join {j}", free_a.len(), free_b.len());
                    compare_all(&eng, &f, &g, &what, &mut seen);
                }
            }
        }
    }
    let summed: Vec<&Seen> = seen.iter().filter(|s| s.summed).collect();
    assert!(summed.len() * 4 > seen.len(), "only {} of {} roots summed", summed.len(), seen.len());
    assert!(summed.iter().any(|s| s.groups > 0), "no summed root had a pair to fuse");
}

/// A root of tens of thousands of pairs in thousands of groups: its fusion
/// sorts the pairs rather than hashing them and tests them against a bitmap
/// of the fused refs, the widths at which the two-step path changes how it
/// works. The relations are dense over a 13-bit block `x` and a 4-bit block
/// `y`, `y` beside 70 free variables so its counts are past `u64`, on both
/// sides of the root.
#[test]
fn a_wide_summed_root_is_the_fused_root() {
    use crate::reduce::contract::pair_fusion::tests::fusion_widths;
    let eng = Engine::new();
    let (x, y, free) = (block(1, 13), block(14, 4), block(18, 70));
    let bal = |v: &[VarId]| Vtree::balanced_over(v).unwrap();
    let y_side = Vtree::join(&bal(&y), &bal(&free)).unwrap();
    let shapes = [
        Arc::new(Vtree::join(&bal(&x), &y_side).unwrap()),
        Arc::new(Vtree::join(&y_side, &bal(&x)).unwrap()),
    ];
    let dense = |seed: u64| -> Vec<Vec<u64>> {
        rows(seed, 1 << 17, 2, 13)
            .into_iter()
            .enumerate()
            .filter(|(_, r)| r[1] & 1 == 0)
            .map(|(i, _)| vec![(i as u64) >> 4, (i as u64) & 15])
            .collect()
    };
    let (sort_min, bitmap_min_plans) = fusion_widths();
    let sparse = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };
    let mut widest = Seen::default();
    for vtree in shapes {
        let c = vtree.leaf_of(y[0]).map(|leaf| {
            let (left, right) = vtree.children(vtree.root());
            if crate::test_helpers::under(&vtree, leaf, left) { left } else { right }
        }).unwrap();
        let (fv, fr) = pack(&[&x, &y], &dense(1));
        let (gv, gr) = pack(&[&x, &y], &dense(2));
        let f = eng.from_models(&vtree, &fv, &fr).unwrap();
        let g = eng.from_models(&vtree, &gv, &gr).unwrap();
        assert_canonical(&f);
        assert_canonical(&g);
        for (forced, thresholds) in [
            (true, sparse),
            (true, SparseThresholds { flat_parents: 1, ..sparse }),
            (false, SparseThresholds::PRODUCTION),
        ] {
            let _forced = ForcedThresholds::install(thresholds);
            let seen = both_ways(&eng, &f, &g, c, &format!("wide, {thresholds:?}"));
            assert!(seen.summed || !forced, "the wide root was not summed under {thresholds:?}");
            if seen.groups > widest.groups {
                widest = seen;
            }
        }
    }
    assert!(widest.pairs >= sort_min, "the widest root held {} pairs", widest.pairs);
    assert!(widest.groups >= bitmap_min_plans, "the widest root fused {} groups", widest.groups);
}

/// A root with no pair: both blocks' products are live, but no pair of
/// the two operands' roots agrees on `y`, so the conjunction is false. And
/// a root with one pair, which is stored in its node's own words. The
/// target is two nodes wide in one operand, so the sweep builds it rather
/// than streaming it.
#[test]
fn an_empty_and_a_one_pair_summed_root_are_the_fused_roots() {
    let eng = Engine::new();
    let (x, y) = (block(1, 4), block(5, 3));
    let vtree = Arc::new(Vtree::join(
        &Vtree::balanced_over(&x).unwrap(),
        &Vtree::balanced_over(&y).unwrap(),
    ).unwrap());
    let (_, right) = vtree.children(vtree.root());
    let at = |v: u64, w: u64| (0..16u64).step_by(2).map(|a| vec![a, if a < 8 { v } else { w }]).collect::<Vec<_>>();
    let (fv, fr) = pack(&[&x, &y], &at(0, 4));
    let (gv, gr) = pack(&[&x, &y], &at(1, 3));
    let f = eng.from_models(&vtree, &fv, &fr).unwrap();
    let g = eng.from_models(&vtree, &gv, &gr).unwrap();
    let (hv, hr) = pack(&[&x, &y], &[vec![2, 1], vec![2, 2]]);
    let h = eng.from_models(&vtree, &hv, &hr).unwrap();
    for d in [&f, &g, &h] {
        assert_canonical(d);
    }
    let _forced = ForcedThresholds::install(threshold_sets()[0]);
    let empty = both_ways(&eng, &f, &g, right, "empty");
    assert!(!empty.summed, "an empty root is left as the sparse route leaves it");
    let summed = eng.and_marginalizing(f.clone(), g.clone(), &[right]).unwrap();
    assert!(summed.is_zero());
    let one = both_ways(&eng, &g, &h, right, "one pair");
    assert!(one.summed, "the one-pair root was not summed");
    let out = eng.and_marginalizing(g.clone(), h.clone(), &[right]).unwrap();
    assert_eq!(eng.model_count(&out).unwrap(), BigUint::from(1u32));
}

/// An operand that already summed out a subtree of free variables: beside
/// the target, the root still sums the target out and the result is the
/// fused one; inside it, the root is built in full and the two paths are one.
#[test]
fn a_summed_root_beside_and_under_an_operand_marginal_level() {
    let eng = Engine::new();
    let (a, b, c) = (block(1, 3), block(4, 3), block(7, 3));
    let free = block(10, 6);
    let mut summed_beside = false;
    for seed in 0..4u64 {
        let f_rows = rows(seed + 500, 40, 3, 3);
        let mut g_rows = rows(seed + 600, 30, 3, 3);
        g_rows.extend(f_rows.iter().step_by(2).cloned());
        for vtree in vtrees(&a, &b, &c, &free, &[]) {
            let (fv, fr) = pack(&[&a, &b, &c], &f_rows);
            let (gv, gr) = pack(&[&a, &b, &c], &g_rows);
            let mut f = eng.from_models(&vtree, &fv, &fr).unwrap();
            let g = eng.from_models(&vtree, &gv, &gr).unwrap();
            let leaf = |v: VarId| vtree.leaf_of(v).unwrap();
            let m = vtree.lca(leaf(free[0]), leaf(free[free.len() - 1]));
            eng.marginalize_levels(&mut f, &[m]).unwrap();
            f.minimize().unwrap();
            assert_canonical(&f);
            assert_canonical(&g);
            let _forced = ForcedThresholds::install(threshold_sets()[0]);
            for c in internal_root_children(&vtree) {
                let seen = both_ways(&eng, &f, &g, c, &format!("seed {seed}, marginal beside or under {c:?}"));
                let under = crate::test_helpers::under(&vtree, m, c);
                assert!(!(under && seen.summed), "a root summed a target holding a marginal level");
                summed_beside |= !under && seen.summed;
            }
        }
    }
    assert!(summed_beside, "no root summed beside a marginal level");
}

/// The candidates in first-reach order with the same kept product reached
/// in a burst, spread out, and once, so the order the pairs take is the one
/// the two-step path's fusion leaves, not a sorted one.
#[test]
fn a_summed_root_keeps_the_two_step_pair_order() {
    let eng = Engine::new();
    let (x, y) = (block(1, 5), block(6, 5));
    let vtree = Arc::new(Vtree::join(
        &Vtree::balanced_over(&x).unwrap(),
        &Vtree::balanced_over(&y).unwrap(),
    ).unwrap());
    let mut seen = Vec::new();
    for seed in 0..6u64 {
        let f_rows: BTreeSet<Vec<u64>> = rows(seed + 700, 300, 2, 5).into_iter().collect();
        let g_rows: BTreeSet<Vec<u64>> = rows(seed + 800, 300, 2, 5).into_iter().collect();
        let f_rows: Vec<Vec<u64>> = f_rows.into_iter().collect();
        let g_rows: Vec<Vec<u64>> = g_rows.into_iter().collect();
        let (fv, fr) = pack(&[&x, &y], &f_rows);
        let (gv, gr) = pack(&[&x, &y], &g_rows);
        let f = eng.from_models(&vtree, &fv, &fr).unwrap();
        let g = eng.from_models(&vtree, &gv, &gr).unwrap();
        compare_all(&eng, &f, &g, &format!("order, seed {seed}"), &mut seen);
    }
    assert!(seen.iter().any(|s| s.summed && s.groups > 1), "no summed root fused two groups");
}
