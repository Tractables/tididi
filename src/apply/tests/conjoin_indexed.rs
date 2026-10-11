//! `and_indexed` and `sum_indexed` against the conjunction they stand for,
//! and the work bound on keyed operands.

use super::*;
use crate::apply::conjoin_indexed::{and_indexed_counted, sum_indexed_counted, IndexedWork};
use crate::diagram::{ChildPair, ColumnAlgebra, LeafLabel, Literal, NodeIdx, PairIndex, SlotPairs};
use crate::test_helpers::check::{check_all_fast, check_determinism, check_no_false_nodes};
use crate::vtree::VtreeIdx;

/// The variables under vtree node `t`, ascending.
fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    let mut out = Vec::new();
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        match vtree.node(t).is_leaf() {
            true => out.push(vtree.leaf_var(t)),
            false => {
                let (l, r) = vtree.children(t);
                stack.extend([l, r]);
            }
        }
    }
    out.sort();
    out
}

/// A random set of codes over the variables under a random internal node,
/// `⊤` over every other variable.
fn local_set(vtree: &Arc<Vtree>, rng: &mut Lcg) -> Tdd {
    let internal: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).collect();
    let t = internal[rng.below(internal.len() as u64) as usize];
    let mut vars = vars_under(vtree, t);
    vars.truncate(4);
    let rows = 1usize << vars.len();
    let keep: Vec<bool> = (0..rows).map(|_| rng.below(3) != 0).collect();
    or_of_cubes(vtree, &vars, |r| keep[r])
}

/// `got` has the models of `f ∧ g` over the whole truth table, keeps the
/// structural invariants, and minimizes to `f ∧ g`'s size.
fn agrees(f: &Tdd, g: &Tdd, got: &Tdd, nvars: u32, label: &str) {
    for mask in 0..1u32 << nvars {
        let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
        assert_eq!(eval(got, &asn), eval(f, &asn) && eval(g, &asn), "{label}: models differ at {asn:?}");
    }
    check_no_false_nodes(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    check_determinism(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut minimized = got.clone();
    minimized.minimize().unwrap();
    check_all_fast(&minimized, label);
    let mut want = and2(f, g);
    want.minimize().unwrap();
    assert_eq!(minimized.pair_count(), want.pair_count(), "{label}: the minimized sizes differ");
}

/// `f ∧ g` through `g`'s index, checked by [`agrees`].
fn indexed(eng: &Engine, f: &Tdd, g: &Tdd, nvars: u32, label: &str) -> Tdd {
    let index = PairIndex::new(g).unwrap();
    let got = eng.and_indexed(f, g, &index).unwrap();
    agrees(f, g, &got, nvars, label);
    got
}

/// The cases: per vtree shape and size, random functions conjoined with
/// random functions, with sets of codes over a block, with cubes, with an
/// unminimized operand holding unreachable nodes, and with constants; `f`
/// local too, so that it is true over whole subtrees and `g`'s nodes there
/// are copied.
fn cases() -> Vec<Tdd> {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x0001_d5ed_2026_1011);
    let mut out = Vec::new();
    for nvars in [3u32, 5, 7] {
        for (shape, vtree) in vtree_shapes(nvars) {
            for round in 0..10 {
                let label = format!("{shape} n={nvars} round {round}");
                let f = rand_conj(&vtree, nvars, 4, 3, round % 2 == 0, &mut rng);
                let g = rand_conj(&vtree, nvars, 4, 3, round % 3 == 0, &mut rng);
                out.push(indexed(&eng, &f, &g, nvars, &format!("{label} any")));
                let set = local_set(&vtree, &mut rng);
                out.push(indexed(&eng, &f, &set, nvars, &format!("{label} g a set")));
                out.push(indexed(&eng, &set, &g, nvars, &format!("{label} f a set")));
                let cube = Tdd::cube(&vtree, [Literal::new(VarId(1 + rng.below(u64::from(nvars)) as u32), rng.coin())]).unwrap();
                out.push(indexed(&eng, &cube, &g, nvars, &format!("{label} f a cube")));
                // An apply's result, unminimized: unreachable nodes in place.
                let raw = eng.and(rand_conj(&vtree, nvars, 3, 3, false, &mut rng), local_set(&vtree, &mut rng)).unwrap();
                out.push(indexed(&eng, &f, &raw, nvars, &format!("{label} g raw")));
                out.push(indexed(&eng, &raw, &g, nvars, &format!("{label} f raw")));
            }
            let g = rand_conj(&vtree, nvars, 4, 3, true, &mut rng);
            out.push(indexed(&eng, &eng.one(&vtree), &g, nvars, &format!("{shape} n={nvars} f true")));
            out.push(indexed(&eng, &g, &eng.one(&vtree), nvars, &format!("{shape} n={nvars} g true")));
            out.push(indexed(&eng, &eng.zero(&vtree), &g, nvars, &format!("{shape} n={nvars} f false")));
            out.push(indexed(&eng, &g, &eng.zero(&vtree), nvars, &format!("{shape} n={nvars} g false")));
        }
    }
    out
}

#[test]
fn and_indexed_is_the_conjunction() {
    cases();
}

/// On implicit levels and on stored ones alike.
#[test]
fn and_indexed_on_implicit_levels() {
    same_as_stored(cases);
}

/// Model counts and the sum of a weighted set of variables, as one value
/// `(count, sum)`: the moments a grouped `count` and `sum(a)` read, `a`'s
/// bits weighted by their place.
struct Moments {
    /// Each variable's weight, by its id; zero for the variables outside `a`.
    weight: Vec<u128>,
}

type Moment = (u128, u128);

impl Moments {
    fn mul(a: Moment, b: Moment) -> Moment {
        (a.0 * b.0, a.1 * b.0 + b.1 * a.0)
    }
}

impl ColumnAlgebra for Moments {
    type Column = Vec<Moment>;
    type Value = Moment;
    fn zero(&self) -> Moment {
        (0, 0)
    }
    fn column(&self, _: VtreeIdx, width: usize) -> Vec<Moment> {
        vec![(0, 0); width]
    }
    fn leaf(&self, _: VtreeIdx, var: VarId, label: LeafLabel, col: &mut Vec<Moment>) {
        let w = self.weight[var.0 as usize];
        col[label as usize] = match label {
            LeafLabel::One => (2, w),
            LeafLabel::Pos => (1, w),
            _ => (1, 0),
        };
    }
    fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<Moment>, right: &Vec<Moment>, out: &mut Vec<Moment>) {
        out[slot] = pairs.fold((0, 0), |acc, (l, r)| {
            let m = Self::mul(left[l], right[r]);
            (acc.0 + m.0, acc.1 + m.1)
        });
    }
    fn read(&self, _: VtreeIdx, col: Vec<Moment>, slot: usize) -> Moment {
        col[slot]
    }
}

/// The value of `f`'s node `x` at `t` with the nodes at `at` valued by
/// `at_col`: `f`'s levels above `at` folded over them, and the subtrees
/// beside the path to `at` evaluated whole.
fn fold_above(f: &Tdd, alg: &Moments, t: VtreeIdx, x: u32, at: VtreeIdx, at_col: &[Moment]) -> Moment {
    if t == at {
        return at_col[x as usize];
    }
    let vtree = f.vtree();
    let (l, r) = vtree.children(t);
    let side = |c: VtreeIdx, y: u32| match under(vtree, at, c) {
        true => fold_above(f, alg, c, y, at, at_col),
        false => f.evaluate_columns_at(c, alg).unwrap()[y as usize],
    };
    f.level(t).pairs_iter_of_idx(x as usize).fold((0, 0), |acc, p| {
        let m = Moments::mul(side(l, p.left.raw()), side(r, p.right.raw()));
        (acc.0 + m.0, acc.1 + m.1)
    })
}

/// Whether node `x` of `f` at `t` holds the assignment `asn` to the
/// variables under `t`.
fn holds(f: &Tdd, t: VtreeIdx, x: u32, asn: &[bool]) -> bool {
    let vtree = f.vtree();
    if vtree.node(t).is_leaf() {
        let v = asn[vtree.leaf_var(t).0 as usize - 1];
        return match x {
            0 => true,
            1 => v,
            _ => !v,
        };
    }
    let (l, r) = vtree.children(t);
    f.level(t).pairs_iter_of_idx(x as usize).any(|p| holds(f, l, p.left.raw(), asn) && holds(f, r, p.right.raw(), asn))
}

/// Slot `x` of `sum_indexed`'s column by brute force: over the assignments
/// to the variables under `at` that node `x` of `f` holds and that extend
/// to a model of `g`, the count and the weighted sum.
fn brute_slot(f: &Tdd, g: &Tdd, alg: &Moments, at: VtreeIdx, x: u32, nvars: u32) -> Moment {
    let under = vars_under(f.vtree(), at);
    let mut acc = (0, 0);
    for mask in 0..1u32 << under.len() {
        let mut asn = vec![false; nvars as usize];
        for (i, v) in under.iter().enumerate() {
            asn[v.0 as usize - 1] = (mask >> i) & 1 == 1;
        }
        if !holds(f, at, x, &asn) {
            continue;
        }
        let outside: Vec<usize> = (0..nvars as usize).filter(|&i| !under.contains(&VarId(i as u32 + 1))).collect();
        let extends = (0..1u32 << outside.len()).any(|z| {
            let mut full = asn.clone();
            for (j, &i) in outside.iter().enumerate() {
                full[i] = (z >> j) & 1 == 1;
            }
            eval(g, &full)
        });
        if extends {
            acc.0 += 1;
            acc.1 += under.iter().enumerate().filter(|&(i, _)| (mask >> i) & 1 == 1).map(|(_, v)| alg.weight[v.0 as usize]).sum::<u128>();
        }
    }
    acc
}

/// Per vtree shape and every internal level `at`, the column against brute
/// force for any `g`, and, for `g` over the variables under `at`, `f`'s
/// levels above folded over it against `and` followed by
/// `evaluate_columns`, under counts and under a weighted sum.
#[test]
fn sum_indexed_is_the_conjunctions_value() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x05ad_2026_1011);
    for nvars in [3u32, 5, 6] {
        for (shape, vtree) in vtree_shapes(nvars) {
            for (t, _, _) in vtree.internal_bottomup() {
                let under = vars_under(&vtree, t);
                // Count alone, then a weighted sum over the block under `at`.
                let mut weight = vec![0u128; nvars as usize + 1];
                let count = Moments { weight: weight.clone() };
                for (i, v) in under.iter().enumerate() {
                    weight[v.0 as usize] = 1 << i;
                }
                let sum = Moments { weight };
                for round in 0..3 {
                    let label = format!("{shape} n={nvars} at {} round {round}", t.idx());
                    let f = match round {
                        0 => rand_conj(&vtree, nvars, 4, 3, true, &mut rng),
                        1 => local_set(&vtree, &mut rng),
                        _ => eng.and(rand_conj(&vtree, nvars, 3, 3, false, &mut rng), local_set(&vtree, &mut rng)).unwrap(),
                    };
                    let any = rand_conj(&vtree, nvars, 4, 3, false, &mut rng);
                    let inside = rand_conj_over(&vtree, &under.iter().map(|v| v.0).collect::<Vec<_>>(), 4, 3, false, &mut rng);
                    for alg in [&count, &sum] {
                        let index = PairIndex::new(&any).unwrap();
                        let col = eng.sum_indexed(&f, &any, &index, t, alg).unwrap();
                        assert_eq!(col.len(), f.reference_slot_count(t), "{label}: a slot per node");
                        for (x, &got) in col.iter().enumerate() {
                            assert_eq!(got, brute_slot(&f, &any, alg, t, x as u32, nvars), "{label}: slot {x}");
                        }
                        let index = PairIndex::new(&inside).unwrap();
                        let col = eng.sum_indexed(&f, &inside, &index, t, alg).unwrap();
                        let want = and2(&f, &inside).evaluate_columns(alg).unwrap();
                        let got = match f.is_zero() {
                            true => (0, 0),
                            false => fold_above(&f, alg, vtree.root(), f.output().local.0, t, &col),
                        };
                        assert_eq!(got, want, "{label}: folded above");
                    }
                }
            }
        }
    }
}

#[test]
fn errors_name_the_operand_the_index_does_not_fit() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = eng.clause(&vtree, [1, -3]).unwrap();
    let g = eng.clause(&vtree, [2, 4]).unwrap();
    let index = PairIndex::new(&g).unwrap();
    // Another diagram on the same vtree.
    let other = eng.clause(&vtree, [-2, 3]).unwrap();
    assert!(matches!(eng.and_indexed(&f, &other, &index), Err(crate::OperationError::IndexMismatch)));
    // `g` changed after the index was built: an apply's result minimized.
    let mut rng = Lcg::new(0xe2_2026_1011);
    let mut raw = loop {
        let raw = rand_conj(&vtree, 4, 5, 3, true, &mut rng);
        let mut minimized = raw.clone();
        minimized.minimize().unwrap();
        if minimized.pair_count() != raw.pair_count() {
            break raw;
        }
    };
    let before = PairIndex::new(&raw).unwrap();
    assert!(eng.and_indexed(&f, &raw, &before).is_ok());
    let pairs = raw.pair_count();
    raw.minimize().unwrap();
    assert_ne!(raw.pair_count(), pairs, "the minimization changes the diagram");
    assert!(matches!(eng.and_indexed(&f, &raw, &before), Err(crate::OperationError::IndexMismatch)));
    // Another vtree, for the operands or for the index.
    let elsewhere = Arc::new(Vtree::balanced(4));
    let g2 = eng.clause(&elsewhere, [2, 4]).unwrap();
    assert!(matches!(eng.and_indexed(&f, &g2, &index), Err(crate::OperationError::VtreeMismatch)));
    let index2 = PairIndex::new(&g2).unwrap();
    assert!(matches!(eng.and_indexed(&f, &g, &index2), Err(crate::OperationError::VtreeMismatch)));
    // A marginal level, a weighted operand, a leaf or a missing `at`.
    let (left, _) = vtree.children(vtree.root());
    let mut summed = g.clone();
    eng.marginalize_levels(&mut summed, &[left]).unwrap();
    assert!(matches!(PairIndex::new(&summed), Err(crate::OperationError::MarginalLevel(_))));
    assert!(matches!(eng.and_indexed(&summed, &g, &index), Err(crate::OperationError::MarginalLevel(_))));
    let count = Moments { weight: vec![0; 5] };
    let leaf = vtree.leaf_of(VarId(1)).unwrap();
    assert!(matches!(eng.sum_indexed(&f, &g, &index, leaf, &count), Err(crate::OperationError::InertOption { .. })));
    assert!(matches!(eng.sum_indexed(&f, &g, &index, VtreeIdx(99), &count), Err(crate::OperationError::LevelNotInVtree(_))));
}

/// The index lists every pair of the reachable nodes once by each side.
#[test]
fn the_index_finds_each_pair_by_its_sides() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x1d_2026_1011);
    for (shape, vtree) in vtree_shapes(6) {
        let mut g = rand_conj(&vtree, 6, 5, 3, true, &mut rng);
        g.minimize().unwrap();
        let index = PairIndex::new(&g).unwrap();
        assert_eq!(index.pair_count(), g.pair_count(), "{shape}");
        assert!(index.heap_bytes() >= 16 * index.pair_count(), "{shape}");
        for (t, _, _) in vtree.internal_bottomup() {
            let level = g.level(t);
            for x in 0..level.node_count() {
                for p in level.pairs_iter_of_idx(x) {
                    assert_eq!(index.owner(t, p), Some(NodeIdx(x as u32)), "{shape} at {}", t.idx());
                    let (l, r) = (NodeIdx(p.left.raw()), NodeIdx(p.right.raw()));
                    assert!(index.with_left(t, l).any(|e| e == (r, NodeIdx(x as u32))));
                    assert!(index.with_right(t, r).any(|e| e == (l, NodeIdx(x as u32))));
                }
            }
            let listed: usize = (0..g.reference_slot_count(vtree.children(t).0)).map(|l| index.with_left(t, NodeIdx(l as u32)).len()).sum();
            assert_eq!(listed, (0..level.node_count()).map(|x| level.pair_count_at(x)).sum::<usize>());
        }
        assert_eq!(index.owner(vtree.root(), ChildPair::new(NodeIdx(u32::MAX >> 1), NodeIdx(0))), None);
        let leaf = vtree.leaf_of(VarId(1)).unwrap();
        assert_eq!(index.owner(leaf, ChildPair::new(NodeIdx(0), NodeIdx(0))), None);
        let _ = eng.and_indexed(&g, &g, &index).unwrap();
    }
}

/// A keyed table: `vtree` with `value` bits on a spine above the key's
/// subtree, the key's bits from `key_bits`, and each key's value.
struct Keyed {
    vtree: Arc<Vtree>,
    /// The variables of a row: the value's bits, then the key's.
    vars: Vec<VarId>,
    value_bits: u32,
    key_bits: u32,
}

impl Keyed {
    /// `value_bits` above a key of `key_bits`, the key one subtree: a
    /// right spine of its bits, or two columns of `split` and the rest
    /// bits joined.
    fn new(value_bits: u32, key_bits: u32, split: Option<u32>) -> Keyed {
        let vars: Vec<VarId> = (1..=value_bits + key_bits).map(VarId).collect();
        let keys = &vars[value_bits as usize..];
        let mut tree = match split {
            None => Vtree::linear_from_order(keys).unwrap(),
            Some(s) => Vtree::join(
                &Vtree::linear_from_order(&keys[..s as usize]).unwrap(),
                &Vtree::linear_from_order(&keys[s as usize..]).unwrap(),
            )
            .unwrap(),
        };
        for &v in vars[..value_bits as usize].iter().rev() {
            tree = Vtree::join(&Vtree::leaf(v), &tree).unwrap();
        }
        Keyed { vtree: Arc::new(tree), vars, value_bits, key_bits }
    }

    /// The value of key `k`.
    fn value(&self, k: u64) -> u64 {
        (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40) & ((1 << self.value_bits) - 1)
    }

    /// The table of `keys` with their values.
    fn table(&self, keys: &[u64]) -> Tdd {
        let rows: Vec<u64> = keys.iter().map(|&k| self.value(k) | (k << self.value_bits)).collect();
        Tdd::from_models(&self.vtree, &self.vars, &rows).unwrap()
    }

    /// The set of `keys`, any value.
    fn keys(&self, keys: &[u64]) -> Tdd {
        let rows: Vec<u64> = keys.to_vec();
        Tdd::from_models(&self.vtree, &self.vars[self.value_bits as usize..], &rows).unwrap()
    }
}

/// The bound of [`Engine::and_indexed`]'s notes, for `keys` keys: at each
/// internal vtree node, `keys` products and `9 · keys + 81` index reads.
fn bound(vtree: &Vtree, keys: u64) -> (u64, u64) {
    let internal = vtree.internal_bottomup().count() as u64;
    (keys * internal, (9 * keys + 81) * internal)
}

/// `f`'s keys fixed while `g` holds 10, 100 and 1000 times as many keys:
/// each walk stays within the bound `f`'s keys set, and the work does not
/// follow `g`. The composite key shares its first column with `f`'s keys
/// in half of the keys added.
#[test]
fn the_work_follows_f_and_not_g() {
    let eng = Engine::new();
    for (split, label) in [(None, "one key column"), (Some(9), "two key columns")] {
        let table = Keyed::new(5, 17, split);
        let mut rng = Lcg::new(0x6e_2026_1011);
        let space = 1u64 << table.key_bits;
        let mut fkeys: Vec<u64> = (0..24).map(|_| rng.below(space)).collect();
        fkeys.sort_unstable();
        fkeys.dedup();
        let f = table.keys(&fkeys);
        // The first key column is the key's low bits.
        let first = (1u64 << split.unwrap_or(0)) - 1;
        let mut works: Vec<IndexedWork> = Vec::new();
        for scale in [96usize, 960, 9_600, 96_000] {
            let mut gkeys = fkeys.clone();
            while gkeys.len() < scale {
                let k = rng.below(space);
                // Half the keys added share a first column with one of `f`'s.
                let k = match split.is_some() && rng.coin() {
                    true => (fkeys[rng.below(fkeys.len() as u64) as usize] & first) | (k & !first),
                    false => k,
                };
                gkeys.push(k);
                if gkeys.len() == scale {
                    gkeys.sort_unstable();
                    gkeys.dedup();
                }
            }
            let g = table.table(&gkeys);
            let index = PairIndex::new(&g).unwrap();
            let (got, work) = and_indexed_counted(&eng, &f, &g, &index).unwrap();
            assert_eq!(got.model_count().unwrap(), BigUint::from(fkeys.len()), "{label} at {scale}: a row per key of f");
            if scale == 96 || scale == 96_000 {
                assert!(eng.equivalent(&got, &eng.and(f.clone(), g.clone()).unwrap()).unwrap(), "{label} at {scale}");
            }
            let (products, probes) = bound(&table.vtree, fkeys.len() as u64);
            assert!(work.products <= products && work.probes <= probes, "{label} at {scale}: {work:?} past {products}, {probes}");
            assert_eq!(work.copies, 0, "{label}: f is true over no internal subtree");
            println!("{label}: g {scale} keys, {} pairs: {work:?}", g.pair_count());
            works.push(work);
        }
        // Flat: the work at a thousand times the keys within a quarter of the least.
        let least = works.iter().map(|w| w.products + w.probes).min().unwrap();
        let most = works.iter().map(|w| w.products + w.probes).max().unwrap();
        assert!(4 * most <= 5 * least, "{label}: {works:?}");
    }
}

/// A grouped sum over a keyed table: the groups' bits on the spine above
/// the value's, `f` the keys with their group, `g` the keys with their
/// value. Each group's count and sum of values, read at the level under
/// the groups, folds into the total, and the work does not follow `g`.
#[test]
fn a_grouped_sum_reads_only_the_keys_of_f() {
    let eng = Engine::new();
    let groups = 3u32;
    let table = Keyed::new(groups + 5, 14, None);
    // The value bits are the low five of the eight above the key; the
    // group bits the three above them.
    let value = |k: u64| table.value(k) & 0x1f;
    let group = |k: u64| (k * 7 + 3) % 8;
    let mut rng = Lcg::new(0x9f_2026_1011);
    let space = 1u64 << table.key_bits;
    let mut fkeys: Vec<u64> = (0..40).map(|_| rng.below(space)).collect();
    fkeys.sort_unstable();
    fkeys.dedup();
    let vars = &table.vars;
    // Group bits are vars 1..=3 (top of the spine), value bits vars 4..=8.
    let frows: Vec<u64> = fkeys.iter().map(|&k| group(k) | (k << 3)).collect();
    let fvars: Vec<VarId> = vars[..3].iter().chain(&vars[8..]).copied().collect();
    let f = Tdd::from_models(&table.vtree, &fvars, &frows).unwrap();
    let mut weight = vec![0u128; vars.len() + 1];
    for i in 0..5 {
        weight[4 + i] = 1 << i;
    }
    let sum = Moments { weight };
    // The level under the group bits.
    let mut at = table.vtree.root();
    for _ in 0..3 {
        at = table.vtree.children(at).1;
    }
    let mut seen = Vec::new();
    for scale in [160usize, 1_600, 16_000] {
        let mut gkeys = fkeys.clone();
        gkeys.extend((0..scale - fkeys.len()).map(|_| rng.below(space)));
        gkeys.sort_unstable();
        gkeys.dedup();
        let gvars: Vec<VarId> = vars[3..].to_vec();
        let grows: Vec<u64> = gkeys.iter().map(|&k| value(k) | (k << 5)).collect();
        let g = Tdd::from_models(&table.vtree, &gvars, &grows).unwrap();
        let index = PairIndex::new(&g).unwrap();
        let (col, work) = sum_indexed_counted(&eng, &f, &g, &index, at, &sum).unwrap();
        let mut got: Vec<Moment> = col.into_iter().filter(|m| m.0 > 0).collect();
        got.sort_unstable();
        let mut want: Vec<Moment> = (0..8u64)
            .map(|c| fkeys.iter().filter(|&&k| group(k) == c).fold((0, 0), |acc, &k| (acc.0 + 1, acc.1 + u128::from(value(k)))))
            .filter(|m| m.0 > 0)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want, "at {scale}: each group's count and sum");
        let (products, probes) = bound(&table.vtree, fkeys.len() as u64);
        assert!(work.products <= products && work.probes <= probes, "at {scale}: {work:?}");
        seen.push(work.products + work.probes);
    }
    let (least, most) = (*seen.iter().min().unwrap(), *seen.iter().max().unwrap());
    assert!(4 * most <= 5 * least, "{seen:?}");
}
