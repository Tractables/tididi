use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::apply::{FoldCodes, FoldMonoid, PrefixFoldError};
use crate::test_helpers::{assert_canonical, eval, Lcg};
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// A running total, each model of the value block worth `3v + 1`.
struct Sum;
impl FoldMonoid for Sum {
    type Value = u64;
    fn value(&self, code: u64) -> u64 { 3 * code + 1 }
    fn combine(&self, a: &u64, b: &u64) -> u64 { a + b }
}

/// A running maximum of the value block's number.
struct Max;
impl FoldMonoid for Max {
    type Value = u64;
    fn value(&self, code: u64) -> u64 { code }
    fn combine(&self, a: &u64, b: &u64) -> u64 { *a.max(b) }
}

fn leaf(v: u32) -> Vtree { Vtree::leaf(VarId(v)) }

/// `codes` on `into`'s `code_vars`.
fn coded(into: &Arc<Vtree>, code_vars: &[VarId], codes: Vec<u64>) -> FoldCodes {
    FoldCodes { into: into.clone(), code_vars: code_vars.to_vec(), codes }
}

fn join(a: Vtree, b: Vtree) -> Vtree { Vtree::join(&a, &b).unwrap() }

fn balanced(vars: &[u32]) -> Vtree {
    let vars: Vec<VarId> = vars.iter().map(|&v| VarId(v)).collect();
    Vtree::balanced_over(&vars).unwrap()
}

/// The right-linear chain of `order` over `low`.
fn chain(order: &[u32], low: Vtree) -> Vtree { order.iter().rev().fold(low, |below, &e| join(leaf(e), below)) }

/// Where the key lies around the chain.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// The key's balanced subtree left of the chain at the root.
    Left,
    /// The chain left of the key's subtree at the root.
    Right,
    /// The first key variable left of the root, the others right of the chain below it.
    Split,
    /// Each key variable a leaf left of the next node down to the chain.
    Spine,
}

const SHAPES: [Shape; 4] = [Shape::Left, Shape::Right, Shape::Split, Shape::Spine];

fn around(shape: Shape, keys: &[u32], chain: Vtree) -> Vtree {
    if keys.is_empty() {
        return chain;
    }
    match shape {
        Shape::Left => join(balanced(keys), chain),
        Shape::Right => join(chain, balanced(keys)),
        Shape::Split if keys.len() > 1 => join(leaf(keys[0]), join(chain, balanced(&keys[1..]))),
        Shape::Split => join(leaf(keys[0]), chain),
        Shape::Spine => keys.iter().rev().fold(chain, |below, &k| join(leaf(k), below)),
    }
}

/// A layout: `nk` key variables, `m` order variables (most significant
/// first), `nv` value variables, and `nc` code variables that reuse the
/// value block's numbers and go on past them.
struct Blocks {
    keys: Vec<u32>,
    order: Vec<u32>,
    values: Vec<u32>,
    codes: Vec<u32>,
}

impl Blocks {
    fn new(nk: u32, m: u32, nv: u32, nc: u32) -> Self {
        Blocks {
            keys: (1..=nk).collect(),
            order: (nk + 1..=nk + m).collect(),
            values: (nk + m + 1..=nk + m + nv).collect(),
            codes: (nk + m + 1..=nk + m + nc).collect(),
        }
    }

    fn vtrees(&self, shape: Shape) -> (Arc<Vtree>, Arc<Vtree>) {
        let vtree = around(shape, &self.keys, chain(&self.order, balanced(&self.values)));
        let into = around(shape, &self.keys, chain(&self.order, balanced(&self.codes)));
        (Arc::new(vtree), Arc::new(into))
    }

    fn vars(list: &[u32]) -> Vec<VarId> { list.iter().map(|&v| VarId(v)).collect() }

    /// The row of key `k`, order `e` and value `v` over the keys, the order
    /// and the value block, each variable its bit.
    fn row(&self, k: u64, e: u64, v: u64) -> u64 {
        let (nk, m) = (self.keys.len(), self.order.len());
        let mut row = k;
        for i in 0..m {
            row |= (e >> (m - 1 - i) & 1) << (nk + i);
        }
        row | v << (nk + m)
    }

    fn all_vars(&self) -> Vec<VarId> {
        Self::vars(&[&self.keys[..], &self.order[..], &self.values[..]].concat())
    }
}

/// The fold of each point by enumeration: per `(k, e)` of `points`, the
/// values of every `(k, e', v)` of `values` with `e' ≤ e` (`<` if strict).
fn expected<M: FoldMonoid<Value = u64>>(
    points: &BTreeSet<(u64, u64)>,
    values: &BTreeSet<(u64, u64, u64)>,
    strict: bool,
    monoid: &M,
) -> BTreeMap<(u64, u64), u64> {
    let mut out = BTreeMap::new();
    for &(k, e) in points {
        let mut acc: Option<u64> = None;
        for &(k2, e2, v) in values {
            if k2 == k && (e2 < e || !strict && e2 == e) {
                let x = monoid.value(v);
                acc = Some(acc.map_or(x, |a| monoid.combine(&a, &x)));
            }
        }
        if let Some(a) = acc {
            out.insert((k, e), a);
        }
    }
    out
}

/// Runs the fold, coding the folds by rank, and checks every assignment of
/// `into` against the enumeration, the result canonical and the size of
/// the diagram `from_models` builds for it. With `raw`, the operands are
/// conjunctions of two supersets of their models, left unminimized.
#[allow(clippy::too_many_arguments)]
fn check<M: FoldMonoid<Value = u64>>(
    blocks: &Blocks,
    shape: Shape,
    points: &BTreeSet<(u64, u64)>,
    values: &BTreeSet<(u64, u64, u64)>,
    strict: bool,
    raw: bool,
    monoid: &M,
    what: &str,
) {
    let eng = Engine::new();
    let (vtree, into) = blocks.vtrees(shape);
    let nk = blocks.keys.len();
    let a_rows: Vec<u64> = points.iter().map(|&(k, e)| blocks.row(k, e, 0)).collect();
    let b_rows: Vec<u64> = values.iter().map(|&(k, e, v)| blocks.row(k, e, v)).collect();
    let all = blocks.all_vars();
    let build = |vars: &[VarId], rows: &[u64]| -> Tdd {
        if !raw {
            return eng.from_models(&vtree, vars, rows).unwrap();
        }
        let have: BTreeSet<u64> = rows.iter().copied().collect();
        let rest: Vec<u64> = (0..1u64 << vars.len()).filter(|r| !have.contains(r)).collect();
        let half = |p: usize| -> Vec<u64> { rows.iter().copied().chain(rest.iter().copied().skip(p).step_by(2)).collect() };
        let (f, g) = (eng.from_models(&vtree, vars, &half(0)).unwrap(), eng.from_models(&vtree, vars, &half(1)).unwrap());
        eng.and(f, g).unwrap()
    };
    let a = build(&all[..nk + blocks.order.len()], &a_rows);
    let b = build(&all, &b_rows);
    let given = RefCell::new(BTreeMap::new());
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    let coding = |folds: &[u64]| -> FoldCodes {
        let sorted: BTreeSet<u64> = folds.iter().copied().collect();
        assert_eq!(sorted.len(), folds.len(), "{what}: each fold once");
        let rank: BTreeMap<u64, u64> = sorted.iter().enumerate().map(|(i, &f)| (f, i as u64)).collect();
        *given.borrow_mut() = rank.clone();
        coded(&into, &code_vars, folds.iter().map(|f| rank[f]).collect())
    };
    let got = eng.prefix_fold(&a, &b, &order, &value_vars, strict, monoid, coding).unwrap();
    assert_canonical(&got);
    let want = expected(points, values, strict, monoid);
    let rank = given.into_inner();
    assert_eq!(rank.len(), want.values().collect::<BTreeSet<_>>().len(), "{what}: the folds coded");
    // Every assignment of `into`.
    let (m, nc) = (blocks.order.len(), blocks.codes.len());
    let n = nk + m + nc;
    let mut rows = Vec::new();
    for bits in 0..1u64 << n {
        let k = bits & ((1 << nk) - 1);
        let mut e = 0;
        for i in 0..m {
            e |= (bits >> (nk + i) & 1) << (m - 1 - i);
        }
        let c = bits >> (nk + m);
        let holds = want.get(&(k, e)).is_some_and(|f| rank[f] == c);
        let asn: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
        assert_eq!(eval(&got, &asn), holds, "{what}: key {k} order {e} code {c}");
        if holds {
            rows.push(bits);
        }
    }
    let out = Blocks::vars(&[&blocks.keys[..], &blocks.order[..], &blocks.codes[..]].concat());
    let built = eng.from_models(&into, &out, &rows).unwrap();
    assert!(eng.equivalent(&got, &built).unwrap(), "{what}");
    assert_eq!(got.pair_count(), built.pair_count(), "{what}: canonical size");
}

/// Points `(k, e)` and values `(k, e, v)`.
type Rows = (BTreeSet<(u64, u64)>, BTreeSet<(u64, u64, u64)>);

/// Random points over the key and order, random values over all three
/// blocks, several at one order often.
fn random(rng: &mut Lcg, blocks: &Blocks, density: u64) -> Rows {
    let (nk, m, nv) = (blocks.keys.len(), blocks.order.len(), blocks.values.len());
    let mut points = BTreeSet::new();
    let mut values = BTreeSet::new();
    for k in 0..1u64 << nk {
        for e in 0..1u64 << m {
            if rng.below(2) == 0 {
                points.insert((k, e));
            }
            for v in 0..1u64 << nv {
                if rng.below(density) == 0 {
                    values.insert((k, e, v));
                }
            }
        }
    }
    (points, values)
}

#[test]
fn folds_agree_with_enumeration() {
    for (nk, m, nv) in [(0, 3, 2), (1, 3, 2), (2, 3, 2), (3, 2, 1), (2, 2, 3)] {
        let blocks = Blocks::new(nk, m, nv, 6);
        for shape in SHAPES {
            for seed in 1..=4u64 {
                let mut rng = Lcg::new(seed * 1000 + u64::from(nk * 100 + m * 10 + nv));
                let (points, values) = random(&mut rng, &blocks, 3 + seed % 3);
                for (strict, raw) in [(false, false), (true, false), (false, true), (true, true)] {
                    let what = format!("{shape:?} k{nk} e{m} v{nv} seed {seed} strict {strict} raw {raw}");
                    check(&blocks, shape, &points, &values, strict, raw, &Sum, &format!("sum {what}"));
                    check(&blocks, shape, &points, &values, strict, raw, &Max, &format!("max {what}"));
                }
            }
        }
    }
}

#[test]
fn points_before_every_value_are_left_out_and_points_after_carry_the_fold() {
    // One key bit, orders 0..8, values 0..4: key 0 has values at orders 2
    // and 5, key 1 none; every point present.
    let blocks = Blocks::new(1, 3, 2, 6);
    let points: BTreeSet<(u64, u64)> = (0..2).flat_map(|k| (0..8).map(move |e| (k, e))).collect();
    let values: BTreeSet<(u64, u64, u64)> = [(0, 2, 1), (0, 5, 3)].into_iter().collect();
    for strict in [false, true] {
        let want = expected(&points, &values, strict, &Max);
        // Orders 0 and 1 (and 2 when strict) have nothing before them; key 1 nothing at all.
        let first = if strict { 3 } else { 2 };
        assert_eq!(want.keys().copied().collect::<Vec<_>>(), (first..8).map(|e| (0, e)).collect::<Vec<_>>());
        for shape in SHAPES {
            check(&blocks, shape, &points, &values, strict, false, &Max, &format!("{shape:?} strict {strict}"));
        }
    }
}

#[test]
fn values_at_one_order_are_folded_together() {
    // Key 0 holds values 1 and 2 at order 1, and 3 at order 4: the sums
    // (3v + 1 each) run 0, 11, 11, 11, 21 from order 1 on.
    let blocks = Blocks::new(1, 3, 2, 6);
    let points: BTreeSet<(u64, u64)> = (0..8).map(|e| (0, e)).collect();
    let values: BTreeSet<(u64, u64, u64)> = [(0, 1, 1), (0, 1, 2), (0, 4, 3)].into_iter().collect();
    let want = expected(&points, &values, false, &Sum);
    assert_eq!(want[&(0, 1)], 4 + 7);
    assert_eq!(want[&(0, 4)], 4 + 7 + 10);
    for shape in SHAPES {
        for strict in [false, true] {
            check(&blocks, shape, &points, &values, strict, false, &Sum, &format!("{shape:?} strict {strict}"));
        }
    }
}

#[test]
fn an_empty_operand_folds_to_false() {
    let blocks = Blocks::new(1, 2, 1, 3);
    let (vtree, into) = blocks.vtrees(Shape::Left);
    let eng = Engine::new();
    let some = eng.from_models(&vtree, &blocks.all_vars(), &[blocks.row(1, 2, 1)]).unwrap();
    let none = eng.from_models(&vtree, &blocks.all_vars(), &[]).unwrap();
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    for (a, b) in [(&none, &some), (&some, &none)] {
        let got = eng.prefix_fold(a, b, &order, &value_vars, false, &Max, |f: &[u64]| coded(&into, &code_vars, f.to_vec())).unwrap();
        assert!(got.is_zero());
        assert!(Arc::ptr_eq(got.vtree(), &into));
    }
    // A point with no value at its key: false too, and no fold is coded.
    let other = eng.from_models(&vtree, &blocks.all_vars(), &[blocks.row(0, 2, 1)]).unwrap();
    let got = eng
        .prefix_fold(&some, &other, &order, &value_vars, false, &Max, |f: &[u64]| {
            assert!(f.is_empty());
            coded(&into, &code_vars, Vec::new())
        })
        .unwrap();
    assert!(got.is_zero());
}

/// The walk's work, counted by the engine's clock, grows with the points
/// times the order's width and with the operands' sizes, not with the
/// self-join of the values on the key, which grows with the square of the
/// points per key.
#[test]
fn work_grows_with_the_points_not_with_the_self_join() {
    let blocks = Blocks::new(2, 10, 4, 16);
    let (vtree, into) = blocks.vtrees(Shape::Left);
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    let mut ratios = Vec::new();
    let mut rng = Lcg::new(7);
    for per_key in [32u64, 64, 128, 256, 512] {
        let eng = Engine::new();
        let mut a_rows = Vec::new();
        let mut b_rows = Vec::new();
        for k in 0..4 {
            let mut orders = BTreeSet::new();
            while orders.len() < per_key as usize {
                orders.insert(rng.below(1 << 10));
            }
            for e in orders {
                a_rows.push(blocks.row(k, e, 0));
                b_rows.push(blocks.row(k, e, 1 + rng.below(15)));
            }
        }
        let all = blocks.all_vars();
        let a = eng.from_models(&vtree, &all[..12], &a_rows).unwrap();
        let b = eng.from_models(&vtree, &all, &b_rows).unwrap();
        let mark = eng.limits().mark();
        let got = eng.prefix_fold(&a, &b, &order, &value_vars, false, &Sum, |f: &[u64]| coded(&into, &code_vars, f.to_vec())).unwrap();
        let work = eng.limits().work_since(mark);
        assert_canonical(&got);
        // Each point keeps its running sum.
        assert_eq!(got.model_count().unwrap(), (4 * per_key).into());
        let bound = a.pair_count() + b.pair_count() + (4 * per_key) as usize * blocks.order.len();
        let self_join = 4 * per_key * (per_key + 1) / 2;
        ratios.push((per_key, work as f64 / bound as f64, work as f64 / self_join as f64));
    }
    let least = ratios.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let most = ratios.iter().map(|r| r.1).fold(0.0, f64::max);
    assert!(most <= 2.0 * least && most <= 16.0, "work per unit of the bound: {ratios:?}");
    // Against the self-join, the work falls as the points per key grow.
    assert!(ratios.last().unwrap().2 * 4.0 < ratios[0].2, "work per pair of the self-join: {ratios:?}");
}

/// Above the chain the walk conjoins the two diagrams on the key: its
/// work grows with the keys, not with the pairs of nodes the two hold
/// there, which grows with the square of the keys when each key leads to a
/// node of its own.
#[test]
fn work_grows_with_the_keys_not_with_their_pairs() {
    let blocks = Blocks::new(10, 4, 2, 12);
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    for shape in [Shape::Left, Shape::Right] {
        let (vtree, into) = blocks.vtrees(shape);
        let mut ratios = Vec::new();
        let mut rng = Lcg::new(11);
        for keys in [64u64, 256, 1024] {
            let eng = Engine::new();
            let mut points = BTreeSet::new();
            let mut values = BTreeSet::new();
            for k in 0..keys {
                // Three orders and a value at each: a fold of its own per key.
                let mut orders = BTreeSet::new();
                while orders.len() < 3 {
                    orders.insert(rng.below(16));
                }
                for e in orders {
                    points.insert((k, e));
                    values.insert((k, e, rng.below(4)));
                }
            }
            let a_rows: Vec<u64> = points.iter().map(|&(k, e)| blocks.row(k, e, 0)).collect();
            let b_rows: Vec<u64> = values.iter().map(|&(k, e, v)| blocks.row(k, e, v)).collect();
            let all = blocks.all_vars();
            let a = eng.from_models(&vtree, &all[..14], &a_rows).unwrap();
            let b = eng.from_models(&vtree, &all, &b_rows).unwrap();
            let mark = eng.limits().mark();
            let got = eng.prefix_fold(&a, &b, &order, &value_vars, false, &Max, |f: &[u64]| coded(&into, &code_vars, f.to_vec())).unwrap();
            let work = eng.limits().work_since(mark);
            assert_canonical(&got);
            assert_eq!(got.model_count().unwrap(), (3 * keys).into());
            let bound = a.pair_count() + b.pair_count() + (3 * keys) as usize * blocks.order.len();
            ratios.push((keys, work as f64 / bound as f64));
        }
        let least = ratios.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        let most = ratios.iter().map(|r| r.1).fold(0.0, f64::max);
        assert!(most <= 2.0 * least && most <= 16.0, "{shape:?}: work per unit of the bound: {ratios:?}");
    }
}

#[test]
fn layouts_the_fold_cannot_read_are_refused() {
    let eng = Engine::new();
    let blocks = Blocks::new(2, 3, 2, 3);
    let (vtree, into) = blocks.vtrees(Shape::Left);
    let f = eng.from_models(&vtree, &blocks.all_vars(), &[blocks.row(1, 2, 1)]).unwrap();
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    let run = |order: &[VarId], value_vars: &[VarId], into: &Arc<Vtree>, code_vars: &[VarId]| {
        eng.prefix_fold(&f, &f, order, value_vars, false, &Max, |f: &[u64]| coded(into, code_vars, f.to_vec()))
    };
    let layout = |r: Result<Tdd, PrefixFoldError>| matches!(r, Err(PrefixFoldError::Layout(_)));
    // The order out of the chain's order, a key variable as the order, the
    // value block short or holding an order variable, an empty block.
    let swapped = [order[1], order[0], order[2]];
    assert!(layout(run(&swapped, &value_vars, &into, &code_vars)));
    assert!(layout(run(&[VarId(1)], &value_vars, &into, &code_vars)));
    assert!(layout(run(&order[..2], &value_vars, &into, &code_vars)));
    assert!(layout(run(&order, &value_vars[..1], &into, &code_vars)));
    assert!(layout(run(&order, &[], &into, &code_vars)));
    assert!(layout(run(&[], &value_vars, &into, &code_vars)));
    let wide: Vec<VarId> = (100..165).map(VarId).collect();
    assert!(layout(run(&order, &value_vars, &into, &wide)));
    // `into` not the vtree around its code subtree, or other code variables.
    let (_, other) = blocks.vtrees(Shape::Right);
    assert!(layout(run(&order, &value_vars, &other, &code_vars)));
    assert!(layout(run(&order, &value_vars, &into, &code_vars[..2])));
    let renamed = Arc::new(around(Shape::Left, &[2, 1], chain(&blocks.order, balanced(&blocks.codes))));
    assert!(layout(run(&order, &value_vars, &renamed, &code_vars)));
    // A variable twice.
    let twice = |r: Result<Tdd, PrefixFoldError>| matches!(r, Err(PrefixFoldError::Variables(VtreeError::OverlappingVariable(_))));
    assert!(twice(run(&order, &[value_vars[0], value_vars[0]], &into, &code_vars)));
    assert!(twice(run(&order, &[value_vars[0], order[2]], &into, &code_vars)));
    assert!(twice(run(&order, &value_vars, &into, &[code_vars[0], code_vars[0], code_vars[1]])));
}

#[test]
fn codes_that_do_not_fit_are_refused() {
    let eng = Engine::new();
    let blocks = Blocks::new(1, 2, 2, 3);
    let (vtree, into) = blocks.vtrees(Shape::Left);
    let rows = [blocks.row(0, 0, 3), blocks.row(0, 1, 2), blocks.row(1, 3, 1)];
    let f = eng.from_models(&vtree, &blocks.all_vars(), &rows).unwrap();
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    let run = |codes: &dyn Fn(&[u64]) -> Vec<u64>| {
        eng.prefix_fold(&f, &f, &order, &value_vars, false, &Sum, |f: &[u64]| coded(&into, &code_vars, codes(f)))
    };
    // Sums 10, 17 and 4: one too few codes, or one past three bits.
    assert!(matches!(run(&|f| f[1..].to_vec()), Err(PrefixFoldError::Codes(_))));
    assert!(matches!(run(&|f| f.iter().map(|&s| s % 8 + 8 * u64::from(s == 17)).collect()), Err(PrefixFoldError::Codes(_))));
    // Two folds may share a code.
    let shared = run(&|f| vec![5; f.len()]).unwrap();
    assert_canonical(&shared);
    assert_eq!(shared.model_count().unwrap(), 3u32.into());
}

/// The coding sees the folds before it names its block: three sums take a
/// block of two variables, their ranks.
#[test]
fn the_code_block_is_as_wide_as_the_folds_need() {
    let eng = Engine::new();
    let blocks = Blocks::new(1, 2, 2, 3);
    let (vtree, _) = blocks.vtrees(Shape::Left);
    let rows = [blocks.row(0, 0, 3), blocks.row(0, 1, 2), blocks.row(1, 3, 1)];
    let f = eng.from_models(&vtree, &blocks.all_vars(), &rows).unwrap();
    let (order, value_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values));
    let coding = |folds: &[u64]| -> FoldCodes {
        let width = (u64::BITS - (folds.len() as u64 - 1).leading_zeros()).max(1);
        let narrow = Blocks::new(1, 2, 2, width);
        let mut sorted = folds.to_vec();
        sorted.sort_unstable();
        let codes = folds.iter().map(|f| sorted.binary_search(f).unwrap() as u64).collect();
        coded(&narrow.vtrees(Shape::Left).1, &Blocks::vars(&narrow.codes), codes)
    };
    let got = eng.prefix_fold(&f, &f, &order, &value_vars, false, &Sum, coding).unwrap();
    assert_canonical(&got);
    // Sums 10, 17 and 4 at (0, 0), (0, 1) and (1, 3): ranks 1, 2 and 0.
    let narrow = Blocks::new(1, 2, 2, 2);
    let want = eng.from_models(got.vtree(), &narrow.all_vars(), &[narrow.row(0, 0, 1), narrow.row(0, 1, 2), narrow.row(1, 3, 0)]).unwrap();
    assert!(eng.equivalent(&got, &want).unwrap());
}

#[test]
fn operands_the_fold_cannot_read_are_refused() {
    let eng = Engine::new();
    let blocks = Blocks::new(1, 2, 2, 3);
    let (vtree, into) = blocks.vtrees(Shape::Left);
    let (twin, _) = blocks.vtrees(Shape::Left);
    let rows = [blocks.row(0, 0, 3), blocks.row(1, 3, 1)];
    let f = eng.from_models(&vtree, &blocks.all_vars(), &rows).unwrap();
    let g = eng.from_models(&twin, &blocks.all_vars(), &rows).unwrap();
    let (order, value_vars, code_vars) = (Blocks::vars(&blocks.order), Blocks::vars(&blocks.values), Blocks::vars(&blocks.codes));
    let run = |a: &Tdd, b: &Tdd| eng.prefix_fold(a, b, &order, &value_vars, false, &Max, |f: &[u64]| coded(&into, &code_vars, f.to_vec()));
    assert!(matches!(run(&f, &g), Err(PrefixFoldError::Operation(OperationError::VtreeMismatch))));
    // The value subtree's level summed out.
    let mut summed = f.clone();
    let value_root: VtreeIdx = vtree.node(vtree.leaf_of(value_vars[0]).unwrap()).parent().unwrap();
    eng.marginalize_levels(&mut summed, &[value_root]).unwrap();
    assert!(matches!(run(&f, &summed), Err(PrefixFoldError::Operation(OperationError::MarginalLevel(_)))));
}
