//! `Tdd::trim`: the diagram moves to the projection of the leaves it names,
//! and every query over the home answers as before.

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::{and2, equiv, exact_weight, rand_conj};
use crate::vtree::rng::Lcg;
use crate::literal;

use super::*;

fn weights(num_vars: u32, rng: &mut Lcg) -> WeightStore {
    let mut w = || BigRational::new((rng.below(7) as i64 - 2).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..=num_vars).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    WeightStore::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// A diagram over `home` after random conditioning, forgetting,
/// conjunction and minimization.
fn random_diagram(home: &Arc<Vtree>, n: u32, rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj(home, n, 4, 3, false, rng);
    for _ in 0..rng.below(4) {
        let var = VarId(1 + rng.below(n as u64) as u32);
        f = match rng.below(4) {
            0 => f.condition_var(var, rng.coin()).unwrap(),
            1 => f.exists_vars(&[var]).unwrap(),
            2 => and2(&f, &rand_conj(home, n, 2, 3, false, rng)),
            _ => {
                f.minimize().unwrap();
                f
            }
        };
    }
    f
}

fn var_subset(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    (1..=n).filter(|_| rng.coin()).map(VarId).collect()
}

/// Every query on `g`, a trimmed copy of `f`, answers as on `f`.
fn same_queries(f: &Tdd, g: &Tdd, n: u32, rng: &mut Lcg, label: &str) {
    assert_eq!(f.model_count().unwrap(), g.model_count().unwrap(), "{label}: model count");
    assert_eq!(f.is_sat().unwrap(), g.is_sat().unwrap(), "{label}: satisfiable");
    let vars = var_subset(n, rng);
    assert_eq!(
        f.projected_model_count(&vars).unwrap(),
        g.projected_model_count(&vars).unwrap(),
        "{label}: projected count over {vars:?}",
    );
    let ws = weights(n, rng);
    let algebra = ws.algebra().clone();
    assert_eq!(f.evaluate(&algebra).unwrap(), g.evaluate(&algebra).unwrap(), "{label}: evaluation");
    let (mut fw, mut gw) = (f.clone(), g.clone());
    fw.set_weights(ws.clone()).unwrap();
    gw.set_weights(ws).unwrap();
    let value = |t: &Tdd| exact_weight(&t.weighted_value().unwrap().unwrap());
    assert_eq!(value(&fw), value(&gw), "{label}: weighted value");
}

#[test]
fn trimmed_copies_answer_as_the_original() {
    let mut trimmed_any = 0;
    for seed in 0..300u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(8) as u32;
        let home = Arc::new(Vtree::random(n, seed));
        let f = random_diagram(&home, n, &mut rng);
        let label = format!("seed {seed}, {n} variables");

        let mut g = f.clone();
        trimmed_any += usize::from(g.trim().unwrap());
        check_trimmed(&g, false).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(!g.trim().unwrap(), "{label}: a second trim moved the diagram");
        same_queries(&f, &g, n, &mut rng, &label);

        if g.vtree().num_nodes() > 0 {
            assert!(Arc::ptr_eq(g.vtree().home().unwrap_or(g.vtree()), &home), "{label}: another home");
            let (back, _) = g.embed(&home, |v| v).unwrap();
            assert!(equiv(&f, &back), "{label}: the embedded copy is another function");
        }

        let mut h = g.clone();
        h.minimize().unwrap();
        h.trim().unwrap();
        check_trimmed(&h, true).unwrap_or_else(|e| panic!("{label}, minimized: {e}"));
        same_queries(&f, &h, n, &mut rng, &format!("{label}, minimized"));
    }
    assert!(trimmed_any > 100, "only {trimmed_any} diagrams moved");
}

#[test]
fn constants_have_no_leaf() {
    let home = Arc::new(Vtree::balanced(5));
    let mut zero = Tdd::clause(&home, [1]).unwrap().condition_var(VarId(1), false).unwrap();
    assert!(zero.trim().unwrap());
    assert!(zero.is_zero() && zero.vtree().num_nodes() == 0);
    assert_eq!(zero.model_count().unwrap(), 0u32.into());

    let mut one = Tdd::one(&home);
    assert!(one.trim().unwrap());
    assert!(one.is_one() && !one.is_zero());
    assert_eq!(one.vtree().num_nodes(), 0);
    assert!(Arc::ptr_eq(one.vtree().home().unwrap(), &home));
    assert_eq!(one.model_count().unwrap(), 32u32.into());
    assert_eq!(one.projected_model_count(&[VarId(2), VarId(4), VarId(2)]).unwrap(), 4u32.into());
    assert!(one.is_sat().unwrap());
    assert!(!one.trim().unwrap());
    check_trimmed(&one, true).unwrap();

    let mut x = literal(&home, 3).unwrap();
    assert!(x.trim().unwrap());
    assert_eq!(x.vtree().num_nodes(), 1);
    assert_eq!(x.model_count().unwrap(), 16u32.into());
    check_trimmed(&x, true).unwrap();
}

/// The top own node whose home node is `h` or below it, by walking the home.
fn own_at_by_walk(own: &Vtree, home: &Vtree, h: VtreeIdx) -> Option<VtreeIdx> {
    let under = |mut a: VtreeIdx| loop {
        if a == h {
            return true;
        }
        match home.node(a).parent() {
            Some(p) => a = p,
            None => return false,
        }
    };
    (0..own.num_nodes() as u32)
        .map(VtreeIdx)
        .filter(|&t| under(own.home_node(t)))
        .find(|&t| own.node(t).parent().is_none_or(|p| !under(own.home_node(p))))
}

#[test]
fn own_at_finds_the_top_own_node_under_a_home_node() {
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed);
        let n = 1 + rng.below(12) as u32;
        let home = Arc::new(Vtree::random(n, seed));
        let mut vars = var_subset(n, &mut rng);
        if vars.is_empty() {
            vars.push(VarId(1));
        }
        let own = Vtree::projection(&home, vars).unwrap();
        for h in (0..home.num_nodes() as u32).map(VtreeIdx) {
            assert_eq!(own.own_at(h), own_at_by_walk(&own, &home, h), "seed {seed}, home node {}", h.0);
        }
        assert_eq!(home.own_at(home.root()), Some(home.root()));
        let empty = Vtree::empty_projection(&home);
        assert_eq!(empty.own_at(home.root()), None);
    }
}

#[test]
fn counts_on_a_projection_are_over_the_home() {
    let home = Arc::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [VarId(2), VarId(5)]).unwrap();
    let f = Tdd::clause(&own, [2, 5]).unwrap();
    assert_eq!(f.model_count().unwrap(), (3u32 << 4).into());
    assert_eq!(f.projected_model_count(&[VarId(2), VarId(1)]).unwrap(), 4u32.into());
    assert_eq!(f.projected_model_count(&[VarId(5)]).unwrap(), 2u32.into());
    let all: Vec<VarId> = (1..=6).map(VarId).collect();
    assert_eq!(f.projected_model_count(&all).unwrap(), f.model_count().unwrap());
    assert!(matches!(f.projected_model_count(&[VarId(7)]), Err(crate::OperationError::VariableNotInVtree(_))));
    let mut on_home = Tdd::clause(&home, [2, 5]).unwrap();
    on_home.minimize().unwrap();
    assert_eq!(on_home.model_count().unwrap(), f.model_count().unwrap());
}

#[test]
fn merged_nodes_and_leaves_named_by_both_literals() {
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    let eng = Engine::new();
    // x1 ↔ x2, its two nodes under one root node beside true over x3, x4.
    let home = Arc::new(Vtree::balanced(4));
    let (l, r) = home.children(home.root());
    let mut b = Tdd::builder(&eng, &home).unwrap();
    let same = b.push(&eng, l, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let differ = b.push(&eng, l, &[ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let t = b.push(&eng, r, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let root = b.push(&eng, home.root(), &[ChildPair::new(same, t), ChildPair::new(differ, t)]).unwrap();
    let f = b.finish(TddNodeId { vtree: home.root(), local: root }).unwrap();
    let mut g = f.clone();
    assert!(g.trim().unwrap());
    check_trimmed(&g, false).unwrap();
    assert_eq!(g.vtree().num_nodes(), 3);
    assert_eq!(g.level(g.vtree().root()).slot_count(), 1);
    assert_eq!(g.model_count().unwrap(), 8u32.into());
    assert!(equiv(&f, &g.embed(&home, |v| v).unwrap().0));

    // (x1 ∨ ¬x1) beside true: x1 is named by both literals, and goes in a
    // second round.
    let two = Arc::new(Vtree::balanced(2));
    let mut b = Tdd::builder(&eng, &two).unwrap();
    let root = b
        .push(&eng, two.root(), &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)])
        .unwrap();
    let mut f = b.finish(TddNodeId { vtree: two.root(), local: root }).unwrap();
    assert!(f.trim().unwrap());
    assert!(f.is_one());
    assert_eq!(f.model_count().unwrap(), 4u32.into());
}

#[test]
fn counters_count_over_the_home() {
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(7) as u32;
        let home = Arc::new(Vtree::random(n, seed));
        let f = random_diagram(&home, n, &mut rng);
        let mut g = f.clone();
        g.trim().unwrap();
        let mut fc = f.counter().unwrap();
        let mut gc = g.counter().unwrap();
        assert_eq!(fc.model_count().unwrap(), gc.model_count().unwrap(), "seed {seed}");
        let pinned: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().leaf_of(v).is_some()).take(2).collect();
        for &var in &pinned {
            let value = Some(rng.coin());
            fc.set_pin(var, value).unwrap();
            gc.set_pin(var, value).unwrap();
        }
        assert_eq!(fc.model_count().unwrap(), gc.model_count().unwrap(), "seed {seed}, pinned {pinned:?}");
        fc.clear_pins();
        gc.clear_pins();
        assert_eq!(fc.count_table(&pinned).unwrap(), gc.count_table(&pinned).unwrap(), "seed {seed}, table");
    }
}
