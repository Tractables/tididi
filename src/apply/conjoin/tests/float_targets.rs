//! Conjunctions that sum out nested targets under the scaled-float
//! arithmetic, whose apply keeps a summed-out level's column until the
//! level's parent has read it.

use std::sync::Arc;

use num_rational::BigRational;

use crate::{Engine, Tdd};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, ScaledFloat, WeightStore};
use crate::test_helpers::{assert_canonical, rand_cnf, rat, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Random literal weights, positive or, with `signed`, of either sign.
fn draw_weights(rng: &mut Lcg, num_vars: u32, signed: bool) -> RationalWeights {
    let mut draw = || {
        let w = rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64);
        if signed && rng.below(4) == 0 { -w } else { w }
    };
    RationalWeights::from_literals(
        &(0..num_vars).map(|_| LiteralWeights { negative: draw(), positive: draw() }).collect::<Vec<_>>(),
    )
}

/// Every internal vtree node not yet marginal in `acc` whose variables no
/// clause in `rest` mentions: what a walk that sums each finished subtree out
/// as it goes asks of one conjunction.
fn finished(tree: &Vtree, acc: &Tdd, rest: &[Vec<i32>]) -> Vec<VtreeIdx> {
    let live: std::collections::HashSet<u32> = rest.iter().flatten().map(|l| l.unsigned_abs()).collect();
    tree.bottomup()
        .filter(|&t| !tree.node(t).is_leaf() && !acc.level(t).is_marginal())
        .filter(|&t| tree.subtree(t).all(|x| match *tree.node(x) {
            VtreeNode::Leaf { var, .. } => !live.contains(&var.0),
            _ => true,
        }))
        .collect()
}

/// The weighted value of `clauses` over `tree`: conjoined one clause at a
/// time, each conjunction summing out every subtree the clauses after it do
/// not touch, nested targets included. Every intermediate result is checked
/// against its store and, minimized, for canonicity.
fn walk(eng: &Engine, tree: &Arc<Vtree>, clauses: &[Vec<i32>], store: &WeightStore) -> Tdd {
    let mut acc = eng.one(tree);
    acc.set_weights(store.empty_like()).unwrap();
    for (k, clause) in clauses.iter().enumerate() {
        let mut g = eng.clause(tree, clause.iter().copied()).unwrap();
        g.set_weights(store.empty_like()).unwrap();
        let targets = finished(tree, &acc, &clauses[k + 1..]);
        acc = eng.and_marginalizing(acc, g, &targets).unwrap();
        if acc.is_zero() {
            return acc;
        }
        for &t in &targets {
            assert!(acc.level(t).is_marginal(), "target {t:?} after clause {k}, {:?}", store.arithmetic());
        }
        acc.weights.as_ref().unwrap().check_levels(&acc.vtree, &acc.levels[..]).unwrap();
        let mut minimized = acc.clone();
        minimized.minimize().unwrap();
        assert_canonical(&minimized);
    }
    let rest = finished(tree, &acc, &[]);
    eng.marginalize_levels(&mut acc, &rest).unwrap();
    acc
}

fn value(eng: &Engine, d: &Tdd) -> crate::diagram::WeightValue {
    eng.weighted_value(d).unwrap().expect("a weighted diagram has a value")
}

/// The scaled-float walk agrees with the exact one to within the rounding
/// of its operations, on formulas whose subtrees finish at every depth.
#[test]
fn nested_targets_match_the_exact_value() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5ca1ed);
    for num_vars in [4u32, 7, 10] {
        for (shape, tree) in vtree_shapes(num_vars) {
            for round in 0..4 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: num_vars as usize + 2, width: 3 });
                let weights = draw_weights(&mut rng, num_vars, round % 2 == 1);
                let exact = walk(&eng, &tree, &clauses, &WeightStore::new(weights.clone(), Arithmetic::ExactRational));
                let scaled = walk(&eng, &tree, &clauses, &WeightStore::new(weights, Arithmetic::ScaledFloat));
                let exact: BigRational = value(&eng, &exact).into_rational();
                let got = *value(&eng, &scaled).as_scaled().expect("a scaled value");
                let want = ScaledFloat::from_rational(&exact);
                let context = format!("{shape}, {num_vars} vars, round {round}: {got:?} against {want:?}");
                if want.is_zero() {
                    assert!(got.to_f64().abs() < 1e-9, "{context}");
                    continue;
                }
                // Positive weights cannot cancel; signed ones can, by little
                // on these sizes.
                let tolerance = if round % 2 == 1 { 1e-9 } else { 1e-13 };
                let ratio = got.mantissa() / want.mantissa() * f64::powi(2.0, (got.exponent() - want.exponent()) as i32);
                assert!((ratio - 1.0).abs() < tolerance, "{context}");
            }
        }
    }
}

/// A conjunction refused at any reserve, the conversion of its kept columns
/// at the end of the sweep included, leaves the engine able to repeat it.
#[test]
fn every_refused_reserve_leaves_the_walk_repeatable() {
    let eng = Engine::new();
    let num_vars = 8;
    let tree = Arc::new(Vtree::balanced(num_vars));
    let mut rng = Lcg::new(17);
    let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 12, width: 3 });
    let store = WeightStore::new(draw_weights(&mut rng, num_vars, false), Arithmetic::ScaledFloat);
    let mut f = eng.one(&tree);
    f.set_weights(store.empty_like()).unwrap();
    let half = clauses.len() / 2;
    for clause in &clauses[..half] {
        let mut g = eng.clause(&tree, clause.iter().copied()).unwrap();
        g.set_weights(store.empty_like()).unwrap();
        f = eng.and(f, g).unwrap();
    }
    let mut g = eng.one(&tree);
    g.set_weights(store.empty_like()).unwrap();
    for clause in &clauses[half..] {
        let mut c = eng.clause(&tree, clause.iter().copied()).unwrap();
        c.set_weights(store.empty_like()).unwrap();
        g = eng.and(g, c).unwrap();
    }
    let targets: Vec<VtreeIdx> = tree.bottomup().filter(|&t| !tree.node(t).is_leaf()).collect();
    let expected = value(&eng, &eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap());
    let mut refusals = 0;
    for n in 0.. {
        eng.limits().refuse_nth_reserve(n);
        let outcome = eng.and_marginalizing(f.clone(), g.clone(), &targets);
        let fired = !eng.limits().refusal_pending();
        eng.limits().grant_every_reserve();
        match outcome {
            Ok(out) => assert_eq!(value(&eng, &out).as_scaled(), expected.as_scaled(), "granted at {n}"),
            Err(_) => {
                refusals += 1;
                let again = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
                assert_eq!(value(&eng, &again).as_scaled(), expected.as_scaled(), "repeated after a refusal at {n}");
            }
        }
        if !fired {
            break;
        }
        assert!(n < 10_000, "never past the last reserve");
    }
    assert!(refusals > 3, "only {refusals} refusal points");
}

/// The scaled-float and signed-log arithmetics agree on a conjunction that
/// sums out every level, nested ones included, of every vtree shape.
#[test]
fn full_sum_out_agrees_with_the_signed_log_domain() {
    let eng = Engine::new();
    let mut rng = Lcg::new(99);
    let num_vars = 9;
    for (shape, tree) in vtree_shapes(num_vars) {
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 14, width: 3 });
        let weights = draw_weights(&mut rng, num_vars, false);
        let mut values = Vec::new();
        for arithmetic in [Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
            let store = WeightStore::new(weights.clone(), arithmetic);
            let (a, b) = clauses.split_at(clauses.len() / 2);
            let build = |part: &[Vec<i32>]| {
                let mut d = eng.one(&tree);
                d.set_weights(store.empty_like()).unwrap();
                for clause in part {
                    let mut c = eng.clause(&tree, clause.iter().copied()).unwrap();
                    c.set_weights(store.empty_like()).unwrap();
                    d = eng.and(d, c).unwrap();
                }
                d
            };
            let targets: Vec<VtreeIdx> = tree.bottomup().filter(|&t| !tree.node(t).is_leaf()).collect();
            let out = eng.and_marginalizing(build(a), build(b), &targets).unwrap();
            let v = value(&eng, &out);
            values.push(match (v.as_log(), v.as_scaled()) {
                (Some(l), _) => (l.log10_abs(), l.sign >= 0),
                (_, Some(s)) => (s.log10_abs(), s.mantissa() >= 0.0),
                _ => unreachable!(),
            });
        }
        assert_eq!(values[0].1, values[1].1, "{shape}: sign");
        assert!(values[0].0 == values[1].0 || (values[0].0 - values[1].0).abs() < 1e-12, "{shape}: {values:?}");
    }
}
