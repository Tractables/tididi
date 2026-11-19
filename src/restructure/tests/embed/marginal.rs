//! Embedding diagrams that have summed levels out.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{
    Arithmetic, LiteralWeights, RationalWeights, TddBuildError, Weights, WeightValue,
};
use crate::limits::LimitConfig;
use crate::restructure::EmbedError;
use crate::test_helpers::{assert_canonical, bounded_log10};
use crate::vtree::{VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

/// The exact weighted value of a diagram.
fn exact(f: &Tdd) -> BigRational {
    f.weighted_value()
        .unwrap()
        .expect("weighted")
        .into_rational()
}

/// A table giving every variable the weights `positive` and `negative`.
fn table(num_vars: usize, positive: BigRational, negative: BigRational) -> RationalWeights {
    RationalWeights::from_literals(&vec![LiteralWeights { negative, positive }; num_vars])
}

/// `(x1 ∨ x2) ∧ (x3 ∨ x4)` on a balanced four-leaf vtree, weighted, with the
/// level over `x1, x2` summed out; and the same formula before either.
fn summed_pair(
    arithmetic: Arithmetic,
    positive: BigRational,
    negative: BigRational,
) -> (Arc<Vtree>, Tdd, Tdd) {
    let formula = |vtree: &Arc<Vtree>| {
        let mut f = Tdd::clause(vtree, [1, 2]).unwrap() & Tdd::clause(vtree, [3, 4]).unwrap();
        f.minimize().unwrap();
        f
    };
    let structural = formula(&Arc::new(Vtree::balanced(4)));
    let weights = Weights::new(table(4, positive, negative), arithmetic);
    let small = Arc::new(Vtree::balanced(4).with_weights(weights).unwrap());
    let mut f = formula(&small);
    let (left, _) = small.children(small.root());
    Engine::new().marginalize_levels(&mut f, &[left]).unwrap();
    assert!(
        f.level(left).is_weight_marginal(),
        "setup: the level over x1, x2 holds values"
    );
    assert_canonical(&f);
    (small, f, structural)
}

/// Every source variable to the odd variables of an eight-leaf vtree: the
/// even ones are free, two of them under the summed-out level's image.
fn spread(v: VarId) -> VarId {
    VarId(2 * v.0 - 1)
}

fn embed_and_marginalize(
    engine: &Engine,
    structural: &Tdd,
    dest: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    target: crate::vtree::VtreeIdx,
) -> Tdd {
    let (mut h, _) = engine.embed(structural, dest, map).unwrap();
    engine.marginalize_levels(&mut h, &[target]).unwrap();
    h
}

#[test]
fn a_summed_out_level_keeps_its_value_scaled_by_the_free_variables_below_it() {
    let (small, f, structural) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let value = exact(&f);
    let store = Weights::new(table(8, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    let big = Arc::new(Vtree::balanced(8).with_weights(store).unwrap());
    let engine = Engine::new();
    let (g, levels) = engine.embed_over(&f, &big, spread).unwrap();
    assert_canonical(&g);
    let (left, _) = small.children(small.root());
    let image = levels.level_of(left);
    assert!(
        g.level(image).is_weight_marginal(),
        "the summed-out level is placed with its values"
    );
    for t in big.bottomup() {
        let below = {
            let mut t = t;
            loop {
                match big.node(t).parent() {
                    Some(p) if p == image => break true,
                    Some(p) => t = p,
                    None => break false,
                }
            }
        };
        if below && !big.node(t).is_leaf() {
            assert!(
                g.level(t).is_marginal(),
                "a level under the image is marginal"
            );
            assert_eq!(g.level(t).slot_count(), 0, "and holds no values of its own");
        }
    }
    // Four free variables, each weighing 1/3 + 1/2 whether under the
    // summed-out level or beside it.
    let free = rat(5, 6) * rat(5, 6) * rat(5, 6) * rat(5, 6);
    assert_eq!(exact(&g), value * free);

    // The same diagram summed out on the destination instead.
    let h = embed_and_marginalize(&engine, &structural, &big, spread, image);
    assert_canonical(&h);
    assert_eq!(exact(&g), exact(&h));
    // Both serve as operands alike.
    let k = || engine.clause(&big, [-6, 8]).unwrap();
    let gk = engine.and(g.clone(), k()).unwrap();
    let hk = engine.and(h.clone(), k()).unwrap();
    assert_canonical(&gk);
    assert_eq!(exact(&gk), exact(&hk));
    assert_eq!(gk.node_count(), hk.node_count());
}

#[test]
fn bounded_values_are_placed_like_exact_ones() {
    for arithmetic in [Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
        let (small, f, structural) = summed_pair(arithmetic, rat(1, 3), rat(1, 2));
        let store = Weights::new(table(8, rat(1, 3), rat(1, 2)), arithmetic);
        let big = Arc::new(Vtree::balanced(8).with_weights(store).unwrap());
        let engine = Engine::new();
        let (g, levels) = engine.embed_over(&f, &big, spread).unwrap();
        assert_canonical(&g);
        let (left, _) = small.children(small.root());
        let h = embed_and_marginalize(
            &engine,
            &structural,
            &big,
            spread,
            levels.level_of(left),
        );
        let value = |t: &Tdd| {
            let v = t.weighted_value().unwrap().expect("weighted");
            let sign = match &v {
                WeightValue::Log(s) => f64::from(s.sign),
                WeightValue::Scaled(s) => s.mantissa().signum(),
                _ => panic!("expected a bounded-precision value"),
            };
            (
                sign,
                bounded_log10(&v).expect("bounded") * std::f64::consts::LN_10,
            )
        };
        let (vg, vh) = (value(&g), value(&h));
        assert_eq!(vg.0, vh.0);
        assert!((vg.1 - vh.1).abs() < 1e-12, "{} against {}", vg.1, vh.1);
        // Exact: (3/4 · 3/4) of a unit table scaled to the weights: check against
        // the exact run of the same construction.
        let (_, exact, _) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
        let expected = self::exact(&exact) * rat(625, 1296);
        let expected = expected.numer().to_string().parse::<f64>().unwrap()
            / expected.denom().to_string().parse::<f64>().unwrap();
        assert!((vg.1.exp() - expected).abs() < 1e-12);
    }
}

#[test]
fn a_summed_out_leaf_is_placed_under_a_new_parent() {
    // Equal weights: the leaf's `¬x` slot canonicalizes onto its `x` slot,
    // so the pass-through over it has twins to contract.
    let store = |n| Weights::new(table(n, rat(1, 3), rat(1, 3)), Arithmetic::ExactRational);
    let formula = |vtree: &Arc<Vtree>| {
        let mut f = Tdd::clause(vtree, [1, 2]).unwrap();
        f.minimize().unwrap();
        f
    };
    let structural = formula(&Arc::new(Vtree::balanced(2)));
    let small = Arc::new(Vtree::balanced(2).with_weights(store(2)).unwrap());
    let mut f = formula(&small);
    let engine = Engine::new();
    let leaf = small.leaf_of(VarId(1)).unwrap();
    engine.marginalize_levels(&mut f, &[leaf]).unwrap();
    assert!(f.level(leaf).is_weight_marginal());
    assert_canonical(&f);
    let value = exact(&f);

    // x1 to 2 and x2 to 4: both land as the mapped side of a pass-through.
    let big = Arc::new(Vtree::balanced(4).with_weights(store(4)).unwrap());
    let (g, levels) = engine.embed_over(&f, &big, |v| VarId(2 * v.0)).unwrap();
    assert_canonical(&g);
    assert!(g.level(levels.level_of(leaf)).is_weight_marginal());
    let free = rat(2, 3) * rat(2, 3);
    assert_eq!(exact(&g), value * free);

    let mut h = embed_and_marginalize(
        &engine,
        &structural,
        &big,
        |v| VarId(2 * v.0),
        levels.level_of(leaf),
    );
    // The equal values leave the parent's `(·, x)` and `(·, ¬x)` nodes twins.
    h.minimize().unwrap();
    assert_canonical(&h);
    assert_eq!(exact(&g), exact(&h));
    // With `x` and `¬x` weighing the same, `1` weighs twice either, and the
    // two constructions fuse the root's pairs into different shapes of the
    // same value; only the values compare.
    let k = || engine.clause(&big, [1, -3]).unwrap();
    let gk = engine.and(g.clone(), k()).unwrap();
    let hk = engine.and(h, k()).unwrap();
    assert_canonical(&gk);
    assert_eq!(exact(&gk), exact(&hk));
}

#[test]
fn a_structural_diagram_takes_the_destination_weights() {
    let small = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&small, [1, 2]).unwrap();
    let store = Weights::new(table(4, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    let weighted = Arc::new(Vtree::balanced(4).with_weights(store).unwrap());
    let big = Arc::new(Vtree::balanced(4));
    let engine = Engine::new();
    let (g, _) = engine.embed_over(&f, &weighted, |v| VarId(2 * v.0)).unwrap();
    assert_canonical(&g);
    assert!(g.weights().is_some());
    let (plain, _) = engine.embed(&f, &big, |v| VarId(2 * v.0)).unwrap();
    assert_eq!(g.node_count(), plain.node_count());
    assert_eq!(
        exact(&g),
        plain.evaluate(&table(4, rat(1, 3), rat(1, 2))).unwrap()
    );
    let (bare, _) = engine.embed_over(&f, &big, |v| VarId(2 * v.0)).unwrap();
    assert!(bare.weights().is_none());
}

#[test]
fn integer_counts_are_refused() {
    let small = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&small, [1, 2]).unwrap();
    let engine = Engine::new();
    engine.marginalize_levels(&mut f, &[small.root()]).unwrap();
    let store = Weights::new(table(4, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    let weighted = Vtree::balanced(4).with_weights(store).unwrap();
    for big in [Vtree::balanced(4), weighted] {
        assert_eq!(
            engine.embed_over(&f, &Arc::new(big), |v| v).unwrap_err(),
            EmbedError::Operation(OperationError::MarginalLevel(small.root())),
        );
    }
}

#[test]
fn weighted_values_need_destination_weights_that_agree_with_the_source() {
    let (small, f, _) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let big = |weights| Arc::new(Vtree::balanced(8).with_weights(weights).unwrap());
    let engine = Engine::new();
    let (left, _) = small.children(small.root());
    assert_eq!(
        engine.embed_over(&f, &Arc::new(Vtree::balanced(8)), spread).unwrap_err(),
        EmbedError::SourceWeights(TddBuildError::WeightedLevelWithoutStore { level: left }),
    );
    let other = Weights::new(table(8, rat(1, 4), rat(1, 2)), Arithmetic::ExactRational);
    assert_eq!(
        engine.embed_over(&f, &big(other), spread).unwrap_err(),
        EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
    );
    for arithmetic in [Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
        let bounded = Weights::new(table(8, rat(1, 3), rat(1, 2)), arithmetic);
        assert_eq!(
            engine.embed_over(&f, &big(bounded), spread).unwrap_err(),
            EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
        );
    }
}

#[test]
fn a_source_the_budget_cannot_copy_refuses_the_embed() {
    let (small, f, _) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let value = exact(&f);
    let engine = Engine::new();
    engine.try_clone(&f).unwrap();
    let copy = engine.limits().meters().in_flight_bytes;
    {
        let _budget = engine
            .limits()
            .scope(LimitConfig::none().with_memory_budget_bytes(Some(copy - 1)));
        assert_eq!(
            engine.embed_over(&f, &small, |v| v).unwrap_err(),
            EmbedError::Operation(OperationError::OverBudget),
        );
    }
    assert_canonical(&f);
    assert_eq!(exact(&f), value);
    let (g, _) = engine.embed_over(&f, &small, |v| v).unwrap();
    assert_canonical(&g);
    assert_eq!(exact(&g), value);
}
