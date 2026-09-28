//! Embedding diagrams that have summed levels out.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, TddBuildError, WeightStore};
use crate::restructure::EmbedError;
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

/// The exact weighted value of a diagram.
fn exact(f: &Tdd) -> BigRational {
    f.weighted_value().unwrap().expect("weighted").into_rational()
}

/// A table giving every variable the weights `positive` and `negative`.
fn table(num_vars: usize, positive: BigRational, negative: BigRational) -> RationalWeights {
    RationalWeights::from_literals(&vec![LiteralWeights { negative, positive }; num_vars])
}

/// `(x1 ∨ x2) ∧ (x3 ∨ x4)` on a balanced four-leaf vtree, weighted, with the
/// level over `x1, x2` summed out; and the same formula before either.
fn summed_pair(arithmetic: Arithmetic, positive: BigRational, negative: BigRational) -> (Arc<Vtree>, Tdd, Tdd) {
    let small = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&small, [1, 2]).unwrap() & Tdd::clause(&small, [3, 4]).unwrap();
    f.minimize().unwrap();
    let structural = f.clone();
    f.set_weights(WeightStore::new(table(4, positive, negative), arithmetic)).unwrap();
    let (left, _) = small.children(small.root());
    Engine::new().marginalize_levels(&mut f, &[left]).unwrap();
    assert!(f.level(left).is_weight_marginal(), "setup: the level over x1, x2 holds values");
    assert_canonical(&f);
    (small, f, structural)
}

/// Every source variable to the odd variables of an eight-leaf vtree: the
/// even ones are free, two of them under the summed-out level's image.
fn spread(v: VarId) -> VarId {
    VarId(2 * v.0 - 1)
}

#[test]
fn a_summed_out_level_keeps_its_value_scaled_by_the_free_variables_below_it() {
    let (small, f, structural) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let value = exact(&f);
    let big = Arc::new(Vtree::balanced(8));
    let engine = Engine::new();
    let store = || WeightStore::new(table(8, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    let (g, levels) = engine.embed_over(&f, &big, spread, Some(store())).unwrap();
    assert_canonical(&g);
    let (left, _) = small.children(small.root());
    let image = levels.level_of(left);
    assert!(g.level(image).is_weight_marginal(), "the summed-out level is placed with its values");
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
            assert!(g.level(t).is_marginal(), "a level under the image is marginal");
            assert_eq!(g.level(t).slot_count(), 0, "and holds no values of its own");
        }
    }
    // Four free variables, each weighing 1/3 + 1/2 whether under the
    // summed-out level or beside it.
    let free = rat(5, 6) * rat(5, 6) * rat(5, 6) * rat(5, 6);
    assert_eq!(exact(&g), value * free);

    // The same diagram summed out on the destination instead.
    let (mut h, _) = engine.embed(&structural, &big, spread).unwrap();
    h.set_weights(store()).unwrap();
    engine.marginalize_levels(&mut h, &[image]).unwrap();
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
fn log_domain_values_are_placed_like_exact_ones() {
    let (small, f, structural) = summed_pair(Arithmetic::SignedLog, rat(1, 3), rat(1, 2));
    let big = Arc::new(Vtree::balanced(8));
    let engine = Engine::new();
    let store = || WeightStore::new(table(8, rat(1, 3), rat(1, 2)), Arithmetic::SignedLog);
    let (g, levels) = engine.embed_over(&f, &big, spread, Some(store())).unwrap();
    assert_canonical(&g);
    let (left, _) = small.children(small.root());
    let (mut h, _) = engine.embed(&structural, &big, spread).unwrap();
    h.set_weights(store()).unwrap();
    engine.marginalize_levels(&mut h, &[levels.level_of(left)]).unwrap();
    let value = |t: &Tdd| *t.weighted_value().unwrap().expect("weighted").as_log().expect("log domain");
    let (vg, vh) = (value(&g), value(&h));
    assert_eq!(vg.sign, vh.sign);
    assert!((vg.ln_abs - vh.ln_abs).abs() < 1e-12, "{} against {}", vg.ln_abs, vh.ln_abs);
    // Exact: (3/4 · 3/4) of a unit table scaled to the weights: check against
    // the exact run of the same construction.
    let (_, exact, _) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let expected = self::exact(&exact) * rat(625, 1296);
    let expected = expected.numer().to_string().parse::<f64>().unwrap() / expected.denom().to_string().parse::<f64>().unwrap();
    assert!((vg.ln_abs.exp() - expected).abs() < 1e-12);
}

#[test]
fn a_summed_out_leaf_is_placed_under_a_new_parent() {
    // Equal weights: the leaf's `¬x` slot canonicalizes onto its `x` slot,
    // so the pass-through over it has twins to contract.
    let small = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&small, [1, 2]).unwrap();
    f.minimize().unwrap();
    let structural = f.clone();
    let store = |n| WeightStore::new(table(n, rat(1, 3), rat(1, 3)), Arithmetic::ExactRational);
    f.set_weights(store(2)).unwrap();
    let engine = Engine::new();
    let leaf = small.leaf_of(VarId(1)).unwrap();
    engine.marginalize_levels(&mut f, &[leaf]).unwrap();
    assert!(f.level(leaf).is_weight_marginal());
    assert_canonical(&f);
    let value = exact(&f);

    // x1 to 2 and x2 to 4: both land as the mapped side of a pass-through.
    let big = Arc::new(Vtree::balanced(4));
    let (g, levels) = engine.embed_over(&f, &big, |v| VarId(2 * v.0), Some(store(4))).unwrap();
    assert_canonical(&g);
    assert!(g.level(levels.level_of(leaf)).is_weight_marginal());
    let free = rat(2, 3) * rat(2, 3);
    assert_eq!(exact(&g), value * free);

    let (mut h, _) = engine.embed(&structural, &big, |v| VarId(2 * v.0)).unwrap();
    h.set_weights(store(4)).unwrap();
    engine.marginalize_levels(&mut h, &[levels.level_of(leaf)]).unwrap();
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
fn a_structural_diagram_takes_the_store_it_is_given() {
    let small = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&small, [1, 2]).unwrap();
    let big = Arc::new(Vtree::balanced(4));
    let engine = Engine::new();
    let store = WeightStore::new(table(4, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    let (g, _) = engine.embed_over(&f, &big, |v| VarId(2 * v.0), Some(store)).unwrap();
    assert_canonical(&g);
    assert!(g.weights().is_some());
    let (plain, _) = engine.embed(&f, &big, |v| VarId(2 * v.0)).unwrap();
    assert_eq!(g.node_count(), plain.node_count());
    assert_eq!(exact(&g), plain.evaluate(&table(4, rat(1, 3), rat(1, 2))).unwrap());
    let (bare, _) = engine.embed_over(&f, &big, |v| VarId(2 * v.0), None).unwrap();
    assert!(bare.weights().is_none());
}

#[test]
fn integer_counts_are_refused() {
    let small = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&small, [1, 2]).unwrap();
    let engine = Engine::new();
    engine.marginalize_levels(&mut f, &[small.root()]).unwrap();
    let big = Arc::new(Vtree::balanced(4));
    let store = WeightStore::new(table(4, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    for weights in [None, Some(store)] {
        assert_eq!(
            engine.embed_over(&f, &big, |v| v, weights).unwrap_err(),
            EmbedError::Operation(OperationError::MarginalLevel(small.root())),
        );
    }
}

#[test]
fn weighted_values_need_a_destination_store_that_agrees_with_the_source() {
    let (small, f, _) = summed_pair(Arithmetic::ExactRational, rat(1, 3), rat(1, 2));
    let big = Arc::new(Vtree::balanced(8));
    let engine = Engine::new();
    let (left, _) = small.children(small.root());
    assert_eq!(
        engine.embed_over(&f, &big, spread, None).unwrap_err(),
        EmbedError::SourceWeights(TddBuildError::WeightedLevelWithoutStore { level: left }),
    );
    let other = WeightStore::new(table(8, rat(1, 4), rat(1, 2)), Arithmetic::ExactRational);
    assert_eq!(
        engine.embed_over(&f, &big, spread, Some(other)).unwrap_err(),
        EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
    );
    let log = WeightStore::new(table(8, rat(1, 3), rat(1, 2)), Arithmetic::SignedLog);
    assert_eq!(
        engine.embed_over(&f, &big, spread, Some(log)).unwrap_err(),
        EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
    );
    let short = WeightStore::new(table(6, rat(1, 3), rat(1, 2)), Arithmetic::ExactRational);
    assert_eq!(
        engine.embed_over(&f, &big, spread, Some(short)).unwrap_err(),
        EmbedError::DestinationWeights(TddBuildError::MissingVariableWeight(VarId(7))),
    );
}
