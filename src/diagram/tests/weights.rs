//! Attaching a weight store: refused once a level holds integer counts.

use std::sync::Arc;

use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::marginal::marginalize;
use crate::test_helpers::{exact_weight, rat};
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

#[test]
#[should_panic(expected = "Tdd::set_weights: level")]
fn a_store_is_refused_after_a_level_was_summed_out_as_counts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    marginalize(&eng, &mut f, &[left]).unwrap();
    let weights: Vec<_> = (0..4).map(|_| (rat(1, 2), rat(1, 3))).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_weights(&weights), Arithmetic::ExactRational));
}

/// A weighted diagram, with the level under the root's left child summed out.
fn weighted_with_a_marginal_level(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(vtree, [1, -2]) & Tdd::clause(vtree, [2, 3]);
    let weights: Vec<_> = (0..4).map(|_| (rat(1, 2), rat(1, 3))).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_weights(&weights), Arithmetic::ExactRational));
    marginalize(eng, &mut f, &[left]).unwrap();
    assert!(f.level(left).is_weight_marginal());
    f
}

#[test]
fn a_build_copying_a_weight_marginal_level_finishes_with_its_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = weighted_with_a_marginal_level(&eng, &vtree);

    let mut b = Tdd::build(&eng, &vtree);
    for (t, _, _) in vtree.internal_bottomup() {
        b.copy_level(t, f.level(t));
    }
    b.set_weights(f.weights().expect("the fixture is weighted").clone());
    let g = b.finish(f.output()).expect("the store holds the copied level's values");
    let value = |t: &Tdd| exact_weight(&crate::query::weighted_value(t).expect("the diagram is weighted"));
    assert_eq!(value(&g), value(&f));
}

#[test]
fn a_build_copying_a_weight_marginal_level_needs_a_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    let (left, _) = vtree.children(vtree.root());

    let mut b = Tdd::build(&eng, &vtree);
    for (t, _, _) in vtree.internal_bottomup() {
        b.copy_level(t, f.level(t));
    }
    match b.finish(f.output()) {
        Ok(_) => panic!("a weight-marginal level with no store was accepted"),
        Err(e) => assert!(matches!(e, crate::diagram::TddBuildError::WeightedLevelWithoutStore { level } if level == left)),
    }
}
