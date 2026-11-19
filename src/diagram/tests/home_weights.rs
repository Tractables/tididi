//! Weights held by a home: checked once, read by every diagram on it, and
//! never carried by one.

use std::sync::Arc;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, TddBuildError, Weights};
use crate::restructure::search::{MinimizePairs, RotationSearchConfig};
use crate::test_helpers::{exact_weight, rat};
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn table(n: usize) -> Weights {
    let literals: Vec<_> = (0..n as i64)
        .map(|i| LiteralWeights { negative: rat(1, 2 + i), positive: rat(1 + i, 3) })
        .collect();
    Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

fn value_of(f: &Tdd) -> num_rational::BigRational {
    exact_weight(&f.weighted_value().unwrap().unwrap())
}

#[test]
fn a_home_takes_weights_once_and_keeps_them() {
    assert_eq!(
        Home::weighted(Vtree::balanced(4), table(3)).unwrap_err(),
        TddBuildError::MissingVariableWeight(VarId(4))
    );
    let home = Home::weighted(Vtree::balanced(4), table(4)).unwrap();
    assert_eq!(home.weights(), Some(&table(4)));
    let own = Vtree::projection(&home, [VarId(1), VarId(3)]).unwrap();
    let again = Home::weighted(Arc::clone(&own), table(4)).unwrap();
    assert!(Arc::ptr_eq(again.vtree(), home.vtree()));
    assert_eq!(Home::weighted(Arc::clone(home.vtree()), table(5)).unwrap_err(), TddBuildError::IncompatibleWeights);
    let plain = Home::new(Vtree::balanced(4));
    assert_eq!(plain.weights(), None);
    assert_eq!(Home::weighted(Arc::clone(plain.vtree()), table(4)).unwrap_err(), TddBuildError::IncompatibleWeights);
    assert!(Home::new(Arc::clone(&own)).weights().is_some());
    // A copy is not a home but keeps the weights, which a rotation of the
    // copy does not change; a home made on it has them.
    let copy = Vtree::clone(home.vtree());
    assert!(!copy.is_home());
    assert_eq!(copy.weights(), home.weights());
    assert_eq!(Home::new(copy).weights(), home.weights());
}

#[test]
fn a_diagram_on_a_weighted_home_reads_the_home_weights_and_carries_none() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::balanced(4), table(4)).unwrap();
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = eng.and_clause(eng.clause(&own, [1, -2]).unwrap(), &[2, 3]).unwrap();
    assert_eq!(f.weights(), home.weights());
    assert_eq!(f.set_weights(table(4)).unwrap_err(), TddBuildError::IncompatibleWeights);
    assert_eq!(f.take_weights().unwrap(), None);
    assert_eq!(f.weights(), home.weights());
    let mut b = Tdd::builder(&eng, &own).unwrap();
    assert_eq!(b.set_weights(table(4)).unwrap_err(), TddBuildError::IncompatibleWeights);

    let mut bare = eng.and_clause(eng.clause(&Arc::new(Vtree::balanced(4)), [1, -2]).unwrap(), &[2, 3]).unwrap();
    bare.set_weights(table(4)).unwrap();
    assert_eq!(value_of(&f), value_of(&bare));
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    assert!(f.has_marginal_level() && f.weights.is_none());
    assert!(f.vtree().bottomup().all(|t| !f.level(t).is_marginal() || f.level(t).is_weight_marginal()));
    assert_eq!(value_of(&f), value_of(&bare));
    crate::test_helpers::assert_canonical(&f);
}

/// The weights of a diagram on a vtree that is not a weighted home: equal
/// ones may be attached again at any time, other ones only while no level
/// is summed out.
#[test]
fn equal_weights_are_attached_again_and_other_weights_refused_after_a_sum() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = eng.and_clause(eng.clause(&vtree, [1, -2]).unwrap(), &[2, 3]).unwrap();
    let other = Weights::new(RationalWeights::unit(4), Arithmetic::ExactRational);
    f.set_weights(other.clone()).unwrap();
    f.set_weights(table(4)).unwrap();
    let before = value_of(&f);
    let (left, _) = vtree.children(vtree.root());
    eng.marginalize_levels(&mut f, &[left]).unwrap();
    f.set_weights(table(4)).unwrap();
    assert_eq!(value_of(&f), before);
    assert_eq!(f.set_weights(other).unwrap_err(), TddBuildError::IncompatibleWeights);
    let scaled = Weights::new(table(4).algebra().clone(), Arithmetic::ScaledFloat);
    assert!(f.set_weights(scaled).is_err());
    assert_eq!(value_of(&f), before);
}

#[test]
fn a_borrowed_weighted_operand_on_a_home_is_not_cloned() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::balanced(8), table(8)).unwrap();
    let mut f = eng.clause(&Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap(), [1, -2, 3]).unwrap();
    let mut g = eng.clause(&Vtree::projection(&home, [VarId(5), VarId(6), VarId(7)]).unwrap(), [-5, 6, 7]).unwrap();
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    eng.marginalize(&mut g, |v| v.0 >= 6).unwrap();
    let holders = Arc::strong_count(&home.weights().unwrap().table);
    let h = eng.and(&f, &g).unwrap();
    assert!(h.weights.is_none());
    assert_eq!(Arc::strong_count(&home.weights().unwrap().table), holders);
    assert_eq!(value_of(&h), value_of(&f) * value_of(&g) / value_of(&Tdd::one(&Vtree::empty_projection(&home))));
}

/// A search that keeps moves on a weighted home's root ends on a home with
/// the same weights.
#[test]
fn a_rotation_search_keeps_the_weights_of_the_home() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::linear(6), table(6)).unwrap();
    let mut f = eng.and_clause(eng.clause(home.vtree(), [1, -4]).unwrap(), &[2, 5]).unwrap();
    f = eng.and_clause(f, &[-3, 6]).unwrap();
    let before = value_of(&f);
    drop(home);
    f.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).unwrap();
    assert_eq!(f.weights(), Some(&table(6)));
    assert_eq!(value_of(&f), before);
    eng.marginalize(&mut f, |v| v.0 % 2 == 0).unwrap();
    assert_eq!(value_of(&f), before);
}
