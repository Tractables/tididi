//! Weights held by a home or a root vtree: checked once and read by every
//! diagram on it.

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
    assert!(Arc::ptr_eq(&again, &home));
    assert_eq!(Home::weighted(Arc::clone(&home), table(5)).unwrap_err(), TddBuildError::IncompatibleWeights);
    let plain = Home::new(Vtree::balanced(4));
    assert_eq!(plain.weights(), None);
    assert_eq!(Home::weighted(Arc::clone(&plain), table(4)).unwrap_err(), TddBuildError::IncompatibleWeights);
    assert!(Home::new(Arc::clone(&own)).weights().is_some());
    // A copy is not a home but keeps the weights, which a rotation of the
    // copy does not change; a home made on it has them.
    let copy = Vtree::clone(&home);
    assert!(!copy.is_home());
    assert_eq!(copy.weights(), home.weights());
    assert_eq!(Home::new(copy).weights(), home.weights());
}

#[test]
fn a_diagram_on_a_weighted_home_reads_the_home_weights() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::balanced(4), table(4)).unwrap();
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = eng.and_clause(eng.clause(&crate::vtree::Home::unchecked(&own), [1, -2]).unwrap(), &[2, 3]).unwrap();
    assert_eq!(f.weights(), home.weights());

    let root = Arc::new(Vtree::balanced(4).with_weights(table(4)).unwrap());
    let g = eng.and_clause(eng.clause(&crate::vtree::Home::unchecked(&root), [1, -2]).unwrap(), &[2, 3]).unwrap();
    assert_eq!(value_of(&f), value_of(&g));
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    assert!(f.has_marginal_level());
    assert_eq!(f.weights(), home.weights());
    assert!(f.vtree().bottomup().all(|t| !f.level(t).is_marginal() || f.level(t).is_weight_marginal()));
    assert_eq!(value_of(&f), value_of(&g));
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn a_borrowed_weighted_operand_on_a_home_is_not_cloned() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::balanced(8), table(8)).unwrap();
    let mut f = eng.clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap()), [1, -2, 3]).unwrap();
    let mut g = eng.clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [VarId(5), VarId(6), VarId(7)]).unwrap()), [-5, 6, 7]).unwrap();
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    eng.marginalize(&mut g, |v| v.0 >= 6).unwrap();
    let holders = Arc::strong_count(&home.weights().unwrap().table);
    let h = eng.and(&f, &g).unwrap();
    assert_eq!(Arc::strong_count(&home.weights().unwrap().table), holders);
    assert_eq!(value_of(&h), value_of(&f) * value_of(&g) / value_of(&Tdd::one(&crate::vtree::Home::unchecked(&Vtree::empty_projection(&home)))));
}

/// A search that keeps moves on a weighted home's root ends on a home with
/// the same weights.
#[test]
fn a_rotation_search_keeps_the_weights_of_the_home() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::linear(6), table(6)).unwrap();
    let mut f = eng.and_clause(eng.clause(&home, [1, -4]).unwrap(), &[2, 5]).unwrap();
    f = eng.and_clause(f, &[-3, 6]).unwrap();
    let before = value_of(&f);
    drop(home);
    f.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).unwrap();
    assert_eq!(f.weights(), Some(&table(6)));
    assert_eq!(value_of(&f), before);
    eng.marginalize(&mut f, |v| v.0 % 2 == 0).unwrap();
    assert_eq!(value_of(&f), before);
}

/// A root that is not a home, with weights: the level sum-outs and placement
/// compute what a weighted home computes.
#[test]
fn a_weighted_root_computes_what_a_weighted_home_computes() {
    assert_eq!(Vtree::balanced(4).with_weights(table(3)).unwrap_err(), TddBuildError::MissingVariableWeight(VarId(4)));
    let home = Home::new(Vtree::balanced(4));
    assert!(Vtree::clone(&home).with_weights(table(4)).is_ok());
    let own = Vtree::projection(&home, [VarId(1), VarId(3)]).unwrap();
    assert_eq!(Vtree::clone(&own).with_weights(table(4)).unwrap_err(), TddBuildError::IncompatibleWeights);
    let again = Vtree::balanced(4).with_weights(table(4)).unwrap();
    assert_eq!(again.clone().with_weights(table(5)).unwrap_err(), TddBuildError::IncompatibleWeights);
    assert_eq!(again.with_weights(table(4)).unwrap().weights(), Some(&table(4)));

    let eng = Engine::new();
    let weighted = Arc::new(Vtree::balanced(6).with_weights(table(6)).unwrap());
    let home = Home::weighted(Vtree::balanced(6), table(6)).unwrap();
    let build = |vtree: &Arc<Vtree>| {
        let f = eng.and_clause(eng.clause(&crate::vtree::Home::unchecked(vtree), [1, -2, 4]).unwrap(), &[2, 3, -6]).unwrap();
        let g = eng.and_clause(eng.clause(&crate::vtree::Home::unchecked(vtree), [-1, 5]).unwrap(), &[3, -4, 6]).unwrap();
        (f, g)
    };
    let (f, g) = build(&weighted);
    let (fh, gh) = build(&home);
    assert_eq!(f.weights(), Some(&table(6)));
    assert_eq!(value_of(&f), value_of(&fh));
    let (left, _) = weighted.children(weighted.root());
    let h = crate::test_helpers::and_marginalizing_under(&eng, &f, &g, &[left]).unwrap();
    let under_left: Vec<VarId> = weighted.leaf_bottomup().filter(|&(l, _)| weighted.in_subtree(l, left)).map(|(_, v)| v).collect();
    let hh = eng.and_marginalizing_where(&fh, &gh, |v| under_left.contains(&v)).unwrap();
    assert!(h.level(left).is_weight_marginal());
    assert_eq!(value_of(&h), value_of(&hh));

}
