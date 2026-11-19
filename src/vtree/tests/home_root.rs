//! A home's root is the projection of its home on every leaf: each entry
//! point that asks whether a diagram is on a home answers as on any other
//! projection.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

/// Home (x1, x2) and x1 ∧ x2 on its root.
fn x_and_y() -> (Home, Tdd) {
    let home = Home::new(Vtree::balanced(2));
    let f = Tdd::cube(home.vtree(), [1, 2]).unwrap();
    (home, f)
}

fn names(error: &OperationError, option: &str, entry: &str) -> bool {
    matches!(error, OperationError::InertOption { option: o, needs } if *o == option && needs.contains(entry))
}

#[test]
fn marginalize_levels_refuses_a_home_root() {
    let (home, mut f) = x_and_y();
    let before = f.clone();
    let error = f.marginalize_levels(&[home.vtree().root()]).unwrap_err();
    assert!(names(&error, "levels", "`marginalize`"), "{error:?}");
    assert!(Engine::new().equivalent(&f, &before).unwrap() && !f.has_marginal_level());
}

#[test]
fn and_marginalizing_by_targets_refuses_a_home_root() {
    let (home, f) = x_and_y();
    let g = Tdd::clause(home.vtree(), [1, -2]).unwrap();
    let error = Engine::new().and_marginalizing(&f, &g, &[home.vtree().root()]).unwrap_err();
    assert!(names(&error, "targets", "`and_marginalizing_where`"), "{error:?}");
}

/// The weighted route of `and_model_count` on a home's root goes through
/// the `_where` form, as on a projection, since targets are refused there:
/// it answers as on a root vtree of the same shape.
#[test]
fn weighted_and_model_count_on_a_home_root() {
    let eng = Engine::new();
    let half = || BigRational::new(1.into(), 2.into());
    let weights = Weights::new(RationalWeights::from_literals(&vec![LiteralWeights { negative: half(), positive: half() }; 2]), Arithmetic::ExactRational);
    let home = Home::weighted(Vtree::balanced(2), weights.clone()).unwrap();
    let plain = Arc::new(Vtree::balanced(2).with_weights(weights).unwrap());
    let count = |vtree: &Arc<Vtree>| {
        let f = Tdd::cube(vtree, [1, 2]).unwrap();
        let g = Tdd::clause(vtree, [1, -2]).unwrap();
        eng.and_model_count(f, g, &[vtree.root()])
    };
    assert_eq!(count(home.vtree()), count(&plain));
}

/// `marginalize` refuses an internal summed-out level that is not a region
/// on a home's root, as on a projection (the smooth API no longer makes
/// one; the internal sum makes it here).
#[test]
fn marginalize_refuses_an_internal_summed_level_on_a_home_root() {
    let (home, mut f) = x_and_y();
    let root = home.vtree().root();
    crate::marginal::marginalize_levels(&Engine::new(), &mut f, &[root]).unwrap();
    assert_eq!(f.marginalize(|_| true), Err(OperationError::MarginalLevel(root)));
}

/// A summed variable leaf on a home's root is placed as on a projection
/// (the smooth API no longer makes one; the internal sum makes it here).
#[test]
fn embed_places_a_summed_leaf_of_a_home_root() {
    let (home, mut f) = x_and_y();
    let eng = Engine::new();
    let x = home.vtree().leaf_of(VarId(1)).unwrap();
    crate::marginal::marginalize_levels(&eng, &mut f, &[x]).unwrap();
    let count = f.model_count().unwrap();
    let into = Arc::new(Vtree::balanced(2));
    let (placed, _) = eng.embed_moving(f, &into, |v| v).map_err(|refused| refused.error).unwrap();
    assert_eq!(placed.model_count().unwrap(), count);
}

/// `tag_level` into a home's root reads the root as a projection: the
/// code's side follows the leaf order, here on the left.
#[test]
fn tag_level_into_a_home_root_with_the_code_on_the_left() {
    let eng = Engine::new();
    let pair = Arc::new(Vtree::balanced(2));
    let f = Tdd::cube(&pair, [1, 2]).unwrap();
    let code = Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap();
    let home = Home::new(Vtree::join(&code, &Vtree::balanced(2)).unwrap());
    let g = eng.tag_level(&f, pair.root(), home.vtree(), |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10]).unwrap();
    let expected = eng.and(Tdd::cube(home.vtree(), [1, 2]).unwrap(), Tdd::cube(home.vtree(), [-3, 4]).unwrap()).unwrap();
    assert!(eng.equivalent(&g, &expected).unwrap());
}
