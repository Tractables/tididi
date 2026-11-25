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
    let f = Tdd::cube(&home, [1, 2]).unwrap();
    (home, f)
}

/// The weighted route of `and_model_count` on a home's root goes through
/// the `_where` form, as on a projection, since targets are refused there,
/// and refuses the weights as a weighted root vtree of the same shape does.
#[test]
fn weighted_and_model_count_on_a_home_root() {
    let eng = Engine::new();
    let half = || BigRational::new(1.into(), 2.into());
    let weights = Weights::new(RationalWeights::from_literals(&vec![LiteralWeights { negative: half(), positive: half() }; 2]), Arithmetic::ExactRational);
    let home = Home::weighted(Vtree::balanced(2), weights).unwrap();
    let count = |vtree: &Arc<Vtree>| {
        let f = Tdd::cube(&crate::vtree::Home::unchecked(vtree), [1, 2]).unwrap();
        let g = Tdd::clause(&crate::vtree::Home::unchecked(vtree), [1, -2]).unwrap();
        crate::test_helpers::and_model_count_under(&eng, f, g, &[vtree.root()])
    };
    assert_eq!(count(&home), Err(OperationError::IncompatibleWeights));
}

/// `marginalize` refuses an internal summed-out level that is not a region
/// on a home's root, as on a projection (the smooth API no longer makes
/// one; the internal sum makes it here).
#[test]
fn marginalize_refuses_an_internal_summed_level_on_a_home_root() {
    let (home, mut f) = x_and_y();
    let root = home.root();
    crate::marginal::tests::marginalize_levels(&Engine::new(), &mut f, &[root]).unwrap();
    assert_eq!(f.marginalize(|_| true), Err(OperationError::MarginalLevel(root)));
}

/// A summed variable leaf on a home's root is placed as on a projection
/// (the smooth API no longer makes one; the internal sum makes it here).
#[test]
fn embed_places_a_summed_leaf_of_a_home_root() {
    let (home, mut f) = x_and_y();
    let eng = Engine::new();
    let x = home.leaf_of(VarId(1)).unwrap();
    crate::marginal::tests::marginalize_levels(&eng, &mut f, &[x]).unwrap();
    let count = f.model_count().unwrap();
    let into = Arc::new(Vtree::balanced(2));
    let (placed, _) = crate::test_helpers::moved(&eng, f, &into, |v| v).unwrap();
    assert_eq!(placed.model_count().unwrap(), count);
}

/// `tag_level` into a home's root reads the root as a projection: the
/// code's side follows the leaf order, here on the left.
#[test]
fn tag_level_into_a_home_root_with_the_code_on_the_left() {
    let eng = Engine::new();
    let pair = Home::new(Vtree::balanced(2));
    let f = Tdd::cube(&pair, [1, 2]).unwrap();
    let code = Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap();
    let home = Home::new(Vtree::join(&code, &Vtree::balanced(2)).unwrap());
    let g = eng.tag_level(&f, pair.root(), &home, |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10]).unwrap();
    let expected = eng.and(Tdd::cube(&home, [1, 2]).unwrap(), Tdd::cube(&home, [-3, 4]).unwrap()).unwrap();
    assert!(eng.equivalent(&g, &expected).unwrap());
}
