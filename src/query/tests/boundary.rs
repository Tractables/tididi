//! The queries at their boundaries: a marginal output level, ⊥, a pin outside
//! the counter.

use crate::vtree::Home;
use super::*;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};

use crate::query::count::{PinSemantics, Retention};
use crate::test_helpers::{assert_canonical, rat};

/// `(x0 ∨ x2) ∧ (¬x1 ∨ x3)` over `balanced(4)`, minimized.
fn two_clauses(vtree: &Arc<Vtree>) -> Tdd {
    let t1 = clause_to_tdd(&crate::vtree::Home::unchecked(vtree), &[Literal::pos(VarId(1)), Literal::pos(VarId(3))]);
    let t2 = clause_to_tdd(&crate::vtree::Home::unchecked(vtree), &[Literal::neg(VarId(2)), Literal::pos(VarId(4))]);
    let mut t = apply_and(t1, t2);
    t.minimize().unwrap();
    t
}

/// Every internal level of `vtree`, bottom-up, which is the order `marginalize_levels`
/// needs to sum the whole diagram out.
fn internal_levels_bottom_up(vtree: &Vtree) -> Vec<VtreeIdx> {
    vtree
        .bottomup_slice()
        .iter()
        .copied()
        .filter(|&t| !vtree.node(t).is_leaf())
        .collect()
}

#[test]
fn a_count_marginal_output_answers_from_its_count() {
    let eng = &Engine::new();
    let vtree = std::sync::Arc::clone(&crate::vtree::Home::new(Vtree::balanced(4)));
    let mut f = two_clauses(&vtree);
    let before = f.model_count().unwrap();
    crate::marginal::tests::marginalize_levels(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].marginal_counts().is_some(), "output level is count-marginal");
    assert_eq!(f.model_count().unwrap(), before);
    assert!(f.is_sat().unwrap());
}

#[test]
fn a_weight_marginal_output_is_satisfiable() {
    let eng = &Engine::new();
    let weights: Vec<_> = (0..4).map(|_| LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }).collect();
    let weights = Weights::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational);
    let vtree = std::sync::Arc::clone(&crate::vtree::Home::weighted(Vtree::balanced(4), weights).unwrap());
    let mut f = two_clauses(&vtree);
    crate::marginal::tests::marginalize_levels(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].is_weight_marginal(), "output level is weight-marginal");
    assert_eq!(f.is_sat(), Ok(true));
}

#[test]
fn the_incremental_count_of_bottom_is_zero() {
    let eng = &Engine::new();
    let vtree = Home::new(Vtree::balanced(3));
    let f = crate::build::constant_zero(eng, &vtree);
    assert!(f.is_zero());
    let mut counter = eng.counter_with(&f, Retention::All, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.model_count().unwrap(), BigUint::ZERO);
}

#[test]
fn weighted_leaf_outputs_are_satisfiable_unless_false() {
    let eng = Engine::new();
    for weight in [rat(0, 1), rat(1, 2)] {
        for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
            let weights = Weights::new(RationalWeights::from_literals(&[
                LiteralWeights { negative: weight.clone(), positive: weight.clone() }
            ]), arithmetic);
            let tree = crate::vtree::Home::new(Vtree::balanced(1).with_weights(weights).unwrap());
            let mut f = eng.literal(&tree, 1).unwrap();
            assert_canonical(&f);
            crate::marginal::tests::marginalize_levels(&eng, &mut f, &[tree.root()]).unwrap();
            assert!(f.level(tree.root()).is_weight_marginal());
            assert_eq!(f.is_sat(), Ok(true));
            f.output.local = crate::diagram::ZERO;
            assert!(!f.is_sat().unwrap());
        }
    }
}

// Padded form: the padded sum-out.
#[test]
fn structural_and_count_marginal_leaf_outputs_remain_satisfiable() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(1));
    for mut f in [eng.one(&crate::vtree::Home::unchecked(&tree)), eng.literal(&crate::vtree::Home::unchecked(&tree), 1).unwrap(), eng.literal(&crate::vtree::Home::unchecked(&tree), -1).unwrap()] {
        assert_canonical(&f);
        assert!(f.is_sat().unwrap());
        crate::marginal::tests::marginalize_levels(&eng, &mut f, &[tree.root()]).unwrap();
        assert!(f.level(tree.root()).marginal_counts().is_some());
        assert!(f.is_sat().unwrap());
    }
}
