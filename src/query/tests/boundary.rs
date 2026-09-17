//! The queries at their boundaries: a marginal output level, ⊥, a pin outside
//! the counter.

use super::*;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};

use crate::query::count::{KeepAllColumns, PinSemantics};
use crate::test_helpers::{assert_canonical, rat};

/// `(x0 ∨ x2) ∧ (¬x1 ∨ x3)` over `balanced(4)`, minimized.
fn two_clauses(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let t1 = clause_to_tdd(eng, vtree, &[Literal::pos(VarId(1)), Literal::pos(VarId(3))]);
    let t2 = clause_to_tdd(eng, vtree, &[Literal::neg(VarId(2)), Literal::pos(VarId(4))]);
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
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = two_clauses(eng, &vtree);
    let before = f.model_count().unwrap();
    eng.marginalize_levels(&mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].marginal_counts().is_some(), "output level is count-marginal");
    assert_eq!(f.model_count().unwrap(), before);
    assert!(f.is_sat().unwrap());
}

#[test]
fn a_weight_marginal_output_is_refused_by_name() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = two_clauses(eng, &vtree);
    let weights: Vec<_> = (0..4).map(|_| LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    eng.marginalize_levels(&mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].is_weight_marginal(), "output level is weight-marginal");
    assert_eq!(f.is_sat(), Err(crate::OperationError::IncompatibleWeights));
}

#[test]
fn the_incremental_count_of_bottom_is_zero() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = crate::build::constant_zero(eng, &vtree);
    assert!(f.is_zero());
    let mut counter = eng.counter_with::<KeepAllColumns>(&f, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.model_count().unwrap(), BigUint::ZERO);
}

#[test]
fn weighted_leaf_outputs_cannot_decide_structural_satisfiability() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(1));
    for weight in [rat(0, 1), rat(1, 2)] {
        for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
            let mut f = eng.literal(&tree, 1).unwrap();
            assert_canonical(&f);
            f.set_weights(WeightStore::new(RationalWeights::from_literals(&[
                LiteralWeights { negative: weight.clone(), positive: weight.clone() }
            ]), arithmetic)).unwrap();
            eng.marginalize_levels(&mut f, &[tree.root()]).unwrap();
            assert!(f.level(tree.root()).is_weight_marginal());
            assert_eq!(f.is_sat(), Err(crate::OperationError::IncompatibleWeights));
            f.output.local = crate::diagram::ZERO;
            assert!(!f.is_sat().unwrap());
        }
    }
}

#[test]
fn structural_and_count_marginal_leaf_outputs_remain_satisfiable() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(1));
    for mut f in [eng.one(&tree), eng.literal(&tree, 1).unwrap(), eng.literal(&tree, -1).unwrap()] {
        assert_canonical(&f);
        assert!(f.is_sat().unwrap());
        eng.marginalize_levels(&mut f, &[tree.root()]).unwrap();
        assert!(f.level(tree.root()).marginal_counts().is_some());
        assert!(f.is_sat().unwrap());
    }
}
