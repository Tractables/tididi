//! The queries at their boundaries: a marginal output level, ⊥, a pin outside
//! the counter.

use super::*;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::marginal::marginalize_levels;
use crate::query::count::{ModelCounter, KeepAllColumns, PinSemantics};
use crate::test_helpers::rat;

/// `(x0 ∨ x2) ∧ (¬x1 ∨ x3)` over `balanced(4)`, minimized.
fn two_clauses(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let t1 = clause_to_tdd(eng, vtree, &[Literal::pos(VarId(0)), Literal::pos(VarId(2))]);
    let t2 = clause_to_tdd(eng, vtree, &[Literal::neg(VarId(1)), Literal::pos(VarId(3))]);
    let mut t = apply_and(t1, t2);
    minimize(&mut t);
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
    let before = f.model_count();
    marginalize_levels(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].marginal_counts().is_some(), "output level is count-marginal");
    assert_eq!(f.model_count(), before);
    assert!(is_sat_minimized(&f));
}

#[test]
#[should_panic(expected = "is_sat_minimized: the output level")]
fn a_weight_marginal_output_is_refused_by_name() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = two_clauses(eng, &vtree);
    let weights: Vec<_> = (0..4).map(|_| LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    marginalize_levels(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].is_weight_marginal(), "output level is weight-marginal");
    let _ = is_sat_minimized(&f);
}

#[test]
fn the_incremental_count_of_bottom_is_zero() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = crate::build::constant_zero(eng, &vtree);
    assert!(f.is_zero());
    let mut counter = ModelCounter::<KeepAllColumns>::new(eng, &f, PinSemantics::Evidence);
    assert_eq!(counter.model_count(eng), BigUint::ZERO);
}
