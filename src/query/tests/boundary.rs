//! The queries at their boundaries: a marginal output level, ⊥, a pin outside
//! the counter.

use super::*;

use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::marginal::marginalize;
use crate::query::count::{IncrementalCounter, KeepAllColumns, SeedConvention};
use crate::test_helpers::rat;

/// `(x0 ∨ x2) ∧ (¬x1 ∨ x3)` over `balanced(4)`, minimized.
fn two_clauses(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let t1 = clause_to_tdd(eng, vtree, &[Literal::pos(VarId(0)), Literal::pos(VarId(2))]);
    let t2 = clause_to_tdd(eng, vtree, &[Literal::neg(VarId(1)), Literal::pos(VarId(3))]);
    let mut t = apply_and(t1, t2);
    minimize(&mut t);
    t
}

/// Every internal level of `vtree`, bottom-up, which is the order `marginalize`
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
    marginalize(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
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
    let weights: Vec<_> = (0..4).map(|_| (rat(1, 2), rat(1, 3))).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_weights(&weights), Arithmetic::ExactRational)).unwrap();
    marginalize(eng, &mut f, &internal_levels_bottom_up(&vtree)).unwrap();
    assert!(f.levels[f.output.vtree.idx()].is_weight_marginal(), "output level is weight-marginal");
    let _ = is_sat_minimized(&f);
}

#[test]
fn the_incremental_count_of_bottom_is_zero() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = crate::build::constant_zero(eng, &vtree);
    assert!(f.is_zero());
    let mut counter = IncrementalCounter::<KeepAllColumns>::new(eng, &f, 3, SeedConvention::Fixed);
    assert_eq!(counter.output_count(eng), BigUint::ZERO);
}

#[test]
#[should_panic(expected = "IncrementalCounter::set_pin: VarId(3) is not below the counter's 3 pins")]
fn a_pin_outside_the_counter_is_refused_by_name() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = constant_one(eng, &vtree);
    let mut counter = IncrementalCounter::<KeepAllColumns>::new(eng, &f, 3, SeedConvention::Fixed);
    counter.set_pin(VarId(3), Some(true));
}
