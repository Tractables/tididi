//! Column allocation failures return errors without unwinding.
use crate::test_helpers::CountVecExt;
use crate::Engine;
use crate::value::{Count, CountVec};

use num_bigint::BigUint;

/// A width whose `u128`/`(u32, BigUint)` byte size overflows `isize::MAX`,
/// so `try_reserve_exact` returns `Err(CapacityOverflow)` *without* touching
/// the allocator — deterministic, no real allocation, no abort.
const OVER_BUDGET_WIDTH: usize = usize::MAX / 8;

#[test]
fn with_width_returns_over_budget() {
    let eng = Engine::new();
    assert!(matches!(CountVec::try_with_width(&eng, OVER_BUDGET_WIDTH), Err(crate::OperationError::OverBudget)));
}

/// The overflow side table grows one entry at a time (it is keyed by slot, so
/// it is never sized to the column's width), but it still routes every growth
/// through the same guarded `reserve`/`reserve_exact` as the fast column. This
/// pins that policy on the side table's element type — a different size than
/// the fast `Vec<u128>` above, so guarded independently.
#[test]
fn big_side_table_reserve_returns_over_budget() {
    let eng = Engine::new();
    let mut v: Vec<(u32, BigUint)> = Vec::new();
    assert_eq!(eng.limits().reserve_exact(&mut v, OVER_BUDGET_WIDTH), Err(crate::OperationError::OverBudget));
}

#[test]
fn clone_guarded_copies_fast_values_exactly() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 4);
    cv.set_i(&eng, 0, Count::Fast(0));
    cv.set_i(&eng, 1, Count::Fast(1));
    cv.set_i(&eng, 2, Count::Big(BigUint::from(u128::MAX))); // sentinel lives in fast[2]
    cv.set_i(&eng, 3, Count::Fast(42));
    let out = cv.clone_guarded(&eng);
    for i in 0..4 {
        assert_eq!(out.get(i).to_count(), cv.get(i).to_count());
    }
}

#[test]
fn clone_guarded_copies_big_overflow_exactly() {
    let eng = Engine::new();
    let mut cv = CountVec::with_width(&eng, 4);
    cv.set_i(&eng, 1, Count::Big(BigUint::from(7u32)));
    cv.set_i(&eng, 3, Count::Big(BigUint::from(u128::MAX)));
    let out = cv.clone_guarded(&eng);
    for i in 0..4 {
        assert_eq!(out.big_val(&eng, i), cv.big_val(&eng, i));
    }
}

/// Refuse every reservation in turn, then resume using the same engine and diagram.
fn marginal_refusals(weighted: bool, overflow: bool) {
    use std::sync::Arc;
    use crate::diagram::{Arithmetic, RationalWeights, Tdd, WeightStore};


    use crate::test_helpers::{assert_canonical, assert_marginal_canonical, compile_clauses};
    use crate::vtree::Vtree;
    use crate::OperationError;

    let eng = Engine::new();
    let n = if overflow { 132 } else { 4 };
    let vtree = Arc::new(Vtree::balanced(n));
    let mut original = if overflow {
        Tdd::one(&vtree)
    } else {
        compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-1, -4]])
    };
    assert_canonical(&original);
    if weighted {
        original.set_weights(WeightStore::new(RationalWeights::unit(n as usize), Arithmetic::ExactRational)).unwrap();
    }
    let targets = if overflow {
        vec![vtree.root()]
    } else {
        let (left, right) = vtree.children(vtree.root());
        vec![left, right]
    };
    let expected_count = (!weighted).then(|| original.model_count().unwrap());
    let expected_weight = weighted.then(|| original.weighted_value().unwrap().unwrap().into_rational_opt().unwrap());
    let check_value = |f: &Tdd| {
        if let Some(expected) = &expected_count { assert_eq!(&eng.model_count(f).unwrap(), expected); }
        if let Some(expected) = &expected_weight { assert_eq!(&f.weighted_value().unwrap().unwrap().into_rational_opt().unwrap(), expected); }
    };
    let mut completed = false;
    let mut partial = false;
    for reserve in 0..2048 {
        let mut f = original.clone();
        eng.limits().refuse_nth_reserve(reserve);
        let result = eng.marginalize_levels(&mut f, &targets);
        eng.limits().grant_every_reserve();
        if reserve == 0 {
            assert_eq!(result, Err(OperationError::OverBudget));
            assert!(!f.has_marginal_level(), "a refused first column must leave the diagram structural");
        }
        check_value(&f);
        match result {
            Ok(()) => { completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        partial |= f.levels[targets[0].idx()].is_marginal()
            && !f.levels[targets[targets.len() - 1].idx()].is_marginal();
        eng.marginalize_levels(&mut f, &targets).unwrap();
        check_value(&f);
        assert_marginal_canonical(&f);
    }
    assert!(completed, "every reservation must be covered");
    if !overflow { assert!(partial, "exercise a refusal after committing the first target"); }
}

#[test]
fn integer_marginalization_recovers_from_every_column_refusal() {
    marginal_refusals(false, false);
}

#[test]
fn weighted_marginalization_recovers_from_every_column_refusal() {
    marginal_refusals(true, false);
}

#[test]
fn overflowing_marginalization_recovers_from_every_column_refusal() {
    marginal_refusals(false, true);
}
