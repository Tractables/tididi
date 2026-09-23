//! Bounded care discovery preserves operands and respects engine limits.
use super::*;
use crate::apply::RestrictionOutcome;
use crate::limits::{LimitConfig, OperationError, StopAt, StopRules};

#[test]
fn abandoned_discovery_returns_the_original_allocation() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let f = eng.clause(&vtree, [1, 2, 3, 4, 5, 6]).unwrap();
    let care = eng.clause(&vtree, [-1, -2, -3]).unwrap();
    assert_canonical(&f);
    assert_canonical(&care);
    for budget in [0, 1] {
        let operand = f.clone();
        let allocation = operand.levels.as_ptr();
        let care_allocation = care.levels.as_ptr();
        let result = eng.restrict_to_care_bounded(operand, &care, budget).unwrap();
        assert!(matches!(result, RestrictionOutcome::Unchanged(_)));
        let g = result.into_tdd();
        assert_eq!(g.levels.as_ptr(), allocation);
        assert_eq!(care.levels.as_ptr(), care_allocation);
        assert_canonical(&g);
        assert!(eng.equivalent(&f, &g).unwrap());
    }
}

#[test]
fn bounded_restriction_agrees_on_care_for_every_assignment() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    for pivot in 1..=6 {
        let f = eng.clause(&vtree, [pivot, pivot % 6 + 1, (pivot + 1) % 6 + 1]).unwrap();
        let left = eng.clause(&vtree, [-pivot, (pivot + 2) % 6 + 1]).unwrap();
        let right = eng.clause(&vtree, [-((pivot + 1) % 6 + 1)]).unwrap();
        assert_canonical(&f);
        assert_canonical(&left);
        assert_canonical(&right);
        // The borrowed care need not have been minimized by its caller.
        let care = eng.and(left, right).unwrap();
        for budget in [0, 1, 3, 16, 256, u64::MAX] {
            let mut g = eng.restrict_to_care_bounded(f.clone(), &care, budget).unwrap().into_tdd();
            assert!(g.pair_count() <= f.pair_count());
            eng.minimize(&mut g).unwrap();
            assert_canonical(&g);
            for mask in 0..64 {
                let assignment: Vec<_> = (0..6).map(|bit| mask & (1 << bit) != 0).collect();
                if eval(&care, &assignment) { assert_eq!(eval(&f, &assignment), eval(&g, &assignment)); }
            }
        }
        let mut checked_care = care;
        eng.minimize(&mut checked_care).unwrap();
        assert_canonical(&checked_care);
    }
}

#[test]
fn an_exhausted_optional_allowance_does_not_hide_cancellation() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let f = eng.clause(&vtree, [1, 2, 3, 4, 5, 6]).unwrap();
    let care = eng.clause(&vtree, [-1, -2, -3]).unwrap();
    assert_canonical(&f);
    assert_canonical(&care);
    for allowance in [0, 1] {
        let limits = LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + allowance)),
            after_pairs: None,
        });
        let _scope = eng.limits().scope(limits);
        assert!(matches!(eng.restrict_to_care_bounded(f.clone(), &care, allowance), Err(OperationError::Stopped)));
    }
}

#[test]
fn skipped_discovery_still_checks_vtree_compatibility() {
    let eng = Engine::new();
    let a = Arc::new(Vtree::balanced(2));
    let b = Arc::new(Vtree::balanced(2));
    let f = eng.clause(&a, [1]).unwrap();
    let care = eng.clause(&b, [1]).unwrap();
    assert_canonical(&f);
    assert_canonical(&care);
    assert!(matches!(eng.restrict_to_care_bounded(f, &care, 0), Err(OperationError::VtreeMismatch)));
}
