use super::*;
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, StopRules, StopAt};
use crate::OperationError;

/// A structural weighted diagram with several internal levels to fold.
fn weighted_fixture(arithmetic: Arithmetic) -> Tdd {
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::clause(&tree, [1, 2, 3]);
    f.set_weights(WeightStore::new(RationalWeights::unit(8), arithmetic)).unwrap();
    crate::test_helpers::assert_canonical(&f);
    f
}

/// Refused scratch reservations are values, and a later read can still succeed.
#[test]
fn weighted_reads_recover_from_budget_and_allocator_refusals() {
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let engine = Engine::new();
        let f = weighted_fixture(arithmetic);
        {
            let _limit = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            assert_eq!(engine.weighted_value(&f).unwrap_err(), OperationError::OverBudget);
        }
        for n in [0, 1] {
            engine.limits().refuse_nth_reserve(n);
            assert_eq!(engine.weighted_value(&f).unwrap_err(), OperationError::OverBudget);
            engine.limits().grant_every_reserve();
        }
        let value = engine.weighted_value(&f).unwrap().unwrap();
        if let Some(log) = value.as_log() {
            assert!((log.log10_abs() - 224f64.log10()).abs() < 1e-12);
        } else {
            assert_eq!(value.as_rational().as_ref(), &crate::test_helpers::rat(224, 1));
        }
        crate::test_helpers::assert_canonical(&f);
    }
}

/// A work stop interrupts a fold and leaves the diagram readable after the limit is removed.
#[test]
fn weighted_reads_stop_during_the_fold() {
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let engine = Engine::new();
        engine.limits().pin_reduce_poll_stride(Some(1));
        let f = weighted_fixture(arithmetic);
        {
            let _limit = engine.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
                unconditional: Some(StopAt::WorkUnits(2)), after_pairs: None,
            }));
            assert_eq!(engine.weighted_value(&f).unwrap_err(), OperationError::Stopped);
            assert!(engine.limits().meters().work_units >= 2);
        }
        assert!(engine.weighted_value(&f).unwrap().is_some());
        crate::test_helpers::assert_canonical(&f);
    }
}

/// The absence of weights is distinct from an operation refusal.
#[test]
fn unweighted_reads_return_none_without_allocating() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::one(&tree);
    crate::test_helpers::assert_canonical(&f);
    let _limit = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert!(engine.weighted_value(&f).unwrap().is_none());
}
