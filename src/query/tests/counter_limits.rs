use std::sync::atomic::{AtomicUsize, Ordering};
use super::*;
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::query::{KeepAllColumns, KeepFrontier, ModelCounter, PinSemantics, Retention};
use crate::test_helpers::assert_canonical;
use crate::OperationError;

#[test]
fn counter_construction_and_ordinary_counts_report_buffer_refusals() {
    let eng = Engine::new();
    let f = Tdd::clause(&Arc::new(Vtree::balanced(8)), [1, 2]);
    assert_canonical(&f);
    {
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(ModelCounter::<KeepAllColumns>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap_err(), OperationError::OverBudget);
        assert_eq!(eng.model_count(&f), Err(OperationError::OverBudget));
    }
    let mut completed = false;
    for reserve in 0..128 {
        eng.limits().refuse_nth_reserve(reserve);
        let result = ModelCounter::<KeepAllColumns>::try_new_on(&eng, &f, PinSemantics::Evidence);
        eng.limits().grant_every_reserve();
        match result {
            Ok(mut counter) => {
                assert_eq!(counter.try_model_count_on(&eng).unwrap(), 192u32.into());
                completed = true;
                break;
            }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed);
    assert_eq!(eng.model_count(&f).unwrap(), 192u32.into());
}

/// Exercise interrupted writes that promote small values into the overflow table.
fn overflow_refusals<R: Retention>() {
    let eng = Engine::new();
    let f = Tdd::one(&Arc::new(Vtree::balanced(132)));
    assert_canonical(&f);
    let expected = BigUint::from(1u32) << 132usize;
    let mut completed = false;
    for reserve in 0..1024 {
        let mut counter = ModelCounter::<R>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
        for var in 0..132 { counter.set_pin(VarId(var), Some(false)).unwrap(); }
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 1u32.into());
        for var in 0..132 { counter.set_pin(VarId(var), None).unwrap(); }
        eng.limits().refuse_nth_reserve(reserve);
        let result = counter.try_model_count_on(&eng);
        eng.limits().grant_every_reserve();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), expected, "reserve {reserve}");
        // A subsequent pin must use the updated cache and keep its overflow values consistent.
        counter.set_pin(VarId(0), Some(false)).unwrap();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), &expected >> 1usize);
        match result {
            Ok(value) => { assert_eq!(value, expected); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed, "the sweep must cover every refresh reservation");
}

#[test]
fn both_counter_policies_recover_from_every_refresh_reservation_failure() {
    overflow_refusals::<KeepAllColumns>();
    overflow_refusals::<KeepFrontier>();
}

/// Stop at each poll in a dirty refresh, then check the retained pins and fresh result.
fn stopped_refreshes<R: Retention>() {
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let f = Tdd::one(&Arc::new(Vtree::balanced(8)));
    assert_canonical(&f);
    let mut completed = false;
    for cut in 0..128 {
        let mut counter = ModelCounter::<R>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 256u32.into());
        counter.set_pin(VarId(0), Some(false)).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let result = {
            let _limit = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(move |_, _| {
                    let call = calls.fetch_add(1, Ordering::Relaxed);
                    if call == cut { StopDecision::Stop } else { StopDecision::Continue }
                }))));
            counter.try_model_count_on(&eng)
        };
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 128u32.into(), "cut {cut}");
        counter.set_pin(VarId(0), None).unwrap();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 256u32.into());
        match result {
            Ok(_) => { assert!(cut > 2); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::Stopped),
        }
    }
    assert!(completed);
}

#[test]
fn both_counter_policies_recover_from_every_refresh_stop() {
    stopped_refreshes::<KeepAllColumns>();
    stopped_refreshes::<KeepFrontier>();
}

#[test]
fn cached_and_constant_counts_observe_stops_without_losing_pins() {
    let eng = Engine::new();
    for f in [Tdd::zero(&Arc::new(Vtree::balanced(3))), Tdd::one(&Arc::new(Vtree::leaf(VarId(0))))] {
        assert_canonical(&f);
        let mut counter = ModelCounter::<KeepAllColumns>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
        counter.set_pin(VarId(0), Some(true)).unwrap();
        let expected = if f.is_zero() { BigUint::ZERO } else { 1u32.into() };
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), expected);
        {
            let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(|_, _| StopDecision::Stop))));
            assert_eq!(counter.try_model_count_on(&eng), Err(OperationError::Stopped));
            assert_eq!(eng.model_count(&f), Err(OperationError::Stopped));
            assert_eq!(ModelCounter::<KeepAllColumns>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap_err(), OperationError::Stopped);
        }
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), expected);
    }
}

#[test]
fn checked_queries_refuse_incompatible_marginal_values() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    use crate::marginal::marginalize_levels;
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let mut integer = Tdd::clause(&tree, [1, 2]);
    assert_canonical(&integer);
    let mut weighted = integer.clone();
    weighted.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    marginalize_levels(&eng, &mut integer, &[tree.root()]).unwrap();
    marginalize_levels(&eng, &mut weighted, &[tree.root()]).unwrap();
    assert!(matches!(eng.evaluate(&integer, &RationalWeights::unit(4)), Err(OperationError::MarginalLevel(_))));
    assert_eq!(eng.model_count(&integer).unwrap(), 12u32.into());
    assert_eq!(eng.model_count(&weighted), Err(OperationError::IncompatibleWeights));
    assert_eq!(ModelCounter::<KeepAllColumns>::try_new_on(&eng, &weighted, PinSemantics::Evidence).unwrap_err(), OperationError::IncompatibleWeights);
}

#[test]
fn counters_sharing_a_context_retain_independent_evidence() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2]);
    assert_canonical(&f);
    let mut first = ModelCounter::<KeepAllColumns>::try_new(&f, PinSemantics::Evidence).unwrap();
    let mut second = ModelCounter::<KeepFrontier>::new(&f, PinSemantics::Evidence);
    assert_eq!(first.try_model_count().unwrap(), 12u32.into());
    second.set_pin(VarId(0), Some(false)).unwrap();
    assert_eq!(second.model_count(), 4u32.into());
    assert_eq!(first.model_count(), 12u32.into());
    first.set_pin(VarId(0), Some(true)).unwrap();
    assert_eq!(first.try_model_count().unwrap(), 8u32.into());
    assert_eq!(second.try_model_count().unwrap(), 4u32.into());
    first.set_pin(VarId(0), None).unwrap();
    assert_eq!(first.model_count(), 12u32.into());
    assert_eq!(second.model_count(), 4u32.into());
}
