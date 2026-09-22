use std::sync::atomic::{AtomicUsize, Ordering};
use super::*;
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::query::{PinSemantics, Retention};
use crate::test_helpers::assert_canonical;
use crate::OperationError;

#[test]
fn counter_construction_and_ordinary_counts_report_buffer_refusals() {
    let eng = Engine::new();
    let f = Tdd::clause(&Arc::new(Vtree::balanced(8)), [1, 2]).unwrap();
    assert_canonical(&f);
    {
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(eng.counter_with(&f, Retention::All, PinSemantics::Evidence).unwrap_err(), OperationError::OverBudget);
        assert_eq!(eng.model_count(&f), Err(OperationError::OverBudget));
    }
    let mut completed = false;
    for reserve in 0..128 {
        eng.limits().refuse_nth_reserve(reserve);
        let result = eng.counter_with(&f, Retention::All, PinSemantics::Evidence);
        eng.limits().grant_every_reserve();
        match result {
            Ok(mut counter) => {
                assert_eq!(counter.model_count().unwrap(), 192u32.into());
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
fn overflow_refusals(retention: Retention) {
    let eng = Engine::new();
    let f = Tdd::one(&Arc::new(Vtree::balanced(132)));
    assert_canonical(&f);
    let expected = BigUint::from(1u32) << 132usize;
    let mut completed = false;
    for reserve in 0..1024 {
        let mut counter = eng.counter_with(&f, retention, PinSemantics::Evidence).unwrap();
        for var in 1..=132 { counter.set_pin(VarId(var), Some(false)).unwrap(); }
        assert_eq!(counter.model_count().unwrap(), 1u32.into());
        for var in 1..=132 { counter.set_pin(VarId(var), None).unwrap(); }
        eng.limits().refuse_nth_reserve(reserve);
        let result = counter.model_count();
        eng.limits().grant_every_reserve();
        assert_eq!(counter.model_count().unwrap(), expected, "reserve {reserve}");
        // A subsequent pin must use the updated cache and keep its overflow values consistent.
        counter.set_pin(VarId(1), Some(false)).unwrap();
        assert_eq!(counter.model_count().unwrap(), &expected >> 1usize);
        match result {
            Ok(value) => { assert_eq!(value, expected); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed, "the sweep must cover every refresh reservation");
}

#[test]
fn both_counter_policies_recover_from_every_refresh_reservation_failure() {
    overflow_refusals(Retention::All);
    overflow_refusals(Retention::Frontier);
}

/// Stop at each poll in a dirty refresh, then check the retained pins and fresh result.
fn stopped_refreshes(retention: Retention) {
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let f = Tdd::one(&Arc::new(Vtree::balanced(8)));
    assert_canonical(&f);
    let mut completed = false;
    for cut in 0..128 {
        let mut counter = eng.counter_with(&f, retention, PinSemantics::Evidence).unwrap();
        assert_eq!(counter.model_count().unwrap(), 256u32.into());
        counter.set_pin(VarId(1), Some(false)).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let result = {
            let _limit = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(move |_, _| {
                    let call = calls.fetch_add(1, Ordering::Relaxed);
                    if call == cut { StopDecision::Stop } else { StopDecision::Continue }
                }))));
            counter.model_count()
        };
        assert_eq!(counter.model_count().unwrap(), 128u32.into(), "cut {cut}");
        counter.set_pin(VarId(1), None).unwrap();
        assert_eq!(counter.model_count().unwrap(), 256u32.into());
        match result {
            Ok(_) => { assert!(cut > 2); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::Stopped),
        }
    }
    assert!(completed);
}

#[test]
fn both_counter_policies_recover_from_every_refresh_stop() {
    stopped_refreshes(Retention::All);
    stopped_refreshes(Retention::Frontier);
}

#[test]
fn cached_and_constant_counts_observe_stops_without_losing_pins() {
    let eng = Engine::new();
    for f in [Tdd::zero(&Arc::new(Vtree::balanced(3))), Tdd::one(&Arc::new(Vtree::leaf(VarId(1))))] {
        assert_canonical(&f);
        let mut counter = eng.counter_with(&f, Retention::All, PinSemantics::Evidence).unwrap();
        counter.set_pin(VarId(1), Some(true)).unwrap();
        let expected = if f.is_zero() { BigUint::ZERO } else { 1u32.into() };
        assert_eq!(counter.model_count().unwrap(), expected);
        {
            let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(|_, _| StopDecision::Stop))));
            assert_eq!(counter.model_count(), Err(OperationError::Stopped));
            assert_eq!(eng.model_count(&f), Err(OperationError::Stopped));
            assert_eq!(eng.counter_with(&f, Retention::All, PinSemantics::Evidence).unwrap_err(), OperationError::Stopped);
        }
        assert_eq!(counter.model_count().unwrap(), expected);
    }
}

#[test]
fn checked_queries_refuse_incompatible_marginal_values() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};

    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let mut integer = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&integer);
    let mut weighted = integer.clone();
    weighted.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    eng.marginalize_levels(&mut integer, &[tree.root()]).unwrap();
    eng.marginalize_levels(&mut weighted, &[tree.root()]).unwrap();
    assert!(matches!(eng.evaluate(&integer, &RationalWeights::unit(4)), Err(OperationError::MarginalLevel(_))));
    assert_eq!(eng.model_count(&integer).unwrap(), 12u32.into());
    assert_eq!(eng.model_count(&weighted), Err(OperationError::IncompatibleWeights));
    assert_eq!(eng.counter_with(&weighted, Retention::All, PinSemantics::Evidence).unwrap_err(), OperationError::IncompatibleWeights);
}

#[test]
fn counters_sharing_a_context_retain_independent_evidence() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let mut first = f.counter_with(Retention::All, PinSemantics::Evidence).unwrap();
    let mut second = f.counter_with(Retention::Frontier, PinSemantics::Evidence).unwrap();
    assert_eq!(first.model_count().unwrap(), 12u32.into());
    second.set_pin(VarId(1), Some(false)).unwrap();
    assert_eq!(second.model_count().unwrap(), 4u32.into());
    assert_eq!(first.model_count().unwrap(), 12u32.into());
    first.set_pin(VarId(1), Some(true)).unwrap();
    assert_eq!(first.model_count().unwrap(), 8u32.into());
    assert_eq!(second.model_count().unwrap(), 4u32.into());
    first.set_pin(VarId(1), None).unwrap();
    assert_eq!(first.model_count().unwrap(), 12u32.into());
    assert_eq!(second.model_count().unwrap(), 4u32.into());
}

/// Setup and fold storage must fit together, not in two independent budgets.
#[test]
fn node_count_export_shares_one_allocation_budget() {
    let engine = Engine::new();
    let vtree = Arc::new(Vtree::leaf(VarId(1)));
    let f = Tdd::one(&vtree);
    assert_canonical(&f);
    let header = std::mem::size_of::<crate::value::CountVec>() as u64;
    let values = (crate::diagram::LEAF_WIDTH * std::mem::size_of::<u128>()) as u64;
    {
        let _scope = engine.limits().scope(LimitConfig::none()
            .with_memory_budget_bytes(Some(header.max(values))));
        assert_eq!(engine.node_counts_u128(&f), Err(OperationError::OverBudget));
    }
    let counts = engine.node_counts_u128(&f).unwrap();
    assert_eq!(counts[vtree.root().idx()], [2, 1, 1]);
    assert_canonical(&f);
}

/// A short fold still reports the work stop at its final polling boundary.
#[test]
fn node_count_export_checks_stops_after_the_fold() {
    use crate::limits::{StopAt, StopRules};
    let engine = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    {
        let stop_at = engine.limits().work_units() + 1;
        let _scope = engine.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(stop_at)), after_pairs: None,
        }));
        assert_eq!(engine.node_counts_u128(&f), Err(OperationError::Stopped));
    }
    let counts = engine.node_counts_u128(&f).unwrap();
    assert_eq!(counts[vtree.root().idx()][f.output().local.idx()], 6);
    assert_canonical(&f);
}

/// Refuse every reservation, including the final array of exported columns.
#[test]
fn node_count_export_recovers_from_each_allocation_refusal() {
    let engine = Engine::new();
    let vtree = Arc::new(Vtree::leaf(VarId(1)));
    let f = Tdd::one(&vtree);
    assert_canonical(&f);
    let expected = f.node_counts_u128().unwrap();
    let mut refusals = 0;
    let mut completed = false;
    for cut in 0..16 {
        engine.limits().refuse_nth_reserve(cut);
        let result = engine.node_counts_u128(&f);
        engine.limits().grant_every_reserve();
        assert_eq!(engine.node_counts_u128(&f).unwrap(), expected);
        match result {
            Err(OperationError::OverBudget) => refusals += 1,
            Ok(counts) => {
                assert_eq!(counts, expected);
                completed = true;
                break;
            }
            other => panic!("reservation {cut}: {other:?}"),
        }
    }
    assert!(completed && refusals >= 3, "setup, column and export buffers must be fallible");
    assert_canonical(&f);
}
