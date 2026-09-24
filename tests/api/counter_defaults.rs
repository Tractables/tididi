use std::sync::Arc;

use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::diagram::{Arithmetic, RationalWeights, WeightStore};
use tididi::limits::{LimitConfig, StopCallback, StopDecision};
use tididi::query::{ModelCounter, PinSemantics, Retention};
use tididi::test_helpers::assert_canonical;
use tididi::vtree::VarId;

#[test]
fn default_counter_matches_evidence_after_pin_changes_and_failed_updates() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let mut counter: ModelCounter<'_> = f.counter().unwrap();
    let mut explicit = f.counter_with(Retention::All, PinSemantics::Evidence).unwrap();
    let mut pins = [None; 3];
    for (var, pin) in [(0, None), (0, Some(true)), (1, Some(true)), (0, Some(false)),
        (1, None), (2, Some(false)), (0, None), (2, None)] {
        pins[var] = pin;
        counter.set_pin(VarId(var as u32 + 1), pin).unwrap();
        explicit.set_pin(VarId(var as u32 + 1), pin).unwrap();
        let expected = (0u32..8).filter(|bits| {
            let assignment: [bool; 3] = std::array::from_fn(|i| bits & (1 << i) != 0);
            (assignment[0] || !assignment[1]) && pins.iter().zip(assignment)
                .all(|(pin, value)| pin.is_none_or(|pin| pin == value))
        }).count();
        assert_eq!(counter.model_count().unwrap(), expected.into());
        assert_eq!(explicit.model_count().unwrap(), expected.into());
        assert_eq!(counter.set_pin(VarId(8), None), Err(OperationError::VariableNotInVtree(VarId(8))));
        assert_eq!(counter.model_count().unwrap(), expected.into());
    }
}

#[test]
fn batch_counter_observes_limits_on_construction_and_cached_reads() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let context = f.context();
    let no_storage = LimitConfig::none().with_memory_budget_bytes(Some(0));
    context.with_limits(no_storage, |engine| {
        assert_eq!(engine.counter(&f).unwrap_err(), OperationError::OverBudget);
    });
    let stop = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&stop);
    let limits = LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
        if trigger.load(Ordering::Relaxed) { StopDecision::Stop } else { StopDecision::Continue }
    })));
    context.with_limits(limits, |engine| {
        let mut counter = engine.counter(&f).unwrap();
        counter.set_pin(VarId(1), Some(false)).unwrap();
        assert_eq!(counter.model_count().unwrap(), 2u32.into());
        stop.store(true, Ordering::Relaxed);
        assert_eq!(engine.counter(&f).unwrap_err(), OperationError::Stopped);
        assert_eq!(counter.model_count(), Err(OperationError::Stopped));
        assert_eq!(f.counter().unwrap().model_count().unwrap(), 6u32.into());
        stop.store(false, Ordering::Relaxed);
        assert_eq!(counter.model_count().unwrap(), 2u32.into());
        counter.set_pin(VarId(1), None).unwrap();
        assert_eq!(counter.model_count().unwrap(), 6u32.into());
    });
}

#[test]
fn persistent_counter_preserves_pins_and_recovers_after_a_refused_binding() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let mut counter = f.counter().unwrap();
    let context = f.context();
    context.run(|engine| {
        let mut batch = counter.bind(engine);
        batch.set_pin(VarId(1), Some(false)).unwrap();
        assert_eq!(batch.model_count().unwrap(), 2u32.into());
    });
    let stopped = LimitConfig::none().with_stop_callback(Some(
        StopCallback::new(|_, _| StopDecision::Stop)));
    context.with_limits(stopped, |engine| {
        let mut batch = counter.bind(engine);
        batch.set_pin(VarId(2), Some(true)).unwrap();
        assert_eq!(batch.model_count(), Err(OperationError::Stopped));
    });
    assert_eq!(counter.model_count().unwrap(), 0u32.into());
    context.run(|engine| {
        let mut batch = counter.bind(engine);
        batch.set_pin(VarId(2), None).unwrap();
        assert_eq!(batch.model_count().unwrap(), 2u32.into());
    });
    counter.set_pin(VarId(1), None).unwrap();
    assert_eq!(counter.model_count().unwrap(), 6u32.into());
}

#[test]
fn persistent_counter_remains_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ModelCounter<'static>>();
    assert_send::<ModelCounter<'static>>();
}

#[test]
fn default_counter_rejects_weighted_marginal_values() {
    let tree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    f.set_weights(WeightStore::new(RationalWeights::unit(3), Arithmetic::ExactRational)).unwrap();
    f.marginalize_levels(&[tree.root()]).unwrap();
    assert_canonical(&f);
    assert_eq!(f.counter().unwrap_err(), OperationError::IncompatibleWeights);
    assert_eq!(Engine::new().counter(&f).unwrap_err(), OperationError::IncompatibleWeights);
}
