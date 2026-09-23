use super::*;
use std::sync::Arc;
use crate::Vtree;
use crate::test_helpers::assert_canonical;

#[test]
fn interrupted_ancestor_growth_leaves_room_for_further_pin_updates() {
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    let mut completed = false;
    for reserve in 0..16 {
        let mut counter = diagram.counter().unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        counter.set_pin(VarId(1), Some(false)).unwrap();
        engine.limits().refuse_nth_reserve(reserve);
        let result = counter.bind(&engine).model_count();
        engine.limits().grant_every_reserve();
        let capacity = counter.state.observations.changed.capacity();
        counter.clear_pins();
        let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(v % 2 == 0))).collect();
        counter.set_pins(&pins).unwrap();
        assert_eq!(counter.state.observations.changed.capacity(), capacity, "pin updates must reuse reserved storage");
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        match result {
            Ok(_) => { completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed);
}

#[test]
fn panicking_refresh_keeps_dirty_membership_reusable() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    engine.limits().pin_reduce_poll_stride(Some(1));
    let mut completed = false;
    for cut in 0..128 {
        let mut counter = diagram.counter().unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        counter.set_pin(VarId(1), Some(false)).unwrap();
        let calls = AtomicUsize::new(0);
        let result = {
            let _limits = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(move |_, _| {
                    assert_ne!(calls.fetch_add(1, Ordering::Relaxed), cut, "interrupted counter refresh");
                    StopDecision::Continue
                }))));
            catch_unwind(AssertUnwindSafe(|| counter.bind(&engine).model_count().unwrap()))
        };
        let capacity = counter.state.observations.changed.capacity();
        counter.clear_pins();
        let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(false))).collect();
        counter.set_pins(&pins).unwrap();
        assert_eq!(counter.state.observations.changed.capacity(), capacity);
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
        counter.set_pin(VarId(1), None).unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(2u32));
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        if result.is_ok() {completed = true; break;}
    }
    assert!(completed);
}

#[test]
fn frontier_evidence_invalidates_without_retaining_a_dirty_worklist() {
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    let mut counter = diagram.counter_with(Retention::Frontier, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.state.observations.changed.capacity(), 0);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
    let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(false))).collect();
    counter.set_pins(&pins).unwrap();
    assert!(!counter.state.observations.evaluated);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
    // Repeating observations leaves the root cached, even if allocations would be refused.
    engine.limits().refuse_nth_reserve(0);
    counter.set_pins(&pins).unwrap();
    assert!(counter.state.observations.evaluated);
    assert_eq!(counter.bind(&engine).model_count().unwrap(), BigUint::from(1u32));
    engine.limits().grant_every_reserve();
    counter.clear_pins();
    assert!(!counter.state.observations.evaluated);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
    counter.clear_pins();
    assert!(counter.state.observations.evaluated);
    assert_eq!(counter.state.observations.changed.capacity(), 0);
    assert!(counter.state.observations.pins.iter().all(|pin| !pin.dirty));
}
