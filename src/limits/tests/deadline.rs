//! `LimitConfig::with_deadline_at_most`: a deadline that only tightens what is
//! already armed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::limits::{LimitConfig, OperationError, StopAt, StopCallback, StopDecision, StopRules};
use crate::vtree::Vtree;
use crate::Engine;

/// A callback that counts its calls and always continues.
fn counting() -> (StopCallback, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let callback = StopCallback::new(move |_, _| {
        seen.fetch_add(1, Ordering::Relaxed);
        StopDecision::Continue
    });
    (callback, calls)
}

#[test]
fn a_deadline_is_armed_where_none_was() {
    let at = Instant::now() + Duration::from_secs(60);
    let config = LimitConfig::none().with_deadline_at_most(at);
    assert_eq!(config.stop_rules(), StopRules::by_time(at));
    assert!(config.stop_callback().is_none(), "a deadline on the wall clock needs no callback");
}

#[test]
fn the_earlier_of_two_deadlines_is_kept() {
    let now = Instant::now();
    let (early, late) = (now + Duration::from_secs(10), now + Duration::from_secs(20));
    let tightened = LimitConfig::none().with_deadline(Some(late)).with_deadline_at_most(early);
    assert_eq!(tightened.stop_rules().unconditional, Some(StopAt::Time(early)));
    let kept = LimitConfig::none().with_deadline(Some(early)).with_deadline_at_most(late);
    assert_eq!(kept.stop_rules().unconditional, Some(StopAt::Time(early)));
    let same = LimitConfig::none().with_deadline(Some(early)).with_deadline_at_most(early);
    assert_eq!(same.stop_rules().unconditional, Some(StopAt::Time(early)));
}

#[test]
fn the_other_settings_survive_a_tightened_deadline() {
    let (callback, calls) = counting();
    let at = Instant::now() + Duration::from_secs(60);
    let conditional = StopRules::default().after_pairs(7, StopAt::WorkUnits(100));
    let config = LimitConfig::none()
        .with_memory_budget_bytes(Some(4096))
        .with_output_node_cap(Some(42))
        .with_stop_rules(conditional)
        .with_stop_callback(Some(callback))
        .with_deadline_at_most(at);
    assert_eq!(config.memory_budget_bytes(), Some(4096));
    assert_eq!(config.output_node_cap(), Some(42));
    assert_eq!(config.stop_rules(), StopRules { unconditional: Some(StopAt::Time(at)), ..conditional });
    let meters = Engine::new().limits().meters();
    assert_eq!(config.stop_callback().unwrap().decide(&meters, at), StopDecision::Continue);
    assert_eq!(calls.load(Ordering::Relaxed), 1, "the callback is the one set before, not a wrapper");
}

/// A work-clock bound keeps its slot, and the deadline is asked through the
/// callback: it stops once the deadline is reached, and before then defers to
/// the callback that was installed, whose decision it returns.
#[test]
fn a_work_clock_bound_is_kept_and_the_deadline_rides_the_callback() {
    let decisions = Arc::new(Mutex::new(vec![StopDecision::Stop, StopDecision::Continue]));
    let replay = Arc::clone(&decisions);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let previous = StopCallback::new(move |_, _| {
        seen.fetch_add(1, Ordering::Relaxed);
        replay.lock().unwrap().pop().unwrap_or(StopDecision::Continue)
    });
    let stops = StopRules { unconditional: Some(StopAt::WorkUnits(0)), after_pairs: Some((7, StopAt::WorkUnits(100))) };
    let at = Instant::now() + Duration::from_secs(60);
    let config = LimitConfig::none()
        .with_stop_rules(stops)
        .with_stop_callback(Some(previous))
        .with_deadline_at_most(at);
    assert_eq!(config.stop_rules(), stops);
    let callback = config.stop_callback().expect("the deadline is a callback here");
    let meters = Engine::new().limits().meters();
    let before = at - Duration::from_secs(1);
    assert_eq!(callback.decide(&meters, before), StopDecision::Continue);
    assert_eq!(callback.decide(&meters, before), StopDecision::Stop, "the earlier callback still decides");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(callback.decide(&meters, at), StopDecision::Stop);
    assert_eq!(callback.decide(&meters, at + Duration::from_secs(1)), StopDecision::Stop);
    assert_eq!(calls.load(Ordering::Relaxed), 2, "past the deadline the earlier callback is not asked");
}

#[test]
fn a_work_clock_bound_without_a_callback_continues_until_the_deadline() {
    let at = Instant::now() + Duration::from_secs(60);
    let config = LimitConfig::none()
        .with_stop_rules(StopRules { unconditional: Some(StopAt::WorkUnits(1_000)), after_pairs: None })
        .with_deadline_at_most(at);
    assert_eq!(config.stop_rules().unconditional, Some(StopAt::WorkUnits(1_000)));
    let callback = config.stop_callback().expect("the deadline is a callback here");
    let meters = Engine::new().limits().meters();
    assert_eq!(callback.decide(&meters, at - Duration::from_secs(1)), StopDecision::Continue);
    assert_eq!(callback.decide(&meters, at), StopDecision::Stop);
}

/// Through `Limits::edit`, a passed deadline stops an operation under a
/// work-clock bound it would not have reached, and dropping the scope restores
/// the bound and the callback as they were.
#[test]
fn an_edited_scope_stops_an_operation_and_restores_the_prior_rules() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let (callback, calls) = counting();
    let stops = StopRules { unconditional: Some(StopAt::WorkUnits(u64::MAX)), after_pairs: None };
    let _outer = eng.limits().scope(LimitConfig::none().with_stop_rules(stops).with_stop_callback(Some(callback)));
    {
        let passed = Instant::now() - Duration::from_secs(1);
        let _stage = eng.limits().edit(|config| config.with_deadline_at_most(passed));
        assert_eq!(eng.cube(&vtree, [1, -2]).unwrap_err(), OperationError::Stopped);
    }
    let restored = eng.limits().armed();
    assert_eq!(restored.stop_rules(), stops);
    let before = calls.load(Ordering::Relaxed);
    let f = eng.cube(&vtree, [1, -2]).expect("the restored rules do not stop");
    crate::test_helpers::assert_canonical(&f);
    assert!(calls.load(Ordering::Relaxed) > before, "the restored callback is the one set before");
    assert_eq!(f.model_count().unwrap(), 4u32.into());
}
