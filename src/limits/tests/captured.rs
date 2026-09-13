use crate::Engine;
use super::*;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};

#[test]
fn engines_keep_independent_captured_schedules_and_restore_them_on_unwind() {
    let first = Engine::new();
    let second = Engine::new();
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let _first = first.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
        if flag.load(Ordering::Relaxed) { StopDecision::Stop } else { StopDecision::Continue }
    }))));
    let _second = second.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Continue))));
    assert!(!first.limits().should_stop());
    cancelled.store(true, Ordering::Relaxed);
    assert!(first.limits().should_stop());
    assert!(!second.limits().should_stop());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _inner = first.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| panic!("schedule failed")))));
        first.limits().should_stop();
    }));
    assert!(result.is_err());
    assert!(first.limits().should_stop());
}

#[test]
fn captured_memory_probes_and_reentrant_installation_release_the_callback_borrow() {
    thread_local! { static ENGINE: Engine = Engine::new(); }
    ENGINE.with(|engine| {
    let bytes = Arc::new(AtomicU64::new(0));
    let observed = Arc::clone(&bytes);
    let probes = MemoryHooks::new(move |n| {
        observed.fetch_add(n, Ordering::Relaxed);
        ENGINE.with(|engine| { let _prior = engine.limits().install(LimitConfig::none()); });
    }, || 41, || Some(SOFT_HEADROOM_MARGIN_BYTES + 141), || {});
    let _installed = engine.limits().scope(LimitConfig::none().with_memory_hooks(probes));
    assert_eq!(engine.limits().headroom(), 100);
    engine.limits().preflight_alloc(37);
    assert_eq!(bytes.load(Ordering::Relaxed), 37);
    engine.limits().preflight_alloc(10);
    assert_eq!(bytes.load(Ordering::Relaxed), 37);
    assert_eq!(Engine::new().limits().headroom(), super::super::memory::VAS_UNLIMITED_HEADROOM);
    });
}

#[test]
fn a_schedule_can_replace_its_own_installation() {
    thread_local! { static ENGINE: Engine = Engine::new(); }
    ENGINE.with(|engine| {
    let _installed = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
        ENGINE.with(|engine| { let _prior = engine.limits().install(LimitConfig::none()); });
        StopDecision::Stop
    }))));
    assert!(engine.limits().should_stop());
    assert!(!engine.limits().should_stop());
    });
}

#[test]
fn engines_and_callback_configurations_can_move_between_threads() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Engine>();
    assert_send::<LimitConfig>();
    assert_sync::<LimitConfig>();
}
