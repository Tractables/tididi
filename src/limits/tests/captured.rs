use crate::Engine;
use super::*;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn engines_keep_independent_captured_schedules_and_restore_them_on_unwind() {
    let first = Engine::new();
    let second = Engine::new();
    let cancelled = Rc::new(Cell::new(false));
    let flag = Rc::clone(&cancelled);
    let _first = first.limits().scope(LimitSet::none().schedule(Some(ScheduleHook::new(move |_, _| {
        if flag.get() { Scheduled::Stop } else { Scheduled::Carry }
    }))));
    let _second = second.limits().scope(LimitSet::none().schedule(Some(ScheduleHook::new(|_, _| Scheduled::Carry))));
    assert!(!first.limits().should_stop());
    cancelled.set(true);
    assert!(first.limits().should_stop());
    assert!(!second.limits().should_stop());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _inner = first.limits().scope(LimitSet::none().schedule(Some(ScheduleHook::new(|_, _| panic!("schedule failed")))));
        first.limits().should_stop();
    }));
    assert!(result.is_err());
    assert!(first.limits().should_stop());
}

#[test]
fn captured_memory_probes_and_reentrant_installation_release_the_callback_borrow() {
    let engine = Rc::new(Engine::new());
    let weak = Rc::downgrade(&engine);
    let bytes = Rc::new(Cell::new(0));
    let observed = Rc::clone(&bytes);
    let probes = MemPressure::new(move |n| {
        observed.set(observed.get() + n);
        let engine = weak.upgrade().unwrap();
        let _prior = engine.limits().install(LimitSet::none());
    }, || 41, || Some(SOFT_HEADROOM_MARGIN_BYTES + 141), || {});
    let _installed = engine.limits().scope(LimitSet::none().mem_pressure(probes));
    assert_eq!(engine.limits().headroom(), 100);
    engine.limits().preflight_alloc(37);
    assert_eq!(bytes.get(), 37);
    engine.limits().preflight_alloc(10);
    assert_eq!(bytes.get(), 37);
    assert_eq!(Engine::new().limits().headroom(), super::super::memory::VAS_UNLIMITED_HEADROOM);
}

#[test]
fn a_schedule_can_replace_its_own_installation() {
    let engine = Rc::new(Engine::new());
    let weak = Rc::downgrade(&engine);
    let _installed = engine.limits().scope(LimitSet::none().schedule(Some(ScheduleHook::new(move |_, _| {
        let engine = weak.upgrade().unwrap();
        let _prior = engine.limits().install(LimitSet::none());
        Scheduled::Stop
    }))));
    assert!(engine.limits().should_stop());
    assert!(!engine.limits().should_stop());
}
