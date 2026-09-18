//! Every setting a `LimitConfig` carries survives `install` and reads back
//! through `armed`.
//!
//! `Limits` keeps each setting in its own cell and copies the two structs
//! field by field, so a seventh setting can be added to `LimitConfig` and left
//! out of one of the copies. The destructuring below is what catches that: a
//! new field makes this file stop compiling until the test names it, and the
//! assertions then say whether the round trip carries it.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::limits::{LimitConfig, Limits, MemoryHooks, StopAt, StopCallback, StopDecision, StopRules};

/// A configuration whose every field differs from the default, so a setting
/// dropped in transit shows up as a mismatch rather than as a default that
/// happens to agree.
fn distinctive() -> (LimitConfig, Arc<AtomicU32>) {
    let calls = Arc::new(AtomicU32::new(0));
    let seen = Arc::clone(&calls);
    let config = LimitConfig::none()
        .with_memory_budget_bytes(Some(4096))
        .with_output_node_cap(Some(77))
        .with_stop_rules(StopRules::default().after_pairs(9, StopAt::WorkUnits(11)))
        .with_deadline(Some(Instant::now() + Duration::from_secs(600)))
        .with_stop_callback(Some(StopCallback::new(move |_, _| {
            seen.fetch_add(1, Ordering::Relaxed);
            StopDecision::Continue
        })))
        .with_memory_hooks(MemoryHooks::new(|_| {}, || 17, || Some(1 << 40), || {}))
        .with_conjunction_progress(true);
    (config, calls)
}

#[test]
fn every_setting_survives_install_and_reads_back_through_armed() {
    let (config, calls) = distinctive();

    // Naming every field: adding one to `LimitConfig` breaks this line, which
    // is the point. Nothing below reads these bindings — the getters are what
    // the round trip is asserted through.
    let LimitConfig {
        memory_budget_bytes: _,
        output_node_cap: _,
        stop: _,
        stop_callback: _,
        memory_hooks: _,
        conjunction_progress: _,
    } = &config;

    let lim = Limits::new();
    let _prior = lim.install(config.clone());
    let back = lim.armed();

    assert_eq!(back.memory_budget_bytes(), config.memory_budget_bytes());
    assert_eq!(back.output_node_cap(), config.output_node_cap());
    assert_eq!(back.stop_rules(), config.stop_rules());
    assert_eq!(back.conjunction_progress_enabled(), config.conjunction_progress_enabled());

    // The callback and the hooks are closures, so they are compared by effect.
    back.stop_callback()
        .expect("the callback came back")
        .decide(&lim.meters(), Instant::now());
    assert_eq!(calls.load(Ordering::Relaxed), 1, "the callback that came back is the one installed");
    assert_eq!(back.memory_hooks().mapped_bytes(), 17, "the hooks that came back are the ones installed");
}

#[test]
fn install_returns_the_whole_prior_set_and_restores_it() {
    let (first, _) = distinctive();
    let lim = Limits::new();
    let _empty = lim.install(first.clone());

    let prior = lim.install(LimitConfig::none());
    assert_eq!(prior.memory_budget_bytes(), first.memory_budget_bytes());
    assert_eq!(prior.output_node_cap(), first.output_node_cap());
    assert_eq!(prior.stop_rules(), first.stop_rules());
    assert_eq!(prior.conjunction_progress_enabled(), first.conjunction_progress_enabled());
    assert!(prior.stop_callback().is_some());
    assert_eq!(prior.memory_hooks().mapped_bytes(), 17);

    // `none()` really cleared what the prior set had armed.
    let now = lim.armed();
    assert_eq!(now.memory_budget_bytes(), None);
    assert_eq!(now.output_node_cap(), None);
    assert_eq!(now.stop_rules(), StopRules::NONE);
    assert!(now.stop_callback().is_none());
    assert!(!now.conjunction_progress_enabled());
    assert_eq!(now.memory_hooks().mapped_bytes(), 0);

    // And putting the prior set back arms all of it again.
    let _restored = lim.install(prior);
    assert_eq!(lim.armed().output_node_cap(), first.output_node_cap());
}
