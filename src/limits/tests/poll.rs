//! The work clock's two paths off a gate: the explicit flush and the drop.

use super::*;
use crate::limits::poll::{ClockReading, CLOCK_POLLS, CLOCK_POLL_WORK};
use crate::limits::OperationError;
use std::time::{Duration, Instant};

/// A gate that ends a level holding less than one stride still charges what it
/// holds. Five production gates never flushed before the drop existed, and the
/// clock lost up to one stride per level of the diagrams they walked.
#[test]
fn a_dropped_gate_charges_the_residue_it_never_flushed() {
    let lim = Limits::new();
    let before = lim.work_units();
    {
        let mut gate = lim.gate_with(1000);
        gate.poll(7).expect("well under the stride, so no poll");
        assert_eq!(lim.work_units(), before, "the gate has not come due");
    }
    assert_eq!(lim.work_units(), before + 7, "the residue reached the clock on drop");
}

/// The flush and the drop take the residue out of the same place, so a gate
/// that does both charges once.
#[test]
fn a_flushed_gate_is_not_charged_again_when_it_drops() {
    let lim = Limits::new();
    let before = lim.work_units();
    {
        let mut gate = lim.gate_with(1000);
        gate.poll(7).unwrap();
        gate.flush().expect("no stop armed");
        assert_eq!(lim.work_units(), before + 7);
    }
    assert_eq!(lim.work_units(), before + 7, "the drop found an empty gate");
}

/// What the flush buys over the drop: a `Drop` cannot return an error, so the
/// cancellation test only happens where the flush is called.
#[test]
fn only_the_flush_reports_a_stop() {
    let lim = Limits::new();
    let _prior = lim.install(
        LimitConfig::none().with_stop_rules(StopRules::default().after_pairs(0, StopAt::WorkUnits(1))),
    );
    let mut gate = lim.gate_with(1000);
    gate.poll(7).expect("under the stride, so nothing is read");
    assert!(matches!(gate.flush(), Err(OperationError::Stopped)));
}

/// A query's last check: `finish` tests cancellation even on an empty gate,
/// and charges only the residue the gate holds.
#[test]
fn finish_tests_the_stop_without_adding_work() {
    let lim = Limits::new();
    let before = lim.work_units();
    lim.gate_with(1000).finish().expect("no stop armed");
    assert_eq!(lim.work_units(), before, "an empty gate charges nothing");
    let mut gate = lim.gate_with(1000);
    gate.poll(7).unwrap();
    gate.finish().expect("no stop armed");
    assert_eq!(lim.work_units(), before + 7, "the residue is charged once");
    let _prior = lim.install(LimitConfig::none().with_stop_rules(
        StopRules::default().after_pairs(0, StopAt::WorkUnits(before + 7)),
    ));
    assert!(matches!(lim.gate_with(1000).finish(), Err(OperationError::Stopped)));
    assert_eq!(lim.work_units(), before + 7);
}

/// `poll_each` is that many polls of one unit: the same charges at the same
/// points, the same stop, the same residue for the drop.
#[test]
fn poll_each_charges_and_stops_as_single_polls_do() {
    fn run(stride: u64, held: u64, units: u64, stop_at: u64, each: bool) -> (Result<(), OperationError>, u64, u64) {
        let lim = Limits::new();
        let _prior = lim.install(
            LimitConfig::none().with_stop_rules(StopRules::default().after_pairs(0, StopAt::WorkUnits(stop_at))),
        );
        let mut gate = lim.gate_with(stride);
        let mut result = gate.poll(held);
        if result.is_ok() {
            result = match each {
                true => gate.poll_each(units),
                false => (0..units).try_for_each(|_| gate.poll(1)),
            };
        }
        let charged = lim.work_units();
        drop(gate);
        (result, charged, lim.work_units())
    }
    for stride in [0, 1, 4, 7] {
        for held in [0, 2, 5] {
            for units in [0, 1, 3, 4, 10, 29] {
                for stop_at in [1, 6, 9, 20, 1000] {
                    assert_eq!(
                        run(stride, held, units, stop_at, true),
                        run(stride, held, units, stop_at, false),
                        "stride {stride}, held {held}, units {units}, stop at {stop_at}",
                    );
                }
            }
        }
    }
}

/// A deadline already past when it is installed stops the first poll, and
/// every poll after it: installing reads the clock afresh, and a reading past
/// the deadline answers every test it stands in for.
#[test]
fn a_spent_deadline_stops_every_poll_from_the_first() {
    let lim = Limits::new();
    let _prior = lim.install(LimitConfig::none().with_deadline(Some(Instant::now() - Duration::from_secs(1))));
    for _ in 0..3 * CLOCK_POLLS {
        assert!(lim.should_stop());
    }
}

/// Between two clock readings a test answers from the last one, which is
/// earlier than the true instant: a deadline passed since is seen at most
/// [`CLOCK_POLLS`] tests late, or at the first test after
/// [`CLOCK_POLL_WORK`] units of work.
#[test]
fn a_test_between_readings_answers_from_the_last_one() {
    let deadline = Instant::now();
    let before = deadline - Duration::from_secs(1);
    let lim = Limits::new();
    let _prior = lim.install(LimitConfig::none().with_deadline(Some(deadline)));
    // A reading taken before the deadline, with three tests left on it.
    lim.clock.set(ClockReading { at: Some(before), polls_left: 3, work_at: lim.work_units() });
    for _ in 0..3 {
        assert!(!lim.should_stop(), "answered from a reading before the deadline");
    }
    assert!(lim.should_stop(), "the fourth test reads the clock");
    assert!(lim.should_stop(), "and the reading past the deadline stands");

    // The work rule: a span of work as long as one dense poll's reads it.
    let lim = Limits::new();
    let _prior = lim.install(LimitConfig::none().with_deadline(Some(deadline)));
    lim.clock.set(ClockReading { at: Some(before), polls_left: CLOCK_POLLS, work_at: lim.work_units() });
    lim.charge_work(CLOCK_POLL_WORK - 1);
    assert!(!lim.should_stop());
    lim.charge_work(1);
    assert!(lim.should_stop(), "a span of CLOCK_POLL_WORK units reads the clock");

    // Without a deadline nothing is read and nothing counted down.
    let lim = Limits::new();
    for _ in 0..3 * CLOCK_POLLS {
        assert!(!lim.should_stop());
    }
    assert!(lim.clock.get().at.is_none());
}

/// A stop callback is asked with the instant it is asked at, every poll: the
/// readings are amortized for the thresholds only.
#[test]
fn a_callback_sees_a_fresh_instant_every_poll() {
    use std::sync::{Arc, Mutex};
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let lim = Limits::new();
    let _prior = lim.install(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, now| {
        log.lock().unwrap().push(now);
        StopDecision::Continue
    }))));
    let start = Instant::now();
    for _ in 0..4 {
        assert!(!lim.should_stop());
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    assert!(seen.iter().all(|&t| t >= start));
}
