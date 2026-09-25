//! The work clock's two paths off a gate: the explicit flush and the drop.

use super::*;
use crate::limits::OperationError;

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
