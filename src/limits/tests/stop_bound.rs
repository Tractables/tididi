//! The stop axis's size-conditional bound, and the output-pair meter it reads.

use super::*;
use crate::Engine;
use crate::limits::OperationError;
use std::time::{Duration, Instant};

fn spent() -> StopAt {
    StopAt::Time(Instant::now() - Duration::from_secs(1))
}

fn unspent() -> StopAt {
    StopAt::Time(Instant::now() + Duration::from_secs(60))
}

fn finish_level(eng: &Engine, exact_pairs: u64) -> Result<(), OperationError> {
    let lim = eng.limits();
    lim.level_settled(exact_pairs);
    lim.level_done(0)
}

/// **The size-conditional bound needs both halves, and it is asked at the
/// INTRA-LEVEL poll — not only at the level boundary.**
///
/// It is the answer to a step whose inputs were small and whose output is not:
/// the caller arms a floor in PAIRS, the unit its own measurement is stated in,
/// as the trigger, and the step's usual share of the wall as the bound. The
/// meter is the output-pair count, which is what lets the check ride the poll
/// the wall already uses from inside the dense loops. An operation that never
/// finishes a level never reaches a level boundary, and that is precisely the
/// operation this is for.
#[test]
fn a_conditional_bound_cuts_at_the_intra_level_poll_once_built_and_out_of_time() {
    let eng = Engine::new();
    let lim = eng.limits();

    // Nothing armed: the poll is inert, whatever the operation has built.
    lim.charge_output_pairs(1 << 20);
    assert_eq!(lim.armed().stop_rules(), StopRules::NONE);
    assert!(!lim.should_stop());
    assert!(finish_level(&eng, 0).is_ok());

    let eng = Engine::new();
    let lim = eng.limits();
    let armed = StopRules::default().after_pairs(1_000, spent());
    let _prior = lim.install(LimitConfig::none().with_stop_rules(armed));
    assert_eq!(lim.armed().stop_rules(), armed);
    // Share spent, output still under the floor: this is a step whose long run
    // is search, which is the whole reason the bound has a size factor at all.
    // It is not cut.
    lim.charge_output_pairs(999);
    assert!(!lim.should_stop(), "under the floor, not at it");
    // At the floor with the share spent: a big diagram, and out of time.
    lim.charge_output_pairs(1);
    assert_eq!(lim.meters().pairs_in_flight, 1_000);
    assert!(lim.should_stop());
    // …and the level boundary reports it as what it is, because it asks that
    // same poll rather than keeping a second meter of its own.
    assert!(matches!(finish_level(&eng, 1_000), Err(OperationError::Stopped)));

    // Built past the floor, but the share it was given is still on the clock —
    // the floor moves who is ELIGIBLE, never when the bound falls.
    let eng = Engine::new();
    let lim = eng.limits();
    let _prior = lim.install(LimitConfig::none().with_stop_rules(StopRules::default().after_pairs(1_000, unspent())));
    lim.charge_output_pairs(2_000);
    assert!(!lim.should_stop());

    // Restoring the previous set puts the axis back, so nothing outside the step
    // it belonged to can be cut by it.
    let prior = lim.install(LimitConfig::none());
    assert!(prior.stop_rules().after_pairs.is_some());
    assert!(!lim.should_stop());
}

/// **A level's unused arena slack does not survive the level.**
///
/// Within a level the meter can only estimate — it reads the arena's CAPACITY,
/// which is seeded before a single pair is emitted. Left standing, that seed
/// would accumulate once per level over the thousands one operation walks, and
/// the meter would grow with the LEVEL COUNT instead of the diagram. So each
/// boundary swaps the estimate for the truth.
#[test]
fn a_finished_level_contributes_its_pairs_not_its_capacity() {
    let eng = Engine::new();
    let lim = eng.limits();
    // Level 1: seeded for 8192, emitted 3.
    lim.charge_output_pairs(8192);
    assert_eq!(lim.meters().pairs_in_flight, 8192, "in flight, capacity is all there is");
    finish_level(&eng, 3).unwrap();
    assert_eq!(lim.meters().pairs_in_flight, 3);
    // Level 2 does the same: the slack does not compound.
    lim.charge_output_pairs(8192);
    finish_level(&eng, 3).unwrap();
    assert_eq!(lim.meters().pairs_in_flight, 6, "two low-survival levels, six pairs");
    // A level that genuinely builds keeps what it built.
    lim.charge_output_pairs(2_000_000);
    finish_level(&eng, 1_500_000).unwrap();
    assert_eq!(lim.meters().pairs_in_flight, 1_500_006);
}

/// **A step that is heavy in BYTES but light in built PAIRS is not eligible.**
///
/// The floor asks how big a diagram this step holds. The byte meter answers a
/// different question — how much memory the operation has claimed, including
/// grids, live scratch and hoisted transients — and metering the rule with it
/// made a tiny node that touched 8 MB of scratch look like a million-pair
/// diagram.
#[test]
fn bytes_are_not_pairs_a_big_operation_that_built_little_is_not_cut() {
    let eng = Engine::new();
    let lim = eng.limits();
    let _prior = lim.install(LimitConfig::none().with_stop_rules(StopRules::default().after_pairs(1_000_000, spent())));
    // A gigabyte of charged memory, and a diagram of 999_999 pairs.
    lim.charge_in_flight(1 << 30);
    lim.charge_output_pairs(999_999);
    assert!(!lim.should_stop(), "eligibility is the pair count, not the byte count");
    // One more pair — the same bound, now genuinely met.
    lim.charge_output_pairs(1);
    assert!(lim.should_stop());
}

/// **A work-shaped bound falls on the work clock and on nothing else.**
///
/// The whole point of the currency: no wall is armed, no time passes, and the
/// cut happens anyway — because the operation did the work.
#[test]
fn a_work_bound_falls_on_the_work_clock_and_not_on_the_wall() {
    let eng = Engine::new();
    let lim = eng.limits();
    let stride = 1u64 << 20;
    let at = lim.meters().work_units.saturating_add(4 * stride);
    let armed = StopRules::default().after_pairs(0, StopAt::WorkUnits(at));
    let _prior = lim.install(LimitConfig::none().with_stop_rules(armed));
    assert_eq!(lim.armed().stop_rules(), armed);
    // Nothing has a wall here, and the floor is zero, so what holds the bound
    // back is the clock alone.
    assert!(!lim.should_stop(), "a work bound fell before its work was done");
    lim.charge_work(3 * stride);
    assert!(!lim.should_stop(), "one stride short is short");
    lim.charge_work(stride);
    assert!(lim.should_stop());
    assert!(matches!(finish_level(&eng, 0), Err(OperationError::Stopped)));

    let _prior = lim.install(LimitConfig::none());
    assert!(!lim.should_stop(), "restoring the previous set left a work bound armed");
}

/// **Clearing the wall does not disarm the size-conditional bound; `without_stop_rules` does.**
///
/// A caller that means "run this without a stop" reaches for the verb that
/// names the axis it knows about, and `deadline(None)` names only the
/// unconditional half. A rope armed on `after` by whatever ran before survives
/// that call and cuts the operation the caller thought it had freed, which
/// reads downstream as an out-of-memory exit rather than as a stop. `without_stop_rules`
/// is the whole-axis verb: no bound, and no schedule to arm one.
#[test]
fn clearing_the_wall_leaves_a_conditional_bound_armed_and_uncut_removes_it() {
    let roped = LimitConfig::none()
        .with_stop_rules(StopRules::default().after_pairs(0, spent()))
        .with_stop_callback(Some(crate::limits::StopCallback::new(|_: &OperationMetrics, _: Instant| StopDecision::Stop)));

    let shielded = roped.clone().with_deadline(None);
    assert!(shielded.stop_rules().after_pairs.is_some(), "deadline names the wall only");
    assert!(shielded.stop_callback().is_some());

    let eng = Engine::new();
    let lim = eng.limits();
    let _prior = lim.install(shielded);
    lim.charge_output_pairs(1);
    assert!(lim.should_stop(), "the rope outlived the call that was meant to free the operation");

    let free = roped.without_stop_rules();
    assert_eq!(free.stop_rules(), StopRules::NONE);
    assert!(free.stop_callback().is_none());

    let eng = Engine::new();
    let lim = eng.limits();
    let _prior = lim.install(free);
    lim.charge_output_pairs(1);
    assert!(!lim.should_stop());
}

/// **A scope restores every axis of the set it displaced, on an unwind too.**
///
/// The restore is the whole point of the guard: a caller that catches a panic
/// and carries on must not inherit a limit armed for the work that panicked.
#[test]
fn a_scope_puts_back_what_it_displaced() {
    let eng = Engine::new();
    let lim = eng.limits();
    let outer = LimitConfig::none().with_memory_budget_bytes(Some(64)).with_output_node_cap(Some(8));
    let _prior = lim.install(outer);

    {
        let _inner = lim.edit(|s| s.with_deadline(Some(unspent().time().unwrap())));
        assert_eq!(lim.armed().memory_budget_bytes(), Some(64), "edit leaves the other axes alone");
        assert!(lim.armed().stop_rules().unconditional.is_some());
    }
    assert!(lim.armed().stop_rules().unconditional.is_none());
    assert_eq!(lim.armed().output_node_cap(), Some(8));

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _inner = lim.scope(LimitConfig::none().with_memory_budget_bytes(Some(1)));
        panic!("the work under the scope gave up");
    }));
    assert!(caught.is_err());
    assert_eq!(lim.armed().memory_budget_bytes(), Some(64), "the unwind restored the enclosing set");
    assert_eq!(lim.armed().output_node_cap(), Some(8));
}

/// **The size-conditional bound is armed on the stop the set already carries.**
#[test]
fn arming_the_size_conditional_bound_leaves_the_unconditional_one_alone() {
    let wall = unspent();
    let floor = spent();
    let armed = LimitConfig::none().with_deadline(wall.time());
    let stop = armed.stop_rules().after_pairs(4, floor);
    let both = armed.with_stop_rules(stop);
    assert_eq!(both.stop_rules().unconditional, Some(wall));
    assert_eq!(both.stop_rules().after_pairs, Some((4, floor)));
}

/// **A mark scopes the monotone clock to an interval, and a work stop is
/// absolute.**
#[test]
fn a_mark_measures_the_work_run_since_it_was_taken() {
    let eng = Engine::new();
    let lim = eng.limits();
    lim.charge_work(10);

    let mark = lim.mark();
    assert_eq!(lim.work_since(mark), 0);
    lim.charge_work(7);
    assert_eq!(lim.work_since(mark), 7, "the interval is measured from the mark, not from zero");
    assert_eq!(lim.work_units(), 17, "the clock itself is never reset");

    assert_eq!(StopAt::WorkUnits(20).time(), None, "no rate converts a work stop to an instant");
}
