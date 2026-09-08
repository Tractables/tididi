//! The stop axes — deadline, decision callback, stall rope — and the check
//! every in-operation poll makes against them.

use super::APPLY_LIMITS;

/// What a scheduled callback ([`ApplyLimitsInstall::schedule`]) concludes when
/// an in-operation poll asks it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Scheduled {
    /// Carry on. The compile is never interrupted and never re-pays anything —
    /// the decision cost it one poll it was making anyway.
    Carry,
    /// Stop here. Surfaces to the caller as [`ApplyError::Deadline`], which is
    /// the unwind path a mid-operation cut already has.
    Stop,
    /// Carry on, under this deadline from here on — a commitment, which replaces
    /// whatever deadline the compile was running under.
    Until(std::time::Instant),
}

/// The work clock ([`ApplyMeters::work_units`]).
#[inline]
fn work_units() -> u64 {
    APPLY_LIMITS.with(|l| l.work_clock.get())
}

/// Add `units` to the compile work clock.
#[inline]
pub(crate) fn charge_compile_work(units: u64) {
    APPLY_LIMITS.with(|l| l.work_clock.set(l.work_clock.get().saturating_add(units)));
}

/// When a stall rope falls — the same rule in whichever currency the run is
/// budgeted in.
///
/// The give-up rule prices a step against the wall it began with, and a wall is
/// what a loaded box makes unreproducible: two runs of one formula reach
/// different steps before the same fraction is gone. `Work` prices it against
/// the compile's own work clock instead, so the cut lands at the same PLACE in
/// the compile on every box. Nothing else about the rule changes, which is why
/// this is one enum on one axis rather than a second rope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLimit {
    /// The rope falls at this instant — the wall-clock rule as it shipped.
    Wall(std::time::Instant),
    /// The rope falls once the work clock ([`ApplyMeters::work_units`]) reaches
    /// this many units.
    Work(u64),
}

impl RopeLimit {
    /// The instant this rope falls at, and `None` for a work-shaped one — for the
    /// callers that can only plan on the clock (the compile's ladder), which must
    /// see nothing rather than a converted guess.
    pub fn wall(self) -> Option<std::time::Instant> {
        match self {
            RopeLimit::Wall(at) => Some(at),
            RopeLimit::Work(_) => None,
        }
    }
}

/// Has the operation in flight reached something that stops it?
///
/// Polled from the apply's cell and scatter loops and from the walks that run
/// between two applies of one step (twin contraction, the ∃-forget batch, the
/// clustering rotation pass). Installing a stop axis is what arms this: with
/// none installed the cost is one thread-local resolution and three `Cell`
/// loads, and the clock is never read.
///
/// The SCHEDULE is asked before the deadline, and the order is load-bearing: a
/// schedule's decision point may CONCLUDE that the compile deserves the rest of
/// the wall ([`Scheduled::Until`]), and asking the deadline first would let a
/// stale, shorter deadline — one an inner scope restored underneath the
/// commitment — cut a compile the schedule has already committed to.
#[inline]
pub(crate) fn deadline_expired() -> bool {
    let (deadline, schedule, stall) =
        APPLY_LIMITS.with(|l| (l.deadline.get(), l.schedule.get(), l.stall_rope.get()));
    if deadline.is_none() && schedule.is_none() && stall.is_none() {
        return false;
    }
    let now = std::time::Instant::now();
    if let Some(decide) = schedule {
        match decide(now) {
            Scheduled::Stop => return true,
            Scheduled::Carry => {}
            Scheduled::Until(wall) => {
                APPLY_LIMITS.with(|l| l.deadline.set(Some(wall)));
                return now >= wall;
            }
        }
    }
    // The stall rope ([`ApplyLimits::stall_rope`]): a cut whose caller made it
    // conditional on the apply having BUILT enough output PAIRS to be priced as
    // diagram growth — the same unit the caller's input floor is stated in — and
    // a floor of zero for the step that was already eligible at the door. It is
    // asked here, beside the deadline, because here is the only place inside a
    // level anything gets asked — an apply that never finishes a level is an
    // apply the level-boundary checks never reach, which is exactly the shape it
    // exists for. The rope is tested first: it is the cheaper half (the clock is
    // already read, the work clock is one TLS load), and before the rope falls
    // the meter does not matter.
    if let Some((floor_pairs, rope)) = stall
        && match rope {
            RopeLimit::Wall(at) => now >= at,
            RopeLimit::Work(at) => work_units() >= at,
        }
        && APPLY_LIMITS.with(|l| l.pairs_in_flight.get()) >= floor_pairs
    {
        return true;
    }
    deadline.is_some_and(|deadline| now >= deadline)
}

/// `true` when any stop axis is installed on this thread.
///
/// Hoisted once per level by the cell kernel so its intra-cell poll costs a
/// local-bool branch. Sound as a hoist: the only writer of a stop axis during an
/// operation is [`deadline_expired`] committing a schedule's
/// [`Scheduled::Until`], and a level that enters with nothing armed reaches no
/// poll that could arm one.
#[inline]
pub(crate) fn any_stop_armed() -> bool {
    APPLY_LIMITS.with(|l| {
        l.deadline.get().is_some() || l.schedule.get().is_some() || l.stall_rope.get().is_some()
    })
}

/// The armed cap on output nodes per apply (one TLS read; the engine only sums
/// output levels when this is `Some`).
#[inline]
pub(crate) fn apply_output_node_cap() -> Option<u64> {
    APPLY_LIMITS.with(|l| l.output_node_cap.get())
}
