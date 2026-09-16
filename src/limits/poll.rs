//! Cooperative cancellation and amortized work accounting.

use super::{OperationError, Limits, StopDecision, StopAt};
use std::time::Instant;

/// Amortization stride for the post-conjunction walks' [`PollGate`]: one poll
/// per ~16384 units, where a unit is one node of the level the walk is standing
/// on (a contracted parent's level, a forget batch's target level, a clustering
/// pivot's pair count).
///
/// Smaller than the conjunction's own strides because the unit is coarser: a
/// whole level's node width rather than one product pair.
pub(super) const REDUCE_POLL_STRIDE: u64 = 1 << 14;

/// The accumulator an in-operation loop polls through: one poll per `stride`
/// units of work, amortizing the check. The stop axis is the only
/// mid-level cut.
pub(crate) struct PollGate {
    work: u64,
    stride: u64,
}

impl PollGate {
    #[inline]
    pub(crate) fn new(stride: u64) -> PollGate {
        PollGate { work: 0, stride }
    }
}

impl Limits {
    /// Add `work` units to `gate` and, once it comes due, charge the work clock
    /// and test cancellation.
    ///
    /// The one amortized cut every in-operation loop makes: the dense level
    /// walk, the sparse scatter and collapse collectors, and the walks that run
    /// between two conjunctions of one step.
    #[inline(always)]
    pub(crate) fn poll(&self, gate: &mut PollGate, work: u64) -> Result<(), OperationError> {
        gate.work += work;
        if gate.work < gate.stride {
            return Ok(());
        }
        let done = std::mem::replace(&mut gate.work, 0);
        self.poll_now(done)
    }

    /// Charge whatever `gate` still holds and test cancellation.
    ///
    /// A gate that spans a whole level ends it holding less than one stride,
    /// and that remainder is real work: without this the clock loses up to one
    /// stride per level, which on a diagram of many small levels is most of the
    /// work there was. Called once, where the gate goes out of scope.
    pub(crate) fn flush_poll(&self, gate: &mut PollGate) -> Result<(), OperationError> {
        let done = std::mem::replace(&mut gate.work, 0);
        if done == 0 {
            return Ok(());
        }
        self.poll_now(done)
    }

    /// The cold half of [`Limits::poll`].
    ///
    /// `done` is what the gate actually held, not the stride it crossed: a
    /// single poll can carry a whole dense row, which may be many strides wide
    /// on its own, and charging one stride per poll would price that row the
    /// same as the narrowest one that trips the gate.
    #[cold]
    fn poll_now(&self, done: u64) -> Result<(), OperationError> {
        self.charge_work(done);
        if self.should_stop() {
            return Err(OperationError::Stopped);
        }
        Ok(())
    }
}

impl Limits {
    /// Ask the callback before checking thresholds, allowing it to replace an
    /// expired rule and let the operation continue.
    #[inline]
    pub(crate) fn should_stop(&self) -> bool {
        let stop = self.stop.get();
        let callback = self.stop_callback.borrow().clone();
        if !stop.armed() && callback.is_none() {
            return false;
        }
        let now = Instant::now();
        let stop = match callback {
            Some(decide) => match decide.decide(&self.meters(), now) {
                StopDecision::Stop => return true,
                StopDecision::Continue => stop,
                StopDecision::ReplaceRules(next) => {
                    self.stop.set(next);
                    next
                }
            },
            None => stop,
        };
        // The size-conditional bound first: it is the cheaper half (the clock is
        // already read) and before it falls the pair meter does not matter.
        if let Some((floor_pairs, at)) = stop.after_pairs
            && self.reached(at, now)
            && self.pairs_in_flight.get() >= floor_pairs
        {
            return true;
        }
        stop.unconditional.is_some_and(|at| self.reached(at, now))
    }

    #[inline]
    fn reached(&self, at: StopAt, now: Instant) -> bool {
        match at {
            StopAt::Time(t) => now >= t,
            StopAt::WorkUnits(units) => self.work_clock.get() >= units,
        }
    }
}
