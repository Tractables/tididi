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
///
/// A gate holds the limits it reports to, so going out of scope charges
/// whatever it still carries. See [`PollGate::flush`] for why the residue
/// matters.
pub(crate) struct PollGate<'a> {
    lim: &'a Limits,
    work: u64,
    stride: u64,
}

impl Limits {
    /// A gate striding at [`Limits::reduce_poll_stride`], which is what every
    /// walk between two conjunctions uses.
    #[inline]
    pub(crate) fn gate(&self) -> PollGate<'_> {
        self.gate_with(self.reduce_poll_stride())
    }

    /// A gate striding at `stride`, for the walks whose unit is finer than a
    /// level: the sparse scatter and the dense cell kernel.
    #[inline]
    pub(crate) fn gate_with(&self, stride: u64) -> PollGate<'_> {
        PollGate { lim: self, work: 0, stride }
    }
}

impl PollGate<'_> {
    /// Add `work` units and, once the gate comes due, charge the work clock and
    /// test cancellation.
    ///
    /// The one amortized cut every in-operation loop makes: the dense level
    /// walk, the sparse scatter and collapse collectors, and the walks that run
    /// between two conjunctions of one step.
    #[inline(always)]
    pub(crate) fn poll(&mut self, work: u64) -> Result<(), OperationError> {
        self.work += work;
        if self.work < self.stride {
            return Ok(());
        }
        let done = std::mem::replace(&mut self.work, 0);
        self.lim.poll_now(done)
    }

    /// Charge whatever the gate still holds and test cancellation now.
    ///
    /// A gate that spans a whole level ends it holding less than one stride,
    /// and that remainder is real work: without it the clock loses up to one
    /// stride per level, which on a diagram of many small levels is most of the
    /// work there was. Dropping the gate charges the same residue, so calling
    /// this buys one thing only — the cancellation test, which a `Drop` cannot
    /// make because it cannot return an error.
    pub(crate) fn flush(&mut self) -> Result<(), OperationError> {
        let done = std::mem::replace(&mut self.work, 0);
        if done == 0 {
            return Ok(());
        }
        self.lim.poll_now(done)
    }
}

impl Drop for PollGate<'_> {
    /// The clock half of [`PollGate::flush`], for the gates that end a level
    /// without one. Work already counted here is never counted twice: both
    /// paths take the residue out of the gate.
    fn drop(&mut self) {
        if self.work > 0 {
            self.lim.charge_work(self.work);
        }
    }
}

impl Limits {
    /// The cold half of [`PollGate::poll`].
    ///
    /// `done` is what the gate actually held, not the stride it crossed: a
    /// single poll can carry a whole dense row, which may be many strides wide
    /// on its own, and charging one stride per poll would price that row the
    /// same as the narrowest one that trips the gate.
    #[cold]
    fn poll_now(&self, done: u64) -> Result<(), OperationError> {
        self.charge_work(done);
        self.check_stop()
    }
}

impl Limits {
    /// Test cancellation now and report it as an error.
    ///
    /// For the checks a walk makes on its own account rather than through a
    /// [`PollGate`]: once at the top of an operation, and once per round of a
    /// loop whose rounds are each large enough that one test apiece costs
    /// nothing. It charges no work, because the walk inside the round charges
    /// its own.
    #[inline]
    pub(crate) fn check_stop(&self) -> Result<(), OperationError> {
        if self.should_stop() {
            return Err(OperationError::Stopped);
        }
        Ok(())
    }

    /// Ask the callback before checking thresholds, allowing it to replace an
    /// expired rule and let the operation continue.
    ///
    /// Read the clock at most once, only for a reached pair floor with a time
    /// threshold, an unconditional deadline, or a callback that needs it.
    /// Work-unit-only rules avoid a clock read on each poll.
    #[inline]
    pub(crate) fn should_stop(&self) -> bool {
        let stop = self.stop.get();
        let callback = self.stop_callback.borrow().clone();
        if !stop.armed() && callback.is_none() {
            return false;
        }
        let mut now = Clock::unread();
        let stop = match callback {
            Some(decide) => match decide.decide(&self.meters(), now.read()) {
                StopDecision::Stop => return true,
                StopDecision::Continue => stop,
                StopDecision::ReplaceRules(next) => {
                    self.stop.set(next);
                    next
                }
            },
            None => stop,
        };
        // The pair meter first: it is the cheaper half now that the clock is
        // read on demand, and below the floor the threshold does not matter.
        if let Some((floor_pairs, at)) = stop.after_pairs
            && self.pairs_in_flight.get() >= floor_pairs
            && self.reached(at, &mut now)
        {
            return true;
        }
        stop.unconditional.is_some_and(|at| self.reached(at, &mut now))
    }

    #[inline]
    fn reached(&self, at: StopAt, now: &mut Clock) -> bool {
        match at {
            StopAt::Time(t) => now.read() >= t,
            StopAt::WorkUnits(units) => self.work_clock.get() >= units,
        }
    }
}

/// The instant one cancellation test runs at, read from the host on first use
/// and not at all when no threshold is a wall-clock one.
struct Clock(Option<Instant>);

impl Clock {
    /// A clock the host has not been asked for yet.
    #[inline]
    fn unread() -> Clock {
        Clock(None)
    }

    /// The instant this test runs at. Every rule in one test sees the same one.
    #[inline]
    fn read(&mut self) -> Instant {
        *self.0.get_or_insert_with(Instant::now)
    }
}
