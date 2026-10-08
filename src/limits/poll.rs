//! Cooperative cancellation and amortized work accounting.

use super::{OperationError, Limits, Quiet, StopDecision, StopAt};
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

    /// Add `units` units as that many polls of one unit would: the clock is
    /// charged and cancellation tested at the same points, each time the gate
    /// comes due. For a loop that charges one unit an item and whose items
    /// have no other effect on the limits, so it can charge them all in one
    /// call, before or after the loop.
    #[inline]
    pub(crate) fn poll_each(&mut self, units: u64) -> Result<(), OperationError> {
        let step = self.stride.max(1);
        let due = step - self.work.min(step);
        if units < due {
            self.work += units;
            return Ok(());
        }
        self.work = 0;
        self.lim.poll_now(step)?;
        let mut left = units - due;
        while left >= step {
            self.lim.poll_now(step)?;
            left -= step;
        }
        self.work = left;
        Ok(())
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

    /// Charge the residue and test cancellation even when the gate is empty:
    /// the last check a query makes before it returns its answer.
    pub(crate) fn finish(mut self) -> Result<(), OperationError> {
        self.lim.charge_work(std::mem::replace(&mut self.work, 0));
        self.lim.check_stop()
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
    /// Charge `units` of the host's own work to the work clock and test
    /// cancellation, as an operation's poll does.
    ///
    /// For computation the host runs on the engine's diagrams outside its
    /// operations, such as a count that walks a diagram itself, which a
    /// [`StopAt::WorkUnits`] threshold should cover as it covers the
    /// operations. The host chooses how many units its work is worth.
    ///
    /// ```
    /// use tididi::Engine;
    /// use tididi::limits::{LimitConfig, StopAt, StopRules};
    ///
    /// let engine = Engine::new();
    /// let limits = engine.limits();
    /// let start = limits.work_units();
    /// let rules = StopRules { unconditional: Some(StopAt::WorkUnits(start + 100)), after_pairs: None };
    /// let _scope = limits.scope(LimitConfig::none().with_stop_rules(rules));
    /// assert!(limits.poll_host_work(60).is_ok());
    /// assert_eq!(limits.poll_host_work(60), Err(tididi::OperationError::Stopped));
    /// assert_eq!(limits.work_units(), start + 120);
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::Stopped`] when an installed stop rule is reached or
    /// the stop callback decides to stop. The units are charged either way.
    pub fn poll_host_work(&self, units: u64) -> Result<(), OperationError> {
        self.poll_now(units)
    }

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
    /// Work-unit-only rules avoid a clock read on each poll, and a poll below
    /// every work-unit threshold reads neither the rules nor the callback.
    #[inline(always)]
    pub(crate) fn should_stop(&self) -> bool {
        if self.quiet.get().holds(self.work_clock.get(), self.pairs_in_flight.get()) {
            return false;
        }
        self.should_stop_now()
    }

    /// [`should_stop`](Self::should_stop) past its quiet bounds.
    #[cold]
    #[inline(never)]
    pub(super) fn should_stop_now(&self) -> bool {
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
                    // The callback may have installed another configuration
                    // while it decided; the bounds follow whichever is in place.
                    self.quiet.set(Quiet::of(next, self.stop_callback.borrow().is_some()));
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

    /// Whether no stop can fire while `work` more units reach the work clock
    /// and `pairs` more the output-pair meter, whenever the polls between
    /// fall: no callback decides, no rule reads the time, and each rule's
    /// threshold lies past the clock's end or its pair floor past the
    /// meter's. A route that knows its work beforehand may then charge it in
    /// one go and poll once.
    pub(crate) fn cannot_stop_within(&self, work: u64, pairs: u64) -> bool {
        if self.stop_callback.borrow().is_some() {
            return false;
        }
        let end = self.work_clock.get().saturating_add(work);
        let before = |at: StopAt| matches!(at, StopAt::WorkUnits(units) if end < units);
        let stop = self.stop.get();
        stop.unconditional.is_none_or(before)
            && stop.after_pairs.is_none_or(|(floor, at)| {
                self.pairs_in_flight.get().saturating_add(pairs) < floor || before(at)
            })
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
