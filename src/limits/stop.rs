//! Cancellation thresholds and callback decisions.

/// An absolute threshold on wall-clock time or the engine's work clock.
///
/// Work units follow operation polls and give a reproducible stopping point
/// for the same operation sequence independently of elapsed wall-clock time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopAt {
    /// The stop falls at this instant.
    Time(std::time::Instant),
    /// The stop falls once the work clock ([`OperationMetrics::work_units`](crate::limits::OperationMetrics::work_units)) reaches
    /// this many units.
    WorkUnits(u64),
}

impl StopAt {
    /// The wall-clock instant, or `None` for a work-clock threshold.
    #[must_use]
    pub fn time(self) -> Option<std::time::Instant> {
        match self {
            StopAt::Time(at) => Some(at),
            StopAt::WorkUnits(_) => None,
        }
    }

}

/// Unconditional and output-pair-dependent bounds on when an operation stops.
///
/// `unconditional` applies regardless of output size. `after_pairs` applies
/// once its threshold is reached and the conjunction has built at least the
/// specified number of output pairs. A zero pair floor makes it unconditional.
/// Both thresholds use [`StopAt`] and may be wall-clock times or work units.
/// Boolean transformations check an armed stop even for constant operands or
/// empty requests; an identity result does not bypass cancellation.
///
/// The pair floor uses [`OperationMetrics::pairs_in_flight`](crate::limits::OperationMetrics::pairs_in_flight),
/// which is reset at operation entry and records the current or last pairwise
/// conjunction within that operation. Other phases of a compound operation
/// may therefore observe its last conjunction's count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StopRules {
    /// The unconditional bound, or `None` to disable it.
    pub unconditional: Option<StopAt>,
    /// `(pairs, at)`: the bound that applies once the operation has built
    /// `pairs` output pairs. `None` for an operation with no size-conditional
    /// bound.
    pub after_pairs: Option<(u64, StopAt)>,
}

impl StopRules {
    /// Nothing armed.
    pub(crate) const NONE: StopRules = StopRules { unconditional: None, after_pairs: None };

    /// Stop unconditionally at this instant.
    #[must_use]
    pub fn by_time(deadline: std::time::Instant) -> StopRules {
        StopRules { unconditional: Some(StopAt::Time(deadline)), after_pairs: None }
    }

    /// Add the size-conditional bound `at`, in force once the operation has
    /// built `pairs` output pairs.
    #[must_use]
    pub fn after_pairs(mut self, pairs: u64, at: StopAt) -> StopRules {
        self.after_pairs = Some((pairs, at));
        self
    }

    /// Is any bound armed?
    #[must_use]
    #[inline]
    pub(crate) fn armed(self) -> bool {
        self.unconditional.is_some() || self.after_pairs.is_some()
    }
}

/// The meter readings under which no stop can fire: while the work clock is
/// below `work`, and the output-pair meter below `pairs` or the clock below
/// `pairs_work`, the rules it was taken from cannot stop an operation.
///
/// A wall-clock threshold or a stop callback can fire at any reading, so its
/// bound is zero; a rule that is not armed bounds nothing. Testing these
/// three numbers is what lets a poll that cannot stop skip the rules and the
/// callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quiet {
    work: u64,
    pairs: u64,
    pairs_work: u64,
}

impl Quiet {
    /// Nothing armed: no reading stops.
    pub(crate) const NONE: Quiet = Quiet { work: u64::MAX, pairs: u64::MAX, pairs_work: u64::MAX };

    /// The bounds of `rules`, with a stop callback installed or not.
    pub(crate) fn of(rules: StopRules, callback: bool) -> Quiet {
        if callback {
            return Quiet { work: 0, pairs: 0, pairs_work: 0 };
        }
        let units = |at: StopAt| match at {
            StopAt::WorkUnits(units) => units,
            StopAt::Time(_) => 0,
        };
        let (pairs, pairs_work) = match rules.after_pairs {
            Some((floor, at)) => (floor, units(at)),
            None => (u64::MAX, u64::MAX),
        };
        Quiet { work: rules.unconditional.map_or(u64::MAX, units), pairs, pairs_work }
    }

    /// Whether no stop can fire at these readings of the work clock and the
    /// output-pair meter.
    #[inline(always)]
    pub(crate) fn holds(self, work: u64, pairs: u64) -> bool {
        work < self.work && (pairs < self.pairs || work < self.pairs_work)
    }
}

/// What a stop callback ([`LimitConfig::with_stop_callback`](crate::limits::LimitConfig::with_stop_callback)) concludes when an
/// in-operation poll asks it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum StopDecision {
    /// Continue under the current stop rules.
    Continue,
    /// Return [`OperationError::Stopped`](crate::OperationError::Stopped).
    Stop,
    /// Replace the stop rules before checking their thresholds.
    ReplaceRules(StopRules),
}

