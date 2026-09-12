//! The one stop axis: when an operation in flight must give up.

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

/// What a stop callback ([`LimitConfig::with_stop_callback`](crate::limits::LimitConfig::with_stop_callback)) concludes when an
/// in-operation poll asks it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum StopDecision {
    /// Carry on. The operation is never interrupted and never re-pays anything
    /// — the decision cost it one poll it was making anyway.
    Continue,
    /// Stop here. Surfaces to the caller as [`OperationError::Stopped`](crate::OperationError::Stopped), which is
    /// the unwind path a mid-operation cut already has.
    Stop,
    /// Carry on, under this stop from here on — a commitment, which replaces
    /// whatever stop the operation was running under.
    ReplaceRules(StopRules),
}

