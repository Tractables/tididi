//! The one stop axis: when an operation in flight must give up.

/// The point a [`Stop`] falls at — the same rule in whichever currency the run
/// is budgeted in.
///
/// A caller that prices work against a wall gets a cut that lands at a
/// different PLACE in the compile on every box, because two runs of one formula
/// reach different points before the same fraction of the wall is gone.
/// [`StopAt::Work`] prices it against the engine's own work clock instead, so
/// the cut is reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopAt {
    /// The stop falls at this instant.
    Wall(std::time::Instant),
    /// The stop falls once the work clock ([`ApplyMeters::work_units`](crate::engine::ApplyMeters::work_units)) reaches
    /// this many units.
    Work(u64),
}

impl StopAt {
    /// The instant this falls at, and `None` for a work-shaped one — for the
    /// callers that can only plan on the clock, which must see nothing rather
    /// than a converted guess.
    #[must_use]
    pub fn wall(self) -> Option<std::time::Instant> {
        match self {
            StopAt::Wall(at) => Some(at),
            StopAt::Work(_) => None,
        }
    }
}

/// When the operation in flight gives up, on one axis with two bounds.
///
/// `wall` is unconditional: past it the operation stops whatever it has built.
/// `after` is conditional on SIZE — past its point, an operation that has built
/// at least `pairs` output pairs stops, and one that has not carries on. A
/// caller that wants to cut a step for spending too long on a big diagram arms
/// the second; a floor of zero makes it unconditional too, which is how a step
/// already big at the door and a step that grows into one ride the same bound.
///
/// The floor is counted in output PAIRS — the unit [`Tdd::size`](crate::Tdd::size) and a caller's
/// own input measurement are already stated in — and not in bytes, which a step
/// that has built no diagram at all can meet through scratch alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stop {
    /// The unconditional bound, or `None` for an operation nothing walls in.
    pub wall: Option<StopAt>,
    /// `(pairs, at)`: the bound that applies once the operation has built
    /// `pairs` output pairs. `None` for an operation with no size-conditional
    /// bound.
    pub after: Option<(u64, StopAt)>,
}

impl Stop {
    /// Nothing armed.
    pub const NONE: Stop = Stop { wall: None, after: None };

    /// Stop unconditionally at `at`.
    #[must_use]
    pub fn at(at: StopAt) -> Stop {
        Stop { wall: Some(at), after: None }
    }

    /// Stop unconditionally at this instant.
    #[must_use]
    pub fn by(deadline: std::time::Instant) -> Stop {
        Stop::at(StopAt::Wall(deadline))
    }

    /// Add the size-conditional bound `at`, in force once the operation has
    /// built `pairs` output pairs.
    #[must_use]
    pub fn after_pairs(mut self, pairs: u64, at: StopAt) -> Stop {
        self.after = Some((pairs, at));
        self
    }

    /// Is any bound armed?
    #[must_use]
    #[inline]
    pub fn armed(self) -> bool {
        self.wall.is_some() || self.after.is_some()
    }
}

/// What a scheduled callback ([`LimitSet::schedule`](crate::engine::LimitSet::schedule)) concludes when an
/// in-operation poll asks it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Scheduled {
    /// Carry on. The operation is never interrupted and never re-pays anything
    /// — the decision cost it one poll it was making anyway.
    Carry,
    /// Stop here. Surfaces to the caller as [`ApplyError::Deadline`](crate::ApplyError::Deadline), which is
    /// the unwind path a mid-operation cut already has.
    Stop,
    /// Carry on, under this stop from here on — a commitment, which replaces
    /// whatever stop the operation was running under.
    Replace(Stop),
}

