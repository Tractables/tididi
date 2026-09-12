//! The in-operation poll: the gate a loop accumulates work in, the stride
//! the post-conjunction walks poll at, and the cut itself.

use super::{ApplyError, Limits};

/// Amortization stride for the post-conjunction walks' [`PollGate`] — one poll
/// per ~16384 units, where a unit is one node of the level the walk is standing
/// on (a contracted parent's level, a forget batch's target level, a clustering
/// pivot's pair count).
///
/// Smaller than the conjunction's own strides because the unit is coarser: a
/// whole level's node width rather than one product pair.
const REDUCE_POLL_STRIDE: u64 = 1 << 14;

/// The post-conjunction walks' amortization stride, or a test's pinned value.
#[inline]
pub(crate) fn reduce_poll_stride(pinned: Option<u64>) -> u64 {
    pinned.unwrap_or(REDUCE_POLL_STRIDE)
}

/// The accumulator an in-operation loop polls through: one poll per `stride`
/// units of work, so the check amortizes to nothing. The stop axis is the only
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
    /// and test the stop axis.
    ///
    /// The one amortized cut every in-operation loop makes: the dense level
    /// walk, the sparse scatter and collapse collectors, and the walks that run
    /// between two conjunctions of one step.
    #[inline(always)]
    pub(crate) fn poll(&self, gate: &mut PollGate, work: u64) -> Result<(), ApplyError> {
        gate.work += work;
        if gate.work < gate.stride {
            return Ok(());
        }
        let done = std::mem::replace(&mut gate.work, 0);
        self.poll_now(done)
    }

    /// Charge whatever `gate` still holds and test the stop axis.
    ///
    /// A gate that spans a whole level ends it holding less than one stride,
    /// and that remainder is real work: without this the clock loses up to one
    /// stride per level, which on a diagram of many small levels is most of the
    /// work there was. Called once, where the gate goes out of scope.
    pub(crate) fn flush_poll(&self, gate: &mut PollGate) -> Result<(), ApplyError> {
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
    fn poll_now(&self, done: u64) -> Result<(), ApplyError> {
        self.charge_work(done);
        if self.should_stop() {
            return Err(ApplyError::Deadline);
        }
        Ok(())
    }
}
