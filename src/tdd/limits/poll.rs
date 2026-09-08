//! The amortized poll ticker and the strides it polls at.

use super::error::ApplyError;
use super::stop::{charge_compile_work, deadline_expired};
#[cfg(test)]
use std::cell::Cell;

/// Amortized cut ticker shared by the dense between-cell loops (stride 1<<16),
/// the sparse-scatter/collapse-collector loops (stride 1<<20 =
/// `APPLY_POLL_STRIDE`) and the reduce walk (`REDUCE_POLL_STRIDE`). One poll per
/// `stride` units of accumulated work: a gated wall-deadline check. See
/// `nxm_deadline_check!` for the deliberate intra-cell exception. The compile
/// path is sequential — there is no cross-thread cancellation, so the wall
/// deadline is the only mid-level cut.
pub(crate) struct PollTicker {
    work: u64,
    stride: u64,
}

impl PollTicker {
    #[inline]
    pub(crate) fn new(stride: u64) -> Self {
        Self { work: 0, stride }
    }

    /// A ticker for the post-apply walks — same amortization, a coarser
    /// cadence. `stride` is a parameter (not the constant) so a test can pin the
    /// counter's cadence without lowering the production one.
    #[inline]
    pub(crate) fn reduce(stride: u64) -> Self {
        Self { work: 0, stride }
    }

    /// Add `inc` units of accumulated work; once `work >= stride`, reset to 0
    /// and poll. (The former one-unit `tick()` had no callers left once the
    /// dense row loop moved to a per-ROW `tick_by(k2)`.)
    #[inline(always)]
    pub(crate) fn tick_by(&mut self, inc: u64) -> Result<(), ApplyError> {
        self.work += inc;
        if self.work >= self.stride {
            let done = self.work;
            self.work = 0;
            self.poll(done)
        } else {
            Ok(())
        }
    }

    #[cold]
    fn poll(&self, done: u64) -> Result<(), ApplyError> {
        // A TEE of the work `tick_by` already counted, not a second counter:
        // this is the amortization point that already exists for reading it,
        // which is why the compile work clock costs the hot path nothing.
        //
        // `done` is what the meter actually held, NOT the stride it crossed. A
        // single `tick_by` carries a whole dense row (`k2`), which can be many
        // strides wide on its own — charging one stride per poll would price
        // that row the same as the narrowest one that trips the meter, and the
        // undercount would fall hardest on exactly the wide-row applies the
        // give-up rule is trying to measure.
        charge_compile_work(done);
        if deadline_expired() {
            return Err(ApplyError::Deadline);
        }
        Ok(())
    }
}

/// Amortization stride for the post-apply walks' [`PollTicker`] — one poll per
/// ~16384 units, where a unit is one node of the level the walk is standing on
/// (a contracted parent's level, a forget batch's target level, a clustering
/// pivot's pair count).
///
/// Smaller than the apply strides because the unit is coarser: the apply counts
/// individual product pairs, these walks count whole levels' node widths, and a
/// worklist of narrow parents would otherwise run millions of pops between two
/// polls. At this cadence the poll (one relaxed load, one TLS read, one
/// `Instant::now`) is under a thousandth of the work it amortizes over even when
/// every popped parent is as small as it can be.
const REDUCE_POLL_STRIDE: u64 = 1 << 14;

#[cfg(test)]
thread_local! {
    /// Test-only override for [`reduce_poll_stride`]. Thread-local, so a test that
    /// pins the cadence cannot race the rest of the suite — the same shape as
    /// `cell::BOTHMARG_COLLAPSE_OVERRIDE`.
    static REDUCE_POLL_STRIDE_OVERRIDE: Cell<Option<u64>> = const { Cell::new(None) };
}

/// The post-apply walks' amortization stride: [`REDUCE_POLL_STRIDE`], or a
/// test's pinned value.
///
/// The hook exists so the amortization itself is testable — a diagram big enough
/// to accumulate 16384 units of contract work before the meter comes due is not a
/// unit test — WITHOUT lowering the production cadence, which is the number the
/// overhead argument is made about.
#[inline]
pub(crate) fn reduce_poll_stride() -> u64 {
    #[cfg(test)]
    if let Some(stride) = REDUCE_POLL_STRIDE_OVERRIDE.with(|c| c.get()) {
        return stride;
    }
    REDUCE_POLL_STRIDE
}

/// Run `body` with the post-apply walks' stride pinned to `stride` on this
/// thread, restoring the prior setting on return. Test-only.
#[cfg(test)]
pub(crate) fn with_reduce_poll_stride<T>(stride: u64, body: impl FnOnce() -> T) -> T {
    crate::tdd::scoped::Scoped::run(&REDUCE_POLL_STRIDE_OVERRIDE, Some(stride), body)
}
