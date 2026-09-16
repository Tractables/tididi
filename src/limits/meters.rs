//! Progress and resource measurements for running operations.

use super::Limits;
use std::time::Instant;

/// Progress recorded when [`LimitConfig::with_conjunction_progress`](crate::limits::LimitConfig::with_conjunction_progress)
/// is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConjunctionProgress {
    /// When the conjunction began.
    pub started_at: std::time::Instant,
    /// The internal vtree level being built, 1-based in bottom-up order; 0
    /// before the first.
    pub level: u32,
    /// How many levels it walks in all.
    pub levels: u32,
}

/// Work and resource measurements returned by [`Limits::meters`](crate::limits::Limits::meters).
/// [`Limits::armed`](crate::limits::Limits::armed) reports the active configuration.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct OperationMetrics {
    /// Bytes the tracked reserves have charged since [`Limits::reset_meters`](crate::limits::Limits::reset_meters)
    /// or the start of the last operation, whichever is later.
    pub in_flight_bytes: u64,
    /// Output pairs built by the pairwise conjunction in flight, or by the last
    /// one finished (capacity for the level being built, exact for finished
    /// levels). Zeroed when any operation starts.
    pub pairs_in_flight: u64,
    /// The work clock: units the operations have polled through. Monotone and
    /// never reset, so an interval is a subtraction of two reads.
    pub work_units: u64,
    /// Bytes asked for by the most recent reserve the allocator refused, or
    /// `None` if none was refused since [`Limits::reset_meters`](crate::limits::Limits::reset_meters).
    /// Reset before a call when using this to distinguish that call's allocator
    /// refusal from a soft-budget refusal: the value persists across operations.
    /// Both refusals surface as [`OperationError::OverBudget`](crate::OperationError::OverBudget).
    pub refused_reserve_bytes: Option<u64>,
    /// Where the watched conjunction in flight stands, or where the last one
    /// ended; `None` before the first watched one. Nothing clears it, so
    /// `started_at` is what tells one conjunction from the next.
    pub conjunction: Option<ConjunctionProgress>,
}

impl Limits {
    /// Charge newly reserved pair capacity without adding work to each pair push.
    /// [`Self::level_settled`] replaces this estimate with the completed level's
    /// actual pair count.
    #[inline]
    pub(crate) fn charge_output_pairs(&self, delta: usize) {
        if delta == 0 {
            return;
        }
        let delta = delta as u64;
        self.pairs_in_flight
            .set(self.pairs_in_flight.get().saturating_add(delta));
        self.pairs_level_charge
            .set(self.pairs_level_charge.get().saturating_add(delta));
    }

    /// Swap the level's charged capacity for the pairs it actually holds, so
    /// only the level in flight is ever an estimate and the arena's slack
    /// cannot accumulate over the thousands of levels one conjunction walks.
    ///
    /// The early-exit routes that skip the per-level tail never settle, so what
    /// they charged comes off at the next boundary instead: the meter reads low
    /// there, which is the direction a size floor tolerates.
    #[inline]
    pub(crate) fn level_settled(&self, exact_pairs: u64) {
        let charged = self.pairs_level_charge.replace(0);
        let total = self.pairs_in_flight.get().saturating_sub(charged);
        self.pairs_in_flight.set(total.saturating_add(exact_pairs));
    }


    /// Whether conjunction progress is being recorded.
    #[inline]
    pub(crate) fn watched(&self) -> bool {
        self.conjunction_progress.get()
    }

    /// A conjunction beginning, over `levels` vtree levels. Clears whatever the
    /// last one left, so a watcher can tell two apart by the instant alone.
    pub(crate) fn merge_began(&self, levels: u32) {
        self.conjunction.set(Some(ConjunctionProgress {
            started_at: Instant::now(),
            level: 0,
            levels,
        }));
    }

    /// Record the current level while keeping the conjunction's start time.
    pub(crate) fn merge_reached(&self, level: u32) {
        if let Some(m) = self.conjunction.get() {
            self.conjunction.set(Some(ConjunctionProgress { level, ..m }));
        }
    }
}
