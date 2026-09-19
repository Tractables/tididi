use super::*;

use std::cell::Cell;

thread_local! {
    /// The thresholds a [`ForcedThresholds`] guard on this thread has installed.
    static FORCED: Cell<Option<SparseThresholds>> = const { Cell::new(None) };
}

/// The thresholds a guard on this thread has installed, if any.
pub(super) fn forced_thresholds() -> Option<SparseThresholds> {
    FORCED.with(Cell::get)
}

/// Thresholds in force on this thread until the guard drops, which restores
/// whatever was in force before it.
pub(super) struct ForcedThresholds {
    prior: Option<SparseThresholds>,
}

impl ForcedThresholds {
    pub(super) fn install(thresholds: SparseThresholds) -> ForcedThresholds {
        ForcedThresholds { prior: FORCED.with(|c| c.replace(Some(thresholds))) }
    }
}

impl Drop for ForcedThresholds {
    fn drop(&mut self) {
        FORCED.with(|c| c.set(self.prior));
    }
}

mod direction;
mod flat_candidates;
mod inner_index;
mod regression;
mod reset_ws;
mod retention;
mod scatter_direction_pool;
mod self_conjunction;
