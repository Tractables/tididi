//! The snapshot of what is armed on this thread and what the meters read.

use super::stop::{RopeLimit, Scheduled};
use super::{APPLY_LIMITS, LAST_REFUSED_RESERVE_BYTES};

/// Where the apply in flight stands, published while a scope with
/// [`ApplyLimitsInstall::watch`] is armed: when it began, the vtree level it is
/// on, and how many levels it has in all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergePosition {
    /// When the apply began.
    pub began: std::time::Instant,
    /// The vtree level the apply has reached (0-based, bottom-up).
    pub level: u32,
    /// How many levels the apply walks in all.
    pub levels: u32,
}

/// A snapshot of the limits armed on this thread and the meters they are
/// checked against. Taken by [`apply_meters`]; a plain `Copy` of every cell,
/// read outside the hot path.
#[derive(Clone, Copy, Debug)]
pub struct ApplyMeters {
    /// The soft byte budget armed by [`set_apply_budget`] or
    /// [`ApplyLimitsInstall::budget`]; `None` when no budget is armed.
    pub budget_remaining: Option<u64>,
    /// Bytes the tracked reserves have charged since [`reset_apply_meters`]
    /// (or apply entry, which zeroes it too).
    pub in_flight_bytes: u64,
    /// Output pairs built by the apply in flight (capacity for the level being
    /// built, exact for finished levels).
    pub pairs_in_flight: u64,
    /// The work clock: units the applies on this thread have polled through.
    /// Monotone and never reset, so an interval is a subtraction of two reads.
    pub work_units: u64,
    /// Bytes asked for by the most recent fallible reserve the allocator
    /// refused, or `None` if none was refused since [`reset_apply_meters`].
    /// This is what tells "the allocator said no" from "the soft budget said
    /// no": both surface as [`ApplyError::OverBudget`].
    pub refused_reserve_bytes: Option<u64>,
    /// The armed wall-clock deadline.
    pub deadline: Option<std::time::Instant>,
    /// The armed decision callback.
    pub schedule: Option<fn(std::time::Instant) -> Scheduled>,
    /// The armed cap on output nodes per apply.
    pub output_node_cap: Option<u64>,
    /// The armed stall rope: `(floor_pairs, rope)`.
    pub stall_rope: Option<(u64, RopeLimit)>,
    /// Where a watched apply stands; `None` outside one.
    pub merge: Option<MergePosition>,
}

/// Snapshot the limits and meters of the current thread.
pub fn apply_meters() -> ApplyMeters {
    APPLY_LIMITS.with(|l| ApplyMeters {
        budget_remaining: l.budget_remaining.get(),
        in_flight_bytes: l.budget_in_flight.get(),
        pairs_in_flight: l.pairs_in_flight.get(),
        work_units: l.work_clock.get(),
        refused_reserve_bytes: LAST_REFUSED_RESERVE_BYTES.with(|c| c.get()),
        deadline: l.deadline.get(),
        schedule: l.schedule.get(),
        output_node_cap: l.output_node_cap.get(),
        stall_rope: l.stall_rope.get(),
        merge: l.merge.get(),
    })
}

/// Set or clear the per-thread soft budget consulted by the tracked reserves.
/// Pass `Some(remaining)` = total budget minus current live bytes, or `None`
/// to disable.
///
/// The raw setter for a caller that re-derives the budget as it goes (a
/// compile loop refreshing `budget − live` after every merge). The scope that
/// owns the budget's lifetime should still be an [`apply_limits`] install
/// (`.budget(None)`), so nothing written here can leak onto the thread once
/// that scope ends.
pub fn set_apply_budget(remaining_bytes: Option<u64>) {
    APPLY_LIMITS.with(|l| l.budget_remaining.set(remaining_bytes));
}

/// Zero the in-flight byte meter and forget any recorded allocator refusal.
///
/// Both are per-apply state that a traversal's applies clear at entry, but
/// tracked reserves also happen between applies, so a traversal that ended
/// inside a huge apply leaves a large total behind, and the next traversal on
/// the thread would charge its first between-apply reserve against that
/// stale total the moment it armed a budget. A traversal calls this at entry
/// so it only ever measures bytes it charged itself and only ever reports a
/// refusal of its own.
pub fn reset_apply_meters() {
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(0));
    LAST_REFUSED_RESERVE_BYTES.with(|c| c.set(None));
}

/// Charge the in-flight meter as an aborted apply would have, without
/// allocating the bytes: the test seam for the ownership rule on
/// [`reset_apply_meters`].
#[cfg(any(test, debug_assertions))]
pub fn charge_apply_in_flight_for_test(bytes: u64) {
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(l.budget_in_flight.get().saturating_add(bytes)));
}
