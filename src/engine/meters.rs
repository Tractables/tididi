//! What a caller reads back: where a watched conjunction stands, and a snapshot
//! of the armed limits and their meters.

use super::stop::{Scheduled, Stop};

/// Where the conjunction in flight stands, published while [`LimitSet::watch`]
/// is armed: when it began, the vtree level it is on, and how many levels it
/// walks in all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergePosition {
    /// When the conjunction began.
    pub began: std::time::Instant,
    /// The vtree level it has reached (0-based, bottom-up).
    pub level: u32,
    /// How many levels it walks in all.
    pub levels: u32,
}

/// A snapshot of the armed limits and the meters they are checked against.
/// Taken by [`Limits::meters`]; a plain `Copy` of every cell, read outside the
/// hot path.
#[derive(Clone, Copy, Debug)]
pub struct ApplyMeters {
    /// The armed soft byte budget; `None` when none is armed.
    pub budget_remaining: Option<u64>,
    /// Bytes the tracked reserves have charged since [`Limits::reset_meters`]
    /// (or operation entry, which zeroes it too).
    pub in_flight_bytes: u64,
    /// Output pairs built by the conjunction in flight (capacity for the level
    /// being built, exact for finished levels).
    pub pairs_in_flight: u64,
    /// The work clock: units the operations have polled through. Monotone and
    /// never reset, so an interval is a subtraction of two reads.
    pub work_units: u64,
    /// Bytes asked for by the most recent reserve the allocator refused, or
    /// `None` if none was refused since [`Limits::reset_meters`]. This is what
    /// tells "the allocator said no" from "the soft budget said no": both
    /// surface as [`ApplyError::OverBudget`].
    pub refused_reserve_bytes: Option<u64>,
    /// The armed stop axis.
    pub stop: Stop,
    /// The armed decision callback.
    pub schedule: Option<fn(&ApplyMeters, std::time::Instant) -> Scheduled>,
    /// The armed cap on output nodes per conjunction.
    pub output_node_cap: Option<u64>,
    /// Where a watched conjunction stands; `None` outside one.
    pub merge: Option<MergePosition>,
}
