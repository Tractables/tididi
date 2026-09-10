//! What a caller reads back: where a watched conjunction stands, and a snapshot
//! of the armed limits and their meters.

/// Where the conjunction in flight stands, published while [`LimitSet::watch`](crate::engine::LimitSet::watch)
/// is armed: when it began, the vtree level it is on, and how many levels it
/// walks in all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeProgress {
    /// When the conjunction began.
    pub started_at: std::time::Instant,
    /// The vtree level it has reached (0-based, bottom-up).
    pub level: u32,
    /// How many levels it walks in all.
    pub levels: u32,
}

/// A snapshot of the meters the armed limits are checked against. Taken by
/// [`Limits::meters`](crate::engine::Limits::meters); a plain `Copy` of every cell, read outside the hot
/// path. What is armed is a separate read, [`Limits::armed`](crate::engine::Limits::armed).
#[derive(Clone, Copy, Debug)]
pub struct ApplyMeters {
    /// Bytes the tracked reserves have charged since [`Limits::reset_meters`](crate::engine::Limits::reset_meters)
    /// (or operation entry, which zeroes it too).
    pub in_flight_bytes: u64,
    /// Output pairs built by the conjunction in flight (capacity for the level
    /// being built, exact for finished levels).
    pub pairs_in_flight: u64,
    /// The work clock: units the operations have polled through. Monotone and
    /// never reset, so an interval is a subtraction of two reads.
    pub work_units: u64,
    /// Bytes asked for by the most recent reserve the allocator refused, or
    /// `None` if none was refused since [`Limits::reset_meters`](crate::engine::Limits::reset_meters). This is what
    /// tells "the allocator said no" from "the soft budget said no": both
    /// surface as [`ApplyError::OverBudget`](crate::ApplyError::OverBudget).
    pub refused_reserve_bytes: Option<u64>,
    /// Where a watched conjunction stands; `None` outside one.
    pub merge: Option<MergeProgress>,
}
