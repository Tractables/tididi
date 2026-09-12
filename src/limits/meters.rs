//! What a caller reads back — where a watched conjunction stands, and a
//! snapshot of the armed limits and their meters — and the scoped charge a
//! transient makes against the byte meter.

use super::Limits;

/// Where the conjunction in flight stands, published while [`LimitConfig::with_conjunction_progress`](crate::limits::LimitConfig::with_conjunction_progress)
/// is armed: when it began, the vtree level it is on, and how many levels it
/// walks in all.
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

/// A snapshot of the meters the armed limits are checked against. Taken by
/// [`Limits::meters`](crate::limits::Limits::meters); a plain `Copy` of every cell, read outside the hot
/// path. What is armed is a separate read, [`Limits::armed`](crate::limits::Limits::armed).
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
    /// `None` if none was refused since [`Limits::reset_meters`](crate::limits::Limits::reset_meters). This is what
    /// tells "the allocator said no" from "the soft budget said no": both
    /// surface as [`OperationError::OverBudget`](crate::OperationError::OverBudget).
    pub refused_reserve_bytes: Option<u64>,
    /// Where the watched conjunction in flight stands, or where the last one
    /// ended; `None` before the first watched one. Nothing clears it, so
    /// `started_at` is what tells one conjunction from the next.
    pub conjunction: Option<ConjunctionProgress>,
}

/// A charge against the in-flight byte meter that is released when the
/// transient it accounts for goes out of scope — including the level's early
/// exits, where a forgotten release would permanently consume headroom the
/// operation no longer uses.
pub(crate) struct ByteCharge<'a> {
    lim: &'a Limits,
    bytes: u64,
}

impl<'a> ByteCharge<'a> {
    /// Charge nothing yet. The transient may end up empty.
    pub(crate) fn none(lim: &'a Limits) -> Self {
        ByteCharge { lim, bytes: 0 }
    }

    /// Record that `bytes` of the charge already made are this transient's to
    /// release.
    pub(crate) fn owe(&mut self, bytes: u64) {
        self.bytes = bytes;
    }
}

impl Drop for ByteCharge<'_> {
    fn drop(&mut self) {
        self.lim.release_bytes(self.bytes);
    }
}
