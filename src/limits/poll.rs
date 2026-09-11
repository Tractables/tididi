//! The amortization stride the post-conjunction walks poll at.

/// Amortization stride for the post-conjunction walks' [`PollGate`](super::PollGate) — one poll
/// per ~16384 units, where a unit is one node of the level the walk is standing
/// on (a contracted parent's level, a forget batch's target level, a clustering
/// pivot's pair count).
///
/// Smaller than the conjunction's own strides because the unit is coarser: the
/// conjunction counts individual product pairs, these walks count whole levels'
/// node widths, and a worklist of narrow parents would otherwise run millions of
/// pops between two polls. At this cadence the poll is under a thousandth of the
/// work it amortizes over even when every popped parent is as small as it can
/// be.
const REDUCE_POLL_STRIDE: u64 = 1 << 14;

/// The post-conjunction walks' amortization stride, or a test's pinned value.
///
/// The hook exists so the amortization itself is testable — a diagram big enough
/// to accumulate 16384 units of contract work before the gate comes due is not a
/// unit test — while the production cadence stays where it is, which is the number
/// the overhead argument is made about.
#[inline]
pub(crate) fn reduce_poll_stride(pinned: Option<u64>) -> u64 {
    pinned.unwrap_or(REDUCE_POLL_STRIDE)
}
