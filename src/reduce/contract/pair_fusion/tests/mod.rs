use super::*;

mod fallible;
mod support;
pub(crate) use support::fuse_pairs;
mod weighted;
mod window;
mod bitmap;
mod record;
mod chunks;

thread_local! {
    /// A chunk size a test pins on its thread ([`pin_chunk_plans`]).
    static CHUNK_PIN: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// The boundaries applied in more than one chunk ([`chunked_boundaries`]).
    static CHUNKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Pin the plans a fusion chunk collects on this thread, `None` for the
/// default; returns the pin before.
pub(crate) fn pin_chunk_plans(plans: Option<usize>) -> Option<usize> {
    CHUNK_PIN.with(|pin| pin.replace(plans))
}

/// The chunk size pinned on this thread, if any.
pub(super) fn pinned_chunk_plans() -> Option<usize> {
    CHUNK_PIN.with(std::cell::Cell::get)
}

/// Count a boundary this thread's fusion applied in more than one chunk.
pub(super) fn note_chunked_boundary() {
    CHUNKED.with(|n| n.set(n.get() + 1));
}

/// The boundaries this thread's fusions applied in more than one chunk.
pub(crate) fn chunked_boundaries() -> usize {
    CHUNKED.with(std::cell::Cell::get)
}

/// The widths at which pair fusion changes how it works on one node: the
/// fewest pairs whose grouping sorts rather than hashes them, and the fewest
/// plans at which the rewrite tests membership on a bitmap. For a caller's
/// test that means to reach both.
pub(crate) fn fusion_widths() -> (usize, usize) {
    (plan::SORT_MIN, rewrite::BITMAP_MIN_PLANS)
}
