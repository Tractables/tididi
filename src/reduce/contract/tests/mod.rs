use super::*;

mod content_twin_forking;
mod fusion_twin;
mod inline_denorm;
mod listing;
mod merge_scratch;

thread_local! {
    /// The listed searches this thread's sweeps have run
    /// ([`note_listed_search`]).
    static LISTED_SEARCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Count a listed search the sweep ran on this thread.
pub(super) fn note_listed_search() {
    LISTED_SEARCHES.with(|n| n.set(n.get() + 1));
}

/// The listed searches this thread's sweeps have run so far, for the tests
/// that check the sweep takes them.
pub(crate) fn listed_searches() -> usize {
    LISTED_SEARCHES.with(std::cell::Cell::get)
}

thread_local! {
    /// The listed pair fusions this thread's sweeps have run
    /// ([`note_listed_fusion`]).
    static LISTED_FUSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Count a pair fusion the sweep ran on a parent's listed nodes only.
pub(super) fn note_listed_fusion() {
    LISTED_FUSIONS.with(|n| n.set(n.get() + 1));
}

/// The listed pair fusions this thread's sweeps have run so far.
pub(crate) fn listed_fusions() -> usize {
    LISTED_FUSIONS.with(std::cell::Cell::get)
}
