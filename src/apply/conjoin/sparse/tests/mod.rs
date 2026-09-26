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

/// The streamed root count's choice a [`ForcedStream`] guard on this thread
/// has installed, and how many times the count has asked for it.
#[derive(Clone, Copy, Default)]
struct StreamPin {
    choice: Option<Option<super::stream::Operand>>,
    asked: u32,
}

thread_local! {
    static STREAM: Cell<StreamPin> = const { Cell::new(StreamPin { choice: None, asked: 0 }) };
}

/// The pinned choice, if any: `Some(None)` builds `c`, `Some(Some(pivot))`
/// streams with `pivot`. Every call is counted, pinned or not.
pub(super) fn forced_stream() -> Option<Option<super::stream::Operand>> {
    STREAM.with(|c| {
        let pin = c.get();
        c.set(StreamPin { asked: pin.asked + 1, ..pin });
        pin.choice
    })
}

/// A streamed root count's choice pinned on this thread until the guard
/// drops; `None` leaves it to the pricing.
pub(super) struct ForcedStream {
    prior: Option<Option<super::stream::Operand>>,
}

impl ForcedStream {
    pub(super) fn install(choice: Option<Option<super::stream::Operand>>) -> ForcedStream {
        STREAM.with(|c| {
            let pin = c.get();
            c.set(StreamPin { choice, ..pin });
            ForcedStream { prior: pin.choice }
        })
    }

    /// How many streamed root counts have reached their choice on this
    /// thread so far.
    pub(super) fn asked() -> u32 {
        STREAM.with(|c| c.get().asked)
    }
}

impl Drop for ForcedStream {
    fn drop(&mut self) {
        STREAM.with(|c| c.set(StreamPin { choice: self.prior, ..c.get() }));
    }
}

mod count_root;
mod counting_sort;
mod direction;
mod emit;
mod flat_candidates;
mod root_gate;
mod inner_index;
mod passthrough;
mod regression;
mod reset_ws;
mod scatter_direction_pool;
mod stream;
mod product_filter;
