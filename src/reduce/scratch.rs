//! The buffers the reduction passes reuse between calls, owned by the engine.
//!
//! Each pass checks its buffers out of a [`Cell`] and puts them back on the way
//! out (see `utils::pool_take`); a pass that bails early simply drops them, and
//! the next call finds the pool empty and starts fresh.

use std::cell::Cell;

use super::contract::content_twin::C2Scratch;
use super::contract::scratch::ContractScratch;
use crate::marg_slots::RefSlotScratch;

/// Every buffer one engine's reductions reuse between calls.
#[derive(Default)]
pub(crate) struct ReduceScratch {
    /// `prune_unreachable`'s flat reachability/remap array.
    pub(crate) prune_remap: Cell<Vec<u32>>,
    /// `prune_unreachable`'s per-level offsets into `prune_remap`.
    pub(crate) prune_level_base: Cell<Vec<usize>>,
    /// `prune_marg_slots`'s per-store slot bookkeeping.
    pub(crate) slot_prune_slots: Cell<Option<RefSlotScratch>>,
    /// `prune_marg_slots`'s slot remap array.
    pub(crate) slot_prune_remap: Cell<Vec<u32>>,
    /// Twin contraction's working set.
    pub(crate) contract: Cell<Option<ContractScratch>>,
    /// Content-twin canonicalization's working set.
    pub(crate) content_twin: Cell<Option<C2Scratch>>,
    /// Test-only allocation-failure injection: consults left before one fires.
    #[cfg(test)]
    pub(crate) fail_countdown: Cell<Option<u32>>,
}

impl ReduceScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.prune_remap.take();
        self.prune_level_base.take();
        self.slot_prune_slots.take();
        self.slot_prune_remap.take();
        self.contract.take();
        self.content_twin.take();
    }
}
