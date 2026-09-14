//! The buffers the reduction passes reuse between calls, owned by the engine.
//!
//! Each pass checks its buffers out of a [`Pool`]; guarded working sets return
//! their bounded capacity automatically on ordinary exits.

use crate::limits::pool::Pool;

use super::contract::content_twin::ContentTwinScratch;
use super::contract::scratch::ContractScratch;
use crate::value::slots::RefSlotScratch;

/// Every buffer one engine's reductions reuse between calls.
#[derive(Default)]
pub(crate) struct ReduceScratch {
    /// `prune_unreachable`'s flat reachability/remap array.
    pub(crate) prune_remap: Pool<Vec<u32>>,
    /// `prune_unreachable`'s per-level offsets into `prune_remap`.
    pub(crate) prune_level_base: Pool<Vec<usize>>,
    /// `prune_value_slots`'s per-store slot bookkeeping.
    pub(crate) slot_prune_slots: Pool<RefSlotScratch>,
    /// `prune_value_slots`'s slot remap array.
    pub(crate) slot_prune_remap: Pool<Vec<u32>>,
    /// Twin contraction's working set.
    pub(crate) contract: Pool<ContractScratch>,
    /// Content-twin canonicalization's working set.
    pub(crate) content_twin: Pool<ContentTwinScratch>,
}

impl ReduceScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.prune_remap.drain();
        self.prune_level_base.drain();
        self.slot_prune_slots.drain();
        self.slot_prune_remap.drain();
        self.contract.drain();
        self.content_twin.drain();
    }
}
