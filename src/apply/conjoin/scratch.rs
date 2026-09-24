//! Workspaces retained between conjunctions.

use super::*;
use crate::limits::{Limits, pool::{Pool, PooledScratch, release_if_oversized}};

#[derive(Default)]
pub(crate) struct ApplyScratch {
    pub(crate) workspace: Pool<ApplyWorkspace>,
    pub(crate) f_identity: Pool<Vec<bool>>,
    pub(crate) g_identity: Pool<Vec<bool>>,
    pub(crate) subvars: Pool<Vec<u32>>,
    pub(crate) marginal_stack: Pool<Vec<crate::vtree::VtreeIdx>>,
    pub(crate) cell_pairs: Pool<Vec<ChildPair>>,
    pub(crate) right_cols: Pool<Vec<ColumnSlice>>,
}

impl ApplyScratch {
    pub(crate) fn drain(&self, lim: &crate::limits::Limits) {
        self.workspace.drain(lim);
        self.f_identity.drain(lim);
        self.g_identity.drain(lim);
        self.subvars.drain(lim);
        self.marginal_stack.drain(lim);
        self.cell_pairs.drain(lim);
        self.right_cols.drain(lim);
    }
}

/// Own the buffers for an entire conjunction; the level sweep borrows them.
#[derive(Default)]
pub(crate) struct ApplyWorkspace {
    pub(super) f_widths: Vec<usize>,
    pub(super) g_widths: Vec<usize>,
    pub(super) f_identity: Vec<bool>,
    pub(super) g_identity: Vec<bool>,
    pub(super) f_pairs_scratch: Vec<ChildPair>,
    pub(super) g_pairs_scratch: Vec<ChildPair>,
    pub(super) products: products::Products,
    pub(super) stream_cache: StreamCache,
    pub(super) prefilter_masks: liveness::PrefilterMaskScratch,
}

impl PooledScratch for ApplyWorkspace {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        [
            capacity_bytes(&self.f_widths),
            capacity_bytes(&self.g_widths),
            capacity_bytes(&self.f_identity),
            capacity_bytes(&self.g_identity),
            capacity_bytes(&self.f_pairs_scratch),
            capacity_bytes(&self.g_pairs_scratch),
            self.products.retained_bytes(),
            self.stream_cache.retained_bytes(),
            self.prefilter_masks.retained_bytes(),
        ].into_iter().sum()
    }

    fn prepare(&mut self) {
        self.f_pairs_scratch.clear();
        self.g_pairs_scratch.clear();
    }

    fn retain(&mut self, lim: &Limits) {
        release_if_oversized(lim, &mut self.f_pairs_scratch);
        release_if_oversized(lim, &mut self.g_pairs_scratch);
        self.products.retain(lim);
        self.prefilter_masks.release_oversized(lim);
        self.stream_cache.retain(lim);
    }
}
