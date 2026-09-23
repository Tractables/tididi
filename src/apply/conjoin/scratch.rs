//! Workspaces retained between conjunctions.

use super::*;
use crate::limits::{Limits, pool::{Pool, PooledScratch, release_if_oversized}};

#[derive(Default)]
pub(crate) struct ApplyScratch {
    pub(crate) workspace: Pool<ApplyWorkspace>,
    pub(crate) left_identity: Pool<Vec<bool>>,
    pub(crate) right_identity: Pool<Vec<bool>>,
    pub(crate) subvars: Pool<Vec<u32>>,
    pub(crate) marginal_stack: Pool<Vec<crate::vtree::VtreeIdx>>,
    pub(crate) cell_pairs: Pool<Vec<ChildPair>>,
    pub(crate) right_cols: Pool<Vec<ColumnSlice>>,
}

impl ApplyScratch {
    pub(crate) fn drain(&self) {
        self.workspace.drain();
        self.left_identity.drain();
        self.right_identity.drain();
        self.subvars.drain();
        self.marginal_stack.drain();
        self.cell_pairs.drain();
        self.right_cols.drain();
    }
}

/// Own the buffers for an entire conjunction; the level sweep borrows them.
#[derive(Default)]
pub(crate) struct ApplyWorkspace {
    pub(super) left_widths: Vec<usize>,
    pub(super) right_widths: Vec<usize>,
    pub(super) left_identity: Vec<bool>,
    pub(super) right_identity: Vec<bool>,
    pub(super) inputs1_scratch: Vec<ChildPair>,
    pub(super) inputs2_scratch: Vec<ChildPair>,
    pub(super) products: products::Products,
    pub(super) stream_cache: StreamCache,
    pub(super) prefilter_masks: liveness::PrefilterMaskScratch,
}

impl PooledScratch for ApplyWorkspace {
    fn prepare(&mut self) {
        self.inputs1_scratch.clear();
        self.inputs2_scratch.clear();
    }

    fn retain(&mut self, lim: &Limits) {
        release_if_oversized(lim, &mut self.inputs1_scratch);
        release_if_oversized(lim, &mut self.inputs2_scratch);
        self.products.retain(lim);
        self.prefilter_masks.release_oversized(lim);
        self.stream_cache.retain(lim);
    }
}
