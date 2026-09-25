//! Workspaces retained between conjunctions.

use super::*;
use crate::limits::Limits;
use crate::execution::pool::{Buffers, Drain, Pool, Pools, PooledScratch, Scratch};

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

impl Pools for ApplyScratch {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        visit(&self.workspace);
        visit(&self.f_identity);
        visit(&self.g_identity);
        visit(&self.subvars);
        visit(&self.marginal_stack);
        visit(&self.cell_pairs);
        visit(&self.right_cols);
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

impl Buffers for ApplyWorkspace {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.f_widths);
        visit(&mut self.g_widths);
        visit(&mut self.f_identity);
        visit(&mut self.g_identity);
        visit(&mut self.f_pairs_scratch);
        visit(&mut self.g_pairs_scratch);
        self.products.buffers(visit);
        self.stream_cache.buffers(visit);
        self.prefilter_masks.buffers(visit);
    }
}

impl PooledScratch for ApplyWorkspace {
    fn prepare(&mut self) {
        self.f_pairs_scratch.clear();
        self.g_pairs_scratch.clear();
    }

    fn retain(&mut self, lim: &Limits) {
        self.stream_cache.discard_columns();
        self.release_oversized(lim);
    }
}
