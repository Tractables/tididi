//! Workspaces retained between conjunctions.

use super::*;
use crate::limits::{Limits, pool::{Buffers, Pool, PooledScratch, Scratch}};

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
