//! The buffers the conjunction reuses between calls.

use super::*;
use crate::engine::pool::Pool;



/// Every buffer one engine's conjunctions reuse between calls.
///
/// The pattern throughout is take-and-return: [`Pool::take`] moves the buffer out
/// (leaving an empty one behind, so nothing is aliased while it is in use) and
/// the caller puts it back when it is done. Capacity is what is being reused —
/// contents are always cleared on take — so a warmed engine allocates nothing
/// on the paths that dominate a compile, and dropping the engine frees the lot.
#[derive(Default)]
pub struct ApplyScratch {
    /// Maps product grid position (i * right_width + j) → compacted local index in output level.
    pub(crate) node_idx: Pool<Vec<u32>>,
    /// Per-level grid descriptor: allocated-or-not plus the base offset into
    /// `node_idx` (see [`LevelGrid`]).
    pub(crate) grids: Pool<Vec<LevelGrid>>,
    /// Tracks which g subtrees are identity (constant-true), reused across calls.
    pub(crate) right_identity: Pool<Vec<bool>>,
    /// Tracks which f subtrees are identity (constant-true), reused across calls.
    pub(crate) left_identity: Pool<Vec<bool>>,
    /// Per-node subvar counts for `init_leaf_identity`'s marginal-constant-true
    /// test. Only filled when the operand has at least one marginal level.
    pub(crate) subvars: Pool<Vec<u32>>,
    /// DFS stack used by `init_leaf_identity` when a marginal subtree is
    /// non-constant-true and its leaf descendants must be marked non-identity.
    pub(crate) marginal_stack: Pool<Vec<crate::vtree::VtreeIdx>>,
    /// Per-level product lists: alive (left_idx, right_idx, prod_idx) entries.
    /// Used by the sparse pipeline and for online density checks.
    pub(crate) product_lists: Pool<Vec<Vec<ProductEntry>>>,
    /// Per-level live counts for online density checking.
    pub(crate) live_counts: Pool<Vec<usize>>,
    /// Per-level flag: true once the product list has been built.
    pub(crate) has_pl: Pool<Vec<bool>>,
    /// Per-level widths of f, pre-cached before identity swaps steal levels.
    pub(crate) left_widths: Pool<Vec<usize>>,
    /// Per-level widths of g, pre-cached before identity swaps steal levels.
    pub(crate) right_widths: Pool<Vec<usize>>,
    /// Decode buffers for one operand cell's marginal-decoded pair list
    /// (`TddLevel::pairs_view_decoded`, which clears them before each fill), one
    /// per operand. Pooled here so they warm up once per engine and the decode
    /// pushes are realloc-free from then on.
    pub(crate) inputs1: Pool<Vec<InputPair>>,
    pub(crate) inputs2: Pool<Vec<InputPair>>,
    /// One streaming-collapse cell's surviving `(lc, rc)` refs
    /// (`cell::StreamCollapse::cell_pairs`, cleared before every cell). Pooled
    /// for the same reason as `inputs1` / `inputs2`.
    pub(crate) cell_pairs: Pool<Vec<InputPair>>,
    /// The per-level g column table (`cell::RightColumns`): one resolved pair
    /// slice per g node, built once before the row sweep so the cell prologue
    /// indexes it instead of re-deriving column `j` on every row. Pure scratch
    /// — it holds descriptors, never pairs — so it is pooled rather than
    /// budget-charged, under the same retain cap as the buffers above.
    pub(crate) right_cols: Pool<Vec<ColumnSlice>>,
    /// The four dead-pair pre-filter masks, as one bundle — see
    /// `liveness::PrefilterMaskScratch`. Were four fresh `Vec<u128>` per apply.
    pub(crate) prefilter_masks: Pool<liveness::PrefilterMaskScratch>,
    /// Per-vtree-node cache of the child columns the streaming-marginal path
    /// computes lazily, of whichever value kind the engine last ran. Populated
    /// by `ensure_level_counts` when a target's child is still explicit, and
    /// cleared at the top of each scheduled `apply_and_fallible` call.
    pub(crate) stream_cache: Pool<StreamCache>,
}

impl ApplyScratch {
    /// Release every retained buffer, leaving the pools empty.
    ///
    /// `take` swaps in an empty `Vec`, so the retained capacity drops here.
    pub(crate) fn drain(&self) {
        self.node_idx.drain();
        self.grids.drain();
        self.right_identity.drain();
        self.left_identity.drain();
        self.subvars.drain();
        self.marginal_stack.drain();
        self.product_lists.drain();
        self.live_counts.drain();
        self.has_pl.drain();
        self.left_widths.drain();
        self.right_widths.drain();
        self.inputs1.drain();
        self.inputs2.drain();
        self.cell_pairs.drain();
        self.right_cols.drain();
        self.prefilter_masks.drain();
        self.stream_cache.drain();
    }
}
