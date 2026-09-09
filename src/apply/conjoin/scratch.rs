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
    /// Maps product grid position (i * k2 + j) → compacted local index in output level.
    pub(crate) node_idx: Pool<Vec<u32>>,
    /// Per-level grid descriptor (kind + base offset into `node_idx`).
    /// Folds the former level-base + NO_GRID sentinel + monotone-flag
    /// triple into a single enum — see `apply_grid::LevelGrid`.
    pub(crate) grids: Pool<Vec<LevelGrid>>,
    /// Tracks which c2 subtrees are identity (constant-true), reused across calls.
    pub(crate) c2_identity: Pool<Vec<bool>>,
    /// Tracks which c1 subtrees are identity (constant-true), reused across calls.
    pub(crate) c1_identity: Pool<Vec<bool>>,
    /// Per-node subvar counts for `init_leaf_identity`'s marginal-constant-true
    /// test. Only filled when the operand has at least one marginal level.
    pub(crate) subvars: Pool<Vec<u32>>,
    /// DFS stack used by `init_leaf_identity` when a marginal subtree is
    /// non-constant-true and its leaf descendants must be marked non-identity.
    pub(crate) marginal_stack: Pool<Vec<crate::vtree::VtreeIdx>>,
    /// Per-level product lists: alive (c1_idx, c2_idx, prod_idx) entries.
    /// Used by the sparse pipeline and for online density checks.
    pub(crate) product_lists: Pool<Vec<Vec<ProductEntry>>>,
    /// Per-level live counts for online density checking.
    pub(crate) live_counts: Pool<Vec<usize>>,
    /// Per-level flag: true once the product list has been built.
    pub(crate) has_pl: Pool<Vec<bool>>,
    /// Per-level widths of c1, pre-cached before identity swaps steal levels.
    pub(crate) c1_widths: Pool<Vec<usize>>,
    /// Per-level widths of c2, pre-cached before identity swaps steal levels.
    pub(crate) c2_widths: Pool<Vec<usize>>,
    /// Decode buffers for one operand cell's marg-decoded pair list
    /// (`TddLevel::pairs_view_decoded`, which clears them before each fill), one
    /// per operand. Pooled here so they warm up once per engine and the decode
    /// pushes are realloc-free from then on.
    pub(crate) inputs1: Pool<Vec<InputPair>>,
    pub(crate) inputs2: Pool<Vec<InputPair>>,
    /// One streaming-collapse cell's surviving `(lc, rc)` refs
    /// (`cell::StreamCollapse::cell_pairs`, cleared before every cell). Pooled
    /// for the same reason as `inputs1` / `inputs2`.
    pub(crate) cell_pairs: Pool<Vec<InputPair>>,
    /// The per-level c2 column table (`cell::C2Columns`): one resolved pair
    /// slice per c2 node, built once before the row sweep so the cell prologue
    /// indexes it instead of re-deriving column `j` on every row. Pure scratch
    /// — it holds descriptors, never pairs — so it is pooled rather than
    /// budget-charged, under the same retain cap as the buffers above.
    pub(crate) c2_cols: Pool<Vec<ColSlice>>,
    /// The four NxM dead-pair pre-filter masks, as one bundle — see
    /// `liveness::NxmMaskScratch`. Were four fresh `Vec<u128>` per apply.
    pub(crate) nxm_masks: Pool<liveness::NxmMaskScratch>,
    /// Per-vtree-node cache of computed counts for the streaming-marginal path
    /// (fast column + lazy BigUint side table, one `CountVec` per level).
    /// Populated lazily by `ensure_level_counts` when a target's child is still
    /// explicit. Cleared at the top of each scheduled `apply_and_fallible` call
    /// (when `marginalize_targets.is_some()`).
    pub(crate) stream_counts: Pool<Vec<Option<CountVec<ApplyBudget>>>>,
    /// Weighted mirror of `stream_counts`: per-vtree-node cache of computed
    /// weights for the streaming-marginal path (one `Vec<WeightVal>` per
    /// level). A concrete second pool because a field cannot be generic over
    /// the fold's column type; kept symmetric with the integer pool by
    /// construction — IDENTICAL take/clear/return semantics
    /// (pooled take, resize-to-`num_nodes`, clear `[..num_nodes]` to `None`,
    /// unbounded `pool_put` on finalize when `marginalize_targets.is_some()`).
    pub(crate) stream_weights: Pool<Vec<Option<Vec<crate::diagram::WeightVal>>>>,
    /// Per-level is_marginal of c1/c2 snapshotted at apply entry, before the
    /// bottom-up sweep mutates operands (an identity-swap steals levels →
    /// is_marginal flips true→false). NOT debug-only: the pass-through carrier
    /// selector reads these to recover a child that was marginal at entry but
    /// got stolen into the output store mid-sweep (see the `ent_c1`/`ent_c2`
    /// disjuncts in the scatter loop).
    pub(crate) marg_entry_c1: std::cell::RefCell<Vec<bool>>,
    pub(crate) marg_entry_c2: std::cell::RefCell<Vec<bool>>,
}

impl ApplyScratch {
    /// Cold buffers: every pool empty.
    pub(crate) fn new() -> ApplyScratch {
        ApplyScratch {
            node_idx: Pool::default(),
            grids: Pool::default(),
            c2_identity: Pool::default(),
            c1_identity: Pool::default(),
            subvars: Pool::default(),
            marginal_stack: Pool::default(),
            product_lists: Pool::default(),
            live_counts: Pool::default(),
            has_pl: Pool::default(),
            c1_widths: Pool::default(),
            c2_widths: Pool::default(),
            inputs1: Pool::default(),
            inputs2: Pool::default(),
            cell_pairs: Pool::default(),
            c2_cols: Pool::default(),
            nxm_masks: Pool::default(),
            stream_counts: Pool::default(),
            stream_weights: Pool::default(),
            marg_entry_c1: std::cell::RefCell::new(Vec::new()),
            marg_entry_c2: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Release every retained buffer, leaving the pools empty.
    ///
    /// `take` swaps in an empty `Vec`, so the retained capacity drops here.
    pub(crate) fn drain(&self) {
        self.node_idx.drain();
        self.grids.drain();
        self.c2_identity.drain();
        self.c1_identity.drain();
        self.subvars.drain();
        self.marginal_stack.drain();
        self.product_lists.drain();
        self.live_counts.drain();
        self.has_pl.drain();
        self.c1_widths.drain();
        self.c2_widths.drain();
        self.inputs1.drain();
        self.inputs2.drain();
        self.cell_pairs.drain();
        self.c2_cols.drain();
        self.nxm_masks.drain();
        self.stream_counts.drain();
        self.stream_weights.drain();
        self.marg_entry_c1.borrow_mut().clear();
        self.marg_entry_c2.borrow_mut().clear();
    }
}
