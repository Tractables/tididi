//! The buffers the conjunction reuses between calls.

use super::*;



/// Every buffer one engine's conjunctions reuse between calls.
///
/// The pattern throughout is take-and-return: `Cell::take` moves the buffer out
/// (leaving an empty one behind, so nothing is aliased while it is in use) and
/// the caller puts it back when it is done. Capacity is what is being reused —
/// contents are always cleared on take — so a warmed engine allocates nothing
/// on the paths that dominate a compile, and dropping the engine frees the lot.
#[derive(Default)]
pub struct ApplyScratch {
    /// Maps product grid position (i * k2 + j) → compacted local index in output level.
    pub(crate) node_idx: Cell<Vec<u32>>,
    /// Per-level grid descriptor (kind + base offset into `node_idx`).
    /// Folds the former level-base + NO_GRID sentinel + monotone-flag
    /// triple into a single enum — see `apply_grid::LevelGrid`.
    pub(crate) grids: Cell<Vec<LevelGrid>>,
    /// Tracks which c2 subtrees are identity (constant-true), reused across calls.
    pub(crate) c2_identity: Cell<Vec<bool>>,
    /// Tracks which c1 subtrees are identity (constant-true), reused across calls.
    pub(crate) c1_identity: Cell<Vec<bool>>,
    /// Per-node subvar counts for `init_leaf_identity`'s marginal-constant-true
    /// test. Only filled when the operand has at least one marginal level.
    pub(crate) subvars: Cell<Vec<u32>>,
    /// DFS stack used by `init_leaf_identity` when a marginal subtree is
    /// non-constant-true and its leaf descendants must be marked non-identity.
    pub(crate) marginal_stack: Cell<Vec<crate::vtree::VtreeIdx>>,
    /// Per-level product lists: alive (c1_idx, c2_idx, prod_idx) entries.
    /// Used by the sparse pipeline and for online density checks.
    pub(crate) product_lists: Cell<Vec<Vec<ProductEntry>>>,
    /// Per-level live counts for online density checking.
    pub(crate) live_counts: Cell<Vec<usize>>,
    /// Per-level flag: true once the product list has been built.
    pub(crate) has_pl: Cell<Vec<bool>>,
    /// Per-level widths of c1, pre-cached before identity swaps steal levels.
    pub(crate) c1_widths: Cell<Vec<usize>>,
    /// Per-level widths of c2, pre-cached before identity swaps steal levels.
    pub(crate) c2_widths: Cell<Vec<usize>>,
    /// Decode buffers for one operand cell's marg-decoded pair list
    /// (`TddLevel::pairs_view_decoded`, which clears them before each fill), one
    /// per operand. Pooled here so they warm up once per engine and the decode
    /// pushes are realloc-free from then on.
    pub(crate) inputs1: Cell<Vec<InputPair>>,
    pub(crate) inputs2: Cell<Vec<InputPair>>,
    /// One streaming-collapse cell's surviving `(lc, rc)` refs
    /// (`cell::StreamCollapse::cell_pairs`, cleared before every cell). Pooled
    /// for the same reason as `inputs1` / `inputs2`.
    pub(crate) cell_pairs: Cell<Vec<InputPair>>,
    /// The per-level c2 column table (`cell::C2Columns`): one resolved pair
    /// slice per c2 node, built once before the row sweep so the cell prologue
    /// indexes it instead of re-deriving column `j` on every row. Pure scratch
    /// — it holds descriptors, never pairs — so it is pooled rather than
    /// budget-charged, under the same retain cap as the buffers above.
    pub(crate) c2_cols: Cell<Vec<ColSlice>>,
    /// The four NxM dead-pair pre-filter masks, as one bundle — see
    /// `liveness::NxmMaskScratch`. Were four fresh `Vec<u128>` per apply.
    pub(crate) nxm_masks: Cell<liveness::NxmMaskScratch>,
    /// Per-vtree-node cache of computed counts for the streaming-marginal path
    /// (fast column + lazy BigUint side table, one `CountVec` per level).
    /// Populated lazily by `ensure_level_counts` when a target's child is still
    /// explicit. Cleared at the top of each scheduled `apply_and_fallible` call
    /// (when `marginalize_targets.is_some()`).
    pub(crate) stream_counts: Cell<Vec<Option<CountVec<ApplyBudget>>>>,
    /// Weighted mirror of `stream_counts`: per-vtree-node cache of computed
    /// weights for the streaming-marginal path (one `Vec<WeightVal>` per
    /// level). A concrete second pool because a field cannot be generic over
    /// the fold's column type; kept symmetric with the integer pool by
    /// construction — IDENTICAL take/clear/return semantics
    /// (pooled take, resize-to-`num_nodes`, clear `[..num_nodes]` to `None`,
    /// unbounded `pool_put` on finalize when `marginalize_targets.is_some()`).
    pub(crate) stream_weights: Cell<Vec<Option<Vec<crate::diagram::WeightVal>>>>,
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
            node_idx: Cell::new(Vec::new()),
            grids: Cell::new(Vec::new()),
            c2_identity: Cell::new(Vec::new()),
            c1_identity: Cell::new(Vec::new()),
            subvars: Cell::new(Vec::new()),
            marginal_stack: Cell::new(Vec::new()),
            product_lists: Cell::new(Vec::new()),
            live_counts: Cell::new(Vec::new()),
            has_pl: Cell::new(Vec::new()),
            c1_widths: Cell::new(Vec::new()),
            c2_widths: Cell::new(Vec::new()),
            inputs1: Cell::new(Vec::new()),
            inputs2: Cell::new(Vec::new()),
            cell_pairs: Cell::new(Vec::new()),
            c2_cols: Cell::new(Vec::new()),
            nxm_masks: Cell::new(liveness::NxmMaskScratch::new()),
            stream_counts: Cell::new(Vec::new()),
            stream_weights: Cell::new(Vec::new()),
            marg_entry_c1: std::cell::RefCell::new(Vec::new()),
            marg_entry_c2: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Release every retained buffer, leaving the pools empty.
    ///
    /// `take` swaps in an empty `Vec`, so the retained capacity drops here.
    pub(crate) fn drain(&self) {
        self.node_idx.take();
        self.grids.take();
        self.c2_identity.take();
        self.c1_identity.take();
        self.subvars.take();
        self.marginal_stack.take();
        self.product_lists.take();
        self.live_counts.take();
        self.has_pl.take();
        self.c1_widths.take();
        self.c2_widths.take();
        self.inputs1.take();
        self.inputs2.take();
        self.cell_pairs.take();
        self.c2_cols.take();
        self.nxm_masks.take();
        self.stream_counts.take();
        self.stream_weights.take();
        self.marg_entry_c1.borrow_mut().clear();
        self.marg_entry_c2.borrow_mut().clear();
    }
}
