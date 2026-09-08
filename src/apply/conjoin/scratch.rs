//! Thread-local scratch pools the conjunction reuses across calls.

use super::*;



// Thread-local scratch buffers reused across apply_and calls.
// Pattern: Cell::take() moves the Vec out, caller uses it, Cell::set() puts it back.
// Grow-only: capacity retained across calls avoids re-allocation. See types.rs for details.
thread_local! {
    /// Maps product grid position (i * k2 + j) → compacted local index in output level.
    pub(super) static SCRATCH_NODE_IDX: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// Per-level grid descriptor (kind + base offset into SCRATCH_NODE_IDX).
    /// Folds the former `SCRATCH_LEVEL_BASE + NO_GRID sentinel + SCRATCH_MONOTONE`
    /// triple into a single enum — see `apply_grid::LevelGrid`.
    pub(super) static SCRATCH_GRIDS: Cell<Vec<LevelGrid>> = const { Cell::new(Vec::new()) };
    /// Tracks which c2 subtrees are identity (constant-true), reused across calls.
    pub(super) static SCRATCH_C2_IDENTITY: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Tracks which c1 subtrees are identity (constant-true), reused across calls.
    pub(super) static SCRATCH_C1_IDENTITY: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Per-node subvar counts for `init_leaf_identity`'s marginal-constant-true
    /// test. Only filled when the operand has at least one marginal level.
    pub(super) static SCRATCH_SUBVARS: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// DFS stack used by `init_leaf_identity` when a marginal subtree is
    /// non-constant-true and its leaf descendants must be marked non-identity.
    pub(super) static SCRATCH_MARGINAL_STACK: Cell<Vec<crate::vtree::VtreeIdx>> = const { Cell::new(Vec::new()) };
    /// Per-level product lists: alive (c1_idx, c2_idx, prod_idx) entries.
    /// Used by the sparse pipeline and for online density checks.
    pub(super) static SCRATCH_PRODUCT_LISTS: Cell<Vec<Vec<ProductEntry>>> = const { Cell::new(Vec::new()) };
    /// Per-level live counts for online density checking.
    pub(super) static SCRATCH_LIVE_COUNTS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Per-level flag: true once the product list has been built.
    pub(super) static SCRATCH_HAS_PL: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Per-level widths of c1, pre-cached before identity swaps steal levels.
    pub(super) static SCRATCH_C1_WIDTHS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Per-level widths of c2, pre-cached before identity swaps steal levels.
    pub(super) static SCRATCH_C2_WIDTHS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Decode buffers for one operand cell's marg-decoded pair list
    /// (`TddLevel::pairs_view_decoded`, which clears them before each fill), one
    /// per operand. Pooled here so they warm up once per thread and the decode
    /// pushes are realloc-free from then on.
    pub(super) static SCRATCH_INPUTS1: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };
    pub(super) static SCRATCH_INPUTS2: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };

    /// One streaming-collapse cell's surviving `(lc, rc)` refs
    /// (`cell::StreamCollapse::cell_pairs`, cleared before every cell). Pooled
    /// for the same reason as `SCRATCH_INPUTS1/2`.
    pub(super) static SCRATCH_CELL_PAIRS: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };

    /// The per-level c2 column table (`cell::C2Columns`): one resolved pair
    /// slice per c2 node, built once before the row sweep so the cell prologue
    /// indexes it instead of re-deriving column `j` on every row. Pure scratch
    /// — it holds descriptors, never pairs — so it is pooled rather than
    /// budget-charged, under the same retain cap as the buffers above.
    pub(super) static SCRATCH_C2_COLS: Cell<Vec<ColSlice>> = const { Cell::new(Vec::new()) };

    /// The four NxM dead-pair pre-filter masks, as one bundle — see
    /// `liveness::NxmMaskScratch`. Were four fresh `Vec<u128>` per apply.
    pub(super) static SCRATCH_NXM_MASKS: Cell<liveness::NxmMaskScratch> =
        const { Cell::new(liveness::NxmMaskScratch::new()) };

    /// Per-vtree-node cache of computed counts for the streaming-marginal path
    /// (fast column + lazy BigUint side table, one `CountVec` per level).
    /// Populated lazily by `ensure_level_counts` when a target's child is still
    /// explicit. Cleared at the top of each scheduled `apply_and_fallible` call
    /// (when `marginalize_targets.is_some()`).
    pub(super) static SCRATCH_STREAM_COUNTS: Cell<Vec<Option<CountVec<ApplyBudget>>>> =
        const { Cell::new(Vec::new()) };

    /// Weighted mirror of `SCRATCH_STREAM_COUNTS`: per-vtree-node
    /// cache of computed weights for the streaming-marginal path (one
    /// `Vec<WeightVal>` per level). A concrete second pool because `thread_local!`
    /// can't be generic over the fold's column type; kept symmetric with the
    /// integer pool by construction — IDENTICAL take/clear/return semantics
    /// (pooled take, resize-to-`num_nodes`, clear `[..num_nodes]` to `None`,
    /// unbounded `pool_put` on finalize when `marginalize_targets.is_some()`).
    pub(super) static SCRATCH_STREAM_WEIGHTS: Cell<Vec<Option<Vec<crate::query::WeightVal>>>> =
        const { Cell::new(Vec::new()) };
}

thread_local! {
    /// Per-level is_marginal of c1/c2 snapshotted at apply entry, before the
    /// bottom-up sweep mutates operands (an identity-swap steals levels →
    /// is_marginal flips true→false). NOT debug-only: the pass-through carrier
    /// selector reads these to recover a child that was marginal at entry but
    /// got stolen into the output store mid-sweep (see the `ent_c1`/`ent_c2`
    /// disjuncts in the scatter loop).
    pub(super) static MARG_ENTRY_C1: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static MARG_ENTRY_C2: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Release every thread-local scratch allocation the apply pipeline retains on
/// THIS thread, giving the next compile a clean allocator slate.
///
/// Called by the recovery driver between a failed compile and its Shannon
/// recovery children (never on a hot path). The panic that unwinds an OOM'd
/// sub-compile drops the `pool_take`-borrowed scratch, but two classes of state
/// survive at full capacity and must be freed explicitly here:
///   - `SPARSE_WS` — a `RefCell`-owned workspace whose bucket arrays measured
///     ~1.8 GiB live at a recovery split (see `reset_sparse_ws`);
///   - the `LEVELS_POOL`/`LEVELS_POOL2` recycle slots (`drop_pools`).
///
/// The `SCRATCH_*` Cell pools below unwind empty on the panic path, but a clean
/// give-up (`Ok(None)`, the WMC memory-pressure signal) returns them full — so
/// they are emptied too, so recovery children never inherit a returned buffer.
/// This is the ONE place the inter-compile scratch reset lives; adding a new
/// apply scratch pool means adding it to this drain.
pub fn reset_apply_scratch() {
    reset_sparse_ws();
    diagram::drop_pools();
    // Drain the Cell-pattern scratch pools (take() leaves Default::default();
    // the taken buffer drops immediately, releasing its capacity).
    pool_take(&SCRATCH_NODE_IDX);
    pool_take(&SCRATCH_GRIDS);
    pool_take(&SCRATCH_C2_IDENTITY);
    pool_take(&SCRATCH_C1_IDENTITY);
    pool_take(&SCRATCH_SUBVARS);
    pool_take(&SCRATCH_MARGINAL_STACK);
    pool_take(&SCRATCH_PRODUCT_LISTS);
    pool_take(&SCRATCH_LIVE_COUNTS);
    pool_take(&SCRATCH_HAS_PL);
    pool_take(&SCRATCH_C1_WIDTHS);
    pool_take(&SCRATCH_C2_WIDTHS);
    pool_take(&SCRATCH_INPUTS1);
    pool_take(&SCRATCH_INPUTS2);
    pool_take(&SCRATCH_CELL_PAIRS);
    pool_take(&SCRATCH_C2_COLS);
    pool_take(&SCRATCH_NXM_MASKS);
    pool_take(&SCRATCH_STREAM_COUNTS);
    pool_take(&SCRATCH_STREAM_WEIGHTS);
}
