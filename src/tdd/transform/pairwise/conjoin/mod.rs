//! Conjunction (AND) of two TDDs via the compacting product construction.
//!
//! Given two TDDs over the same vtree, produces a TDD for their conjunction.
//! The product construction pairs every node from c1 with every node from c2
//! at each vtree level, computes the conjunction of their input pairs, and
//! omits dead nodes (compaction). See `docs/tdd.md` for details.

use std::cell::Cell;
use std::sync::Arc;

use crate::vtree::VtreeIdx;
use super::grid::LevelGrid;
use super::leaf::CONJOIN_GRID;
use crate::tdd::types::{self, *};
use crate::tdd::utils::{pool_put, pool_put_bounded, pool_take, release_if_oversized};

pub mod budget;
mod child_lookup; // Representation-specialized child lookups (sparse-conjunction kernels)
// Re-export budget items needed by external callers (pub / pub(crate)) and by
// sibling submodules (which use `super::name` — the `use budget::*` below
// brings everything pub(super)+ into this module's namespace).
pub use budget::ApplyError;
pub(crate) use budget::{
    budget_reserve_exact, reserve_pairs_for_emit, try_push, try_resize,
    DEAD, try_resize_dead2,
};
// Consumed by the downstream compiler crate (apply-deadline / OOM-budget install).
//
// The budget re-exports are split in two by API class, and the split is load
// bearing: `#[doc(hidden)]` on a definition governs that definition, so a
// non-hidden `pub use` beside it is an independent decision about the parent
// module's surface. Documented API goes through this block; hidden plumbing
// goes through the `#[doc(hidden)] pub use` blocks below, one per group, so
// that adding a name to the wrong block is visible in review rather than only
// in the rendered docs.
pub use budget::{
    set_apply_budget,
    apply_budget_remaining,
    apply_in_flight_bytes,
    clear_last_refused_reserve,
    last_refused_reserve_bytes,
    reset_apply_in_flight,
    apply_limits,
    MemPressure,
    // The read side of the give-up rule's conditional rope, for the tests that
    // pin what a scope armed (the rule itself lives in the downstream compiler).
    apply_stall_rope,
    RopeLimit,
    apply_output_node_cap,
};
// Apply-engine scratch and test-only meters (freeze class B2): callable across
// the crate boundary, off the documented surface.
#[doc(hidden)]
pub use budget::{
    charge_apply_in_flight_for_test,
    // Nameable so a downstream scope can HOLD an install across a decision it
    // makes mid-compile (the canopy root ladder's commitment), rather than only
    // in the `let _g = ...` shape where inference suffices.
    ApplyLimitsGuard,
    enable_apply_deadline_check,
    apply_pairs_in_flight,
    // The work the applies have polled through, for the same upstream rule: its
    // budget is a share of one currency or the other, and this is the other.
    compile_work_units,
};
// What an armed decision callback concludes, and the position the apply in
// flight publishes for the caller that armed one to read. The rule and its
// state stay upstream; what this crate lends is the poll to stand on.
pub use budget::Scheduled;
#[doc(hidden)]
pub use budget::{
    apply_schedule,
    watch_merges,
    merge_position,
};
// The reduce walk's half of the deadline machinery: the same amortized ticker
// this module's cell loops use, on its own arming cell. Re-exported at
// crate visibility so `tdd::minimize` polls through ONE implementation rather
// than growing a private copy of the counter.
pub(crate) use budget::{reduce_poll_stride, PollTicker};
#[cfg(test)]
pub(crate) use budget::with_reduce_poll_stride;
// Reachable across the crate boundary by the downstream compiler crate's tests
// (which arm/observe the apply-deadline check); not `#[cfg(test)]`-gated because
// dependency crates never compile with `cfg(test)`. All four definitions are
// `#[doc(hidden)]`, so the re-export is too.
#[doc(hidden)]
pub use budget::{
    apply_deadline_check_enabled, enable_reduce_deadline_check,
    reset_apply_deadline_check_for_test, reset_reduce_deadline_check_for_test,
};
use budget::*;

mod cell;
use cell::{
    C2Columns, CellCtx, ColSlice,
    run_level_rows_marg, run_level_rows_marg_sparse, run_level_rows_plain,
    run_level_rows_stream_count,
};
// Test-only gate override; re-exported so the gate-off streaming parity
// regression (query_tests.rs) can force `TIDIDI_BOTHMARG_NOCOLLAPSE` behavior
// without racing the env memoization.
#[cfg(test)]
pub(crate) use cell::with_bothmarg_collapse_forced;

mod sparse;
use sparse::{ProductEntry, is_self_conjunction, fill_identity_product_list, apply_sparse_level, apply_leaf_levels, compute_apply_output, release_sparse_ws_if_large, reset_sparse_ws};

// Identity/constant-true detection + per-level identity fast paths (extracted).
mod identity;
use identity::{init_leaf_identity, try_level_fast_paths, FastPathResult};
#[cfg(debug_assertions)]
use identity::debug_assert_marg_schedule;
// Consumed only by the `apply_tests` submodule's `use super::*` glob (marginal
// constant-true unit tests); production callers live inside `identity`.
#[cfg(test)]
use identity::level_marginal_is_constant_true;

// Apply setup (phases 1-3) → `ApplySetup` (extracted).
mod setup;
use setup::{ApplySetup, apply_and_setup};

// Grid-reclaim allocator + grid/product-list materialization helpers (extracted).
mod grid_alloc;
use grid_alloc::{materialize_dense_child, grid_alloc, grid_free, grid_free_child, ensure_product_list};

// Per-level marg classification plan + NxM dead-pair masks (extracted).
mod marg_plan;
use marg_plan::{MargPlan, plan_marg_level, build_nxm_masks};

// Spine-bounded ("restricted") apply: the O(spine) batch merge. Same apply
// core, restricted level set. Documented public API — the module is private, so
// this re-export IS the surface, and it is deliberately not `doc(hidden)`.
mod restrict;
pub use restrict::{try_apply_and_batch, BatchMerge, RebuiltMax};
use restrict::Restrict;


// Thread-local scratch buffers reused across apply_and calls.
// Pattern: Cell::take() moves the Vec out, caller uses it, Cell::set() puts it back.
// Grow-only: capacity retained across calls avoids re-allocation. See types.rs for details.
thread_local! {
    /// Maps product grid position (i * k2 + j) → compacted local index in output level.
    static SCRATCH_NODE_IDX: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// Per-level grid descriptor (kind + base offset into SCRATCH_NODE_IDX).
    /// Folds the former `SCRATCH_LEVEL_BASE + NO_GRID sentinel + SCRATCH_MONOTONE`
    /// triple into a single enum — see `apply_grid::LevelGrid`.
    static SCRATCH_GRIDS: Cell<Vec<LevelGrid>> = const { Cell::new(Vec::new()) };
    /// Tracks which c2 subtrees are identity (constant-true), reused across calls.
    static SCRATCH_C2_IDENTITY: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Tracks which c1 subtrees are identity (constant-true), reused across calls.
    static SCRATCH_C1_IDENTITY: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Per-node subvar counts for `init_leaf_identity`'s marginal-constant-true
    /// test. Only filled when the operand has at least one marginal level.
    static SCRATCH_SUBVARS: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// DFS stack used by `init_leaf_identity` when a marginal subtree is
    /// non-constant-true and its leaf descendants must be marked non-identity.
    static SCRATCH_MARGINAL_STACK: Cell<Vec<crate::vtree::VtreeIdx>> = const { Cell::new(Vec::new()) };
    /// Per-level product lists: alive (c1_idx, c2_idx, prod_idx) entries.
    /// Used by the sparse pipeline and for online density checks.
    static SCRATCH_PRODUCT_LISTS: Cell<Vec<Vec<ProductEntry>>> = const { Cell::new(Vec::new()) };
    /// Per-level live counts for online density checking.
    static SCRATCH_LIVE_COUNTS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Per-level flag: true once the product list has been built.
    static SCRATCH_HAS_PL: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Per-level widths of c1, pre-cached before identity swaps steal levels.
    static SCRATCH_C1_WIDTHS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Per-level widths of c2, pre-cached before identity swaps steal levels.
    static SCRATCH_C2_WIDTHS: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Decode buffers for one operand cell's marg-decoded pair list
    /// (`TddLevel::pairs_view_decoded`, which clears them before each fill), one
    /// per operand. These used to be declared inside the per-level loop, so every
    /// level of every apply re-grew them from empty one push at a time —
    /// callgrind attributed 1,234 `finish_grow` calls per canopy leaf (1.8% of
    /// the window) to that. Pooled here they warm up once per thread and the
    /// decode pushes are realloc-free from then on.
    static SCRATCH_INPUTS1: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };
    static SCRATCH_INPUTS2: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };

    /// One streaming-collapse cell's surviving `(lc, rc)` refs
    /// (`cell::StreamCollapse::cell_pairs`, cleared before every cell). Pooled
    /// for the same reason as `SCRATCH_INPUTS1/2`: it used to be built from
    /// empty at every streaming level.
    static SCRATCH_CELL_PAIRS: Cell<Vec<InputPair>> = const { Cell::new(Vec::new()) };

    /// The per-level c2 column table (`cell::C2Columns`): one resolved pair
    /// slice per c2 node, built once before the row sweep so the cell prologue
    /// indexes it instead of re-deriving column `j` on every row. Pure scratch
    /// — it holds descriptors, never pairs — so it is pooled rather than
    /// budget-charged, under the same retain cap as the buffers above.
    static SCRATCH_C2_COLS: Cell<Vec<ColSlice>> = const { Cell::new(Vec::new()) };

    /// The four NxM dead-pair pre-filter masks, as one bundle — see
    /// `liveness::NxmMaskScratch`. Were four fresh `Vec<u128>` per apply.
    static SCRATCH_NXM_MASKS: Cell<liveness::NxmMaskScratch> =
        const { Cell::new(liveness::NxmMaskScratch::new()) };

    /// Per-vtree-node cache of computed counts for the streaming-marginal path
    /// (fast column + lazy BigUint side table, one `CountVec` per level).
    /// Populated lazily by `ensure_level_counts` when a target's child is still
    /// explicit. Cleared at the top of each scheduled `apply_and_fallible` call
    /// (when `marginalize_targets.is_some()`).
    static SCRATCH_STREAM_COUNTS: Cell<Vec<Option<CountVec<ApplyBudget>>>> =
        const { Cell::new(Vec::new()) };

    /// Weighted (`--weighted`) mirror of `SCRATCH_STREAM_COUNTS`: per-vtree-node
    /// cache of computed weights for the streaming-marginal path (one
    /// `Vec<WeightVal>` per level). A concrete second pool because `thread_local!`
    /// can't be generic over the fold's column type; kept symmetric with the
    /// integer pool by construction — IDENTICAL take/clear/return semantics
    /// (pooled take, resize-to-`num_nodes`, clear `[..num_nodes]` to `None`,
    /// unbounded `pool_put` on finalize when `marginalize_targets.is_some()`).
    static SCRATCH_STREAM_WEIGHTS: Cell<Vec<Option<Vec<crate::tdd::query::semiring::WeightVal>>>> =
        const { Cell::new(Vec::new()) };
}

thread_local! {
    /// Per-level is_marginal of c1/c2 snapshotted at apply entry, before the
    /// bottom-up sweep mutates operands (an identity-swap steals levels →
    /// is_marginal flips true→false). NOT debug-only: the pass-through carrier
    /// selector reads these to recover a child that was marginal at entry but
    /// got stolen into the output store mid-sweep (see the `ent_c1`/`ent_c2`
    /// disjuncts in the scatter loop).
    static MARG_ENTRY_C1: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static MARG_ENTRY_C2: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Release every thread-local scratch allocation the apply pipeline retains on
/// THIS thread, giving the next compile a clean allocator slate.
///
/// Called by `log_and_purge_recovery` between a failed compile and its Shannon
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
#[doc(hidden)]
pub fn reset_apply_scratch() {
    reset_sparse_ws();
    types::drop_pools();
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


mod liveness;
// `bucket_shift`/`build_live_cols_bitmask`/`build_reach_masks` are consumed by
// `marg_plan::build_nxm_masks` via `super::liveness::…`, not directly here.



mod stream;
use stream::{StreamLevelState, stream_marginal_eligible, build_stream_state, commit_stream_state};
use crate::tdd::counts::{ApplyBudget, CountVec};

/// Drop the operand-side `Vec`s of a dead operand-child level.
///
/// Called at the *start* of each iteration `t` in `apply_and_fallible`'s
/// bottom-up loop to release `c1.levels[left_idx/right_idx]` and
/// `c2.levels[left_idx/right_idx]`. Children of the current `t` are
/// guaranteed dead at this point: the post-order traversal already visited
/// them in earlier iterations, no future iteration walks into their
/// pairs/nodes/ext (only `c?_widths[child_idx]` is read, and that is a flat
/// usize array snapshot precomputed before the loop).
///
/// **Drop at the START of the iteration, never at the end.** The widest-level
/// output reserve (`budget_reserve_exact`, a single multi-GB allocation) fires
/// mid-iteration; freeing the children before it is what lets the allocator
/// recycle their slabs for the output grows. Dropping after it instead
/// recovers only a fraction of the peak — confirmed by a measured A/B.
///
/// **Keep it unconditional and check-free.** Gating the drop on a per-call
/// size *estimate* costs far more solves than the skipped drops save. The `Vec`s
/// would be freed at apply return anyway; this only moves the drop earlier.
/// The one gate that survives that verdict is the exact-and-trivial test below:
/// it reads the three arenas' own `capacity()` (no estimate, no heuristic) and
/// only declines to free a level whose arenas together fit in a single Vec
/// minimum allocation — i.e. a level where the "release the slab for the output
/// reserve to recycle" motive above has nothing to release.
///
/// Why it matters: on a vtree with far more levels than the operands' support
/// touches (the indicator-deferred def fold crosses a ~360K-level accumulator
/// with a batch whose spine is a few thousand levels), nearly every level is an
/// identity pass-through holding one node and one pair. Freeing those is a
/// `free()` per level per operand per merge that buys back nothing, and it also
/// strips the level pool of its warm arenas so the next apply re-`malloc`s them
/// one at a time. Retention is bounded: the skipped arenas are at Vec's minimum
/// allocation, and the only level arrays that survive an apply are the two the
/// pool parks (`return_levels` / `return_levels2`).
///
/// This is the conservative end of the "keep the warm pool" idea the
/// peak-memory work sketched but never landed: that sketch proposed retaining
/// anything under the pool's own
/// per-arena byte cap, which is orders of magnitude more generous. The gate
/// here retains only what a `free()` would not meaningfully return, so the
/// GiB-class operand levels that motivated the unconditional drop are still
/// dropped unconditionally.
///
/// `marginal_counts` / `marginal_counts_big` are left alone — they're
/// `Option<Vec<_>>`, small relative to nodes/pairs/ext, and `is_marginal()`
/// (which checks `marginal_counts.is_some()`) stays accurate so the
/// marginal-schedule assert still functions on dropped levels. That assert's
/// `subtree_dump` will show `c1.nodes=0` for dropped descendants, but that's a
/// diagnostic-only quality issue on a crash path.
#[inline]
fn drop_dead_operand_level(level: &mut crate::tdd::types::TddLevel) {
    // Nothing worth releasing: the three arenas together hold no more than one
    // Vec minimum allocation (a 1-node / 1-pair identity pass-through level
    // rounds up to 4 slots of each = 64 B). Freeing that returns no slab the
    // output reserve can use, and costs a `free()` now plus a `malloc()` when
    // the pooled level is refilled. Exact capacity reads, not an estimate — see
    // the "Keep it unconditional and check-free" note above for why the
    // distinction is the whole point.
    let bytes = level.nodes.capacity() * std::mem::size_of::<TddNodeData>()
        + level.pairs.capacity() * std::mem::size_of::<InputPair>()
        + level.ext.capacity() * std::mem::size_of::<crate::tdd::types::ExtMulti>();
    // Leave the level completely untouched on this branch — including
    // `dead_pairs`, which stays consistent with the `pairs` arena it counts
    // garbage in. (The unconditional path can zero it only *because* it empties
    // `pairs` in the same breath.)
    if bytes <= 64 { return; }
    level.nodes = Vec::new();
    level.pairs = Vec::new();
    level.ext = Vec::new();
    // No arena left to sweep, so no garbage to remember.
    level.dead_pairs = 0;
}

/// Set `live_counts[t_idx] = v` while keeping the running `out_nodes_so_far`
/// (== `live_counts.iter().sum()`) in sync in O(1). Every write to
/// `live_counts` MUST go through here so the output-node-cap check can read the
/// counter instead of re-summing all levels each boundary (was O(levels²)). The
/// delta form (subtract the old value, add the new) is robust to re-writes; a
/// `debug_assert_eq!` at the cap check cross-validates against the full sum.
#[inline(always)]
fn bump_live_count(
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    t_idx: usize,
    v: usize,
) {
    *out_nodes_so_far = *out_nodes_so_far - live_counts[t_idx] as u64 + v as u64;
    live_counts[t_idx] = v;
}

/// Shared post-output bookkeeping for a freshly-built sparse level's output:
/// refresh the live-node count, mark its product list as populated, and shrink
/// its now-final arrays. Common tail of the two sparse-output emit sites
/// (`apply_sparse_level` and `run_level_rows_marg_sparse`) in
/// `apply_and_fallible_inner`; each site's own pre-tail cleanup
/// (`release_sparse_ws_if_large` / `grid_free`) stays at the call site since
/// it isn't shared.
#[inline(always)]
fn finish_sparse_output(
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    has_pl: &mut [bool],
    level: &mut TddLevel,
    t_idx: usize,
) {
    bump_live_count(live_counts, out_nodes_so_far, t_idx, level.nodes.len());
    has_pl[t_idx] = true;
    level.shrink_arrays();
}

/// Conjunction of two TDDs over the same vtree, with optional marginalization.
///
/// Implements the apply algorithm from §4 of the paper:
/// a level-by-level product construction that yields a fresh canonical TDD.
/// `c1` and `c2` are mutable because per-level scratch / packed encodings may be
/// stripped as their information moves into the output — the underlying TDDs are
/// not semantically modified.
///
/// # Arguments
///
/// - `marginalize_targets`: optional `&[bool]` indexed by `VtreeIdx`. `true` at
///   index `t` requests that the output's level `t` be turned into a *marginal*
///   level (Boolean structure replaced with per-node model counts) during this
///   apply. `None` requests no
///   marginalization.
///
/// # Per-level dispatch
///
/// For each vtree level, the product `k1 × k2` is built in one of several
/// modes, picked locally per level:
///
/// - **Identity fast path** (one operand is constant-true at this subtree):
///   `mem::swap` the other side's level into the output. Zero work, no
///   allocation. Detection propagates bottom-up via `identity_levels`.
/// - **Self-conjunction** at the top: short-circuit `f ∧ f → f.clone()` and
///   skip the entire traversal.
/// - **Sparse mode** (`k1 * k2 > SPARSE_THRESHOLD`): scatter-filter-dedup over
///   live products only. The
///   reverse-index buckets live in `sparse::SparseWorkspace` (thread-local).
/// - **Dense mode** (default): iterate the `k1 × k2` grid with 1×1 / N×1 / 1×N
///   / N×M specializations. The dense scratch is a single flat `node_idx`
///   array reused across levels via `Vec::with_capacity` + lazy DEAD-fill.
///
/// Leaf levels are handled separately by `apply_leaf_levels`, which fills the
/// grid from the static `CONJOIN_GRID` (a 3×3 conjunction table).
///
/// # `OverBudget` recovery contract
///
/// Returns `Err(ApplyError::OverBudget)` if any growth step would push cumulative
/// scratch + output past the soft budget held in
/// [`set_apply_budget`]. Callers (the vsplit / restart-split drivers) take this
/// as the signal to roll back to a `recovery_snap` and try a case-split. The
/// per-apply in-flight counter (`ApplyLimits::budget_in_flight`) is reset at the top of
/// every call so prior apply growth doesn't leak into this one's budget check.
///
/// Other failure modes (allocator OOM not gated by the budget) also bubble up
/// as `OverBudget` — the infallible wrapper [`apply_and`] panics
/// rather than handle them.
///
/// # End-of-apply slot tagging
///
/// This function is also the tagging wrapper around
/// the apply core: the end-of-apply chokepoint where every
/// persisted marg-side ref in the freshly-built result gets its slot tag
/// (bit 30) set. This runs *after* all intra-apply structural reads (which use
/// raw indices) and *before* the result reaches minimize / canon / a
/// subsequent apply / query — exactly the boundary the strict decode assert in
/// `resolve_marg_ref` audits. Idempotent, so the accumulator's repeated
/// re-tagging across batches is harmless.
///
/// # Operand-state contract
///
/// **On `Err`, `c1` and `c2` are CONSUMED / left in an
/// unspecified state.** The bottom-up loop drains dead operand-child levels in
/// place as it goes (`drop_dead_operand_level`), so on an `Err(OverBudget)` /
/// `Err(Deadline)` an unknown prefix of both operands' levels has already been
/// stolen. Callers MUST NOT reuse `c1`/`c2` after an `Err` — rebuild them (from
/// a clone taken before the call) if a retry is needed. All production callers
/// (e.g. vsplit recovery) already treat the operands as trashed on `Err`. On
/// `Ok`, the operands are likewise spent (their
/// levels moved into the result / recycled); the contract is the same, it just
/// matters most on the error path where a naive caller might try to reuse them.
pub(crate) fn apply_and_fallible(
    c1: &mut Tdd,
    c2: &mut Tdd,
    marginalize_targets: Option<&[bool]>,
) -> Result<Tdd, ApplyError> {
    // NB: no operand swap-to-narrower here. That optimization lives ONLY in the
    // owned wrappers (`try_apply_and`), NOT on this
    // shared borrowed path. Order-sensitive callers reach apply through here,
    // and a swap would silently rebind their per-operand bookkeeping to the
    // wrong side. B5's "unify the orientation" premise was false: the
    // borrowed/owned asymmetry is intentional.
    let mut out = apply_and_fallible_inner(c1, c2, marginalize_targets, None)?;
    // Apply emits marg refs self-describing (bit-30 set = inline count;
    // bit-30 clear = bare slot), so a bit-30-clear ref is never an
    // already-inline count here — declare self-describing.
    crate::tdd::types::tag_all_marg_side_slots(&mut out, None);
    Ok(out)
}

/// Spine-bounded variant of [`apply_and_fallible`]: the SAME apply core, run
/// over the restricted level set `restrict.rebuild` and merged back into `c1`'s
/// own level array. See the `restrict` module for what `R` is and why the
/// result is bit-identical to the unrestricted apply.
///
/// Only `restrict::try_apply_and_batch` calls this; it owns the decline
/// checks that make the restriction sound.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a buffer reservation is refused.
fn apply_and_fallible_restricted(
    c1: &mut Tdd,
    c2: &mut Tdd,
    restrict: &Restrict<'_>,
) -> Result<Tdd, ApplyError> {
    let mut out = apply_and_fallible_inner(c1, c2, None, Some(restrict))?;
    // Restricted tagger domain: `tag_all_marg_side_slots` only does work at a
    // STRUCTURAL level with at least one MARGINAL child, and every such level
    // is in `R` by construction (that is what `AncClosure(P)` collects). Off
    // `R` neither the level nor its children changed, so the sweep there would
    // re-derive the accumulator's existing tags — restricting it is
    // result-identical, not merely sound.
    crate::tdd::types::tag_all_marg_side_slots_at(&mut out, None, Some(restrict.rebuild));
    Ok(out)
}

/// Return all pool-owned scratch buffers from a completed apply to their pools.
///
/// Called just before `compute_apply_output` returns.
/// Heavy buffers (`node_idx`, individual product lists) are capped to
/// `MAX_LEVEL_ARENA_BYTES` before returning so a single wide apply doesn't
/// leave GiB-scale allocations cached in the thread-local pools.
/// Zero logic: pure pool-return.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn apply_and_finalize(
    node_idx: Vec<u32>,
    grids: Vec<LevelGrid>,
    c2_identity: Vec<bool>,
    c1_identity: Vec<bool>,
    mut product_lists: Vec<Vec<ProductEntry>>,
    live_counts: Vec<usize>,
    has_pl: Vec<bool>,
    c1_widths: Vec<usize>,
    c2_widths: Vec<usize>,
    stream_computed: Vec<Option<CountVec<ApplyBudget>>>,
    stream_computed_weights: Vec<Option<Vec<crate::tdd::query::semiring::WeightVal>>>,
    inputs1_scratch: Vec<InputPair>,
    inputs2_scratch: Vec<InputPair>,
    mut nxm_masks: liveness::NxmMaskScratch,
    marginalize_targets: Option<&[bool]>,
) {
    // Heavy scratch pools — cap retained capacity to avoid GiB-scale RSS bloat
    // surviving wide applies. Small per-level buffers stay warm.
    pool_put_bounded(&SCRATCH_NODE_IDX, node_idx, MAX_LEVEL_ARENA_BYTES);
    pool_put(&SCRATCH_GRIDS, grids);
    pool_put(&SCRATCH_C2_IDENTITY, c2_identity);
    pool_put(&SCRATCH_C1_IDENTITY, c1_identity);
    for pl in &mut product_lists {
        release_if_oversized(pl, MAX_LEVEL_ARENA_BYTES);
    }
    pool_put(&SCRATCH_PRODUCT_LISTS, product_lists);
    pool_put(&SCRATCH_LIVE_COUNTS, live_counts);
    pool_put(&SCRATCH_HAS_PL, has_pl);
    pool_put(&SCRATCH_C1_WIDTHS, c1_widths);
    pool_put(&SCRATCH_C2_WIDTHS, c2_widths);
    pool_put_bounded(&SCRATCH_INPUTS1, inputs1_scratch, MAX_LEVEL_ARENA_BYTES);
    pool_put_bounded(&SCRATCH_INPUTS2, inputs2_scratch, MAX_LEVEL_ARENA_BYTES);
    // Same retention rule as the buffers above, applied to the bundle's four
    // fields (see `NxmMaskScratch::release_oversized`).
    nxm_masks.release_oversized();
    pool_put(&SCRATCH_NXM_MASKS, nxm_masks);
    if marginalize_targets.is_some() {
        pool_put(&SCRATCH_STREAM_COUNTS, stream_computed);
        pool_put(&SCRATCH_STREAM_WEIGHTS, stream_computed_weights);
    }
}

/// The bottom-up sweep's level walk: the tuned lazy `internal_bottomup` iterator
/// (the byte-identical main-compile order) or the spine-bounded apply's
/// restricted level list. A two-variant enum, not `Box<dyn Iterator>` — that box
/// cost one heap allocation per apply and an indirect `next()` per level, while
/// the loop body below is ONE loop either way.
enum LevelWalk<'a, I> {
    Depth(I),
    /// Spine-bounded apply: `R` in `topo_pos` order (the `Depth` order with the
    /// levels that would take an identity fast path removed).
    Restricted(std::slice::Iter<'a, VtreeIdx>, &'a crate::vtree::Vtree),
}

impl<'a, I: Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)>> Iterator for LevelWalk<'a, I> {
    type Item = (VtreeIdx, VtreeIdx, VtreeIdx);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LevelWalk::Depth(it) => it.next(),
            LevelWalk::Restricted(it, vtree) => it.next().map(|&t| {
                let (l, r) = vtree.children(t);
                (t, l, r)
            }),
        }
    }
}

/// Conservative per-cell byte factor for the apply's product grid: pairs (8B)
/// + nodes (8B) + scratch (4–8B) ≈ 24B. Shared by the in-apply predictive budget
/// check (above) and [`predict_conjoin_dense_cells`] so a pre-apply size gate
/// and the apply itself agree on the byte conversion.
pub(crate) const APPLY_BYTES_PER_CELL: u64 = 24;

/// Byte ceiling on each of the two per-level exact reserves (`level.nodes` and
/// `level.pairs`, both sized from that level's exact upper bound).
///
/// The bounds — `k1 × k2` live cells, `|c1.pairs| × |c2.pairs|` emitted pairs —
/// are exact but loose: most levels have low survival, so an uncapped reserve
/// would routinely grab orders of magnitude more than the level ends up using
/// (and charge every byte of it to the soft budget). Capping bounds the
/// over-allocation per level; a level that outgrows the cap keeps growing
/// through the ordinary fallible push path, and `shrink_arrays` at
/// `finalize_level` hands the unused tail back (it shrinks at cap > 4 × len).
/// ONE definition — the two element-count caps below derive from it.
const LEVEL_RESERVE_CAP_BYTES: usize = 64 * 1024;

/// [`LEVEL_RESERVE_CAP_BYTES`] in `TddNodeData`s — the `level.nodes` arm.
const LEVEL_RESERVE_NODES_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<TddNodeData>();

/// [`LEVEL_RESERVE_CAP_BYTES`] in `InputPair`s — the `level.pairs` arm.
const LEVEL_RESERVE_PAIRS_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<InputPair>();


/// Per-level emit-growth MODE decision before emit — walk-free: the mode is
/// chosen from the caller's `emit_pair_bound` alone, with no pre-sizing pass
/// over the level. `emit_pair_bound` must be a SOUND upper bound on the pairs
/// this level can emit: the dense walk passes `c1_pairs × c2_pairs` (every
/// product pair emits at most once), the clause apply passes its per-level
/// worst case (up to 3 c_t pairs + 1 d_t pair per input pair).
///
/// When the bound of a huge level exceeds `DENSE_GROWTH_DECISION_THRESHOLD` and
/// the worst-case Vec-doubling transient is not provably affordable, arms the
/// bounded-increment growth mode (`set_pairs_bounded_growth` → the increment
/// policy at the `try_push_pair_into` / `reserve_pairs_for_emit` choke points).
/// No counting walk runs in either mode.
///
/// Disarms first, so the armed mode is a pure function of THIS level's bound for
/// every caller. The dense cell-build loop disarms on the routes that skip this
/// call, so every level gets exactly one disarm.
#[inline(always)]
pub(super) fn decide_emit_growth_mode(
    stream_marginal: bool,
    emit_pair_bound: u128,
) {
    set_pairs_bounded_growth(false);
    // When the bound exceeds the threshold, the resulting `level.pairs` may be
    // large enough that Vec's 2× doubling growth would peak at 3× the final size
    // (alloc new + copy + free old) — the mode decision below bounds that
    // transient. Small levels keep plain doubling (~hundreds of MiB transient
    // at the 128 M-iteration threshold — acceptable, same regime the old
    // precount also skipped) and never pay the headroom read.
    //
    // `!stream_marginal` gate: streaming targets truncate pairs per cell, so
    // peak transient is bounded by single-cell pair count and Vec-doubling on
    // `level.pairs` is irrelevant.
    if !stream_marginal && emit_pair_bound > DENSE_GROWTH_DECISION_THRESHOLD as u128 {
        // ── Mode decision: plain doubling vs bounded-increment growth ────
        //
        //   PLAIN DOUBLING — `emit_pair_bound` is a SOUND upper bound on emitted
        //   pairs, so the worst-case doubling transient is ≤ 3 × it × pair_bytes.
        //   When even that fits the headroom, nothing to protect: leave the
        //   default doubling growth. Per-push `try_push` accounting still
        //   enforces any budget during emit.
        //
        //   BOUNDED GROWTH — the doubling transient is NOT provably
        //   affordable (near-cap level: canary mc2020_131's 1.25 G-entry
        //   level under the 31 GiB MCC cap, 3 × 1.25 G × 8 B = 30 GB ≥
        //   headroom). Flag the level so the emit's `level.pairs` growth
        //   (`try_push_pair_into` per push, `reserve_pairs_for_emit` per node)
        //   takes headroom-aware increments: transient = current + increment
        //   instead of doubling's 3×current — which is what lets the canary
        //   survive 31 GiB with no walk.
        //
        // Headroom is `apply_headroom_bytes_or_vas()`: the soft-budget
        // remaining when one is armed (segmented compile — unchanged
        // semantics), else `RLIMIT_AS − current VAS usage` (the competition /
        // run_benchmark 32 GiB cap), else a large finite value when RLIMIT_AS
        // is unlimited. Count-safe throughout (growth policy never changes
        // output).
        let pair_bytes = std::mem::size_of::<InputPair>() as u128;
        let headroom = apply_headroom_bytes_or_vas() as u128;
        if emit_pair_bound.saturating_mul(3).saturating_mul(pair_bytes) >= headroom {
            // Near-cap: bounded-increment emit growth for this level.
            set_pairs_bounded_growth(true);
        }
    }
}

/// Reclaim the two consumed child grids — dead once this level is built (each
/// node has exactly one parent). Sparse mode only; dense mode owns one upfront
/// contiguous block and must not be freed piecemeal. Invoked at every
/// level-finishing exit.
#[inline(always)]
fn reclaim_child_grids(
    might_use_sparse: bool,
    grids: &mut [LevelGrid],
    free_regions: &mut Vec<(usize, usize)>,
    c1_widths: &[usize],
    c2_widths: &[usize],
    left_idx: usize,
    right_idx: usize,
) {
    if might_use_sparse {
        grid_free_child(grids, free_regions, c1_widths, c2_widths, left_idx);
        grid_free_child(grids, free_regions, c1_widths, c2_widths, right_idx);
    }
}

/// Ensure `product_lists[ci]` is populated. Tries the cheap identity fast path
/// first (constant-true operand → the product list is just the non-identity
/// operand's nodes); falls back to scanning the dense grid. Used on both the
/// sparse and dense paths of the level loop.
#[inline]
#[allow(clippy::too_many_arguments)]
fn ensure_product_list_for_child(
    ci: usize, k1: usize, k2: usize,
    c1_identity: &[bool], c2_identity: &[bool],
    grids: &[LevelGrid], node_idx: &[u32],
    product_lists: &mut [Vec<ProductEntry>], has_pl: &mut [bool],
) -> Result<(), ApplyError> {
    if has_pl[ci] { return Ok(()); }
    if !fill_identity_product_list(
        k1, k2,
        c2_identity[ci], c1_identity[ci],
        &mut product_lists[ci],
        &mut has_pl[ci],
    )? {
        ensure_product_list(
            ci, k1, k2,
            grids, node_idx,
            &mut product_lists[ci], has_pl,
        )?;
    }
    Ok(())
}

/// Mark a level's pass-through inline-emit flags (#45).
///
/// If this level was built via pass-through, its marginal-side pair fields hold
/// inline counts carried verbatim from the carrier operand (already emitted),
/// NOT fresh slots. Mark them so the end-of-apply tagger's emit arm skips
/// re-emitting (which would misread an inline count as a slot index →
/// miscount). Guarded on `!is_marginal()`: a level that became marginal during
/// its build had its markers reset by `make_marginal` and has no structural
/// pairs to describe. `left_passthrough`/`right_passthrough` are emit-gated, so
/// this is a no-op in baseline.
///
/// Shared by the general per-level tail (`finalize_level`) and the sparse
/// one-marginal-child route, which returns before that tail runs.
#[inline(always)]
fn mark_passthrough_inlined(level: &mut TddLevel, left_passthrough: bool, right_passthrough: bool) {
    if (left_passthrough || right_passthrough) && !level.is_marginal() {
        if left_passthrough { level.set_marg_inlined_left(true); }
        if right_passthrough { level.set_marg_inlined_right(true); }
    }
}

/// Per-level tail after the cell-build route dispatch (extraction 5).
///
/// Covers: stream commit (`commit_stream_state`), `live_counts` update,
/// `grids[t_idx]` tagging, `shrink_arrays`, and the pass-through
/// inline-emit flags (`mark_passthrough_inlined`).
///
/// Grid reclamation ([`reclaim_child_grids`]) stays at the call site — the
/// three early-exit routes reclaim without running this tail at all, so it
/// cannot fold in here.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn finalize_level(
    stream_state: &mut Option<StreamLevelState>,
    t: VtreeIdx,
    t_idx: usize,
    t_base: usize,
    might_use_sparse: bool,
    left_passthrough: bool,
    right_passthrough: bool,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    grids: &mut Vec<LevelGrid>,
    live_counts: &mut Vec<usize>,
    out_nodes_so_far: &mut u64,
) {
    // Commit streaming-marginal emit: convert level to marginal_counts.
    // Must happen before the `levels[t_idx]` reborrows below; the local
    // `level: &mut TddLevel` borrow ends at last use above (in the j-loop).
    if let Some(st) = stream_state.take() {
        commit_stream_state(st, t, t_idx, vtree, levels);
    }

    // Record live count for parent density checks (only when sparse mode possible).
    // Use `width()` so streaming-marginal levels (nodes.len() == 0 after
    // make_marginal) report their actual alive-cell count.
    if might_use_sparse {
        bump_live_count(live_counts, out_nodes_so_far, t_idx, levels[t_idx].width());
    }
    // Dense emit wrote node_idx in (i, j) row-major order keyed by
    // level.nodes.len() at each emission, so live cells are strictly
    // monotone → eligible for the H1 sort-skip at parent levels.
    grids[t_idx] = LevelGrid::DenseStrict { base: t_base };

    levels[t_idx].shrink_arrays();

    // Output-pair meter: this level is done, so its arena's CAPACITY estimate
    // gives way to the pairs it actually holds. See `settle_output_pairs`.
    budget::settle_output_pairs(levels[t_idx].pairs.len());

    mark_passthrough_inlined(&mut levels[t_idx], left_passthrough, right_passthrough);
}

fn apply_and_fallible_inner(
    c1: &mut Tdd,
    c2: &mut Tdd,
    marginalize_targets: Option<&[bool]>,
    restrict: Option<&Restrict<'_>>,
) -> Result<Tdd, ApplyError> {
    mem_eager_reclaim();
    assert!(
        Arc::ptr_eq(&c1.vtree, &c2.vtree),
        "apply_and requires TDDs with the same vtree"
    );
    assert_eq!(
        c1.output.vtree, c2.output.vtree,
        "apply_and requires TDDs with outputs at the same vtree node"
    );

    // Reset per-apply meters: the in-flight byte counter, so cumulative
    // capacity-grow accounting via `budget_reserve(_exact)` starts fresh —
    // without this the counter would conflate growth across calls and trip
    // OverBudget spuriously on a small later apply.
    budget::reset_meters();

    // Self-conjunction short-circuit: f ∧ f = f.
    if is_self_conjunction(c1, c2) {
        return Ok(c1.clone());
    }

    // Early return for ZERO inputs: x ∧ ZERO = ZERO.
    // Avoids allocating levels, level_base, and node_idx for unsatisfiable operands.
    let vtree = Arc::clone(&c1.vtree);
    let num_nodes = vtree.num_nodes();
    if c1.is_zero() || c2.is_zero() {
        let levels = types::take_levels(num_nodes);
        return Ok(Tdd::with_levels(
            vtree,
            levels,
            TddNodeId { vtree: c1.output.vtree, local: ZERO },
        ));
    }

    let ApplySetup {
        mut levels, mut grids, c1_widths, c2_widths,
        min_grid, sparsity_factor, might_use_sparse,
        mut stream_computed,
        mut node_idx, mut grid_end,
        mut product_lists, mut live_counts, mut has_pl,
        any_entry_marginal,
    } = apply_and_setup(c1, c2, &vtree, num_nodes, marginalize_targets, restrict)?;
    // Running sum of `live_counts`, kept in O(1) via `bump_live_count` at every
    // write site so the output-node-cap check reads it instead of re-summing all
    // levels each boundary. `live_counts` starts all-zero (pooled + resized with
    // 0 fill), so this holds the invariant `out_nodes_so_far == live_counts.sum()`
    // from init.
    let mut out_nodes_so_far: u64 = 0;

    // Grid-reclaim free-list. In sparse mode, each
    // densified level's `node_idx` grid region is dead once its single parent has
    // consumed it; returning those regions here lets later levels reuse them
    // instead of bumping `grid_end` forever. This caps the arena at the live-grid
    // frontier (~2 GiB on mc2024_track1_041) rather than the cumulative sum of
    // every densified level (~24.5 GiB → past the 32 GiB wall). (base, len) pairs;
    // unused (stays empty) in dense mode, which pre-sizes node_idx upfront.
    let mut free_regions: Vec<(usize, usize)> = Vec::new();

    // Weighted (`--weighted`) streaming scratch: per-batch BigRational values for
    // streaming-target levels whose children are still explicit. Pooled via
    // `SCRATCH_STREAM_WEIGHTS`, the concrete weighted mirror of the integer
    // `SCRATCH_STREAM_COUNTS` pool — same take/clear/return discipline, so pooled
    // reuse can't leak a stale weight into a later apply. Empty when not
    // marginalizing.
    let mut stream_computed_weights: Vec<Option<Vec<crate::tdd::query::semiring::WeightVal>>> =
        if marginalize_targets.is_some() {
            pool_take(&SCRATCH_STREAM_WEIGHTS)
        } else {
            Vec::new()
        };
    if marginalize_targets.is_some() {
        if stream_computed_weights.len() < num_nodes {
            stream_computed_weights.resize_with(num_nodes, || None);
        }
        // Clear any stale entries from prior calls within `num_nodes` (mirrors the
        // integer scratch reset; dropping the old `Some(vec)` frees its weights).
        for slot in stream_computed_weights[..num_nodes].iter_mut() { *slot = None; }
    }

    // ── Identity tracking ─────────────────────────────────────────────────
    //
    // c2_identity[t] is true when c2 computes constant-true at subtree t,
    // meaning c1's nodes pass through unchanged (x ∧ 1 = x). This lets the
    // product construction skip the expensive per-node inner loop at
    // identity levels — just copy c1's nodes directly (via mem::swap on c1).
    //
    // c1_identity[t] is the symmetric case: c1 is constant-true at subtree t,
    // so c2's nodes pass through unchanged (1 ∧ x = x). At identity levels,
    // c2's nodes are cloned to the output (c2 is immutable, can't swap).
    // This is critical for conjoin_children(left, right) where c1 = left
    // child TDD has identity at right-subtree levels and vice versa.
    //
    // A leaf is identity iff only the One label (local index 0) is referenced by
    // parent pairs. An internal node is identity iff width == 1 AND both
    // children are identity. With implicit leaf representation, we precompute
    // leaf identity by scanning parent pairs for non-One references.
    let mut c2_identity = pool_take(&SCRATCH_C2_IDENTITY);
    let mut c1_identity = pool_take(&SCRATCH_C1_IDENTITY);
    if let Some(r) = restrict {
        // Restricted mode derives both identity vectors from the spine
        // certificate instead of scanning every leaf's parent pairs twice:
        //
        // * `c2_identity[t] = !on_spine[t]` — the batch is width-1
        //   constant-true off its spine by construction, which is exactly the
        //   fixpoint the generic path's FP1 accretes at every off-spine level.
        // * `c1_identity` all-false — it is read ONLY by FP2's guard (the other
        //   readers are the declined sparse routes and `plan_marg_level`'s
        //   `left_pt_c2`, whose `c2_ref` conjunct is false because the batch
        //   has no marginal levels and no `marg_inlined_*` flags). Restricted
        //   mode never takes a fast path, so an all-false vector just means
        //   every level in `R` is rebuilt — and a rebuild against a width-1
        //   constant-true operand reproduces the carried level.
        //
        // Only `touched` entries are ever read, so only they are written; the
        // pooled buffers keep whatever stale values they had elsewhere.
        try_resize(&mut c2_identity, num_nodes, false)?;
        try_resize(&mut c1_identity, num_nodes, false)?;
        for &t in r.touched {
            c2_identity[t.idx()] = !r.on_spine[t.idx()];
            c1_identity[t.idx()] = false;
        }
    } else {
        init_leaf_identity(&mut c2_identity, c2, &vtree, num_nodes)?;
        init_leaf_identity(&mut c1_identity, c1, &vtree, num_nodes)?;
    }

    // `c{1,2}_identity` is lazily accreted, so it can read false for a child
    // that is structurally identity, sending that child to the dense-grid
    // fallback instead of pass-through. Completing the predicate is a measured
    // NO-GO — the misses are rare and land on tiny grids; don't retry it
    // without a bench showing real grid savings.

    // ── Node-index monotonicity tracking ─────────────────────────────────
    //
    // Each producer tags its level with a `LevelGrid` variant. Only the
    // `DenseStrict` variant (set by dense-internal emit and identity
    // shortcuts) guarantees row-major strict monotonicity of live cells;
    // leaves (`CONJOIN_GRID`) and scatter-materialised grids (`DenseWeak`)
    // do not. Nothing consumes that guarantee today — pair lists are unordered
    // sets and twin contraction is order-independent (see the NOTE in
    // types.rs), so no emit site needs to produce a particular order. The
    // variant tagging is retained as cheap producer metadata.

    // Scratch buffers for decoded pair slices: when the operand level is packed
    // (packed-pairs feature), these hold a materialized copy of ONE cell's pairs
    // (transient — `pairs_view_decoded` clears before every fill). Taken from
    // the thread pool and cleared here, so they carry only capacity across
    // applies; returned in `apply_and_finalize`.
    let mut inputs1_scratch: Vec<InputPair> = pool_take(&SCRATCH_INPUTS1);
    let mut inputs2_scratch: Vec<InputPair> = pool_take(&SCRATCH_INPUTS2);
    inputs1_scratch.clear();
    inputs2_scratch.clear();

    // ── NxM dead-pair pre-filter scratch space ──────────────────────────
    //
    // Reused across internal levels (see `build_live_cols_bitmask` and
    // `build_reach_masks` for what each slot means). Left and right sides are
    // tracked independently because their child widths (and therefore the
    // column-bucket shift, see `bucket_shift`) can differ. Taken from the thread
    // pool so the four buffers carry only capacity across applies — both
    // builders clear and refill their whole length, so nothing else survives;
    // returned (capped) at the tail alongside the other apply scratch.
    let mut nxm_masks: liveness::NxmMaskScratch = pool_take(&SCRATCH_NXM_MASKS);

    // ── Bottom-up product construction ─────────────────────────────────
    //
    // At each vtree level, compute c1[i] ∧ c2[j] for all (i,j) node pairs.
    // Leaf levels: static CONJOIN_GRID lookup (no stored nodes, no iteration).
    // Internal levels: either dense grid iteration or sparse scatter pipeline,
    // chosen online based on children's product density.

    apply_leaf_levels(
        &vtree, &c1_widths, &c2_widths, &mut grids, &mut node_idx,
        &mut grid_end, &mut live_counts, &mut out_nodes_so_far, might_use_sparse,
        restrict.map(|r| r.leaf_children),
    )?;

    // Restricted mode: stand in for the FP1 pass the generic loop would have
    // run at every off-`R` INTERNAL child of a rebuilt level. FP1 there carries
    // the accumulator's level through by reference (which restricted mode gets
    // for free — the output array IS the accumulator's) and leaves behind two
    // observable side effects the rebuild above `t` reads:
    //   * the identity grid `node_idx[base + i] = i` (dense layout), or a
    //     `Sparse` tag the parent densifies via `materialize_dense_child`
    //     (bump-allocator layout — same as generic, which also leaves FP1'd
    //     levels ungridded under `might_use_sparse`);
    //   * `live_counts[x] = k_carrier`, which the parent's online density check
    //     divides by. Seeding it is not optional: a zero live count would send
    //     a level to the sparse route the generic path keeps dense.
    if let Some(r) = restrict {
        for &x in r.touched {
            let xi = x.idx();
            if r.in_rebuild[xi] || vtree.node(VtreeIdx(xi as u32)).is_leaf() {
                continue;
            }
            let k1 = c1_widths[xi];
            debug_assert_eq!(
                c2_widths[xi], 1,
                "spine-bounded merge: off-spine level {xi} is not width-1 in the \
                 batch — the batch spine certificate is wrong"
            );
            if might_use_sparse {
                bump_live_count(&mut live_counts, &mut out_nodes_so_far, xi, k1);
            } else {
                let base = grids[xi].base_unchecked();
                for idx in 0..k1 {
                    node_idx[base + idx] = idx as u32;
                }
                grids[xi] = LevelGrid::DenseStrict { base };
            }
        }
    }

    // Leaf marginalization: a marginal vtree LEAF is never visited as a `t` by
    // the bottom-up loop, so — unlike a marginal internal child — its OUTPUT level
    // is never marked marginal. Seed it here from the operands. A marginalized
    // leaf var is PRIVATE (summed only once its every clause is compiled), so the
    // other operand is identity at that leaf; the parent's marginal-child dispatch
    // then routes Route A and the passthrough path carries the carrier's inline
    // `MargRef` refs through verbatim. The output store stays empty (all leaf
    // counts are inline at the parent).
    // Under `--weighted` the leaf's counts are NOT inline at the parent: the
    // weighted leaf-marg installs a real per-slot column in the (compile-global,
    // vtree-indexed) `WeightStore` and leaves the parent's bare leaf-label refs to
    // decode as `MargRef::Slot`. That column is PINNED — immutable, label-ordered,
    // exactly `LEAF_WIDTH` slots, never compacted / erased / appended to by any
    // pass — so the output level reports `LEAF_WIDTH` and this only re-flags it.
    //
    // `canon_leaves` collects the leaves flagged weight-marginal on ONE operand's
    // authority: the other operand was structural there, so its genuine leaf-LABEL
    // refs flow through `CONJOIN_GRID` into the output and may not be canonical
    // (`marginalize::leaf_canon_map`). They are canonicalized once the output's
    // pair lists are final — the bottom-up loop BELOW emits them, so there is
    // nothing to rewrite here yet. When BOTH operands are weight-marginal both
    // sides are already canonical and the grid is closed over each canon class
    // (`{One,Pos}`, `{One,Neg}`, `{One}` are each closed under ∧), so nothing is
    // recorded.
    // Restricted mode skips this sweep entirely. The output level array is
    // merged back into the ACCUMULATOR's, so a marginal accumulator leaf keeps
    // its own level (already marginal) rather than needing to be re-seeded; the
    // batch has no marginal levels at all; and `weight_ctx_active()` is a
    // decline, so `canon_leaves` would stay empty. The only consumer of the
    // seeded flag inside the loop is the "output child is marginal" test, which
    // reads through to `c1.levels[..]` under a restriction (see below).
    let mut canon_leaves: Vec<usize> = Vec::new();
    for (leaf, _) in vtree.leaf_bottomup() {
        if restrict.is_some() {
            break;
        }
        let li = leaf.idx();
        let c1m = c1.levels[li].is_marginal();
        let c2m = c2.levels[li].is_marginal();
        if c1m || c2m {
            debug_assert!(
                (c1m && c2m) || (c1m && c2_identity[li]) || (c2m && c1_identity[li]),
                "marginal leaf {li} conjoined with a non-identity operand \
                 (var not private?): c1m={c1m} c2m={c2m} \
                 c1_id={} c2_id={}",
                c1_identity[li], c2_identity[li],
            );
            let w1 = c1.levels[li].is_weight_marginal();
            let w2 = c2.levels[li].is_weight_marginal();
            if w1 || w2 {
                // PIN INVARIANT (`marginalize::marginalize_leaf_weighted`): a
                // weight-marginal LEAF's column is an IMMUTABLE, label-ordered,
                // exactly-`LEAF_WIDTH` cache of `WeightStore::leaf_val`. No pass
                // compacts, erases, reorders or appends to it — slot-prune,
                // dup-resolve's twin fold, weighted p-fusion and the subsumption
                // reclaim all decline at leaves — so the output level's slot count
                // is `LEAF_WIDTH`, full stop.
                //
                // Flagging it directly (rather than reading the column's length)
                // is what makes this robust: a `map_or(0, len)` read reports width
                // 0 whenever the global column happens not to be installed for
                // this vtree index, and a width-0 weight-marginal leaf is silently
                // skipped by `marginalize_batch_weighted` and read as an empty
                // column by the streaming child view — dropping the leaf's entire
                // mass with no error anywhere.
                let leaf_slots = crate::tdd::types::LEAF_WIDTH;
                debug_assert!(
                    crate::tdd::transform::unary::marginalize::with_weight_ctx(|ws| {
                        ws.level(li).is_none_or(|v| v.len() == leaf_slots)
                    }),
                    "weight-marginal leaf {li}: WeightStore column is not the \
                     pinned {leaf_slots}-slot leaf_val cache",
                );
                debug_assert!(
                    (!w1 || c1.levels[li].width() == leaf_slots)
                        && (!w2 || c2.levels[li].width() == leaf_slots),
                    "weight-marginal leaf {li}: operand slot carriers \
                     (c1={}, c2={}) disagree with LEAF_WIDTH ({leaf_slots})",
                    c1.levels[li].width(), c2.levels[li].width(),
                );
                levels[li].make_marginal_weighted_with_slots(leaf_slots as u32);
                if w1 != w2 {
                    canon_leaves.push(li);
                }
            } else {
                levels[li].make_marginal(Vec::new(), None);
            }
        }
    }

    let internal_iter = match restrict {
        // Restricted mode walks `R` in `topo_pos` order — the generic
        // `internal_bottomup()` order with the fast-path levels removed.
        Some(r) => LevelWalk::Restricted(r.rebuild.iter(), &vtree),
        // The tuned lazy bottom-up iterator: no allocation, no reorder.
        None => LevelWalk::Depth(vtree.internal_bottomup()),
    };
    // Where this apply has got to, for a caller watching one long merge from
    // outside it (`budget::merge_position`). The level COUNT is the only thing
    // that costs a walk, so it is taken inside the gate; past that it is one
    // store per level and no clock at all.
    let watched = budget::merges_watched();
    if watched {
        budget::merge_began(vtree.internal_bottomup().count() as u32);
    }
    let mut level_k: u32 = 0;
    // The loop body runs inside an immediately-invoked closure so a single seam
    // catches `?` propagation from the many allocation sites below.
    let loop_result: Result<(), ApplyError> = (|| {
    for (t, left, right) in internal_iter {
        if watched {
            level_k += 1;
            budget::merge_reached(level_k);
        }
        // The per-level-boundary cut check (gated wall deadline, then
        // output-node cap), in order — see `budget::check_level_boundary`
        // for the per-step rationale (why the out_nodes_so_far-sum stays lazily
        // gated).
        budget::check_level_boundary(
            out_nodes_so_far,
            &live_counts,
        )?;

        let k1 = c1_widths[t.idx()];
        let k2 = c2_widths[t.idx()];
        let t_idx = t.idx();
        let left_idx = left.idx();
        let right_idx = right.idx();

        // Lever 6 — drop dead operand-child levels at START of iteration.
        // Replaces Lever 5b's end-of-iter drops: at end-of-iter, the
        // 6.51 GB output.level.pairs reserve at line ~2871 has already
        // fired with children alive; start-of-iter drops release them
        // BEFORE the reserve so peak-heap window excludes them. Safe:
        // c?_widths is a precomputed flat array (line 1775), and no
        // reads of c1/c2.levels[left/right_idx].nodes/pairs/ext occur
        // in this iteration's body (only c1.level(t)/c2.level(t)). The
        // marginal-assert diagnostic subtree_dump (line ~2091) will show
        // c1.nodes=0 for already-dropped children, but the assert itself
        // (c1.level(t).is_marginal()) is correct. Drop preserves
        // marginal_counts so is_marginal() stays accurate.
        // Restricted mode does NOT drop operand child levels: `c1` is the
        // accumulator and its off-`R` levels ride through into the output
        // verbatim (the output array IS c1's, merged at the tail). Generically
        // these drops are free — an FP1'd child was already swapped out of `c1`,
        // so the call sees an empty placeholder — but under a restriction the
        // level is still live data.
        //
        // Nor does it attempt an identity fast path: `R` is by construction the
        // set of levels where none fires (see the `restrict` module), and the
        // OUTPUT-child marginality the guards read lives in `c1.levels[..]`
        // here, not in the fresh `levels[..]`.
        if restrict.is_none() {
            drop_dead_operand_level(&mut c1.levels[left_idx]);
            drop_dead_operand_level(&mut c1.levels[right_idx]);
            drop_dead_operand_level(&mut c2.levels[left_idx]);
            drop_dead_operand_level(&mut c2.levels[right_idx]);

            // Identity fast paths: FP1 (c1 carrier / c2 identity), FP2 (symmetric),
            // and the 0-width orphan-marginal case. See `try_level_fast_paths` for
            // the full guard logic.
            match try_level_fast_paths(
                c1, c2, t,
                k1, k2, t_idx, left_idx, right_idx,
                might_use_sparse,
                &mut levels, &mut c1_identity, &mut c2_identity,
                &mut live_counts, &mut out_nodes_so_far, &mut grids, &mut node_idx,
            )? {
                FastPathResult::Taken => {
                    reclaim_child_grids(might_use_sparse, &mut grids, &mut free_regions, &c1_widths, &c2_widths, left_idx, right_idx);
                    continue;
                }
                FastPathResult::NotTaken => {}
            }
        }

        // Marginalize-schedule invariant — ALWAYS ON (the gate is two `is_marginal()`
        // bool reads per level, negligible vs the apply work). If either operand's
        // level t is marginal (mc-mode pair structure replaced with model counts), the
        // identity fast-paths above MUST have consumed it: a marginal level conjoins
        // soundly ONLY with an identity (non-constraining) counterpart, and that case
        // is taken by `try_level_fast_paths` and `continue`s. Reaching this dense path
        // with a marginal level therefore means the OTHER operand still constrains
        // node t — i.e. a variable was summed out of one operand while still live in
        // the other. That is an invalid conjoin; the dense path below would deref
        // `nodes[i]` on an empty Vec (SIGSEGV) or silently miscount. It can only arise
        // from a marginalize-schedule bug (`compute_marginalize_at`), never from a
        // correct run — so fail loudly rather than corrupt the count.
        //
        // Was debug-only from 2026-05-17 until a SIGSEGV in the segment-compile fold
        // (conjoining a marginalized operand with a partner that still referenced the
        // summed-out var) showed release builds had NO guard here.
        // The violation is the ASYMMETRIC case: exactly one operand marginalized node
        // t while the OTHER still carries a real (non-identity) function over t's
        // variables. The symmetric both-marginal case (both summed out the same scope)
        // is sound and handled below / by the both-marginal-width-1 fast path; the
        // marginal-vs-identity case is consumed by the identity fast paths above. So we
        // only fault on marginal-vs-constraining.
        let c1_marg = c1.level(t).is_marginal();
        let c2_marg = c2.level(t).is_marginal();
        let c1_identity_at_t = c1_identity[left_idx] && c1_identity[right_idx];
        let c2_identity_at_t = c2_identity[left_idx] && c2_identity[right_idx];
        let violation = (c1_marg && !c2_marg && !c2_identity_at_t)
            || (c2_marg && !c1_marg && !c1_identity_at_t);
        if violation {
            // Rich subtree dump (expensive string build + /tmp file) only in debug.
            #[cfg(debug_assertions)]
            debug_assert_marg_schedule(
                c1, c2, t, left, right, &vtree,
                k1, k2, left_idx, right_idx,
                &c1_widths, &c2_widths, &c1_identity, &c2_identity,
            );
            panic!(
                "apply_and marginalize-schedule violation at vtree node {t:?} \
                 (left={left:?} right={right:?}): one operand marginalized this node \
                 while the other still constrains it \
                 (c1.marg={c1_marg}, c2.marg={c2_marg}, c1_id[L,R]={},{}, c2_id[L,R]={},{}). \
                 A variable was summed out of one operand while still live in the other \
                 — a marginalize-schedule bug; see compute_marginalize_at. This conjoin \
                 is invalid and would corrupt the model count.",
                c1_identity[left_idx],
                c1_identity[right_idx],
                c2_identity[left_idx],
                c2_identity[right_idx],
            );
        }

        let k1_left = c1_widths[left_idx];
        let k2_left = c2_widths[left_idx];
        let k1_right = c1_widths[right_idx];
        let k2_right = c2_widths[right_idx];

        // ── Online density check ─────────────────────────────────────
        //
        // Only when might_use_sparse: decide dense vs sparse based on children's
        // actual product density. Children are already processed (bottom-up order),
        // so live_counts are available.
        // A marginal child's pair ref is a *count payload* (inline count or
        // tagged slot), NOT a structural node index. The sparse reverse index
        // buckets parent nodes by `decode_marg_coord(ref)` (see
        // `build_reverse_index`), which is only a valid per-node key when the
        // payload happens to be per-node-distinct — true for the legacy slot
        // encoding, FALSE under inline, where every equal-count ref decodes to
        // the same value and distinct marginal children collapse into one
        // bucket (dropping multiplicity → model-count corruption, e.g. mc007
        // ×4). The dense path indexes by the real grid position and is immune.
        // So: marginal levels always take the dense path.
        let left_child_marg = levels[left_idx].is_marginal()
            || c1.levels[left_idx].is_marginal()
            || c2.levels[left_idx].is_marginal();
        let right_child_marg = levels[right_idx].is_marginal()
            || c1.levels[right_idx].is_marginal()
            || c2.levels[right_idx].is_marginal();
        let marg_child = left_child_marg || right_child_marg;
        let use_sparse = might_use_sparse && !marg_child && {
            let max_left = (k1_left * k2_left) as u128;
            let max_right = (k1_right * k2_right) as u128;
            let live_l = live_counts[left_idx] as u128;
            let live_r = live_counts[right_idx] as u128;
            k1 * k2 > min_grid
                && max_left > 0 && max_right > 0
                && sparsity_factor * live_l * live_r < max_left * max_right
        };

        // Sparse bottom-up build for an *exactly-one*-marginal-child level whose
        // output is STRUCTURAL (the OOM case). The dense path allocates a full
        // k1*k2 slab even though the marginal side never kills a pair (it is a
        // pass-through carrier) — so the live structural sibling alone governs
        // survival and the slab is mostly DEAD. Instead, drive the build from the
        // structural sibling, emit a `product_list` of the surviving cells, and
        // tag this level Sparse; the grandparent densifies lazily via ensure_grid.
        //
        // Gated to the structural case only: the explicit `!is_marg_target`
        // conjunct is what excludes marginalize targets — one-marginal-child
        // TARGETS do exist and are common (measured: thousands of such sites
        // across the marginalize/recovery test suites), so do not assume the
        // conjunct is dead. XOR + !is_marg_target is exactly the structural
        // one-marginal-child level. Both-marginal and target levels fall through
        // to the existing dense Route A unchanged.
        let use_sparse_marg = might_use_sparse
            && (left_child_marg ^ right_child_marg)
            && !marginalize_targets.is_some_and(|a| a[t_idx])
            && k1 * k2 > min_grid;

        if use_sparse {
            // Ensure children have product lists for the scatter pipeline.
            ensure_product_list_for_child(
                left_idx, k1_left, k2_left,
                &c1_identity, &c2_identity, &grids, &node_idx,
                &mut product_lists, &mut has_pl,
            )?;
            ensure_product_list_for_child(
                right_idx, k1_right, k2_right,
                &c1_identity, &c2_identity, &grids, &node_idx,
                &mut product_lists, &mut has_pl,
            )?;

            // Disjoint borrows of three product lists (left, right, output).
            let [pl_left, pl_right, pl_output] = product_lists
                .get_disjoint_mut([left_idx, right_idx, t_idx])
                .expect("left_idx, right_idx, t_idx must be distinct");
            let is_marg_target = marginalize_targets.is_some_and(|arr| arr[t_idx]);
            apply_sparse_level(
                t, left, right, c1, c2,
                &mut levels, &c1_widths, &c2_widths,
                pl_left,
                pl_right,
                pl_output,
                vtree.node(VtreeIdx(left_idx as u32)).is_leaf(),
                vtree.node(VtreeIdx(right_idx as u32)).is_leaf(),
                is_marg_target,
            )?;
            // Release oversized bucket Vecs to avoid retaining peak allocations.
            release_sparse_ws_if_large();
            finish_sparse_output(&mut live_counts, &mut out_nodes_so_far, &mut has_pl, &mut levels[t_idx], t_idx);

            reclaim_child_grids(might_use_sparse, &mut grids, &mut free_regions, &c1_widths, &c2_widths, left_idx, right_idx);
            continue;
        }

        // ── Route prediction (before materializing children) ──────────────
        //
        // Compute the marg-plan flags now: they read only level metadata, no
        // child grid (the grid-reading NxM masks are deferred to
        // `build_nxm_masks` below, which runs only when `nxm` — i.e. the
        // general path, where the grids are materialized). Knowing the route
        // here is what lets the dense path skip `ensure_grid` for a sparse
        // child when the level will take the plain-dense emit.
        let MargPlan {
            left_pt_c1, right_pt_c1,
            left_passthrough, right_passthrough,
            left_mask, right_mask,
            nxm,
        } = plan_marg_level(
            c1, c2, t,
            t_idx, left_idx, right_idx,
            &levels, &c1_identity, &c2_identity,
            any_entry_marginal,
        );
        // ── Dense path: ensure children have grids ───────────────────
        //
        // Only when might_use_sparse: if a child was processed by the sparse
        // pipeline (no grid), materialize its grid (`ensure_grid`).
        //
        // Note: this branch uses `fill_identity_product_list` directly (not
        // `ensure_product_list_for_child!`) because on the dense path we know
        // the child is sparse — no grid to scan — so the identity fast path
        // is the only way to build the product list.
        // Threaded out of the alloc block below: the k2-row scratch base used by
        // the sparse-marg path (0 when that path is not taken).
        let mut sparse_marg_row_base = 0usize;
        if might_use_sparse {
            if grids[left_idx].is_sparse() {
                materialize_dense_child(
                    left_idx, k1_left, k2_left,
                    c2_identity[left_idx], c1_identity[left_idx],
                    &mut has_pl[left_idx], &mut product_lists[left_idx],
                    &mut grids, &mut node_idx, &mut grid_end, &mut free_regions,
                )?;
            }
            if grids[right_idx].is_sparse() {
                materialize_dense_child(
                    right_idx, k1_right, k2_right,
                    c2_identity[right_idx], c1_identity[right_idx],
                    &mut has_pl[right_idx], &mut product_lists[right_idx],
                    &mut grids, &mut node_idx, &mut grid_end, &mut free_regions,
                )?;
            }

            // Bump-allocate grid for this level. Kind will be overwritten to
            // DenseStrict at the end of the dense emit loop below; use a
            // placeholder variant (DenseWeak) until then so consumers that
            // peek here (e.g. debug_assert paths) see a consistent base.
            //
            // Sparse-marg path: allocate only a single reused k2-row scratch
            // instead of the dense k1*k2 slab. `run_level_rows_marg_sparse`
            // processes one structural row at a time into this scratch, records
            // the surviving cells into the output product_list, then frees the
            // scratch — the dense slab is never materialized. The level is tagged
            // Sparse here; the grandparent densifies it lazily via ensure_grid.
            let cells = if use_sparse_marg { k2 } else { k1 * k2 };
            let base = grid_alloc(&mut node_idx, &mut grid_end, &mut free_regions, cells)?;
            if use_sparse_marg {
                grids[t_idx] = LevelGrid::Sparse;
            } else {
                grids[t_idx] = LevelGrid::DenseWeak { base };
            }
            sparse_marg_row_base = base;
        }

        // For the sparse-marg path `grids[t_idx]` is Sparse (no slab base), so the
        // row scratch base is threaded out of the alloc block explicitly.
        let t_base = if use_sparse_marg {
            sparse_marg_row_base
        } else {
            grids[t_idx].base_unchecked()
        };

        // Non-identity internal: DEAD-fill is interleaved with product construction
        // below (per-row fill before each row's cells are computed). This keeps the
        // active row in L1 cache during process_cell, avoiding the cache pollution
        // from a single bulk fill of the entire k1*k2 grid.
        // Child grid dimensions: used to compute flat grid positions.
        // Grid position for child product (a,b) = child_base + a * k2_child + b
        let k2_left = k2_left as u32;
        let k2_right = k2_right as u32;
        let left_base = grids[left_idx].base_unchecked();
        let right_base = grids[right_idx].base_unchecked();

        // Marg flags (`left_pt_*`, passthrough, masks, `nxm`) were computed
        // above, before child materialization, to decide the route. Only the
        // grid-reading NxM liveness masks are deferred to here — they need the
        // materialized child grids, and `nxm` implies the general (non-sparse-
        // served) path, so both grids exist.
        if nxm {
            build_nxm_masks(
                c2, t,
                k2,
                k1_left, k2_left as usize, k2_right as usize,
                left_base, right_base, right_idx,
                left_passthrough, right_passthrough,
                left_mask, right_mask,
                &node_idx, &c1_widths,
                &mut nxm_masks.live_left_cols, &mut nxm_masks.reach_c2_left,
                &mut nxm_masks.live_right_cols, &mut nxm_masks.reach_c2_right,
            )?;
        }

        // Streaming-eligibility gate (see `stream_marginal_eligible`), read by
        // the emit-growth mode decision and threaded into the row loop below.
        let stream_marginal = stream_marginal_eligible(marginalize_targets, t_idx);

        let mut stream_state: Option<StreamLevelState> = build_stream_state(
            t_idx, left_idx, right_idx, k1, k2,
            marginalize_targets, &vtree, &mut levels,
            &mut stream_computed,
            &mut stream_computed_weights,
        )?;

        // ── Dedicated marginal-child dispatch ──
        // Decide HERE (post-cascade, where child marginality is final; pre-`level`
        // borrow, where we can still read `levels[child]`) whether this level has
        // EXACTLY ONE marginal child. If so, and the gate is on, the cell-build
        // row-loop below branches to the dedicated no-grid path instead of the
        // general product grid. Setup (widths, bases, node_idx, stream_state) and
        // finalize (dedup/permute, shrink, pack) are SHARED — only the inner cell
        // build differs. `^` (xor) = exactly one; both-marginal (deep marginal)
        // and neither (pure structural) fall through to the general body.
        // Under a restriction the OUTPUT level of an off-`R` child is the
        // accumulator's own level (it is never copied into the fresh `levels`
        // array — the two are merged at the tail), so read through to
        // `c1.levels[..]`. Levels IN `R` are structural in the accumulator by
        // construction, so the extra disjunct is inert for them.
        let (left_marg_now, right_marg_now) = if restrict.is_some() {
            (levels[left_idx].is_marginal() || c1.levels[left_idx].is_marginal(),
             levels[right_idx].is_marginal() || c1.levels[right_idx].is_marginal())
        } else {
            (levels[left_idx].is_marginal(), levels[right_idx].is_marginal())
        };
        // Dispatch on AT LEAST ONE marginal child (OR, not XOR): the dedicated
        // path tags whichever side(s) are marginal and carries them verbatim, so
        // a both-marginal level (pure Σ left_count × right_count, no structural
        // product) is handled identically — and routing it here keeps it off the
        // legacy inline tagger that otherwise corrupts it under inline-emit.
        let marg_child_dispatch = left_marg_now || right_marg_now;

        // A streaming marginalize target collapses to Σ left × right per cell —
        // no downstream structure survives. Build each alive cell's scalar
        // directly from the surviving (lc, rc) refs, never materializing a
        // product node (see `run_level_rows_stream_count`). Covers EVERY
        // Route A streaming shape (any marginal-child pattern, integer and
        // weighted — D2 stage 1): which side is marginal is carried by
        // `stream_state` itself, and `MargLookup` degrades to a dense grid
        // read on a non-pass-through side. The streaming gate lives in
        // `stream_marginal_eligible` (`stream_state` is None when off), so
        // `stream_state.is_some()` alone decides here (D2 stage 2).
        let marg_stream_collapse = marg_child_dispatch && stream_state.is_some();

        // `t` and its two vtree children are three distinct nodes of a tree, so
        // `t_idx`, `left_idx` and `right_idx` name three disjoint level slots —
        // split them apart in one step. That is what lets the streaming row
        // loops read the child count columns IN PLACE while the output level is
        // exclusively borrowed; snapshotting them instead doubled a wide
        // marginal child's storage (an 8 GiB single alloc at 536M slots) at
        // exactly the moment streaming exists to relieve.
        //
        // The split's borrow of `levels` must end before the per-level tail
        // retakes it — every use of the three below is inside the row loop.
        let [level, left_level, right_level] = levels
            .get_disjoint_mut([t_idx, left_idx, right_idx])
            .expect("a vtree node and its two children are distinct level indices");
        let (left_level, right_level) = (&*left_level, &*right_level);

        // Pre-reserve capacity for OR nodes. `k1*k2` is the EXACT upper bound
        // (one node per live cell; compaction only removes dead ones), so
        // reserving it once replaces the Vec-doubling ladder the old
        // `max(k1, k2)` seed left behind. Capped at `LEVEL_RESERVE_CAP_BYTES`
        // because most levels are low-survival — beyond the cap the over-
        // allocation would dwarf the real node count, and growth past it
        // continues through the ordinary fallible `try_push` path. `finalize_level`
        // calls `shrink_arrays`, which hands the unused tail straight back.
        // Never below the old seed. Fallible: under tight budgets, even this
        // baseline reservation may exceed the remaining VAS.
        let nodes_reserve = k1
            .saturating_mul(k2)
            .min(LEVEL_RESERVE_NODES_CAP)
            .max(k1.max(k2));
        budget_reserve(&mut level.nodes, nodes_reserve)?;

        // ── Cell-build shared context ────────────────────────────────────
        //
        // Built here (before the route dispatch) so it can be shared by both
        // row-loop routes without re-stating its 15 fields in each site.  All
        // inputs are available at this point (widths/bases computed above,
        // masks/flags from `plan_marg_level`).
        //
        // Route A calls run_level_rows_marg (forward order, marg path).
        // Route B calls run_level_rows_plain (forward order, plain path).
        //
        // Resolve every c2 column ONCE for this level; `process_cell` (all
        // routes) then indexes the table instead of re-deriving column j's
        // slice on every row i. On identity-mask levels the descriptors are
        // zero-copy borrows of c2's own storage; on marg-mask levels they
        // point into a decode arena the table owns and budget-charges. `None`
        // (marginal-encoded c2, or the budget rejecting the arena) falls back
        // to the per-cell derivation — never worse than doing it per cell.
        let c2_cols = C2Columns::build(c2.level(t), k2, left_mask, right_mask);
        let cell_ctx = CellCtx {
            t_base, k2, left_base, right_base,
            k2_left, k2_right,
            left_passthrough, right_passthrough,
            left_pt_c1, right_pt_c1,
            nxm, left_mask, right_mask,
            live_left_cols: &nxm_masks.live_left_cols,
            reach_c2_left: &nxm_masks.reach_c2_left,
            live_right_cols: &nxm_masks.live_right_cols,
            reach_c2_right: &nxm_masks.reach_c2_right,
            c2_cols: c2_cols.as_ref(),
            deadline_armed: budget::apply_deadline_check_enabled(),
        };

        // ── Emit-growth mode decision ────────────────────────────────────
        // Exactly one disarm per level on every route: `decide_emit_growth_mode`
        // disarms on entry for the routes that call it, and the else-arm covers
        // the routes that skip it — so a previous level's near-cap bounded-growth
        // decision can never leak into this level's growth events.
        //
        // See `decide_emit_growth_mode` for the mode logic and cost rationale.
        // The mode decision only matters for the dense emit into `level.pairs`,
        // so it is skipped on the routes that don't do one: sparse-marg and
        // stream-collapse. Skipping is sound — the mode is a growth-policy
        // optimization, not a correctness step; the per-push budget checks in the
        // emit still apply.
        if !use_sparse_marg && !marg_stream_collapse {
            // THE per-level emit-pair bound: every product pair emits at most
            // once, so `|c1.pairs| × |c2.pairs|` bounds this level's emit. Used
            // twice — once to pick the growth mode, once to size the pairs
            // arena — computed once so the two can never disagree.
            let emit_pair_bound = (c1.level(t).pairs.len() as u128)
                .saturating_mul(c2.level(t).pairs.len() as u128);
            decide_emit_growth_mode(stream_marginal, emit_pair_bound);
            // Seed `level.pairs` at that bound instead of letting it double from
            // empty on every level. Same cap/rationale as the nodes reserve
            // above: low-survival levels would over-allocate wildly past it, so
            // the reserve stops there and the emit's own `try_push_pair_into`
            // choke point (still under the growth mode just armed) carries any
            // level that outgrows it; `shrink_arrays` reclaims the tail at
            // `finalize_level`. Only the routes that DO emit into `level.pairs`
            // reserve — the sparse-marg and stream-collapse routes never touch it.
            let pairs_reserve =
                emit_pair_bound.min(LEVEL_RESERVE_PAIRS_CAP as u128) as usize;
            let pre_pairs_cap = level.pairs.capacity();
            budget_reserve(&mut level.pairs, pairs_reserve)?;
            // Output-pair meter: this bulk seed is real arena capacity the
            // emit walk will not charge again. See `ApplyLimits::pairs_in_flight`.
            budget::account_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
        } else {
            set_pairs_bounded_growth(false);
        }

        // ── Cell-build row loop ──────────────────────────────────────────
        // Branch on the dedicated marginal-child path. See `cell_ctx` above.
        if use_sparse_marg {
            // Sparse bottom-up build (exactly-one-marginal-child, structural
            // output). Runs the shared emit kernel (`process_cell` + MargLookup),
            // but writes into the k2-row scratch (cell_ctx.t_base == row_base,
            // called with i=0 so grid_pos == row_base + j) and records each
            // surviving cell into the output product_list instead of a dense slab.
            let c1_level_t: &TddLevel = c1.level(t);
            let c2_level_t: &TddLevel = c2.level(t);
            run_level_rows_marg_sparse(
                k1,
                c1_level_t, c2_level_t, &cell_ctx,
                &mut inputs1_scratch, &mut inputs2_scratch,
                level, &mut node_idx,
                &mut product_lists[t_idx],
            )?;
            // Reclaim the transient k2-row scratch; the level is Sparse (its
            // product_list is the authoritative representation, densified lazily
            // by the grandparent's ensure_grid) and never reads the dense slab.
            grid_free(&mut free_regions, t_base, k2);
            // Reuse the split's `level` rather than reborrowing `levels[t_idx]`:
            // the child halves of the split are still in scope here.
            finish_sparse_output(&mut live_counts, &mut out_nodes_so_far, &mut has_pl, level, t_idx);
            // This route returns before `finalize_level`, so run its inline-emit
            // marking here. Exactly one side is the marginal pass-through (XOR gate).
            mark_passthrough_inlined(level, left_passthrough, right_passthrough);
            reclaim_child_grids(might_use_sparse, &mut grids, &mut free_regions, &c1_widths, &c2_widths, left_idx, right_idx);
            continue;
        }
        if marg_child_dispatch {
            // Route A: Dedicated marginal-parent build. A level with at least one
            // marginal child is built by a forward-order grid loop running the
            // SHARED cell kernel (`process_cell` with MargLookup sides) — the
            // exact per-cell logic the general path uses. Isolating marginal
            // levels here lets the general product-grid path (Route B) assume no
            // child is marginal.
            //
            // Refs are emitted in the exact encoding the general path produces, so
            // this level still relies on the end-of-apply tagger
            // (`tag_all_marg_side_slots`) for its final marginal-side counts —
            // identical to the general path.
            //
            // Both c1.level(t) and c2.level(t) are immutable borrows into their
            // respective levels vecs. `level` is a mutable borrow into the OUTPUT
            // `levels[t_idx]`, which is a separate allocation from c1 and c2.
            // We pre-take both operand borrows here while `level` is not yet live,
            // then re-lend them to the called function as plain `&TddLevel` refs.
            let c1_level_t: &TddLevel = c1.level(t);
            let c2_level_t: &TddLevel = c2.level(t);
            if marg_stream_collapse {
                // Streaming target → collapse at source, skipping product-node
                // materialization. The gate guarantees `stream_state` is Some;
                // the walker picks the integer or weighted fold from it.
                let left_marg = child_lookup::MargLookup::left(&cell_ctx);
                let right_marg = child_lookup::MargLookup::right(&cell_ctx);
                run_level_rows_stream_count(
                    k1,
                    c1_level_t, c2_level_t, &cell_ctx,
                    &mut inputs1_scratch, &mut inputs2_scratch,
                    &mut node_idx,
                    &left_marg, &right_marg,
                    stream_state.as_mut().unwrap(),
                    left_idx, right_idx, &vtree, left_level, right_level,
                    &stream_computed, &stream_computed_weights,
                )?;
            } else {
                // Non-streaming marginal-child level (`stream_state` None —
                // includes gate-off, which `stream_marginal_eligible` maps to
                // "don't stream"): plain materializing build. One-marginal-child
                // MARGINALIZE TARGETS are common (the ~4k-hit stage-T probe,
                // 2026-07-02) but always stream, so they take the collapse
                // walker above, never this arm.
                run_level_rows_marg(
                    k1,
                    c1_level_t, c2_level_t, &cell_ctx,
                    &mut inputs1_scratch, &mut inputs2_scratch,
                    level, &mut node_idx,
                )?;
            }
        } else {
        // No-marginal-leakage guard (tier-0, every build incl. release). Every
        // parent of a marginal child level is dispatched to Route A above, so
        // Route B must never see a *non-empty* marginal child. The general path's
        // marg-ref decode has been deleted on the strength of this invariant; the
        // assert stays always-on so a future leak aborts loudly instead of
        // silently miscounting.
        //
        // EXEMPTION — a `width()==0` marginal child is benign. The deleted decode
        // only mattered for a width>0 marginal child: the parent carries inline
        // MargRefs into its cells that the general grid path can't interpret. A
        // 0-width marginal level has NO cells, so the parent has NO refs into it
        // (`t` is itself 0-width over that child) and the row-loop — which indexes
        // the OUTPUT child grids, never the operand's marginal store — does no
        // work for it. It is semantically identical to a 0-width *non*-marginal
        // child. Such levels are a legitimate transient state minted by the apply
        // streaming commit (`commit_stream_state`) and leaf-marginalization, and
        // can disagree across two pool members (one went through `restrict`,
        // the other didn't) — which is exactly the `WS_FAST_REDUCE` conjoin that
        // used to trip this assert.
        let marg_wide = |lvl: &TddLevel| lvl.is_marginal() && lvl.width() > 0;
        cheap_assert!(
            !left_marg_now && !right_marg_now
                && !marg_wide(&c1.levels[left_idx]) && !marg_wide(&c1.levels[right_idx])
                && !marg_wide(&c2.levels[left_idx]) && !marg_wide(&c2.levels[right_idx]),
            "general product-grid path reached with a non-empty marginal child \
             (t_idx={t_idx} l={left_idx} r={right_idx}): the dedicated \
             marginal-parent dispatch was bypassed"
        );
        // Route B: Plain (non-marginal-child) row-loop.
        // Pre-take the operand borrows before `level` (a `&mut` into the OUTPUT
        // `levels`) is active. Both row drivers below read them immutably.
        let c1_level_t: &TddLevel = c1.level(t);
        let c2_level_t: &TddLevel = c2.level(t);
        // Branch hoist: when all four level-invariant guards hold, dispatch to the
        // simplified dense path (no streaming, no nxm, no passthrough). Otherwise
        // fall through to the general path.
        let plain_dense = stream_state.is_none()
            && !cell_ctx.nxm
            && !cell_ctx.left_passthrough
            && !cell_ctx.right_passthrough;
        // Dense child lookups. A DenseLookup inlines to the original
        // `get_unchecked` positional index.
        let left_dense = child_lookup::DenseLookup { base: cell_ctx.left_base, k2: cell_ctx.k2_left };
        let right_dense = child_lookup::DenseLookup { base: cell_ctx.right_base, k2: cell_ctx.k2_right };
        macro_rules! run_plain {
            ($dense:literal, $l:expr, $r:expr) => {
                run_level_rows_plain::<$dense, _, _>(
                    k1,
                    c1_level_t, c2_level_t, &cell_ctx,
                    &mut inputs1_scratch, &mut inputs2_scratch,
                    level, &mut node_idx,
                    $l, $r,
                )?
            };
        }
        if plain_dense {
            run_plain!(true, &left_dense, &right_dense);
        } else if stream_state.is_some() {
            // Streaming-marginal level with no marginal child (leaf children at
            // the lowest levels): collapse to Σ left × right per cell WITHOUT
            // building+truncating a product node — the Route B analogue of
            // `marg_stream_collapse`; the walker picks the integer or weighted
            // fold from `stream_state`. The general (non-plain-dense) arm
            // always sees two dense-served children, and streaming forces this
            // arm (`plain_dense` requires `stream_state` None), so the dense
            // lookups are correct. The streaming gate lives in
            // `stream_marginal_eligible`, so `stream_state` alone decides here.
            run_level_rows_stream_count(
                k1,
                c1_level_t, c2_level_t, &cell_ctx,
                &mut inputs1_scratch, &mut inputs2_scratch,
                &mut node_idx,
                &left_dense, &right_dense,
                stream_state.as_mut().unwrap(),
                left_idx, right_idx, &vtree, left_level, right_level,
                &stream_computed, &stream_computed_weights,
            )?;
        } else {
            run_plain!(false, &left_dense, &right_dense);
        }
        } // end `else` of marg_child_dispatch (Route B row loop)

        // Per-level tail: stream commit, live_counts, grid tag, shrink,
        // pass-through flags. See `finalize_level`.
        finalize_level(
            &mut stream_state,
            t, t_idx,
            t_base,
            might_use_sparse,
            left_passthrough, right_passthrough,
            &vtree, &mut levels, &mut grids, &mut live_counts, &mut out_nodes_so_far,
        );

        // Lever 6 supersedes Lever 5b end-of-iter drops — start-of-iter
        // drop on the NEXT iteration releases this iteration's children
        // before its reserve/scatter fires.
        reclaim_child_grids(might_use_sparse, &mut grids, &mut free_regions, &c1_widths, &c2_widths, left_idx, right_idx);
    }
    Ok(())
    })();
    loop_result?;

    // EQUAL-VALUE LEAF-REF CANONICALIZATION, apply-side mirror of
    // `marginalize::marginalize_leaf_weighted`'s pass and sharing its ONE walk.
    // Runs here, not at the flag site above: the parent level's pairs are emitted
    // by the bottom-up loop that just finished, so this is the first point at
    // which they are final.
    //
    // Scope is the leaves recorded above — flagged weight-marginal on ONE
    // operand's authority. The structural operand contributes leaf-side refs that
    // never passed through the canon map, and `CONJOIN_GRID` carries them into the
    // output unchanged wherever the marginal side reads `One`. Rewriting them onto
    // the canonical slot of their value class is value-preserving (same column
    // entry) and is what lets the contraction that follows this apply see the
    // parent's `(·, Pos)` / `(·, Neg)` branches as twins.
    {
        use crate::tdd::transform::unary::marginalize as marg;
        for &li in &canon_leaves {
            let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(li as u32)) else { continue };
            let Some(parent) = vtree.node(VtreeIdx(li as u32)).parent() else { continue };
            // A marginal parent folded the leaf's bases into its own aggregate — no
            // leaf-side pairs remain to rewrite (same guard as the marginalize pass).
            if levels[parent.idx()].is_marginal() {
                continue;
            }
            // Exact domain only: `WeightKey::Log` compares `f64` bit patterns, so
            // "equal" there is representation identity, not value identity.
            let canon = marg::with_weight_ctx(|ws| {
                (!ws.is_log()).then(|| marg::leaf_canon_map(&marg::leaf_column_vals(ws, var)))
            });
            let Some(canon) = canon else { continue };
            if canon == [0, 1, 2] {
                continue; // no equal-valued slots — the walk would rewrite nothing
            }
            let (pl, _) = vtree.children(parent);
            marg::canonicalize_leaf_refs_at_parent(
                &mut levels[parent.idx()],
                pl.idx() == li,
                &canon,
            );
        }
    }

    let out_local = compute_apply_output(
        c1, c2, &grids, &node_idx, &c2_widths,
        &c1_identity, &c2_identity, &has_pl, &product_lists,
    );
    let out_vtree = c1.output.vtree;

    // Stale-grid FALSE guard. A real (materializing) conjoin whose product is
    // FALSE leaves the output level with zero materialized slots, but
    // `compute_apply_output`'s grid branch reads a stale `node_idx` cell (0,
    // never DEAD-filled in sparse mode) and returns local 0 — an index past the
    // level's slot count. `prune`/`minimize` then index one-past-end of the
    // `remap` arena and panic (prune.rs classic_mark). An `out_local` that
    // cannot fit in the root level's *effective* width is exactly that stale
    // read: the product is FALSE, so emit ZERO (prune early-returns on is_zero,
    // sidestepping the OOB). A valid result always indexes an existing slot
    // (out_local < width), and constant-TRUE keeps width ≥ 1 at every internal
    // level, so this never misfires on a true result.
    //
    // Mirror prune's indexing exactly: `effective_width` (LEAF_WIDTH for leaves,
    // marginal_counts.len() for marginal levels, node count otherwise) — NOT raw
    // `width()` — so the guard fires on the same one-past-end that prune would,
    // including marginal roots under WS_MARGINALIZE and leaf roots.
    let out_local = {
        let out_ti = out_vtree.idx();
        let eff_width = if vtree.node(VtreeIdx(out_ti as u32)).is_leaf() {
            crate::tdd::types::LEAF_WIDTH
        } else {
            levels[out_ti].width()
        };
        if out_local != ZERO && (out_local.0 as usize) >= eff_width {
            ZERO
        } else {
            out_local
        }
    };

    apply_and_finalize(
        node_idx, grids, c2_identity, c1_identity,
        product_lists, live_counts, has_pl,
        c1_widths, c2_widths,
        stream_computed,
        stream_computed_weights,
        inputs1_scratch,
        inputs2_scratch,
        nxm_masks,
        marginalize_targets,
    );

    let output = TddNodeId { vtree: out_vtree, local: out_local };
    let Some(r) = restrict else {
        return Ok(Tdd::with_levels(vtree, levels, output));
    };

    // ── Restricted tail: merge `R` back into the accumulator's array ─────
    //
    // Every level OFF `R` rode through untouched — it is still the
    // accumulator's own level, byte-for-byte, in the accumulator's own
    // allocation. Moving the `|R|` rebuilt levels across is the whole
    // "output" step; the fresh array goes back to the pool with only empty
    // levels in it (the leaf-marginal seeding sweep, the one other writer,
    // is skipped under a restriction).
    //
    // SWAP rather than assign: the accumulator's superseded level at `t` goes
    // back into the fresh array, which is what heads to the level pool a few
    // lines later. Its `nodes`/`pairs` arenas are then reused by the next
    // merge's rebuild instead of being freed here and reallocated there.
    for &t in r.rebuild {
        let ti = t.idx();
        std::mem::swap(&mut c1.levels[ti], &mut levels[ti]);
    }
    std::mem::swap(&mut c1.levels, &mut levels);

    // Contract seed: `R`, not every internal level. `R` is ancestor-closed
    // (`S` is; `AncClosure(P)` is by construction), so its complement is
    // DESCENDANT-closed — an off-`R` level's own pairs, its parent's pairs and
    // its whole subtree are bit-identical to the accumulator's. A contraction
    // sweep seeded at an off-`R` level would therefore re-run the accumulator's
    // own last sweep on the same bytes and fire nothing. Whatever the
    // accumulator still owed is carried over rather than dropped
    // (`with_levels_dirty`'s obligation 2). Same argument, same shape, as
    // `conjoin_clause::try_apply_and_clause`'s seed.
    let mut dirty_contract = std::mem::take(&mut c1.dirty_contract);
    let mut dirty_leaf_contract = std::mem::take(&mut c1.dirty_leaf_contract);
    dirty_contract.reserve(r.rebuild.len());
    dirty_leaf_contract.reserve(r.rebuild.len());
    for &t in r.rebuild {
        dirty_contract.push(t.0);
        dirty_leaf_contract.push(t.0);
    }
    Ok(Tdd::with_levels_dirty(vtree, levels, output, dirty_contract, dirty_leaf_contract))
}

// Specialized TDD × clause conjunction lives in the sibling module
// `super::conjoin_clause` (declared in `transform/pairwise/mod.rs`); it reaches
// this module's `ApplyError`/`DEAD`/`try_push`/`try_resize_dead2` via their
// crate-visible re-exports above.

/// Conjoin two TDDs that share the same vtree.
///
/// Both operands are CONSUMED: the algorithm drains their level arenas as it
/// walks bottom-up and recycles the storage into the result. Clone one first if
/// you need to keep it.
///
/// Infallible: an allocation refusal panics. Use [`try_apply_and`] to recover,
/// or to marginalize while conjoining.
///
/// A deadline shield (`apply_limits().deadline(None)`) is installed for the
/// call's lifetime and the prior deadline restored on drop, so a vtree-level
/// deadline check cannot surface as `Err(Deadline)` inside the `expect` below
/// and panic. Auxiliary applies are bounded constructions meant to run to
/// completion; only the fallible entry honors the deadline.
///
/// # Panics
///
/// Panics on allocator OOM (`ApplyError::OverBudget`).
pub fn apply_and(c1: Tdd, c2: Tdd) -> Tdd {
    let _shield = apply_limits().deadline(None).apply();
    try_apply_and(c1, c2, None)
        .expect("apply_and: allocator OOM in infallible entry — use try_apply_and to recover")
}

/// Conjoin two TDDs that share the same vtree, reporting a refusal instead of
/// panicking on it. The production conjunction entry.
///
/// Both operands are CONSUMED — the algorithm drains their level arenas as it
/// goes and recycles the storage into the result — on `Err` as well as on `Ok`.
/// Clone one first if you need to keep it, and never reuse an operand after a
/// call.
///
/// `marginalize_targets`, when given, names the vtree nodes whose levels the
/// bottom-up loop should emit as streaming-marginal instead of explicit.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a buffer reservation is refused
/// (allocator failure or the configured soft budget would be exceeded),
/// `Err(ApplyError::OutputCap)` on the output-node cap, or
/// `Err(ApplyError::Deadline)` on the scoped deadline or an armed decision
/// callback that concluded the compile should stop.
pub fn try_apply_and(
    mut c1: Tdd,
    mut c2: Tdd,
    marginalize_targets: Option<&[bool]>,
) -> Result<Tdd, ApplyError> {
    // Checked before the swap and the self-conjunction shortcut, both of which
    // can return without ever reaching `apply_and_fallible_inner`.
    assert!(
        Arc::ptr_eq(&c1.vtree, &c2.vtree),
        "apply_and requires TDDs with the same vtree"
    );
    assert_eq!(
        c1.output.vtree, c2.output.vtree,
        "apply_and requires TDDs with outputs at the same vtree node"
    );
    // Operand swap: make c2 the narrower operand. The c2-identity fast path
    // checks k2 == 1 first — the narrower operand is more likely to have
    // width 1 at subtree levels, skipping more product constructions.
    // Secondary benefit: shorter grid rows (width k2) improve cache locality.
    //
    // Kept HERE (owned path only), NOT pushed down into `apply_and_fallible`:
    // the borrowed path has order-sensitive callers that must not be swapped.
    // See the note in `apply_and_fallible`.
    //
    // NB (2026-06-13): the reframe proposed flipping this
    // to put the NARROWER operand on c1 (the held-flat inputs1 side) to shrink
    // the dense read-side peak. REFUTED empirically — a gated flip raised peak
    // RSS by +5.9%/+12.3% on the big-transient instances (033/052) and was
    // neutral on 080, never better. The premise is false: inputs1 is decoded
    // per-c1-*node* (`pairs_view_decoded` in the row loops, cell.rs), so operand max_width
    // (a node count) does not size the held buffer; and this c2=narrower
    // orientation is already the RSS-better one via the k2==1 fast-path. Keep
    // the compute-orientation as-is. See grouped-pairs THREAD for the data.
    if c2.max_width() > c1.max_width() {
        std::mem::swap(&mut c1, &mut c2);
    }
    // Self-conjunction: f ∧ f = f. Owned variant avoids the clone in apply_and.
    if is_self_conjunction(&c1, &c2) {
        types::return_levels2(std::mem::take(&mut c2.levels));
        return Ok(c1);
    }
    let result = apply_and_fallible(&mut c1, &mut c2, marginalize_targets);
    types::return_levels(std::mem::take(&mut c1.levels));
    types::return_levels2(std::mem::take(&mut c2.levels));
    result
}

#[cfg(test)]
#[path = "../apply_tests.rs"]
mod apply_tests;
