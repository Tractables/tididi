//! Dense cell/row engine for the apply product construction.
//!
//! Contains the merged per-cell product-walk kernel (`process_cell`, generic
//! over a [`ChildLookup`] per side and a [`PairSink`] action), the ONE row-loop
//! driver behind every build route (`run_level_rows`, generic over a
//! [`CellAction`]) with its four route entry points (`run_level_rows_marg`,
//! `run_level_rows_marg_sparse`, `run_level_rows_stream_count`,
//! `run_level_rows_plain`), the streaming per-cell folds ([`StreamCellFold`]),
//! and the product-node emitter (`emit_product_node`).
//!
//! Also houses `CellCtx` (the per-level loop-invariant context struct) and
//! the row-mask fold (`row_alive_masks`). The amortized between-cell
//! cancel/deadline poll is `budget::PollTicker` (shared with the sparse
//! join); see `nxm_deadline_check!` below for the deliberate intra-cell
//! exception that is NOT part of that shared ticker.

use crate::tdd::types::{InputPair, TddLevel, TddNodeData, ExtMulti, LocalNodeIdx,
    MAX_LEVEL_ARENA_BYTES};
use crate::tdd::counts::{ApplyBudget, CountVec, IntFold, WeightFold};
use crate::tdd::query::WeightVal;
use crate::tdd::utils::{pool_put_bounded, pool_take};
use super::{
    ApplyError, DEAD,
    try_push, try_push_pair_into,
};
use crate::tdd::limits::{budget_reserve_exact, unaccount_transient_bytes};
use super::stream::{attach_children, StreamLevelState, StreamPayload, StreamState};
use super::child_lookup::{ChildLookup, MargLookup};
use super::sparse::{ProductEntry, C1NodeIdx, C2NodeIdx, ProdNodeIdx};

/// Amortized wall-clock deadline cut for the intra-cell N×M product loops.
///
/// The per-cell `budget::PollTicker` check in the row loops fires only *between*
/// `process_cell` calls. A single very wide cell can push or count hundreds of
/// millions of (p1,p2) pairs *inside one call*, so a branch deadline
/// (projected-cutset conditioning) can't slice it — the apply grinds
/// minutes/GiB on one cell (track-3 029). This bumps a per-outer-iteration work
/// accumulator and, every ~1M pairs of work, consults the gated `ApplyLimits::deadline`.
///
/// "Outer iteration" is per arm: the N×M arms bump once per c1 pair (by the c2
/// pair count), while the N×1 / 1×N arms have a single sweep whose pair count is
/// known up front, so they bump ONCE before it. Either way the poll costs at
/// most one branch per outer iteration and never one per pair.
///
/// `$armed` is [`CellCtx::deadline_armed`]: whether any stop axis is installed,
/// hoisted once per level, so with none installed this is a single
/// predicted-not-taken local-bool branch per OUTER iteration. On expire it
/// returns `Err(Deadline)`; the partial level is discarded with the aborted
/// apply, never counted.
///
/// Deliberate exception to `PollTicker` (the shared between-cell / sparse
/// ticker): this poll is intra-cell, so it needs a counter the ticker's
/// per-cell granularity cannot give it.
macro_rules! nxm_deadline_check {
    ($armed:expr, $work:expr, $inc:expr) => {
        if $armed {
            $work += ($inc) as u64;
            if $work >= (1u64 << 20) {
                // A TEE of the work this loop already counted, not a second
                // counter: this is the amortization point that already exists
                // for reading it. Charge what the meter HELD, not the cadence it
                // crossed — a single `$inc` can be many cadences wide, and
                // pricing it as one would undercount exactly the large cells the
                // give-up rule exists to catch.
                crate::tdd::limits::charge_compile_work($work);
                $work = 0;
                if crate::tdd::limits::deadline_expired() {
                    return Err(ApplyError::Deadline);
                }
            }
        }
    };
}

#[cfg(test)]
thread_local! {
    /// Test-only override for `bothmarg_collapse_enabled`, scoped by
    /// `with_bothmarg_collapse_forced`.
    static BOTHMARG_COLLAPSE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with the streaming gate forced to `enabled` on this thread, so
/// the gate-off parity test can exercise the no-streaming fallback.
#[cfg(test)]
pub(crate) fn with_bothmarg_collapse_forced<T>(enabled: bool, body: impl FnOnce() -> T) -> T {
    crate::tdd::scoped::Scoped::run(&BOTHMARG_COLLAPSE_OVERRIDE, Some(enabled), body)
}

/// Streaming gate for a level whose operands are both marginal: always on
/// (the alternative — materialize, then marginalize after the apply — is
/// count-identical at a higher peak, and exists only as the test comparison).
pub(super) fn bothmarg_collapse_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = BOTHMARG_COLLAPSE_OVERRIDE.with(|c| c.get()) {
        return forced;
    }
    true
}

/// Per-level loop-invariant context passed to all cell/row processing functions.
pub(super) struct CellCtx<'a> {
    /// Flat base offset of the output level's grid in `node_idx`.
    pub t_base: usize,
    /// Number of c2 nodes at this level (column count of the product grid).
    pub k2: usize,
    /// Flat base offset of the left child's grid in `node_idx`.
    pub left_base: usize,
    /// Flat base offset of the right child's grid in `node_idx`.
    pub right_base: usize,
    /// c2 column count for the left child grid.
    pub k2_left: u32,
    /// c2 column count for the right child grid.
    pub k2_right: u32,
    /// True when the left child carries a pass-through marginal field
    /// (one operand is identity at left, the other is the marginal carrier).
    pub left_passthrough: bool,
    /// True when the right child carries a pass-through marginal field.
    pub right_passthrough: bool,
    /// Carrier selector for left pass-through: true ⇒ carry `p1.left`, i.e. c1
    /// is the carrier. "Carrier" is defined in `marg_plan`.
    pub left_pt_c1: bool,
    /// Carrier selector for right pass-through: true ⇒ carry `p1.right`.
    pub right_pt_c1: bool,
    /// True when both operands have multi-pair nodes (NxM dead-pair pre-filter active).
    pub nxm: bool,
    /// `limits::any_stop_armed()`, read once per level: with nothing installed
    /// the intra-cell poll is a local-bool test. Hoisting is value-identical
    /// because no poll inside a level can install the first stop axis.
    pub deadline_armed: bool,
    /// Decode mask for the left child's pair fields: `MARG_VALUE_MASK` when that
    /// child is marginal, so a bit-30 inline tag is stripped and the remaining
    /// payload is read as a coordinate; `u32::MAX` otherwise. See
    /// `MARG_OVERFLOW_TAG` for the encoding.
    pub left_mask: u32,
    /// Decode mask for the right child's pair fields, as `left_mask`.
    pub right_mask: u32,
    /// Per-c1-row live-column bitmasks (indexed by c1 left-child node idx).
    pub live_left_cols: &'a [u128],
    /// Per-j reach bitmasks for c2's left child references.
    pub reach_c2_left: &'a [u128],
    /// Per-c1-row live-column bitmasks for the right child.
    pub live_right_cols: &'a [u128],
    /// Per-j reach bitmasks for c2's right child references.
    pub reach_c2_right: &'a [u128],
    /// Per-level c2 column table — every column's pair slice resolved ONCE
    /// (see [`C2Columns`]). `Some` on every level the table could be built
    /// for; `None` ⇒ `process_cell` re-derives column `j`'s slice per cell,
    /// as before (marginal-encoded c2 level, or the marg arena declined by
    /// the budget).
    pub c2_cols: Option<&'a C2Columns>,
}

mod columns;
pub(super) use columns::{C2Columns, ColSlice};
mod kernel;
pub(super) use kernel::*;
mod rows;
pub(super) use rows::*;

#[cfg(test)]
#[path = "../cell_tests.rs"]
mod tests;
