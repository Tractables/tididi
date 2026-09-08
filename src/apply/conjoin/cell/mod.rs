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
//! cancel/deadline poll is a shared [`PollGate`]; the intra-cell N×M arm
//! keeps a gate of its own, at a finer cadence.

use crate::diagram::{InputPair, TddLevel, TddNodeData, ExtMulti, LocalNodeIdx,
    MAX_LEVEL_ARENA_BYTES};
use crate::counts::{ApplyBudget, CountVec, IntFold, WeightFold};
use crate::query::WeightVal;
use crate::utils::{pool_put_bounded, pool_take};
use crate::engine::Limits;
use super::{ApplyError, DEAD, try_push_pair_into};
use super::stream::{attach_children, StreamLevelState, StreamPayload, StreamState};
use super::child_lookup::{ChildLookup, MargLookup};
use super::sparse::{ProductEntry, C1NodeIdx, C2NodeIdx, ProdNodeIdx};

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
    crate::scoped::Scoped::run(&BOTHMARG_COLLAPSE_OVERRIDE, Some(enabled), body)
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
    pub c2_cols: Option<&'a C2Columns<'a>>,
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
