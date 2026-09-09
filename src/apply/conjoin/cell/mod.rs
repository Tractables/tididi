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
//! the row-mask fold (`row_alive_masks`). One [`PollGate`] serves the whole
//! level: every cell charges the pairs it walks and the row loop adds a unit
//! per cell, so the work clock counts pairs and the stop axis is asked mid-cell
//! on a cell wide enough to need it.

use crate::diagram::{InputPair, TddLevel, TddNodeData, MultiPairRange, NodeIdx,
    MAX_LEVEL_ARENA_BYTES};
use crate::value_fold::{IntFold, WeightFold};
use crate::engine::ApplyBudget;
use crate::engine::Engine;

mod rows_stream;
pub(crate) use rows_stream::run_level_rows_stream_count;
use super::{ApplyError, DEAD, try_push_pair_into};
use super::stream::{attach_children, StreamLevelState, StreamState};
use crate::value_fold::ValueDomain;
use super::child_lookup::{ChildLookup, MargLookup};
use super::marg_plan::{SidePlan, Sides};
use super::sparse::{ProductEntry, LeftNodeIdx, RightNodeIdx, ProductNodeIdx};

#[cfg(test)]
thread_local! {
    /// Test-only override for `both_marginal_collapse_enabled`, scoped by
    /// `with_bothmarg_collapse_forced`.
    static BOTHMARG_COLLAPSE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with the streaming gate forced to `enabled` on this thread, so
/// the gate-off parity test can exercise the no-streaming fallback.
#[cfg(test)]
pub(crate) fn with_bothmarg_collapse_forced<T>(enabled: bool, body: impl FnOnce() -> T) -> T {
    crate::thread_local_override::Scoped::run(&BOTHMARG_COLLAPSE_OVERRIDE, Some(enabled), body)
}

/// Streaming gate for a level whose operands are both marginal: always on
/// (the alternative — materialize, then marginalize after the apply — is
/// count-identical at a higher peak, and exists only as the test comparison).
pub(super) fn both_marginal_collapse_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = BOTHMARG_COLLAPSE_OVERRIDE.with(|c| c.get()) {
        return forced;
    }
    true
}

/// Everything the cell walk needs about ONE child side of a level: how the
/// side is read ([`SidePlan`]), where its product grid lives, and its NxM
/// liveness masks.
#[derive(Clone, Copy)]
pub(super) struct ChildPlan<'a> {
    /// Carrier and decode for this side.
    pub plan: SidePlan,
    /// Flat base offset of this child's grid in `node_idx`.
    pub base: usize,
    /// g column count for this child's grid (its row stride).
    pub right_width: u32,
    /// Per-f-row live-column bitmasks (indexed by f child node idx).
    pub live_cols: &'a [u128],
    /// Per-j reach bitmasks for g's references to this child.
    pub reach: &'a [u128],
}

/// Per-level loop-invariant context passed to all cell/row processing functions.
pub(super) struct CellCtx<'a> {
    /// Flat base offset of the output level's grid in `node_idx`.
    pub output_grid_base: usize,
    /// Number of g nodes at this level (column count of the product grid).
    pub right_width: usize,
    /// True when both operands have multi-pair nodes (NxM dead-pair pre-filter active).
    pub both_multi_pair: bool,
    /// The two child sides. The kernel reaches them as `.left` / `.right`
    /// only — never by a runtime `Side`, which would put a branch in the walk.
    pub sides: Sides<ChildPlan<'a>>,
    /// Per-level g column table — every column's pair slice resolved ONCE
    /// (see [`RightColumns`]). `Some` on every level the table could be built
    /// for; `None` ⇒ `process_cell` re-derives column `j`'s slice per cell,
    /// as before (marginal-encoded g level, or the marg arena declined by
    /// the budget).
    pub c2_cols: Option<&'a RightColumns<'a>>,
}

mod columns;
pub(super) use columns::{RightColumns, ColumnSlice};
mod kernel;
pub(super) use kernel::*;
mod rows;
pub(super) use rows::*;

#[cfg(test)]
#[path = "../cell_tests.rs"]
mod tests;
