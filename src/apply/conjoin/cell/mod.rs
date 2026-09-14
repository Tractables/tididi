//! Dense cell/row engine for the apply product construction.
//!
//! Contains the merged per-cell product-walk kernel (`process_cell`, generic
//! over a [`ChildLookup`] per side and a [`PairSink`] action), the one row-loop
//! driver behind every build route (`run_level_rows`, generic over a
//! [`CellAction`]) with its four route entry points (`run_level_rows_marginal`,
//! `run_level_rows_marginal_sparse`, `run_level_rows_stream_count`,
//! `run_level_rows_plain`), the streaming per-cell folds ([`StreamCellFold`](rows_stream::StreamCellFold)),
//! and the product-node emitter (`emit_product_node`).
//!
//! Also houses `CellCtx` (the per-level loop-invariant context struct) and
//! the row-mask fold (`row_alive_masks`). One [`PollGate`](crate::limits::PollGate) serves the whole
//! level: every cell charges the pairs it walks and the row loop adds a unit
//! per cell, so the work clock counts pairs and the stop axis is asked mid-cell
//! on a cell wide enough to need it.

use crate::diagram::{ChildPair, TddLevel, EncodedNode, MultiPairRange};
use crate::value::{IntFold, WeightFold};
use crate::engine::Engine;

mod rows_stream;
pub(crate) use rows_stream::run_level_rows_stream_count;
use super::{OperationError, NO_PRODUCT, try_push_pair_into};
use super::streaming_marginal::{attach_children, StreamEnv, StreamLevelState, StreamState};
use crate::value::ValueDomain;
use super::child_lookup::{ChildLookup, MarginalLookup};
use super::marginal_plan::SidePlan;
use crate::diagram::Sides;
use super::sparse::{ProductEntry, LeftNodeIdx, RightNodeIdx, ProductNodeIdx};

/// Everything the cell walk needs about one child side of a level: how the
/// side is read ([`SidePlan`]), where its product grid lives, and its dead-pair
/// liveness masks.
#[derive(Clone, Copy)]
pub(super) struct ChildPlan<'a> {
    /// Carrier and decode for this side.
    pub(crate) plan: SidePlan,
    /// Flat base offset of this child's grid in `node_idx`.
    pub(crate) base: usize,
    /// g column count for this child's grid: the distance between the starts of
    /// two consecutive rows.
    pub(crate) stride: u32,
    /// Per-f-row live-column bitmasks (indexed by f child node idx).
    pub(crate) live_cols: &'a [u128],
    /// Per-j reach bitmasks for g's references to this child.
    pub(crate) reach: &'a [u128],
}

/// Per-level loop-invariant context passed to all cell/row processing functions.
pub(super) struct CellCtx<'a> {
    /// Flat base offset of the output level's grid in `node_idx`.
    pub(crate) output_grid_base: usize,
    /// Number of g nodes at this level (column count of the product grid).
    pub(crate) right_width: usize,
    /// True when both operands have multi-pair nodes (dead-pair pre-filter active).
    pub(crate) both_multi_pair: bool,
    /// The two child sides. The kernel reaches them as `.left` / `.right`
    /// only — never by a runtime `Side`, which would put a branch in the walk.
    pub(crate) sides: Sides<ChildPlan<'a>>,
    /// Per-level g column table — every column's pair slice resolved once
    /// (see [`RightColumns`]). `Some` on every level the table could be built
    /// for; `None` ⇒ `process_cell` re-derives column `j`'s slice per cell
    /// (marginal-encoded g level, or the marginal arena declined by the
    /// budget).
    pub(crate) right_cols: Option<&'a RightColumns<'a>>,
}

mod columns;
pub(super) use columns::{RightColumns, ColumnSlice};
mod kernel;
pub(super) use kernel::*;
mod rows;
pub(super) use rows::*;

#[cfg(test)]
mod tests;
