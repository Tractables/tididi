//! Streaming-marginal emit support for the scheduled `apply_and_fallible` path.
//!
//! When `marginalize_targets[t_idx]` is true during the bottom-up scatter, the
//! dense path emits the output level as `marginal_counts` directly — for each
//! alive cell, build the pairs into `level.pairs` (so `process_cell` is reused
//! unchanged), compute `Σ counts_left[lc] * counts_right[rc]`, then truncate
//! the pairs/node away before the next cell. Peak transient is bounded by the
//! largest single cell, not Σ pairs. The post-apply `marginalize_batch` sees
//! the level as already-marginal and skips it.
//!
//! The same fold the marginalization cascade runs, on a `&[TddLevel]` slice:
//! apply's output is still being built, so there is no finished `Tdd` to hand
//! the cascade's entry points.
//!
//! Integer and weighted streaming share one driver, generic over
//! [`crate::value::ValueDomain`]; the value kind is chosen at runtime in
//! [`build_stream_state`] and in `cell::run_level_rows_stream_count`.

use crate::diagram;
use crate::diagram::{NodeIdx, SideView, ValueRef};
use crate::diagram::WeightVal;
use crate::diagram::WeightStore;
use crate::engine::Engine;
use super::{ApplyError, TddLevel, InputPair, Sides};

pub(crate) use crate::value::COUNT_OVERFLOW;
use crate::value::{
    ColumnRetention, Count, CountRef, CountVec, FoldInput, FoldScope, InternalLevel, IntFold, StreamChild,
    ValueDomain, WeightFold,
};
use crate::diagram::Tdd;
use crate::limits::{RecoveryPanic, ReservePolicy};
use crate::vtree::VtreeIdx;
use crate::limits::ApplyBudget;

use crate::value::StreamCache;
mod fold;
pub(crate) use fold::*;
mod count;
mod weight;
mod level;
pub(crate) use level::*;
