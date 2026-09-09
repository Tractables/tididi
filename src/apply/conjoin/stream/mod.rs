//! Streaming-marginal emit support for the scheduled `apply_and_fallible` path.
//!
//! When `marginalize_targets[t_idx]` is true during the bottom-up scatter, the
//! dense path emits the output level as `marginal_counts` directly — for each
//! alive cell, build the pairs into `level.pairs` (so process_cell! is reused
//! verbatim), compute `Σ counts_left[lc] * counts_right[rc]`, then truncate
//! the pairs/node away before the next cell. Peak transient is bounded by the
//! largest single cell, not Σ pairs. The post-apply `marginalize_batch` sees
//! the level as already-marginal and skips it.
//!
//! The same fold the marginalization cascade runs, on a `&[TddLevel]` slice:
//! apply's output is still being built, so there is no finished `Tdd` to hand
//! the cascade's entry points.
//!
//! # One driver, two value domains
//!
//! Integer and weighted streaming share ONE driver, written against
//! [`crate::value_fold::ValueDomain`] — the same contract the cascade uses, so
//! a domain answers each question once for both. Everything here — the state
//! build, the per-cell push/remap, the commit precondition — is generic over
//! `F: ValueDomain` and monomorphized at the ONE runtime branch in
//! [`build_stream_state`] (and its mirror in
//! `cell::run_level_rows_stream_count`, the row-loop dispatch point).

use crate::diagram;
use crate::diagram::{NodeIdx, SideView, ValueRef};
use crate::diagram::WeightVal;
use crate::diagram::WeightStore;
use crate::engine::Engine;
use super::{ApplyError, TddLevel, InputPair};

pub(crate) use crate::value_fold::STREAM_OVERFLOW;
use crate::value_fold::{
    ColumnRetention, Count, CountRef, CountVec, InternalLevel, IntFold, StreamChild,
    ValueDomain, WeightFold,
};
use crate::diagram::Tdd;
use crate::engine::{RecoveryPanic, ReservePolicy};
use crate::vtree::VtreeIdx;
use crate::engine::ApplyBudget;

mod cache;
pub(crate) use cache::StreamCache;
mod fold;
pub(crate) use fold::*;
mod count;
pub(crate) use count::*;
mod weight;
mod level;
pub(crate) use level::*;
