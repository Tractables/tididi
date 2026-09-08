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
//! Mirrors `ensure_counts` / `read_marginal_count` but operates on
//! a `&[TddLevel]` slice — apply_and's output is still being built, so we
//! can't hand a finished `Tdd` to the existing helpers.
//!
//! # One driver, two value kinds
//!
//! Integer and weighted streaming share ONE driver.
//! The value-kind axis is [`crate::tdd::counts::MargFold`] (scalar + column
//! contract, `counts.rs`) extended here by [`StreamPayload`], which adds the
//! four things the apply-side driver needs and that genuinely differ between
//! the two kinds:
//!
//! | hook | why it must stay per-kind |
//! |---|---|
//! | `fold_node` | the child READERS differ (lazy `CountRead` vs `Cow<WeightVal>`); `MargRef::Inline` is integer-side only |
//! | `child_view` | the borrow shape and the marginal-LEAF semantics differ (integer: always a `CountRef` into the child's storage, fixed `[2,1,1]` slots for an empty inline store; weighted: `Cow`, since the `WeightStore` column and the semiring leaf bases can only be produced owned) |
//! | `fold_cell` | integer carries the u128-fast-path/`BigUint`-overflow discipline; rationals cannot overflow, so the weighted fold is a single clean pass |
//! | `store_level` | integer commits raw `(fast, big)` arrays into the level (no reshaping — both sides hold the same sparse side table); weighted commits slot count + `WeightStore` payload |
//!
//! Everything else — the ensure walk, the descendant cascade, the state
//! build, the per-cell push/remap, the commit precondition — is written once,
//! generic over `F: StreamPayload`, and monomorphized at the ONE runtime
//! branch in [`build_stream_state`] (and its mirror in
//! `cell::run_level_rows_stream_count`, the row-loop dispatch point).

use crate::vtree::VtreeIdx;
use crate::tdd::types;
use crate::tdd::types::{decode_marg_coord, MargRef, MARG_VALUE_MASK, MARG_OVERFLOW_TAG};
use crate::tdd::query::WeightVal;
use crate::tdd::weight_store::WeightStore;
use super::{ApplyError, budget_reserve_exact, TddLevel, InputPair, LeafLabel};
use super::cell::bothmarg_collapse_enabled;

pub(crate) use crate::tdd::counts::STREAM_OVERFLOW;
use crate::tdd::counts::{
    ensure_fold_walk, ApplyBudget, ColumnRetention, Count, CountRead, CountRef, CountVec, IntFold,
    MargFold, WeightFold,
};

mod fold;
pub(crate) use fold::*;
mod count;
pub(crate) use count::*;
mod weight;
mod level;
pub(crate) use level::*;
