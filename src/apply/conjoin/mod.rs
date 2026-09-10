//! Conjunction (AND) of two diagrams via the compacting product construction.
//!
//! Given two diagrams over the same vtree, produces a diagram for their conjunction.
//! The product construction pairs every node from f with every node from g
//! at each vtree level, computes the conjunction of their input pairs, and
//! omits dead nodes (compaction). See `docs/tdd.md` for details.

use std::sync::Arc;

use crate::vtree::VtreeIdx;
use super::grid::LevelGrid;
use super::leaf::CONJOIN_GRID;
use crate::diagram::{self, *};

pub(crate) mod budget;
mod child_lookup; // Representation-specialized child lookups (sparse-conjunction kernels)
use crate::engine::Engine;
use crate::error::ApplyError;
use budget::*;

mod cell;
use cell::{
    RightColumns, CellCtx, ChildPlan, ColumnSlice,
    run_level_rows_marginal, run_level_rows_marginal_sparse, run_level_rows_plain,
    run_level_rows_stream_count,
};

mod sparse;
pub(crate) use sparse::{reset_sparse_ws, SparseWorkspace};
use sparse::{ProductEntry, is_self_conjunction, apply_sparse_level, apply_leaf_levels, compute_apply_output, release_sparse_ws_if_large};

// Identity/constant-true detection + per-level identity fast paths (extracted).
mod identity;
use identity::{take_level_fast_path, FastPathResult};
#[cfg(debug_assertions)]
use identity::marginal_schedule_dump;
// Consumed only by the `apply_tests` submodule's `use super::*` glob (marginal
// constant-true unit tests); production callers live inside `identity`.
#[cfg(test)]
use identity::level_marginal_is_constant_true;

// Apply setup (phases 1-3) → `ApplyRun` (extracted).
mod setup;
use setup::{apply_and_setup, ApplyRun, LevelShape};

// Per-level marginal classification plan + dead-pair masks (extracted).
pub(crate) mod marginal_plan;
use marginal_plan::{MarginalPlan, SidePlan, Sides, plan_marginal_level, build_side_masks};

// MergeScope-bounded ("restricted") apply: the O(spine) batch merge.
mod restrict;
pub use restrict::{conjoin_batch, BatchMergeOutcome, MergeScope};
use restrict::Restrict;
pub(crate) use restrict::RestrictScratch;

mod scratch;
pub(crate) use scratch::ApplyScratch;
pub(crate) mod plan;
pub(crate) mod targets;
use targets::MarginalTargets;
mod route;
use route::*;
use plan::ApplyPlan;
mod grid_arena;
pub(in crate::apply::conjoin) use grid_arena::{GridArena, GridBase};
mod output;
use output::*;
mod drive;
use drive::*;
pub(crate) use drive::apply_and_fallible;

mod liveness;
// `bucket_shift`/`build_live_cols_bitmask`/`build_reach_masks` are consumed by
// `marginal_plan::build_prefilter_masks` via `super::liveness::…`, not directly here.

mod streaming_marginal;
use streaming_marginal::{StreamCache, StreamLevelState, build_stream_state, commit_stream_state};

/// Conjoin two diagrams that share the same vtree.
///
/// Both operands are CONSUMED: the algorithm drains their level arenas as it
/// walks bottom-up and recycles the storage into the result. Clone one first if
/// you need to keep it.
///
/// Infallible: an allocation refusal panics. Use `conjoin_owned` to recover,
/// or to marginalize while conjoining.
///
/// Runs on limits of its own, with nothing armed, so a stop poll cannot surface
/// as `Err(Deadline)` inside the `expect` below and panic. Auxiliary
/// conjunctions are bounded constructions meant to run to completion; only the
/// fallible entry honors a caller's limits.
///
/// # Panics
///
/// Panics on allocator OOM (`ApplyError::OverBudget`).
pub(crate) fn apply_and(f: Tdd, g: Tdd) -> Tdd {
    Engine::new()
        .and(f, g)
        .expect("apply_and: allocator OOM in infallible entry — use Engine::and to recover")
}

/// Conjoin two diagrams that share the same vtree, reporting a refusal instead of
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
pub(crate) fn conjoin_owned(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    marginalize_targets: Option<&[bool]>,
) -> Result<Tdd, ApplyError> {
    // Checked before the swap and the self-conjunction shortcut, both of which
    // can return without ever reaching `apply_and_fallible_inner`.
    assert!(
        Arc::ptr_eq(&f.vtree, &g.vtree),
        "apply_and requires TDDs with the same vtree"
    );
    assert_eq!(
        f.output.vtree, g.output.vtree,
        "apply_and requires TDDs with outputs at the same vtree node"
    );
    // Operand swap: make g the narrower operand. The g-identity fast path
    // checks right_width == 1 first — the narrower operand is more likely to have
    // width 1 at subtree levels, skipping more product constructions.
    // Secondary benefit: shorter grid rows (width right_width) improve cache locality.
    //
    // Kept HERE (owned path only), not pushed down into `apply_and_fallible`:
    // the borrowed path has order-sensitive callers that must not be swapped.
    // See the note in `apply_and_fallible`.
    //
    // The orientation is not arbitrary and the opposite one is worse: `inputs1`
    // is decoded per f NODE, so putting the narrower operand on f does not
    // shrink the held buffer, and it forfeits the right_width == 1 fast path.
    if g.max_width() > f.max_width() {
        std::mem::swap(&mut f, &mut g);
    }
    // Self-conjunction: f ∧ f = f, on the same structural test as the borrowed
    // entry (see `is_self_conjunction`). Owned variant avoids the clone.
    if is_self_conjunction(&f, &g) {
        diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut g.levels));
        return Ok(f);
    }
    let result = apply_and_fallible(eng, &mut f, &mut g, MarginalTargets::new(marginalize_targets));
    diagram::return_levels(eng, diagram::PoolSlot::First, std::mem::take(&mut f.levels));
    diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut g.levels));
    result
}

/// The conjunction entry points on a caller's engine.
impl crate::engine::Engine {
    /// Conjoin two diagrams over the same vtree.
    ///
    /// Both operands are consumed on `Err` as well as on `Ok`: the product
    /// construction drains their level arenas as it walks bottom-up and
    /// recycles the storage into the result. Clone one first if you need to
    /// keep it, and never reuse an operand after a call.
    ///
    /// # Errors
    ///
    /// [`ApplyError::OverBudget`] when a buffer reservation is refused (the
    /// allocator or the armed soft budget), [`ApplyError::OutputCap`] on the
    /// output-node cap, [`ApplyError::Deadline`] on the armed deadline or a
    /// stop decision.
    ///
    /// # Panics
    ///
    /// Panics if the operands do not share a vtree, or their outputs sit at
    /// different vtree nodes.
    pub fn and(&self, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin::conjoin_owned(self, f, g, None)
    }

    /// [`Engine::and`], emitting the named vtree levels as streaming-marginal
    /// instead of explicit — the levels are summed out as the product is
    /// built rather than in a pass after it.
    ///
    /// `targets` is indexed by [`VtreeIdx`]: `true` at index `t` marginalizes
    /// the output's level `t`.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_marginalizing(&self, f: Tdd, g: Tdd, targets: &[bool]) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin::conjoin_owned(self, f, g, Some(targets))
    }

    /// Conjoin a small batch into a large accumulator by rebuilding only the
    /// levels the batch can reach — the ancestor closure of `spine`.
    ///
    /// Declines rather than fails when the shape does not suit the restricted
    /// merge, returning both operands untouched in
    /// [`BatchMergeOutcome::Declined`] for the caller to conjoin the ordinary way.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_batch(
        &self,
        acc: Tdd,
        batch: Tdd,
        spine: &MergeScope<'_>,
    ) -> Result<BatchMergeOutcome, ApplyError> {
        crate::apply::conjoin::conjoin_batch(self, acc, batch, spine)
    }
}

#[cfg(test)]
#[path = "apply_tests.rs"]
mod apply_tests;

#[cfg(test)]
#[path = "marginal_level_tests.rs"]
mod marginal_level_tests;

#[cfg(test)]
#[path = "identity_tests.rs"]
mod identity_tests;
