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
use crate::diagram::{self, *};

pub(super) mod budget;
mod child_lookup; // Representation-specialized child lookups (sparse-conjunction kernels)
use crate::engine::Engine;
use crate::error::ApplyError;
use budget::*;

mod cell;
use cell::{
    C2Columns, CellCtx, ColSlice,
    run_level_rows_marg, run_level_rows_marg_sparse, run_level_rows_plain,
    run_level_rows_stream_count,
};
// Test-only gate override; re-exported so the gate-off streaming parity
// regression (query_tests.rs) can disable streaming eligibility.
#[cfg(test)]
pub(crate) use cell::with_bothmarg_collapse_forced;

mod sparse;
pub(crate) use sparse::{reset_sparse_ws, SparseWorkspace};
use sparse::{ProductEntry, is_self_conjunction, apply_sparse_level, apply_leaf_levels, compute_apply_output, release_sparse_ws_if_large};

// Identity/constant-true detection + per-level identity fast paths (extracted).
mod identity;
use identity::{try_level_fast_paths, FastPathResult};
#[cfg(debug_assertions)]
use identity::debug_assert_marg_schedule;
// Consumed only by the `apply_tests` submodule's `use super::*` glob (marginal
// constant-true unit tests); production callers live inside `identity`.
#[cfg(test)]
use identity::level_marginal_is_constant_true;

// Apply setup (phases 1-3) → `ApplyRun` (extracted).
mod setup;
use setup::{apply_and_setup, ApplyRun, LevelShape};





// Per-level marg classification plan + NxM dead-pair masks (extracted).
mod marg_plan;
use marg_plan::{MargPlan, plan_marg_level, build_nxm_masks};

// Spine-bounded ("restricted") apply: the O(spine) batch merge. Same apply
mod restrict;
pub use restrict::{conjoin_batch, BatchMerge, RebuiltMax, Spine};
use restrict::Restrict;
pub(crate) use restrict::RestrictScratch;


mod scratch;
pub use scratch::ApplyScratch;
pub(crate) mod plan;
mod route;
use route::*;
use plan::{ApplyPlan, FullPlan, RestrictedPlan};
mod grid_arena;
use grid_arena::*;
mod output;
use output::*;
mod drive;
use drive::*;
pub(crate) use drive::apply_and_fallible;

mod liveness;
// `bucket_shift`/`build_live_cols_bitmask`/`build_reach_masks` are consumed by
// `marg_plan::build_nxm_masks` via `super::liveness::…`, not directly here.



mod stream;
use stream::{StreamLevelState, build_stream_state, commit_stream_state};
use crate::counts::{ApplyBudget, CountVec};





















/// Conjoin two TDDs that share the same vtree.
///
/// Both operands are CONSUMED: the algorithm drains their level arenas as it
/// walks bottom-up and recycles the storage into the result. Clone one first if
/// you need to keep it.
///
/// Infallible: an allocation refusal panics. Use [`conjoin_owned`] to recover,
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
pub fn apply_and(f: Tdd, g: Tdd) -> Tdd {
    Engine::new()
        .and(f, g)
        .expect("apply_and: allocator OOM in infallible entry — use Engine::and to recover")
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
pub fn conjoin_owned(
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
    // checks k2 == 1 first — the narrower operand is more likely to have
    // width 1 at subtree levels, skipping more product constructions.
    // Secondary benefit: shorter grid rows (width k2) improve cache locality.
    //
    // Kept HERE (owned path only), NOT pushed down into `apply_and_fallible`:
    // the borrowed path has order-sensitive callers that must not be swapped.
    // See the note in `apply_and_fallible`.
    //
    // The orientation is not arbitrary and the opposite one is worse: `inputs1`
    // is decoded per f NODE, so putting the narrower operand on f does not
    // shrink the held buffer, and it forfeits the k2 == 1 fast path.
    if g.max_width() > f.max_width() {
        std::mem::swap(&mut f, &mut g);
    }
    // Self-conjunction: f ∧ f = f, on the same structural test as the borrowed
    // entry (see `is_self_conjunction`). Owned variant avoids the clone.
    if is_self_conjunction(&f, &g) {
        diagram::return_levels2(eng, std::mem::take(&mut g.levels));
        return Ok(f);
    }
    let result = apply_and_fallible(eng, &mut f, &mut g, marginalize_targets);
    diagram::return_levels(eng, std::mem::take(&mut f.levels));
    diagram::return_levels2(eng, std::mem::take(&mut g.levels));
    result
}

#[cfg(test)]
#[path = "apply_tests.rs"]
mod apply_tests;

#[cfg(test)]
#[path = "marginal_level_tests.rs"]
mod marginal_level_tests;
