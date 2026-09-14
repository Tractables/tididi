//! Conjunction of two diagrams via the compacting product construction.
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
use crate::limits::OperationError;
use budget::*;

mod cell;
use cell::{
    RightColumns, CellCtx, ChildPlan, ColumnSlice,
    run_level_rows_marginal, run_level_rows_marginal_sparse, run_level_rows_plain,
    run_level_rows_stream_count, RowLoop, RowScratch,
};

mod sparse;
pub(crate) use sparse::{reset_sparse_ws, SparseWorkspace};
use sparse::{
    ProductEntry, is_self_conjunction, apply_sparse_level, apply_leaf_levels,
    compute_apply_output, release_sparse_ws_if_large,
};

// Identity/constant-true detection and the per-level identity fast paths.
mod identity;
use identity::{init_leaf_identity, take_level_fast_path, FastPathResult};

// Apply setup → `ApplyRun`.
mod setup;
use setup::{apply_and_setup, ApplyRun, LevelShape};

// Per-level marginal classification plan and dead-pair masks.
pub(crate) mod marginal_plan;
use marginal_plan::{ChildGrid, MarginalPlan, SidePlan, plan_marginal_level, build_side_masks};

// Seeding the output's marginal vtree leaves from the operands, before the
// bottom-up loop.
mod leaf_seed;

mod scratch;
pub(crate) use scratch::ApplyScratch;
pub(crate) mod targets;
use targets::MarginalTargets;
mod route;
use route::*;
mod grid_arena;
pub(in crate::apply::conjoin) use grid_arena::{GridArena, GridBase};
mod output;
use output::*;
mod drive;
pub(crate) use drive::apply_and_fallible;
use drive::Sweep;

mod liveness; // Used by `marginal_plan::build_side_masks`.

pub(crate) mod streaming_marginal;
use crate::value::StreamCache;
use streaming_marginal::{StreamEnv, StreamLevelState, build_stream_state, commit_stream_state};

/// Conjoin two diagrams that share the same vtree.
///
/// Both operands are consumed: the algorithm drains their level arenas as it
/// walks bottom-up and recycles the storage into the result. Clone one first if
/// you need to keep it.
///
/// Panics on invalid operands or allocation refusal; [`Engine::and`] returns the error.
///
/// Runs on a fresh engine with nothing armed, so a caller's deadline or stop
/// is not polled; only the fallible entry honors limits.
///
/// # Panics
///
/// Panics on operand incompatibility or allocation refusal.
pub(crate) fn apply_and(f: Tdd, g: Tdd) -> Tdd {
    Engine::new()
        .and(f, g)
        .expect("apply_and: operation refused; use Engine::and to handle errors")
}

/// Conjoin two diagrams that share the same vtree, reporting a refusal instead of
/// panicking on it. The production conjunction entry.
///
/// Both operands are consumed — the algorithm drains their level arenas as it
/// goes and recycles the storage into the result — on `Err` as well as on `Ok`.
/// Clone one first if you need to keep it, and never reuse an operand after a
/// call.
///
/// `marginalize_targets`, when given, names the vtree nodes whose levels the
/// bottom-up loop should emit as streaming-marginal instead of explicit.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if a buffer reservation is refused
/// (allocator failure or the configured soft budget would be exceeded),
/// `Err(OperationError::OutputCap)` on the output-node cap, or
/// `Err(OperationError::Stopped)` on the scoped deadline or an armed decision
/// callback that concluded the compile should stop.
pub(crate) fn conjoin_owned(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    marginalize_targets: Option<&[bool]>,
) -> Result<Tdd, OperationError> {
    // Checked before the swap and the self-conjunction shortcut, both of which
    // can return without ever reaching `apply_and_fallible_inner`.
    crate::apply::check_conjunction_operands(&f, &g)?;
    crate::apply::prepare_weights([&mut f, &mut g])?;
    // Make `g` the narrower operand: the identity fast path tests
    // `right_width == 1` first, so the narrower side on the right takes it at
    // more levels, and grid rows (width `right_width`) get shorter. Only this
    // owned entry swaps; `apply_and_fallible`'s callers track operands by side.
    if g.max_width() > f.max_width() {
        std::mem::swap(&mut f, &mut g);
    }
    // Self-conjunction: f ∧ f = f. The test is structural equality of every
    // explicit level, not pointer identity, and it declines on any marginal
    // level — see `is_self_conjunction`, where the soundness of both choices
    // is stated.
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
    /// Return the conjunction of two diagrams sharing the same vtree allocation.
    ///
    /// Both operands are consumed on success and on error; clone an operand first
    /// if it is needed afterward. The result is correct for counting but may retain
    /// unreachable nodes and twins; [`try_minimize`](crate::reduce::try_minimize)
    /// establishes canonical form when required.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let either = engine.clause(&tree, [1, 2])?;
    /// let not_third = engine.literal(&tree, -3)?;
    /// let f = engine.and(either, not_third)?;
    /// assert_eq!(engine.model_count(&f)?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Weights and marginal levels
    ///
    /// Attached weight tables and arithmetic must agree. A structural operand
    /// without weights inherits the other operand's table; stored integer counts
    /// cannot be reweighted. Marginal levels remain marginal, and conjunction is
    /// valid there only when the other operand imposes no further constraint on
    /// the summed-out variables. When both operands are marginal at a level, the
    /// caller must ensure one represents the constant-true function on that subtree;
    /// this condition is checked in debug builds.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] for different vtree allocations,
    /// [`OperationError::RootMismatch`] for different output levels,
    /// [`OperationError::IncompatibleWeights`] for different weight interpretations,
    /// or [`OperationError::MarginalLevel`] when a structural operand constrains
    /// variables the other has summed out.
    ///
    /// Resource refusals are [`OperationError::OverBudget`],
    /// [`OperationError::OutputCap`], or [`OperationError::Stopped`].
    /// [`Limits::scope`](crate::limits::Limits::scope) shows how to install limits.
    pub fn and(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        crate::apply::conjoin::conjoin_owned(self, f, g, None)
    }

    /// Conjoin two diagrams and replace selected subtrees with marginal values.
    ///
    /// `targets` names vtree nodes; order and duplicates do not matter. Internal
    /// targets can be summed out during product construction; leaf targets are
    /// handled afterward. The result preserves the conjunction's count, or its
    /// fixed weighted value when weights are attached.
    ///
    /// Every target's remaining structure is discarded, including after identity
    /// and self-conjunction shortcuts; the false diagram stays false. See
    /// [`marginalize_levels`](crate::marginal::marginalize_levels) for the operations
    /// that remain valid after structure is discarded.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = tree.children(tree.root());
    /// let f = engine.clause(&tree, [1, 2])?;
    /// let g = engine.clause(&tree, [3, 4])?;
    /// let counted = engine.and_marginalizing(f, g, &[left])?;
    /// assert!(counted.level(left).is_marginal());
    /// assert_eq!(engine.model_count(&counted)?, 9u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The operand and resource errors of [`Engine::and`], plus
    /// [`OperationError::LevelNotInVtree`] for an invalid target, checked before
    /// product construction. Both operands are consumed on every outcome.
    pub fn and_marginalizing(
        &self,
        mut f: Tdd,
        mut g: Tdd,
        targets: &[VtreeIdx],
    ) -> Result<Tdd, OperationError> {
        crate::apply::check_conjunction_operands(&f, &g)?;
        f.check_level_indices(targets)?;
        crate::apply::prepare_weights([&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        // The apply core asks "is level `t` a target?" once per level it emits,
        // so the membership array is derived here, once, at the cost the caller
        // would pay to build it.
        let vtree = Arc::clone(f.vtree());
        let mut mask = Vec::new();
        self.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf();
        }
        let mut out = crate::apply::conjoin::conjoin_owned(self, f, g, Some(&mask))?;
        // Streaming can finish every target; only retained structure needs the pass.
        if !out.is_zero() && targets.iter().any(|&t| !out.level(t).is_marginal()) {
            crate::marginal::marginalize_levels(self, &mut out, targets)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
