//! Conjunction of two diagrams via the compacting product construction.
//!
//! Given two diagrams over the same vtree, produces a diagram for their conjunction.
//! The product construction pairs every node from f with every node from g
//! at each vtree level, computes the conjunction of their input pairs, and
//! omits dead nodes (compaction). See `docs/tdd.md` for details.

use std::sync::Arc;

use crate::vtree::VtreeIdx;
use super::CONJOIN_GRID;
use crate::diagram::{self, *};

pub(crate) mod budget;
mod child_lookup; // Representation-specialized child lookups (sparse-conjunction kernels)
use crate::Engine;
use crate::limits::OperationError;
use budget::*;

mod cell;
use cell::{
    RightColumns, CellCtx, ChildPlan, ColumnSlice,
    run_level_rows_marginal, run_level_rows_marginal_sparse, run_level_rows_plain,
    run_level_rows_stream_count, RowLoop, RowScratch,
};

mod sparse;
pub(crate) use sparse::SparseWorkspace;
use sparse::{apply_sparse_level, count_sparse_level, CandidateFold, Passthrough};

// Identity/constant-true detection and the per-level identity fast paths.
mod identity;
pub(crate) use identity::init_leaf_identity;
use identity::take_level_fast_path;

// Apply setup → `ApplyRun`.
mod setup;
use setup::{apply_and_setup, ApplyRun, LevelShape, Operands};

// Per-level marginal classification plan and dead-pair masks.
mod marginal_plan;
use marginal_plan::{ChildGrid, MarginalPlan, SidePlan, plan_marginal_level, build_side_masks};

// The leaf levels, before the bottom-up loop.
mod leaf_seed;
use leaf_seed::apply_leaf_levels;

mod scratch;
pub(crate) use scratch::ApplyScratch;
pub(crate) use setup::VtreeMask;
mod route;
use route::*;
mod grid_arena;
mod products;
pub(in crate::apply::conjoin) use grid_arena::GridBase;
mod output;
use output::*;
mod quantify;
mod drive;
pub(crate) use drive::apply_and_fallible;
use drive::{apply_and_core, Conjoined};
use drive::Sweep;
mod filter;

mod liveness;

pub(crate) mod streaming_marginal;
use crate::value::StreamCache;
use streaming_marginal::{StreamEnv, StreamLevelState, build_stream_state, commit_stream_state};

/// Panicking conjunction used by `BitAnd` and test fixtures.
/// The checked entry point is [`and`].
pub(crate) fn apply_and(f: Tdd, g: Tdd) -> Tdd {
    and(f, g)
        .expect("apply_and: operation refused; use tididi::and to handle errors")
}

/// Conjoin owned operands, recycling their levels as the bottom-up walk
/// proceeds, and collapsing every subtree in `quantified` to the constant-true
/// node instead of building it — see [`quantify`].
///
/// The flag says whether the sweep ran, which is what decides whether the
/// caller's quantification has collapsed levels to account for: the shortcuts
/// for a false operand and for `f ∧ f` return without collapsing anything.
/// Operand validation and allocation, cancellation and output-cap errors follow
/// [`Engine::and`]. Both inputs are consumed on every outcome.
pub(crate) fn conjoin_on(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    quantified: VtreeMask<'_>,
) -> Result<(Tdd, bool), OperationError> {
    // Checked before the swap and the self-conjunction shortcut, both of which
    // can return without ever reaching `apply_and_fallible`.
    crate::apply::check_vtree(&f, &g)?;
    crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
    conjoin_checked(eng, f, g, VtreeMask::default(), quantified)
}

/// True when `f` and `g` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, ranges)` on every level is
/// sufficient. This is structural equality, not pointer identity — but it is
/// only sound when no level is marginal, since a marginal level hides its
/// content outside `nodes`/`pairs` where the structural test cannot see it.
fn is_self_conjunction(f: &Tdd, g: &Tdd) -> bool {
    // The shortcut lets `f ∧ g` return `f.clone()` when the operands are the
    // same function. It is a pure perf optimization, never needed for
    // correctness. A marginal level clears `nodes`/`pairs` (integer-marginal) or
    // `pairs` (weight-marginal) and moves its real content into
    // `marginal_counts`/the external weight store — which this structural test
    // does not compare. Two operands agreeing on every explicit level but
    // differing in marginal mass (or holding a marginal×marginal unsound
    // schedule the callers debug-assert against) would compare equal and
    // silently drop one side's content. Bail whenever either operand carries any
    // marginal level.
    if f.levels.iter().any(|l| l.is_marginal()) || g.levels.iter().any(|l| l.is_marginal()) {
        return false;
    }
    f.output == g.output
        && f.levels.iter().zip(g.levels.iter()).all(|(l1, l2)| {
            // `ranges` too: equal nodes+pairs with a differently-arranged `ranges` table
            // is a different function.
            l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ranges == l2.ranges
        })
}

/// [`conjoin_on`] after its operand checks, summing out the levels in
/// `targets`: for a caller that has already validated and weight-aligned the
/// operands.
pub(crate) fn conjoin_checked(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
) -> Result<(Tdd, bool), OperationError> {
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
        let _op = eng.limits().enter()?;
        diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
        return Ok((f, false));
    }
    let zero = f.is_zero() || g.is_zero();
    let result = conjoin_recycling(eng, f, g, targets, quantified, None);
    result.map(|out| (out, !zero))
}

/// Run the sweep over `f` and `g`, then hand both operands' level arrays back
/// to the pool whatever the outcome.
fn conjoin_recycling(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
) -> Result<Tdd, OperationError> {
    let result = apply_and_fallible(eng, &mut f, &mut g, targets, quantified, filter);
    diagram::return_levels(eng, diagram::PoolSlot::First, std::mem::take(&mut f.levels).into_vec());
    diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
    result
}

/// Return the conjunction of two diagrams sharing the same vtree allocation.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are consumed on success and on error; clone an operand first
/// if it is needed afterward. The result is correct for counting but may retain
/// unreachable nodes and twins; [`Tdd::minimize`]
/// establishes canonical form when required.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, literal, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let either = Tdd::clause(&vtree, [1, 2])?;
/// let not_third = literal(&vtree, -3)?;
/// let f = and(either, not_third)?;
/// assert_eq!(f.model_count()?, 3u32.into());
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
/// [`OperationError::IncompatibleWeights`] for different weight interpretations,
/// or [`OperationError::MarginalLevel`] when a structural operand constrains
/// variables the other has summed out.
///
/// Allocation refusal is reported as [`OperationError::OverBudget`].
/// For explicit resource limits, use [`Context::with_limits`](crate::Context::with_limits)
/// and the supplied engine's operations.
pub fn and(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.and(f, g))
}

impl crate::Engine {
    /// Run [`and`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        conjoin_on(self, f, g, VtreeMask::default()).map(|(out, _)| out)
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
    /// [`Tdd::marginalize_levels`] for the operations
    /// that remain valid after structure is discarded.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = vtree.children(vtree.root());
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
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
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        f.check_level_indices(targets)?;
        crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
        // The apply core asks "is level `t` a target?" once per level it emits,
        // so the membership array is derived here, once, at the cost the caller
        // would pay to build it.
        let vtree = Arc::clone(f.vtree());
        let mut mask = Vec::new();
        self.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf();
        }
        let (mut out, _) = conjoin_checked(
            self, f, g, VtreeMask::new(Some(&mask)), VtreeMask::default(),
        )?;
        // Streaming can finish every target; only retained structure needs the pass.
        if !out.is_zero() && targets.iter().any(|&t| !out.level(t).is_marginal()) {
            self.marginalize_levels(&mut out, targets)?;
        }
        Ok(out)
    }

    /// Count the models of `f ∧ g` without keeping the conjunction.
    ///
    /// The result is `model_count(and_marginalizing(f, g, targets))`:
    /// `targets` are summed out during product construction as there, which
    /// changes how much structure is built, never the count. Where the root
    /// of the conjunction is one product built by the sparse route, its
    /// pairs are folded into the count as they are found instead of being
    /// stored, so the largest level of a join that ends in a count is never
    /// held. Otherwise the conjunction is built and counted.
    ///
    /// Integer counts only: with weights attached, the result is the count
    /// of the conjunction built with them, as [`Engine::model_count`] gives it.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// assert_eq!(engine.and_model_count(f, g, &[])?, 9u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`Engine::and_marginalizing`] and of
    /// [`Engine::model_count`]. Both operands are consumed on every outcome.
    pub fn and_model_count(
        &self,
        mut f: Tdd,
        mut g: Tdd,
        targets: &[VtreeIdx],
    ) -> Result<num_bigint::BigUint, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        f.check_level_indices(targets)?;
        if f.weights.is_some() || g.weights.is_some() {
            let out = self.and_marginalizing(f, g, targets)?;
            return self.model_count(&out);
        }
        crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
        let vtree = Arc::clone(f.vtree());
        let mut mask = Vec::new();
        self.limits().try_resize(&mut mask, vtree.num_nodes(), false)?;
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf();
        }
        // The swap and the self-conjunction shortcut of `conjoin_checked`.
        if g.max_width() > f.max_width() {
            std::mem::swap(&mut f, &mut g);
        }
        if is_self_conjunction(&f, &g) {
            let count = self.model_count(&f);
            diagram::return_levels(self, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
            return count;
        }
        let result = apply_and_core(
            self, &mut f, &mut g, VtreeMask::new(Some(&mask)), VtreeMask::default(), None, true,
        );
        diagram::return_levels(self, diagram::PoolSlot::First, std::mem::take(&mut f.levels).into_vec());
        diagram::return_levels(self, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
        match result? {
            Conjoined::Counted(count) => Ok(count),
            Conjoined::Built(out) => self.model_count(&out),
        }
    }
}

#[cfg(test)]
mod tests;
