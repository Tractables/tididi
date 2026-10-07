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
    run_level_rows_complete, run_level_rows_marginal, run_level_rows_marginal_sparse, run_level_rows_plain,
    run_level_rows_stream_count, PlainLookups, RowLoop, RowScratch,
};

mod sparse;
pub(crate) use sparse::SparseWorkspace;
use sparse::{apply_sparse_level, count_sparse_level, sum_sparse_level, CandidateFold, ChildSum, Passthrough};

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
use drive::{apply_and_core, Conjoined, ConjoinMode};
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

/// Make `g` the narrower operand, and say whether the two were swapped.
///
/// The identity fast path tests `right_width == 1` first, so the narrower side
/// on the right takes it at more levels, and grid rows (width `right_width`)
/// get shorter. Only the owned entries swap; `apply_and_fallible`'s callers
/// track operands by side.
fn narrower_right(f: &mut Tdd, g: &mut Tdd) -> bool {
    let swap = g.max_width() > f.max_width();
    if swap {
        std::mem::swap(f, g);
    }
    swap
}

/// [`conjoin_on`] after its operand checks, summing out the levels in
/// `targets`: for a caller that has already validated and weight-aligned the
/// operands.
pub(crate) fn conjoin_checked(
    eng: &Engine,
    f: Tdd,
    g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
) -> Result<(Tdd, bool), OperationError> {
    conjoin_checked_as(eng, f, g, targets, quantified, ConjoinMode::Build)
        .map(|(out, nonzero)| (out.diagram(), nonzero))
}

/// [`conjoin_checked`] with the sweep run in `mode`, [`ConjoinMode::Build`]
/// or [`ConjoinMode::Sum`]; the self-conjunction shortcut returns
/// [`Conjoined::Built`].
fn conjoin_checked_as(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    mode: ConjoinMode,
) -> Result<(Conjoined, bool), OperationError> {
    narrower_right(&mut f, &mut g);
    // Self-conjunction: f ∧ f = f. The test is structural equality of every
    // explicit level, not pointer identity, and it declines on any marginal
    // level — see `is_self_conjunction`, where the soundness of both choices
    // is stated.
    if is_self_conjunction(&f, &g) {
        let _op = eng.limits().enter()?;
        diagram::return_levels(eng, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
        return Ok((Conjoined::Built(f), false));
    }
    let zero = f.is_zero() || g.is_zero();
    let result = conjoin_recycling_as(eng, f, g, targets, quantified, None, mode);
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

/// [`conjoin_recycling`] with the sweep run in `mode`.
fn conjoin_recycling_as(
    eng: &Engine,
    mut f: Tdd,
    mut g: Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
    mode: ConjoinMode,
) -> Result<Conjoined, OperationError> {
    let result = apply_and_core(eng, &mut f, &mut g, targets, quantified, filter, mode);
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

/// A conjunction [`Engine::and_restoring`](crate::Engine::and_restoring)
/// refused, with both operands as they were given.
#[derive(Debug)]
pub struct AndRefused {
    /// Why the conjunction was refused.
    pub error: OperationError,
    /// The first operand, unchanged.
    pub f: Tdd,
    /// The second operand, unchanged.
    pub g: Tdd,
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

    /// Run [`Engine::and`], giving both operands back unchanged when it is
    /// refused.
    ///
    /// A caller that retries after a stop would otherwise clone both operands
    /// before each conjunction. Here the sweep keeps the operands whole until
    /// the result exists, which holds their storage as long as a clone would,
    /// without the copy. Operands with a weight table or a marginal level are
    /// cloned first.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::limits::{LimitConfig, StopCallback, StopDecision};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [-2, 3])?;
    /// let refused = {
    ///     let stop = StopCallback::new(|_, _| StopDecision::Stop);
    ///     let _stopped = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(stop)));
    ///     engine.and_restoring(f, g).expect_err("every poll stops")
    /// };
    /// assert_eq!(engine.model_count(&refused.f)?, 12u32.into());
    /// let both = engine.and_restoring(refused.f, refused.g).map_err(|r| r.error)?;
    /// assert_eq!(engine.model_count(&both)?, 8u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`Engine::and`], each with both operands.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn and_restoring(&self, f: Tdd, g: Tdd) -> Result<Tdd, AndRefused> {
        let (mut f, mut g) = (f, g);
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(error) => return Err(AndRefused { error, f, g }),
        };
        if let Err(error) = crate::apply::check_vtree(&f, &g) {
            return Err(AndRefused { error, f, g });
        }
        let plain = |t: &Tdd| t.weights.is_none() && !t.levels.iter().any(TddLevel::is_marginal);
        if !plain(&f) || !plain(&g) {
            return self.and(f.clone(), g.clone()).map_err(|error| AndRefused { error, f, g });
        }
        let swapped = narrower_right(&mut f, &mut g);
        if is_self_conjunction(&f, &g) {
            diagram::return_levels(self, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
            return Ok(f);
        }
        match drive::apply_and_kept(self, &mut f, &mut g) {
            Ok(out) => {
                diagram::return_levels(self, diagram::PoolSlot::First, std::mem::take(&mut f.levels).into_vec());
                diagram::return_levels(self, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
                Ok(out)
            }
            Err(error) => {
                if swapped {
                    std::mem::swap(&mut f, &mut g);
                }
                Err(AndRefused { error, f, g })
            }
        }
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
    /// A target built as structure is summed out once the conjunction is
    /// built, and the pairs above it that then name the same node on their
    /// other side are fused into one. With a single target on an unweighted
    /// conjunction, where each operand has one node at the target's parent
    /// and the sparse route builds that one product, the parent instead adds
    /// each pair's count into its fused pair as the pair is found, so the
    /// pairs the fusion would remove are never stored; the diagram is the
    /// one the other order leaves.
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
        // One internal target, on an unweighted conjunction: where each
        // operand has one node at the target's parent, that level may sum
        // the target out as it is built (`ConjoinMode::Sum`).
        let summable = match targets.split_first() {
            Some((&c, rest)) => (rest.iter().all(|&t| t == c)
                && !vtree.node(c).is_leaf()
                && vtree.node(c).parent().is_some()
                && f.weights.is_none()
                && g.weights.is_none()
                && !two_step_forced())
            .then_some(c),
            None => None,
        };
        let mode = match summable {
            Some(c) => ConjoinMode::Sum(c),
            None => ConjoinMode::Build,
        };
        let (out, _) = conjoin_checked_as(self, f, g, VtreeMask::new(Some(&mask)), VtreeMask::default(), mode)?;
        let mut out = match out {
            // The root summed the target out as it found its pairs, leaving
            // what the pass below and its pair fusion leave but for the
            // value slots no reference names any more (see
            // `sparse::ChildSum::write_root`).
            Conjoined::Summed(mut out) => {
                note_summed();
                #[cfg(debug_assertions)]
                if let Some(parent) = targets.first().and_then(|&c| vtree.node(c).parent()) {
                    crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(
                        &out, Some(&[parent]), "and_marginalizing",
                    );
                }
                crate::reduce::slot_prune::prune_value_slots(self, &mut out);
                return Ok(out);
            }
            other => other.diagram(),
        };
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
    /// held. Where both operands are one node at the root, the count may
    /// also skip one child of the root: it is summed from that child's own
    /// children, pair by pair against the other child's counts, when that
    /// walk is priced below building the child. Otherwise the conjunction is
    /// built and counted.
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
        // A target at the root sums the whole conjunction into its count,
        // which is what is returned anyway: it is dropped, so the root can be
        // counted instead of built.
        for &t in targets {
            mask[t.idx()] = !vtree.node(t).is_leaf() && t != vtree.root();
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
            self, &mut f, &mut g, VtreeMask::new(Some(&mask)), VtreeMask::default(), None, ConjoinMode::Count,
        );
        diagram::return_levels(self, diagram::PoolSlot::First, std::mem::take(&mut f.levels).into_vec());
        diagram::return_levels(self, diagram::PoolSlot::Second, std::mem::take(&mut g.levels).into_vec());
        match result? {
            Conjoined::Counted(count) => Ok(count),
            Conjoined::Built(out) | Conjoined::Summed(out) => self.model_count(&out),
        }
    }
}

#[cfg(test)]
mod tests;

// A test sends `and_marginalizing` down the two-step path to compare the
// root that sums its child out against it, and counts the roots that did.
#[cfg(test)]
use tests::{note_summed, two_step_forced};

#[cfg(not(test))]
#[inline(always)]
fn two_step_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_summed() {}

// A test reads every child side from the grid, as the oracle for the
// arithmetic lookups on complete children, and counts the levels that read a
// side by arithmetic and the meter charges of their reserved arenas.
#[cfg(test)]
use tests::{grid_lookups_forced, note_lookups, note_scheduled_charge};

#[cfg(not(test))]
#[inline(always)]
fn grid_lookups_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_lookups(_lookups: PlainLookups) {}

#[cfg(not(test))]
#[inline(always)]
fn note_scheduled_charge() {}
