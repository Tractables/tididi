//! Existential quantification of variables from a diagram.
//!
//! `∃X.T` is one bottom-up regroup of the levels above X's leaves, rewritten in
//! place. It never calls apply or negate, and it costs one pass over those
//! levels rather than a product of two full diagrams — one pass for the whole
//! request, not one per variable. Both `∃X.T` and `∀X.T` (`¬∃X.¬T`) need every
//! level structural.

use crate::Engine;

use crate::apply::negate::negate_on;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::reduce::ReductionPlan;
use crate::vtree::{VarId, Vtree, VtreeIdx};

mod structural;

/// Existentially quantify every variable in `vars` out of `f` in one sweep,
/// ending with `reduction`.
fn exists_vars_on(
    eng: &Engine,
    f: Tdd,
    vars: &[VarId],
    reduction: ReductionPlan<'_>,
) -> Result<Tdd, OperationError> {
    let _op = eng.limits().enter()?;
    // Caller input, so it is answered before any work and before the ⊥
    // shortcut: the same request is refused whatever the operand happens to be.
    let targets = quantification_targets(eng, f.vtree(), vars)?;
    exists_targets_on(eng, f, &targets, reduction)
}

/// Validate the entire request and retain each leaf once.
pub(super) fn quantification_targets(eng: &Engine, vtree: &Vtree, vars: &[VarId]) -> Result<Vec<VtreeIdx>, OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    let mut targets = Vec::new();
    for &var in vars {
        let leaf = vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        lim.try_push(&mut targets, leaf)?;
        gate.poll(1)?;
    }
    targets.sort_unstable();
    targets.dedup();
    gate.flush()?;
    Ok(targets)
}

/// Quantify prepared leaves without repeating validation, ending with
/// `reduction`; with no target, `f` is returned as it is, unreduced.
pub(super) fn exists_targets_on(
    eng: &Engine,
    f: Tdd,
    targets: &[VtreeIdx],
    reduction: ReductionPlan<'_>,
) -> Result<Tdd, OperationError> {
    if targets.is_empty() {
        return Ok(f);
    }
    structural::exists_leaves_structural(eng, f, targets, reduction)
}

impl crate::Engine {
    /// Run [`Tdd::exists_var`](crate::Tdd::exists_var) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    /// The rewrite checks allocation and cancellation while regrouping nodes. Its
    /// output cap counts emitted intermediate nodes, including the final root union.
    pub fn exists_var(&self, f: Tdd, x: VarId) -> Result<Tdd, OperationError> {
        exists_vars_on(self, f, &[x], ReductionPlan::default())
    }

    /// Run [`Tdd::exists_vars`](crate::Tdd::exists_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn exists_vars(&self, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        exists_vars_on(self, f, vars, ReductionPlan::default())
    }

    /// [`exists_vars`](Self::exists_vars) with the reduction it ends with
    /// chosen explicitly.
    ///
    /// Every plan returns the same function. [`ReductionPlan::Prune`] leaves
    /// every node reachable without establishing canonical form: it skips the
    /// twin contraction, so the result can be larger than the minimized one,
    /// and a later [`minimize`](Self::minimize) finishes the job.
    /// [`ReductionPlan::Contract`] runs no prune and leaves the result owing
    /// the other passes, which a later `minimize` runs. An empty
    /// `vars`, a false `f`, and a request that quantifies every variable return
    /// without running the plan.
    ///
    /// # Errors
    ///
    /// As [`exists_vars`](Self::exists_vars).
    pub fn exists_vars_with(&self, f: Tdd, vars: &[VarId], plan: ReductionPlan<'_>) -> Result<Tdd, OperationError> {
        exists_vars_on(self, f, vars, plan)
    }

    /// Run [`Tdd::forall_vars`](crate::Tdd::forall_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn forall_vars(&self, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        let targets = quantification_targets(self, f.vtree(), vars)?;
        if targets.is_empty() {
            return Ok(f);
        }
        // `¬∃X.¬f`; the complement needs every level structural. Minimizing
        // the projection keeps the second make-full pass small.
        let complement = negate_on(self, f)?;
        let projected = exists_targets_on(self, complement, &targets, ReductionPlan::default())?;
        self.negate(projected)
    }
}
