//! Existential quantification of variables from a diagram.
//!
//! `∃X.T` is one bottom-up regroup of the levels above X's leaves, rewritten in
//! place. It never calls apply or negate, so it is sound when a level elsewhere
//! is marginal, and it costs one pass over those levels rather than a product
//! of two full diagrams — one pass for the whole request, not one per
//! variable.

use crate::Engine;

use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeIdx};

mod structural;

/// The implementation behind [`Engine::exists_var`](crate::Engine::exists_var).
pub(crate) fn exists_var_on(eng: &Engine, f: Tdd, x: VarId) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    // Caller input, so it is answered before any work and before the ⊥ shortcut:
    // the same request is refused whatever the operand happens to be.
    let leaf_idx = f.vtree.leaf_of(x).ok_or(OperationError::VariableNotInVtree(x))?;
    exists_leaves_on(eng, f, &[leaf_idx], &[])
}

/// Quantify validated leaf indices on the operand's unchanged vtree.
///
/// `collapsed` names the levels a fused conjunction already reduced to the
/// single `⊤` node; empty for an operand nothing has quantified yet.
fn exists_leaves_on(eng: &Engine, f: Tdd, targets: &[VtreeIdx], collapsed: &[bool]) -> Result<Tdd, OperationError> {
    eng.limits().check_stop()?;
    if f.is_zero() {
        return Ok(f);
    }
    structural::exists_leaves_structural(eng, f, targets, collapsed)
}

/// Existentially quantify every variable in `vars` out of `f` in one sweep.
pub(crate) fn exists_vars_on(eng: &Engine, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    let targets = quantification_targets(eng, f.vtree(), vars)?;
    exists_targets_on(eng, f, &targets, &[])
}

/// Validate the entire request and retain each leaf once.
pub(super) fn quantification_targets(eng: &Engine, vtree: &Vtree, vars: &[VarId]) -> Result<Vec<VtreeIdx>, OperationError> {
    let lim = eng.limits();
    lim.check_stop()?;
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

/// Quantify prepared leaves without repeating validation. `collapsed` is
/// [`exists_leaves_on`]'s.
pub(super) fn exists_targets_on(
    eng: &Engine,
    f: Tdd,
    targets: &[VtreeIdx],
    collapsed: &[bool],
) -> Result<Tdd, OperationError> {
    if targets.is_empty() {
        return Ok(f);
    }
    exists_leaves_on(eng, f, targets, collapsed)
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
        crate::apply::project::exists_var_on(self, f, x)
    }

    /// Run [`Tdd::exists_vars`](crate::Tdd::exists_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn exists_vars(&self, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_vars_on(self, f, vars)
    }
}
