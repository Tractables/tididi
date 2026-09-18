//! Existential quantification of variables from a diagram.
//!
//! `∃x.T` is a leaf-to-root regroup of the levels on x's path, rewritten in
//! place. It never calls apply or negate, so it is sound when a level off that
//! path is marginal, and it costs one pass over the path rather than a product
//! of two full diagrams.

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
    exists_leaf_on(eng, f, leaf_idx)
}

/// Quantify a validated leaf index on the operand's unchanged vtree.
fn exists_leaf_on(eng: &Engine, f: Tdd, leaf_idx: VtreeIdx) -> Result<Tdd, OperationError> {
    eng.limits().check_stop()?;
    if f.is_zero() {
        return Ok(f);
    }
    structural::exists_var_structural(eng, f, leaf_idx)
}

/// Existentially quantify every variable in `vars` out of `f`, one at a time.
pub(crate) fn exists_vars_on(eng: &Engine, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    let targets = quantification_targets(eng, f.vtree(), vars)?;
    exists_targets_on(eng, f, &targets)
}

/// Validate the entire request and retain each leaf once in first-occurrence order.
pub(super) fn quantification_targets(eng: &Engine, tree: &Vtree, vars: &[VarId]) -> Result<Vec<VtreeIdx>, OperationError> {
    let lim = eng.limits();
    lim.check_stop()?;
    let mut gate = lim.gate();
    let mut targets = Vec::new();
    for (position, &var) in vars.iter().enumerate() {
        let leaf = tree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        lim.try_push(&mut targets, (leaf, position))?;
        gate.poll(1)?;
    }
    targets.sort_unstable_by_key(|&(leaf, position)| (leaf, position));
    targets.dedup_by_key(|(leaf, _)| *leaf);
    targets.sort_unstable_by_key(|&(_, position)| position);
    gate.flush()?;
    let mut leaves = Vec::new();
    lim.reserve_exact(&mut leaves, targets.len())?;
    leaves.extend(targets.into_iter().map(|(leaf, _)| leaf));
    Ok(leaves)
}

/// Quantify prepared leaves without repeating validation or changing their order.
pub(super) fn exists_targets_on(eng: &Engine, mut f: Tdd, targets: &[VtreeIdx]) -> Result<Tdd, OperationError> {
    for &leaf in targets {
        f = exists_leaf_on(eng, f, leaf)?;
    }
    Ok(f)
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
