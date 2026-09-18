//! Existential quantification of variables from a diagram.
//!
//! Two rewrites compute the same function and the choice between them is
//! [`QuantificationStrategy`]:
//!
//! - **Structural** — a leaf-to-root in-place regroup that never calls apply or
//!   negate, so it is sound where the cofactor rewrite is not. This is what
//!   quantification does unless the caller asks otherwise.
//! - **Cofactor-OR** — `∃x.T = T[x←⊤] ∨ T[x←⊥]`, with the cofactors computed by
//!   rewriting parent-level pair lists that reference x's leaf. The disjunction
//!   is De Morgan, and a negation fills its operand out to full structure, so
//!   the intermediate product is the grid of two dense diagrams: a level whose
//!   children have widths `a` and `b` costs `a * b` cells. It stays close to
//!   the operand's size only when every internal vtree node has a leaf child.

use crate::Engine;

use crate::apply::condition::{condition_leaf, Polarity};
use crate::apply::disjoin::disjoin_owned;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeIdx};

mod structural;

/// How existential quantification is computed; the Boolean result is the same.
///
/// Ordinary quantification uses [`Automatic`](Self::Automatic). Every choice
/// honors the resource restrictions of [`Engine::exists_var`];
/// [`CofactorOr`](Self::CofactorOr) additionally requires structural input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantificationStrategy {
    /// The rewrite the library picks, today [`Structural`](Self::Structural):
    /// it is sound on a marginal level, and its cost is one pass over the
    /// variable's leaf-to-root path rather than a product of two full diagrams.
    Automatic,
    /// Regroup nodes along the variable's leaf-to-root path, without constructing
    /// a pair of cofactors or their disjunction.
    Structural,
    /// Disjoin the two cofactors of the variable. Materializes both of them and
    /// their disjunction, which negation fills out to full structure, so the
    /// intermediate diagrams can be much larger than the operand and the result.
    CofactorOr,
}

/// The implementation behind [`Engine::exists_var`](crate::Engine::exists_var).
pub(crate) fn exists_var_on(eng: &Engine, f: Tdd, x: VarId, how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    // Caller input, so it is answered before any work and before the ⊥ shortcut:
    // the same request is refused whatever the operand happens to be.
    let leaf_idx = f.vtree.leaf_of(x).ok_or(OperationError::VariableNotInVtree(x))?;
    exists_leaf_on(eng, f, leaf_idx, how)
}

/// Quantify a validated leaf index on the operand's unchanged vtree.
fn exists_leaf_on(eng: &Engine, f: Tdd, leaf_idx: VtreeIdx, how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    eng.limits().check_stop()?;
    if f.is_zero() {
        return Ok(f);
    }
    if how != QuantificationStrategy::CofactorOr {
        return structural::exists_var_structural(eng, f, leaf_idx);
    }
    // The disjunction negates, which is unsound across a marginal level.
    f.require_structure()?;
    // One cofactor is rewritten in `f`'s own arenas and the other in a copy
    // reserved through the engine, so a diagram too large to duplicate is
    // refused here.
    let copy = f.try_clone_on(eng)?;
    let pos_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Positive)?;
    let neg_cofactor = condition_leaf(eng, copy, leaf_idx, Polarity::Negative)?;
    disjoin_owned(eng, pos_cofactor, neg_cofactor)
}

/// Existentially quantify every variable in `vars` out of `f`, one at a time.
pub(crate) fn exists_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    let targets = quantification_targets(eng, f.vtree(), vars)?;
    exists_targets_on(eng, f, &targets, how)
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
pub(super) fn exists_targets_on(eng: &Engine, mut f: Tdd, targets: &[VtreeIdx], how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    for &leaf in targets {
        f = exists_leaf_on(eng, f, leaf, how)?;
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
    pub fn exists_var(&self, f: Tdd, x: VarId) -> Result<Tdd, OperationError> {
        self.exists_var_with_strategy(f, x, QuantificationStrategy::Automatic)
    }

    /// Run [`Tdd::exists_var_with_strategy`](crate::Tdd::exists_var_with_strategy) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    /// The structural rewrite checks allocation and cancellation while regrouping
    /// nodes. Its output cap counts emitted intermediate nodes, including the final
    /// root union.
    pub fn exists_var_with_strategy(&self, f: Tdd, x: VarId, how: QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_var_on(self, f, x, how)
    }

    /// Run [`Tdd::exists_vars`](crate::Tdd::exists_vars) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn exists_vars(&self, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        self.exists_vars_with_strategy(f, vars, QuantificationStrategy::Automatic)
    }

    /// Run [`Tdd::exists_vars_with_strategy`](crate::Tdd::exists_vars_with_strategy) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn exists_vars_with_strategy(&self, f: Tdd, vars: &[VarId], how: QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_vars_on(self, f, vars, how)
    }
}
