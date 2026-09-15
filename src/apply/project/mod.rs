//! Existential quantification of variables from a diagram.
//!
//! Two rewrites compute the same function and the choice between them is
//! [`QuantificationStrategy`]:
//!
//! - **Cofactor-OR** — `∃x.T = T[x←⊤] ∨ T[x←⊥]`, with the cofactors computed by
//!   rewriting parent-level pair lists that reference x's leaf.
//! - **Structural** — a leaf-to-root in-place regroup that never calls apply or
//!   negate, so it is sound where the cofactor rewrite is not.

use crate::engine::Engine;

use crate::apply::condition::{condition_leaf, Polarity};
use crate::apply::disjoin::disjoin_owned;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeIdx};

mod structural;

/// How existential quantification is computed; the Boolean result is the same.
///
/// Ordinary quantification uses [`Automatic`](Self::Automatic).
/// [`Structural`](Self::Structural) avoids building two cofactors and their disjunction, which can be useful when
/// that intermediate representation is too large. Both choices honor the
/// resource and marginal-level restrictions of [`Engine::exists_var`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantificationStrategy {
    /// Cofactor-OR on a diagram with no marginal level, the structural
    /// rewrite otherwise: the disjunction negates, which is unsound across a
    /// marginal level.
    Automatic,
    /// Regroup nodes along the variable's leaf-to-root path, without constructing
    /// a pair of cofactors or their disjunction.
    Structural,
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
    if eng.limits().should_stop() { return Err(OperationError::Stopped); }
    if f.is_zero() {
        return Ok(f);
    }
    if how == QuantificationStrategy::Structural || f.levels.iter().any(|l| l.is_marginal()) {
        return structural::exists_var_structural(eng, f, leaf_idx);
    }
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
    if lim.should_stop() { return Err(OperationError::Stopped); }
    let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
    let mut targets = Vec::new();
    for (position, &var) in vars.iter().enumerate() {
        let leaf = tree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        lim.try_push(&mut targets, (leaf, position))?;
        lim.poll(&mut gate, 1)?;
    }
    targets.sort_unstable_by_key(|&(leaf, position)| (leaf, position));
    targets.dedup_by_key(|(leaf, _)| *leaf);
    targets.sort_unstable_by_key(|&(_, position)| position);
    lim.flush_poll(&mut gate)?;
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

/// Existentially quantify `x` out of the diagram, using the vtree context with no limits armed.
///
/// Uses [`QuantificationStrategy::Automatic`]; [`exists_var_with_strategy`] selects
/// a rewrite explicitly. The operand is borrowed and cloned.
/// [`Engine::exists_var`] takes ownership, reuses workspace and returns errors.
///
/// This is existential quantification over a variable, which is not what
/// [`marginalize_levels`](crate::marginal::marginalize_levels) does: that sums a vtree
/// *level* out into per-node counts and leaves the model count unchanged.
///
/// # Panics
///
/// Panics on any error reported by [`Engine::exists_var`].
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::apply::exists_var;
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);  // x1 ∧ x2
/// assert_eq!(f.model_count(), 2u32.into());
///
/// // ∃x2. (x1 ∧ x2) is x1. The vtree still carries x2, now free, so the
/// // count over the whole vtree doubles.
/// let g = exists_var(&f, VarId(1));
/// assert_eq!(g.model_count(), 4u32.into());
/// ```
#[must_use]
pub fn exists_var(f: &Tdd, x: VarId) -> Tdd {
    exists_var_with_strategy(f, x, QuantificationStrategy::Automatic)
}

/// Existentially quantify `x` with an explicit rewrite strategy.
///
/// Borrowing, vtree and result semantics are those of [`exists_var`].
///
/// # Panics
///
/// Panics on any error reported by [`Engine::exists_var_with_strategy`].
#[must_use]
pub fn exists_var_with_strategy(f: &Tdd, x: VarId, how: QuantificationStrategy) -> Tdd {
    f.clone().exists_var_with_strategy(x, how)
        .expect("exists_var_with_strategy: use Engine::exists_var_with_strategy to handle a refusal or a variable outside the vtree")
}

/// Existentially quantify every variable in `vars`, one at a time, using the
/// vtree context with no limits armed.
///
/// Uses [`QuantificationStrategy::Automatic`]; [`exists_vars_with_strategy`] selects
/// a rewrite explicitly. The operand is borrowed and cloned.
/// [`Engine::exists_vars`] takes ownership, reuses workspace and returns errors.
///
/// # Panics
///
/// Panics on any error reported by [`Engine::exists_vars`].
#[must_use]
pub fn exists_vars(f: &Tdd, vars: &[VarId]) -> Tdd {
    exists_vars_with_strategy(f, vars, QuantificationStrategy::Automatic)
}

/// Existentially quantify `vars` with an explicit rewrite strategy.
///
/// Borrowing, vtree and result semantics are those of [`exists_vars`].
///
/// # Panics
///
/// Panics on any error reported by [`Engine::exists_vars_with_strategy`].
#[must_use]
pub fn exists_vars_with_strategy(f: &Tdd, vars: &[VarId], how: QuantificationStrategy) -> Tdd {
    f.clone().exists_vars_with_strategy(vars, how)
        .expect("exists_vars_with_strategy: use Engine::exists_vars_with_strategy to handle a refusal or a variable outside the vtree")
}

/// The projection entry points on a caller's engine.
impl crate::engine::Engine {
    /// Existentially quantify `x`: a minimized diagram for ∃x. f.
    ///
    /// Uses [`QuantificationStrategy::Automatic`];
    /// [`Engine::exists_var_with_strategy`] selects a rewrite explicitly.
    ///
    /// The vtree is unchanged, so `x` remains a variable, now free, and
    /// [`Tdd::model_count`] still ranges over it: each model of ∃x. f over the
    /// remaining variables is counted twice. To count over the remaining
    /// variables only, divide by 2 (by 2^k after projecting k distinct variables).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = engine.cube(&tree, [1, 2])?; // x1 ∧ x2
    /// # tididi::test_helpers::assert_canonical(&f);
    /// assert_eq!(f.model_count(), 2u32.into());
    /// let g = engine.exists_var(f, VarId(1))?;
    /// assert_eq!(g.model_count(), 4u32.into()); // x1, with x2 and x3 free
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, the rule [`Engine::and`]
    /// states: one cofactor is rewritten in `f`'s own level arenas. Clone it
    /// first if you need to keep it. A ⊥ operand comes back unchanged, and a
    /// diagram whose output sits at `x`'s own leaf gives ⊤.
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] when `x` is not a variable of `f`'s
    /// vtree, reported before any work is done,
    /// [`OperationError::OverBudget`] when a reservation is refused, the second
    /// cofactor's copy of the diagram included, [`OperationError::OutputCap`] on
    /// the output-node cap, [`OperationError::Stopped`] on the armed deadline or a
    /// stop decision.
    ///
    /// [`OperationError::MarginalLevel`] if the target leaf or a rewritten
    /// ancestor is marginal, or an ancestor has a marginal grandchild.
    pub fn exists_var(&self, f: Tdd, x: VarId) -> Result<Tdd, OperationError> {
        self.exists_var_with_strategy(f, x, QuantificationStrategy::Automatic)
    }

    /// Existentially quantify `x` with an explicit rewrite strategy.
    ///
    /// Vtree, ownership and result semantics are those of [`Engine::exists_var`].
    ///
    /// The structural rewrite — every call with [`QuantificationStrategy::Structural`],
    /// and an [`QuantificationStrategy::Automatic`] call on a diagram with a marginal
    /// level — checks allocations and cancellation while regrouping nodes,
    /// then reduces the result on this engine. Its output cap counts emitted
    /// intermediate nodes, including the final root union.
    /// Marginal levels off the path from `x`'s leaf to the root are carried
    /// through unchanged.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Engine::exists_var`].
    pub fn exists_var_with_strategy(&self, f: Tdd, x: VarId, how: QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_var_on(self, f, x, how)
    }

    /// Remove dependence on `vars` by allowing either value of each variable.
    ///
    /// Uses [`QuantificationStrategy::Automatic`];
    /// [`Engine::exists_vars_with_strategy`] selects a rewrite explicitly.
    ///
    /// An assignment to the remaining variables satisfies the result when at least
    /// one extension satisfies `f`. This is Boolean existential quantification:
    /// multiple satisfying extensions count as one remaining assignment.
    ///
    /// The vtree stays fixed. Each distinct quantified variable becomes free, so a
    /// model count over the remaining variables divides the result's full count by
    /// `2^k`, where `k` is the number of distinct quantified variables. This differs
    /// from [`marginalize_levels`](crate::marginal::marginalize_levels), which sums
    /// the contributions of the extensions and preserves the original count.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let f = engine.clause(&tree, [1, 2])?; // three satisfying assignments
    /// let vars = [VarId(0)];
    /// let projected = engine.exists_vars(f, &vars)?;
    /// assert!(engine.equivalent(&projected, &engine.one(&tree))?);
    /// let remaining_count = engine.model_count(&projected)? >> vars.len();
    /// assert_eq!(remaining_count, 2u32.into()); // both values of x2 have an extension
    /// # tididi::test_helpers::assert_canonical(&projected);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// Each distinct variable is processed once, in first-occurrence order;
    /// an empty slice returns `f` unchanged. Nonempty
    /// calls minimize the result as described by [`Engine::exists_var`]. The operand
    /// is consumed on success and on error, and attached weights are retained.
    ///
    /// # Errors
    ///
    /// As [`Engine::exists_var`]. Every variable is validated before the first
    /// quantification; preparing the request also honors allocation and stop limits.
    pub fn exists_vars(&self, f: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        self.exists_vars_with_strategy(f, vars, QuantificationStrategy::Automatic)
    }

    /// Existentially quantify `vars` with an explicit rewrite strategy.
    ///
    /// Vtree, ownership and ordering semantics are those of [`Engine::exists_vars`].
    /// The strategy applies to every distinct variable in the request.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Engine::exists_vars`].
    pub fn exists_vars_with_strategy(&self, f: Tdd, vars: &[VarId], how: QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_vars_on(self, f, vars, how)
    }
}
