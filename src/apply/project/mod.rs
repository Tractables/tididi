//! Existential quantification of variables from a diagram.
//!
//! Two rewrites compute the same function and the choice between them is
//! [`QuantificationStrategy`]:
//!
//! - **Cofactor-OR** — `∃x.T = T[x←⊤] ∨ T[x←⊥]`, with the cofactors computed by
//!   rewriting every parent-level pair list that references x's leaf. Fast,
//!   and the default on diagrams it can handle.
//! - **Structural** — a leaf-to-root in-place regroup that never calls apply or
//!   negate, so it is sound where the cofactor rewrite is not.

use crate::engine::Engine;

use crate::apply::condition::{condition_leaves, Polarity};
use crate::apply::disjoin::disjoin_owned;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::vtree::VarId;

mod structural;

/// The algorithm used for existential quantification.
///
/// The two agree on every diagram both accept, so this is a cost/robustness
/// choice, not a semantic one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantificationStrategy {
    /// Cofactor-OR on a diagram with no marginal level, the structural
    /// rewrite otherwise: the disjunction negates, which is unsound across a
    /// marginal level.
    Automatic,
    /// The structural rewrite always. Slower where the cofactor rewrite
    /// applies, but it copies levels verbatim where cofactoring holds a second
    /// copy of the diagram and then negates, so it peaks lower.
    Structural,
}

/// The implementation behind [`Engine::exists_var`](crate::Engine::exists_var).
pub(crate) fn exists_var_on(eng: &Engine, f: Tdd, x: VarId, how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    // Caller input, so it is answered before any work and before the ⊥ shortcut:
    // the same request is refused whatever the operand happens to be.
    let leaf_idx = f.vtree.leaf_of(x).ok_or(OperationError::VariableNotInVtree(x))?;
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
    let pos_cofactor = condition_leaves(eng, f, &[leaf_idx], Polarity::Positive)?;
    let neg_cofactor = condition_leaves(eng, copy, &[leaf_idx], Polarity::Negative)?;
    disjoin_owned(eng, pos_cofactor, neg_cofactor)
}

/// Existentially quantify every variable in `vars` out of `f`, one at a time.
pub(crate) fn exists_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], how: QuantificationStrategy) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    let mut result = f;
    for &x in vars {
        result = exists_var_on(eng, result, x, how)?;
    }
    Ok(result)
}

/// Existentially quantify `x` out of the diagram, on a transient engine with no limits armed.
///
/// `f` is borrowed and cloned. [`Engine::exists_var`] is this operation on a
/// caller's engine, and states the contract: it keeps the per-level buffers
/// warm between calls, takes the operand by value, and hands a refused
/// allocation or a variable outside the vtree back instead of panicking.
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
/// use tididi::apply::{exists_var, QuantificationStrategy};
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);  // x1 ∧ x2
/// assert_eq!(f.model_count(), 2u32.into());
///
/// // ∃x2. (x1 ∧ x2) is x1. The vtree still carries x2, now free, so the
/// // count over the whole vtree doubles.
/// let g = exists_var(&f, VarId(1), QuantificationStrategy::Automatic);
/// assert_eq!(g.model_count(), 4u32.into());
/// ```
#[must_use]
pub fn exists_var(f: &Tdd, x: VarId, how: QuantificationStrategy) -> Tdd {
    exists_var_on(&Engine::new(), f.clone(), x, how)
        .expect("exists_var: use Engine::exists_var to handle a refusal or a variable outside the vtree")
}

/// Existentially quantify every variable in `vars`, one at a time, on a
/// transient engine with no limits armed.
///
/// `f` is borrowed and cloned. [`Engine::exists_vars`] is this operation on a
/// caller's engine.
///
/// # Panics
///
/// Panics on any error reported by [`Engine::exists_vars`].
#[must_use]
pub fn exists_vars(f: &Tdd, vars: &[VarId], how: QuantificationStrategy) -> Tdd {
    exists_vars_on(&Engine::new(), f.clone(), vars, how)
        .expect("exists_vars: use Engine::exists_vars to handle a refusal or a variable outside the vtree")
}

/// The projection entry points on a caller's engine.
impl crate::engine::Engine {
    /// Existentially quantify `x`: a minimized diagram for ∃x. f, using the
    /// rewrite `how` selects.
    ///
    /// The vtree is unchanged, so `x` remains a variable, now free, and
    /// [`Tdd::model_count`] still ranges over it: each model of ∃x. f over the
    /// remaining variables is counted twice. To count over the remaining
    /// variables only, divide by 2 (by 2^k after projecting k variables).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::Tdd;
    /// use tididi::vtree::{VarId, Vtree};
    /// use tididi::Engine;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
    /// assert_eq!(f.model_count(), BigUint::from(2u32));
    /// // ∃x2. (x1 ∧ x2) == x1: forgetting x2 frees it, doubling the count.
    /// let g = eng.exists_var(f, VarId(1), tididi::apply::QuantificationStrategy::Automatic).unwrap();
    /// assert_eq!(g.model_count(), BigUint::from(4u32));
    ///
    /// // A byte budget of zero refuses the second cofactor's copy.
    /// let _armed = eng.limits().scope(tididi::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let h = Tdd::clause(&wide, [1, -2]) & Tdd::clause(&wide, [2, 3]);
    /// match eng.exists_var(h, VarId(1), tididi::apply::QuantificationStrategy::Automatic) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, tididi::OperationError::OverBudget),
    /// }
    /// ```
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, the rule [`Engine::and`]
    /// states: one cofactor is rewritten in `f`'s own level arenas. Clone it
    /// first if you need to keep it. A ⊥ operand comes back unchanged, and a
    /// diagram whose output sits at `x`'s own leaf gives ⊤.
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
    /// [`OperationError::VariableNotInVtree`] when `x` is not a variable of `f`'s
    /// vtree, reported before any work is done,
    /// [`OperationError::OverBudget`] when a reservation is refused, the second
    /// cofactor's copy of the diagram included, [`OperationError::OutputCap`] on
    /// the output-node cap, [`OperationError::Stopped`] on the armed deadline or a
    /// stop decision.
    ///
    /// [`OperationError::MarginalLevel`] if the target leaf or a rewritten
    /// ancestor is marginal, or an ancestor has a marginal grandchild.
    pub fn exists_var(&self, f: Tdd, x: VarId, how: crate::apply::QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_var_on(self, f, x, how)
    }

    /// Existentially quantify every variable in `vars`, one at a time.
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::exists_var`]. Each variable is checked against the vtree
    /// only when its turn comes, so an unknown variable late in `vars` fails
    /// after the earlier ones were projected.
    ///
    /// # Errors
    ///
    /// As [`Engine::exists_var`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{OperationError, Engine, Tdd};
    /// # use tididi::apply::QuantificationStrategy;
    /// # use tididi::limits::LimitConfig;
    /// # use tididi::vtree::{VarId, Vtree};
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);
    /// let g = engine.exists_vars(f, &[VarId(0), VarId(1)], QuantificationStrategy::Automatic).unwrap();
    /// assert_eq!(g.model_count(), 16u32.into());   // every variable is free now
    ///
    /// // A byte budget of zero refuses the first cofactor copy.
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let h = Tdd::clause(&wide, [1, -2]) & Tdd::clause(&wide, [2, 3]);
    /// let _armed = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    /// match engine.exists_vars(h, &[VarId(0), VarId(1)], QuantificationStrategy::Automatic) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, OperationError::OverBudget),
    /// }
    /// ```
    pub fn exists_vars(&self, f: Tdd, vars: &[VarId], how: crate::apply::QuantificationStrategy) -> Result<Tdd, OperationError> {
        crate::apply::project::exists_vars_on(self, f, vars, how)
    }
}
