//! Existential projection (∃-forget) of variables from a diagram.
//!
//! Two rewrites compute the same function and the choice between them is
//! [`Projection`]:
//!
//! - **Cofactor-OR** — `∃x.T = T[x←⊤] ∨ T[x←⊥]`, with the cofactors computed by
//!   rewriting every parent-level pair list that references x's leaf. Fast,
//!   and the default on diagrams it can handle.
//! - **Structural** — a leaf-to-root in-place regroup that never calls apply or
//!   negate, so it is sound where the cofactor rewrite is not.

use crate::engine::Engine;

use crate::apply::condition::{condition_leaves, Polarity};
use crate::apply::disjoin::disjoin_owned;
use crate::limits::ApplyError;
use crate::diagram::Tdd;
use crate::vtree::VarId;

mod structural;

/// Which rewrite an ∃-forget uses.
///
/// The two agree on every diagram both accept, so this is a cost/robustness
/// choice, not a semantic one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Projection {
    /// Cofactor-OR on a diagram with no marginal level, the structural
    /// rewrite otherwise: the disjunction negates, which is unsound across a
    /// marginal level.
    Automatic,
    /// The structural rewrite always. Slower where the cofactor rewrite
    /// applies, but it copies levels verbatim where cofactoring holds a second
    /// copy of the diagram and then negates, so it peaks lower.
    Structural,
}

/// The implementation behind [`Engine::project_var`](crate::Engine::project_var).
pub(crate) fn project_var_on(eng: &Engine, f: Tdd, x: VarId, how: Projection) -> Result<Tdd, ApplyError> {
    let _op = eng.limits().begin_operation();
    // Caller input, so it is answered before any work and before the ⊥ shortcut:
    // the same request is refused whatever the operand happens to be.
    let leaf_idx = f.vtree.leaf_of(x).ok_or(ApplyError::VariableNotInVtree(x))?;
    if f.is_zero() {
        return Ok(f);
    }
    if how == Projection::Structural || f.levels.iter().any(|l| l.is_marginal()) {
        return structural::project_var_structural(eng, f, x, leaf_idx);
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
pub(crate) fn project_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], how: Projection) -> Result<Tdd, ApplyError> {
    let _op = eng.limits().begin_operation();
    let mut result = f;
    for &x in vars {
        result = project_var_on(eng, result, x, how)?;
    }
    Ok(result)
}

/// Sum `x` out of the structure, on a transient engine with no limits armed.
///
/// `f` is borrowed and cloned. [`Engine::project_var`] is this operation on a
/// caller's engine, and states the contract: it keeps the per-level buffers
/// warm between calls, takes the operand by value, and hands a refused
/// allocation or a variable outside the vtree back instead of panicking.
///
/// This is existential quantification over a variable, which is not what
/// [`marginalize`](crate::marginal::marginalize) does: that sums a vtree
/// *level* out into per-node counts and leaves the model count unchanged.
///
/// # Panics
///
/// Panics if `x` is not a variable of `f`'s vtree, if an allocation is
/// refused, and where [`Engine::project_var`] panics.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::apply::{project_var, Projection};
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);  // x1 ∧ x2
/// assert_eq!(f.model_count(), 2u32.into());
///
/// // ∃x2. (x1 ∧ x2) is x1. The vtree still carries x2, now free, so the
/// // count over the whole vtree doubles.
/// let g = project_var(&f, VarId(1), Projection::Automatic);
/// assert_eq!(g.model_count(), 4u32.into());
/// ```
#[must_use]
pub fn project_var(f: &Tdd, x: VarId, how: Projection) -> Tdd {
    project_var_on(&Engine::new(), f.clone(), x, how)
        .expect("project_var: use Engine::project_var to handle a refusal or a variable outside the vtree")
}

/// Sum every variable in `vars` out of the structure, one at a time, on a
/// transient engine with no limits armed.
///
/// `f` is borrowed and cloned. [`Engine::project_vars`] is this operation on a
/// caller's engine.
///
/// # Panics
///
/// Panics if any of `vars` is not a variable of `f`'s vtree, if an
/// allocation is refused, and where [`Engine::project_var`] panics.
#[must_use]
pub fn project_vars(f: &Tdd, vars: &[VarId], how: Projection) -> Tdd {
    project_vars_on(&Engine::new(), f.clone(), vars, how)
        .expect("project_vars: use Engine::project_vars to handle a refusal or a variable outside the vtree")
}

/// The projection entry points on a caller's engine.
impl crate::engine::Engine {
    /// Sum `x` out of the structure: a minimized diagram for ∃x. f, using the
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
    /// let g = eng.project_var(f, VarId(1), tididi::apply::Projection::Automatic).unwrap();
    /// assert_eq!(g.model_count(), BigUint::from(4u32));
    ///
    /// // A byte budget of zero refuses the second cofactor's copy.
    /// let _armed = eng.limits().scope(tididi::limits::LimitSet::none().budget(Some(0)));
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let h = Tdd::clause(&wide, [1, -2]) & Tdd::clause(&wide, [2, 3]);
    /// match eng.project_var(h, VarId(1), tididi::apply::Projection::Automatic) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, tididi::ApplyError::OverBudget),
    /// }
    /// ```
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, the rule [`Engine::and`]
    /// states: one cofactor is rewritten in `f`'s own level arenas. Clone it
    /// first if you need to keep it. A ⊥ operand comes back unchanged, and a
    /// diagram whose output sits at `x`'s own leaf gives ⊤.
    ///
    /// The structural rewrite — every call with [`Projection::Structural`],
    /// and an [`Projection::Automatic`] call on a diagram with a marginal
    /// level — rewrites `f`'s own levels, charging nothing, then reduces the
    /// result on this engine, where the reduction can be refused or cut.
    /// Marginal levels off the path from `x`'s leaf to the root are carried
    /// through unchanged.
    ///
    /// # Errors
    ///
    /// [`ApplyError::VariableNotInVtree`] when `x` is not a variable of `f`'s
    /// vtree, reported before any work is done,
    /// [`ApplyError::OverBudget`] when a reservation is refused, the second
    /// cofactor's copy of the diagram included, [`ApplyError::OutputCap`] on
    /// the output-node cap, [`ApplyError::Deadline`] on the armed deadline or a
    /// stop decision.
    ///
    /// # Panics
    ///
    /// The structural rewrite panics if a level on the path from `x`'s leaf to
    /// the root is marginal (`x` is already summed out), or is the grandparent
    /// of a marginal level. The cofactor rewrite panics if `x`'s leaf level or
    /// its parent is marginal, which [`Projection::Automatic`] never reaches.
    pub fn project_var(&self, f: Tdd, x: VarId, how: crate::apply::Projection) -> Result<Tdd, ApplyError> {
        crate::apply::project::project_var_on(self, f, x, how)
    }

    /// Sum every variable in `vars` out of the structure, one at a time.
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::project_var`]. Each variable is checked against the vtree
    /// only when its turn comes, so an unknown variable late in `vars` fails
    /// after the earlier ones were projected.
    ///
    /// # Errors
    ///
    /// As [`Engine::project_var`].
    ///
    /// # Panics
    ///
    /// As [`Engine::project_var`], for any variable in `vars`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::apply::Projection;
    /// # use tididi::limits::LimitSet;
    /// # use tididi::vtree::{VarId, Vtree};
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]);
    /// let g = engine.project_vars(f, &[VarId(0), VarId(1)], Projection::Automatic).unwrap();
    /// assert_eq!(g.model_count(), 16u32.into());   // every variable is free now
    ///
    /// // A byte budget of zero refuses the first cofactor copy.
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let h = Tdd::clause(&wide, [1, -2]) & Tdd::clause(&wide, [2, 3]);
    /// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
    /// match engine.project_vars(h, &[VarId(0), VarId(1)], Projection::Automatic) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
    /// }
    /// ```
    pub fn project_vars(&self, f: Tdd, vars: &[VarId], how: crate::apply::Projection) -> Result<Tdd, ApplyError> {
        crate::apply::project::project_vars_on(self, f, vars, how)
    }
}
