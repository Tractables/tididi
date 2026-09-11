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

use crate::apply::condition::{condition_leaf, Polarity};
use crate::apply::disjoin::disjoin_owned;
use crate::error::ApplyError;
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
    /// Cofactor-OR where it is sound, the structural rewrite where it is not.
    ///
    /// The cofactor rewrite implements `a ∨ b` as `¬(¬a ∧ ¬b)`, and negating
    /// across a marginal level is unsound: a width>1 marginal on both sides has
    /// no pair structure to conjoin, and a width-1 marginal trips the apply's
    /// marginal-child dispatch. So the presence of any marginal level selects
    /// the structural rewrite, which carries such levels verbatim without ever
    /// dereferencing them. Projection never *creates* marginal levels — only
    /// marginalization does — so one scan of the input decides a whole batch.
    Automatic,
    /// The structural rewrite always, marginal levels or not.
    ///
    /// The reason to ask for it on a diagram the cofactor rewrite would accept
    /// is memory: cofactoring holds a second copy of the diagram and then
    /// negates, so it peaks well above the structural regroup, which copies
    /// levels verbatim. On [`Engine::project_var`] the cofactor rewrite reports
    /// a copy it cannot make rather than taking the process down, so the choice
    /// is peak against speed; the free functions have no limits to report to and
    /// panic instead. It is the slower of the two on structures the cofactor
    /// rewrite handles, so this is for a caller that has already decided
    /// robustness beats speed.
    Structural,
}

/// The implementation behind [`Engine::project_var`](crate::Engine::project_var).
pub(crate) fn project_var_on(eng: &Engine, f: Tdd, x: VarId, how: Projection) -> Result<Tdd, ApplyError> {
    // Caller input, so it is answered before any work and before the ⊥ shortcut:
    // the same request is refused whatever the operand happens to be.
    let leaf_idx = f.vtree.leaf_of(x).ok_or(ApplyError::VariableNotInVtree(x))?;
    if f.is_zero() {
        return Ok(f);
    }
    if how == Projection::Structural || f.levels.iter().any(|l| l.is_marginal()) {
        return Ok(structural::project_var_structural(&f, x, leaf_idx));
    }
    let vtree = &f.vtree;
    // Sound iff no ancestor of x's leaf is marginal. Levels in disjoint
    // sub-vtrees may be marginal without affecting correctness — but the
    // marginal scan above has already routed any such diagram to the structural
    // rewrite, so reaching here with one at all is a caller error.
    let mut ancestor = vtree.node(leaf_idx).parent();
    while let Some(idx) = ancestor {
        if f.levels[idx.idx()].is_marginal() {
            panic!(
                "project_var: variable {:?} has a marginal ancestor at vtree index {:?}. \
                 Call project_var before marginalization, or use compile_cnf (non-mc mode).",
                x, idx
            );
        }
        ancestor = vtree.node(idx).parent();
    }

    // One cofactor is rewritten in `f`'s own arenas and the other in a copy, so
    // the two of them are the peak. The copy is reserved through the engine:
    // a diagram too large to duplicate is a refusal here, at the request,
    // rather than an allocator abort no caller can catch.
    let copy = f.try_clone_on(eng)?;
    let mut pos_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Positive)?;
    let mut neg_cofactor = condition_leaf(eng, copy, leaf_idx, Polarity::Negative)?;
    // The store travels with the diagram. Each cofactor carries one, but the
    // disjunction negates, and negation copies levels without the side table,
    // so the store is moved across by hand. No values change on the way: this
    // path runs only when no level is marginal, so nothing in the store is
    // referenced by anything being rewritten.
    let ws = pos_cofactor.detach_weights().or_else(|| neg_cofactor.detach_weights());
    let mut out = disjoin_owned(eng, pos_cofactor, neg_cofactor)?;
    out.weights = ws;
    Ok(out)
}

/// Existentially quantify all variables in `vars`, one at a time.
/// Returns a fully minimized diagram representing ∃vars. t.
pub(crate) fn project_vars_on(eng: &Engine, f: Tdd, vars: &[VarId], how: Projection) -> Result<Tdd, ApplyError> {
    let mut result = f;
    for &x in vars {
        result = project_var_on(eng, result, x, how)?;
    }
    Ok(result)
}

/// Sum `x` out of the structure, on a transient engine with no limits armed.
///
/// [`Engine::project_var`] is this operation on a caller's engine: it keeps the
/// per-level buffers warm between calls, takes the operand by value, and hands
/// a refused allocation or a variable outside the vtree back instead of
/// panicking.
///
/// This is existential quantification over a variable, which is not what
/// [`marginalize`](crate::marginal::marginalize) does: that sums a vtree
/// *level* out into per-node counts and leaves the model count unchanged.
///
/// # Panics
///
/// Panics if `x` is not a variable of `f`'s vtree, and if an allocation is
/// refused.
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
/// [`Engine::project_vars`] is this operation on a caller's engine.
///
/// This is existential quantification over variables, which is not what
/// [`marginalize`](crate::marginal::marginalize) does: that sums a vtree
/// *level* out into per-node counts and leaves the model count unchanged.
///
/// # Panics
///
/// Panics if any of `vars` is not a variable of `f`'s vtree, and if an
/// allocation is refused.
#[must_use]
pub fn project_vars(f: &Tdd, vars: &[VarId], how: Projection) -> Tdd {
    project_vars_on(&Engine::new(), f.clone(), vars, how)
        .expect("project_vars: use Engine::project_vars to handle a refusal or a variable outside the vtree")
}

/// The projection entry points on a caller's engine.
impl crate::engine::Engine {
    /// Returns a fully minimized canonical diagram representing ∃x. t.
    ///
    /// Precondition: no ancestor of x's leaf may be a marginal level (i.e., must
    /// be called on a full/non-mc diagram). A variable `t.vtree` does not carry
    /// is an error rather than a precondition.
    ///
    /// This is existential quantification over a variable, which is not what
    /// [`marginalize`](crate::marginal::marginalize) does: that sums a vtree
    /// *level* out into per-node counts and leaves the model count unchanged.
    ///
    /// Count convention: the result keeps `t.vtree` unchanged, so `x` remains a
    /// (now don't-care) variable and [`Tdd::model_count`] still ranges over it —
    /// each satisfying assignment of ∃x. t over the remaining variables is counted
    /// twice (once per value of `x`). To count over the remaining variables only,
    /// divide by 2 (by 2^k after projecting k variables).
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
    /// let _armed = eng.limits().scope(tididi::engine::LimitSet::none().budget(Some(0)));
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
    /// first if you need to keep it.
    ///
    /// # Errors
    ///
    /// [`ApplyError::VariableNotInVtree`] when `x` is not a variable of `t`'s
    /// vtree, reported before any work is done,
    /// [`ApplyError::OverBudget`] when a reservation is refused — including the
    /// second cofactor's copy of the diagram, which is where a projection of a
    /// diagram too large to duplicate gives up — [`ApplyError::OutputCap`] on
    /// the output-node cap, [`ApplyError::Deadline`] on the armed deadline or a
    /// stop decision.
    pub fn project_var(&self, f: Tdd, x: VarId, how: crate::apply::Projection) -> Result<Tdd, ApplyError> {
        crate::apply::project::project_var_on(self, f, x, how)
    }

    /// Sum every variable in `vars` out of the structure, one at a time.
    ///
    /// `f` is consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::project_var`].
    ///
    /// # Errors
    ///
    /// As [`Engine::project_var`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::apply::Projection;
    /// # use tididi::engine::LimitSet;
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
