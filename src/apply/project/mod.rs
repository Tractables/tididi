//! Existential projection (∃-forget) of variables from a diagram.
//!
//! Two rewrites compute the same function and the choice between them is
//! [`Projection`]:
//!
//! - **Cofactor-OR** — `∃x.T = T[x←⊤] ∨ T[x←⊥]`, with the cofactors computed by
//!   rewriting every parent-level pair list that references x's leaf. Fast, and
//!   the measured default on diagrams it can handle.
//! - **Structural** — a leaf-to-root in-place regroup that never calls apply or
//!   negate, so it is sound where the cofactor rewrite is not.

use crate::engine::Engine;

use crate::apply::apply_or;
use crate::apply::condition::{condition_leaf, Polarity};
use crate::diagram::{LeafLabel, NodeIdx, Tdd};
use crate::vtree::VarId;

mod structural;

pub(crate) const POS: NodeIdx = NodeIdx(LeafLabel::Pos as u32);
pub(crate) const NEG: NodeIdx = NodeIdx(LeafLabel::Neg as u32);
pub(crate) const ONE: NodeIdx = NodeIdx(LeafLabel::One as u32);

/// Which rewrite an ∃-forget uses.
///
/// The two agree on every diagram both accept, so this is a cost/robustness
/// choice, not a semantic one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// is memory: cofactoring negates, negation clones the whole diagram, and a
    /// clone that cannot be served calls `handle_alloc_error` — an abort no
    /// caller can catch. The structural rewrite copies levels verbatim and
    /// cannot fail that way. It is the slower of the two on structures the
    /// cofactor rewrite handles, so this is for a caller that has already
    /// decided robustness beats speed.
    Structural,
}

/// The implementation behind [`Engine::project_var`](crate::Engine::project_var).
pub(crate) fn project_var_on(eng: &Engine, f: &Tdd, x: VarId, how: Projection) -> Tdd {
    if f.is_zero() {
        return f.clone();
    }
    if how == Projection::Structural || f.levels.iter().any(|l| l.is_marginal()) {
        return structural::project_var_structural(f, x);
    }
    let vtree = &f.vtree;
    assert!(
        x.idx() < vtree.num_vars() as usize,
        "project_var: variable {:?} is not in the vtree (var_to_leaf len={})",
        x,
        vtree.num_vars()
    );
    let leaf_idx = vtree.leaf_of(x).expect("the vtree carries this variable");
    assert!(
        vtree.node(leaf_idx).is_leaf(),
        "project_var: var_to_leaf[{:?}] = {:?} is not a leaf node",
        x,
        leaf_idx
    );
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

    let mut pos_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Positive);
    let mut neg_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Negative);
    // The store travels with the diagram. Each cofactor is a clone of `f` and
    // carries one, but the disjunction negates, and negation copies levels
    // without the side table, so the store is moved across by hand. No values
    // change on the way: this path runs only when no level is marginal, so
    // nothing in the store is referenced by anything being rewritten.
    let ws = pos_cofactor.take_weights().or_else(|| neg_cofactor.take_weights());
    let mut out = apply_or(pos_cofactor, neg_cofactor);
    out.weights = ws;
    out
}

/// Existentially quantify all variables in `vars`, one at a time.
/// Returns a fully minimized diagram representing ∃vars. t.
pub(crate) fn project_vars_on(eng: &Engine, f: &Tdd, vars: &[VarId], how: Projection) -> Tdd {
    let mut result = f.clone();
    for &x in vars {
        result = project_var_on(eng, &result, x, how);
    }
    result
}

/// Sum `x` out of the structure, on a transient engine.
///
/// [`Engine::project_var`] is this operation on a caller's engine, where the
/// per-level buffers stay warm between calls.
#[must_use]
pub fn project_var(f: &Tdd, x: VarId, how: Projection) -> Tdd {
    project_var_on(&Engine::new(), f, x, how)
}

/// Sum every variable in `vars` out of the structure, one at a time, on a
/// transient engine.
///
/// [`Engine::project_vars`] is this operation on a caller's engine.
#[must_use]
pub fn project_vars(f: &Tdd, vars: &[VarId], how: Projection) -> Tdd {
    project_vars_on(&Engine::new(), f, vars, how)
}

/// The projection entry points on a caller's engine.
impl crate::engine::Engine {
    /// Returns a fully minimized canonical diagram representing ∃x. t.
    ///
    /// Precondition: `x` must be a leaf in `t.vtree`, and no ancestor of x's leaf
    /// may be a marginal level (i.e., must be called on a full/non-mc diagram).
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
    /// let g = eng.project_var(&f, VarId(1), tididi::apply::Projection::Automatic);
    /// assert_eq!(g.model_count(), BigUint::from(4u32));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a variable present in `t.vtree`.
    #[must_use]
    pub fn project_var(&self, f: &Tdd, x: VarId, how: crate::apply::Projection) -> Tdd {
        crate::apply::project::project_var_on(self, f, x, how)
    }

    /// Sum every variable in `vars` out of the structure, one at a time.
    #[must_use]
    pub fn project_vars(&self, f: &Tdd, vars: &[VarId], how: crate::apply::Projection) -> Tdd {
        crate::apply::project::project_vars_on(self, f, vars, how)
    }
}
