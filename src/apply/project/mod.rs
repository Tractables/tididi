//! Existential projection (∃-forget) of variables from a TDD.
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
    /// marginal-child dispatch. So the presence of ANY marginal level selects
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

    let mut pos_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Pos);
    let mut neg_cofactor = condition_leaf(eng, f, leaf_idx, Polarity::Neg);
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
/// Returns a fully minimized TDD representing ∃vars. t.
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
