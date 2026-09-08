//! Existential projection of a single variable from a TDD.
//!
//! Implements `∃x.T` via cofactor-OR decomposition:
//!   `project_var(T`, x)  =  `apply_or(T`[x←⊤], T[x←⊥])
//!
//! The cofactors are computed by rewriting every parent-level pair list that
//! references x's leaf, replacing Pos/Neg labels with One (value fixed) or
//! dropping them (variable excluded), then minimizing.

use std::cell::Cell;

use crate::scoped::Scoped;

use crate::apply::apply_or;
use crate::apply::condition::{condition_leaf, Polarity};
use crate::diagram::{LeafLabel, LocalNodeIdx, Tdd};
use crate::vtree::VarId;

mod scoped;

pub use scoped::*;

pub(crate) const POS: LocalNodeIdx = LocalNodeIdx(LeafLabel::Pos as u32);
pub(crate) const NEG: LocalNodeIdx = LocalNodeIdx(LeafLabel::Neg as u32);
pub(crate) const ONE: LocalNodeIdx = LocalNodeIdx(LeafLabel::One as u32);

/// Returns a fully minimized canonical TDD representing ∃x. t.
///
/// Precondition: `x` must be a leaf in `t.vtree`, and no ancestor of x's leaf
/// may be a marginal level (i.e., must be called on a full/non-mc TDD).
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
/// use tididi::apply::project_var;
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
/// assert_eq!(f.model_count(), BigUint::from(2u32));
/// // ∃x2. (x1 ∧ x2) == x1: forgetting x2 frees it, doubling the count.
/// let g = project_var(&f, VarId(1));
/// assert_eq!(g.model_count(), BigUint::from(4u32));
/// ```
///
/// # Panics
///
/// Panics if `x` is not a variable present in `t.vtree`.
pub fn project_var(f: &Tdd, x: VarId) -> Tdd {
    if f.is_zero() {
        return f.clone();
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
    // project_var is sound iff no ancestor of x's leaf has a marginal level.
    // Levels in disjoint sub-vtrees may be marginal without affecting correctness.
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

    let pos_cofactor = condition_leaf(f, leaf_idx, Polarity::Pos);
    let neg_cofactor = condition_leaf(f, leaf_idx, Polarity::Neg);
    apply_or(pos_cofactor, neg_cofactor)
}

/// Existentially quantify all variables in `vars` from TDD `t`, one at a time.
/// Returns a fully minimized TDD representing ∃vars. t.
pub fn project_vars(f: &Tdd, vars: &[VarId]) -> Tdd {
    let mut result = f.clone();
    for &x in vars {
        result = project_var(&result, x);
    }
    result
}


/// Crash-safe-and-fast existential forget: dispatch per call between the fast
/// cofactor path (`project_vars`) and the crash-safe scoped path
/// (`project_vars_scoped`).
///
/// The cofactor `apply_or` path (`project_var`) panics when a **width>1
/// marginal** level survives into both cofactors: marginal×marginal at k>1 has
/// no pair structure to conjoin (identity fast-paths need k==1). The scoped
/// path carries such sibling levels verbatim without dereferencing them, so it
/// never crashes — but its ownership-regroup fan-out makes it slower on
/// structures that the cofactor path handles fine, and defaulting to it was
/// measured to lose solves on diagrams that have no wide marginal level to
/// protect.
///
/// So: use scoped whenever ANY marginal level is present, else the faster
/// cofactor path. The cofactor path implements `apply_or` via De Morgan
/// (`a∨b = ¬(¬a ∧ ¬b)`), and *negating* across a marginal level is unsound /
/// crashes — not only the width>1 marginal×marginal case (`mc2025_track1_189_bva`)
/// but also width-1 marginals, which trip the apply marginal-child dispatch
/// ("general product-grid path reached with a marginal child")
/// seen on `mc2026_track3_169` under during-compile forget. So the safe predicate
/// is "any marginal level", not "width>1". Scoped carries marginal levels
/// verbatim without negating, so it is sound on all of them. Projection never
/// *creates* marginal levels (only marginalization does), so one scan of the
/// input covers the whole batch.
pub fn project_vars_gated(t: &Tdd, vars: &[VarId]) -> Tdd {
    let has_marginal = t.levels.iter().any(|l| l.is_marginal());
    // PREFER_SCOPED_PROJECTION: forces the negation-free scoped path even with no
    // marginal level. Set by the driver for the duration of a
    // conditioning-branch compile: the cofactor path's `negate_tdd` clones the
    // whole TDD and `handle_alloc_error`-ABORTS (uncatchable, rc=134) when a
    // branch's negation balloons — observed on track-3 091 (a single 16 GiB
    // clone at a depth-4 branch). Scoped carries levels verbatim without negating,
    // so it never balloons that way. We force it ONLY inside conditioning (not the
    // single-shot path, where a blind scoped default measured −2 solves): a
    // conditioning branch is already a fallback on a hard instance, its
    // post-preprocessing/min-depth structure is small, and converting an uncatchable abort
    // into a sound (slower) compile is strictly better there.
    let force_scoped = PREFER_SCOPED_PROJECTION.with(|c| c.get());
    if has_marginal || force_scoped {
        project_vars_scoped(t, vars)
    } else {
        project_vars(t, vars)
    }
}

thread_local! {
    /// When true, `project_vars_gated` takes the negation-free scoped projection
    /// path regardless of marginal-level presence. Installed by the driver
    /// around conditioning-branch compiles to avoid the cofactor path's
    /// uncatchable big-negation abort. See `project_vars_gated`.
    pub(crate) static PREFER_SCOPED_PROJECTION: Cell<bool> = const { Cell::new(false) };
}

/// Forces scoped projection for its lifetime; the prior setting is restored
/// on drop, so nested compiles are safe.
pub struct ScopedProjectionGuard(#[allow(dead_code)] Scoped<Cell<bool>>);
impl ScopedProjectionGuard {
    /// Force scoped projection until the guard drops.
    pub fn new() -> Self {
        ScopedProjectionGuard(Scoped::install(&PREFER_SCOPED_PROJECTION, true))
    }
}
impl Default for ScopedProjectionGuard {
    fn default() -> Self {
        Self::new()
    }
}
