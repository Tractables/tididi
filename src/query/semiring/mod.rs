//! Generic bottom-up TDD evaluation parameterized by a semiring.
//!
//! `evaluate(&tdd, &sr)` performs the same bottom-up traversal as
//! `query::compute_node_counts`, but with all arithmetic delegated to a
//! `EvalAlgebra` impl. The production impl is `RationalWeights` (exact
//! arbitrary-precision rational WMC, Track 4 PWMC).
//!
//! Note: `query::model_count` (the production path) uses a hybrid
//! u128/BigUint scheme that requires per-node overflow detection — its
//! *storage* stays specialized and does not fit cleanly in the semiring
//! abstraction here. The count discipline itself (the sentinel, the
//! exact-max promotion rule, the lazy `BigUint` side table) now lives in
//! `crate::counts` (`Count`/`CountVec`), with the fold-level
//! unification across this integer path and the weighted path. `EvalAlgebra`
//! remains the whole-diagram
//! `evaluate` oracle — a traversal-level trait, not a fold-level one.

mod rational;
mod weight;

pub use rational::RationalWeights;
pub use weight::{SignedLog, WeightVal};
pub(crate) use weight::{weight_key, WeightKey, WeightMap};

use crate::counts::ColumnRetention;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::engine::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

/// Commutative semiring over `Value`, with leaf values keyed by
/// `(VarId, LeafLabel)` so weight-table semirings (e.g. WMC) can
/// look up per-variable weights.
///
/// The receiver is `&self` so an impl can hold a table it reads from (a weight
/// table, say); a stateless semiring is a unit struct.
///
/// `LeafLabel::Zero` is never passed to `leaf` — `evaluate` short-circuits
/// it to `zero()` directly.
pub trait EvalAlgebra {
    /// The semiring's carrier type.
    type Value: Clone;
    /// The additive identity.
    fn zero(&self) -> Self::Value;
    /// Value of leaf `label` for variable `var`. `LeafLabel::Zero` is never
    /// passed here — `evaluate` short-circuits it to `zero()`.
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value;
    /// Accumulate `other` into `acc` (the semiring `+`).
    fn add_assign(&self, acc: &mut Self::Value, other: &Self::Value);
    /// The semiring product of `a` and `b`.
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
}

/// Bottom-up evaluate the TDD under semiring `sr`. Returns the value of
/// the output node (or `sr.zero()` for the constant-zero TDD).
///
/// **Precondition: no level of `tdd` is marginal.** A marginal level stores
/// values rather than pairs, and this traversal reads pairs only, so a
/// marginalized diagram evaluates to `zero()` or panics on an inline ref
/// depending on how its refs are encoded. Use `query::model_count` for a
/// marginalized diagram.
pub fn evaluate<S: EvalAlgebra>(tdd: &Tdd, sr: &S) -> S::Value {
    debug_assert!(
        tdd.levels.iter().all(|l| !l.is_marginal()),
        "evaluate: the diagram has a marginal level, which this traversal cannot read",
    );
    if tdd.is_zero() {
        return sr.zero();
    }
    let eng = crate::engine::Engine::new();
    let fold = Evaluate(sr);
    let mut cols: Vec<Vec<S::Value>> = (0..tdd.vtree.num_nodes())
        .map(|i| fold.alloc(&eng, tdd.effective_width(VtreeIdx(i as u32))))
        .collect();
    fold_bottom_up_unpolled(&fold, &eng, tdd, &mut cols, ColumnRetention::Frontier, |_, _| {});
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
    cols[out_t][out_i].clone()
}

/// [`evaluate`] as an instance of the shared bottom-up walk.
struct Evaluate<'a, S>(&'a S);

impl<S: EvalAlgebra> LevelFold for Evaluate<'_, S> {
    type Value = S::Value;
    type Col = Vec<S::Value>;

    fn alloc(&self, _eng: &Engine, width: usize) -> Vec<S::Value> {
        vec![self.0.zero(); width]
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<S::Value>, i: usize, v: S::Value) {
        col[i] = v;
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> S::Value {
        match label {
            LeafLabel::Zero => self.0.zero(),
            _ => self.0.leaf(var, label),
        }
    }

    /// Unreachable under the precondition. A frozen level stores model counts,
    /// and an arbitrary semiring has no way to say what a count is worth: the
    /// algebra promises a value per LEAF, not an embedding of ℕ. Weighted
    /// evaluation of a frozen diagram is `marginal::weighted_value`, which
    /// reads the store the weighted freeze wrote.
    fn frozen_column(&self, _eng: &Engine, _tdd: &Tdd, t: VtreeIdx, _col: &mut Vec<S::Value>) {
        unreachable!(
            "evaluate: level {t:?} is marginal, which this traversal cannot read \
             (see the precondition on `evaluate`)"
        )
    }

    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<S::Value>>,
        right: Side<'_, Vec<S::Value>>,
    ) -> S::Value {
        self.sum_over_pairs(pairs, left, right)
    }
}

impl<S: EvalAlgebra> PairAlgebra for Evaluate<'_, S> {
    fn zero(&self) -> S::Value {
        self.0.zero()
    }
    fn read(&self, col: &Vec<S::Value>, i: usize) -> S::Value {
        col[i].clone()
    }
    fn inline(&self, _count: u32) -> S::Value {
        unreachable!("evaluate: a marginal level's inline ref (see the precondition)")
    }
    fn add_assign(&self, acc: &mut S::Value, v: &S::Value) {
        self.0.add_assign(acc, v);
    }
    fn mul(&self, a: &S::Value, b: &S::Value) -> S::Value {
        self.0.mul(a, b)
    }
}
