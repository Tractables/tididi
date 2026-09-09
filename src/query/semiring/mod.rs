//! Bottom-up evaluation of a whole diagram in a caller's algebra.
//!
//! `evaluate(&tdd, &sr)` walks the diagram the way `node_counts` does
//! and delegates every arithmetic step to an
//! [`EvalAlgebra`](crate::diagram::semiring::EvalAlgebra) impl. The algebra and
//! its exact-rational instance live in [`crate::diagram::semiring`], below the
//! diagram they value; only the walk is here.
//!
//! `model_count` does not go through this trait. It carries a u128 count with a
//! lazy `BigUint` side table and needs per-node overflow detection, which an
//! arbitrary algebra cannot express; that discipline lives in `crate::value_fold`.
//! `EvalAlgebra` is the whole-diagram oracle, not the per-fold contract.

use crate::value_fold::ColumnRetention;
use crate::diagram::semiring::EvalAlgebra;
use crate::diagram::*;
use crate::diagram::PairsIter;
use crate::engine::Engine;
use crate::vtree::{VarId, VtreeIdx};

use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

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
