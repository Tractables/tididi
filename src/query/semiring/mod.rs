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

use crate::diagram::*;
use crate::vtree::{VarId, VtreeIdx};

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

    let mut counts: Vec<Vec<S::Value>> = (0..tdd.vtree.num_nodes())
        .map(|i| vec![sr.zero(); tdd.effective_width(VtreeIdx(i as u32))])
        .collect();
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());

    for (t, var) in tdd.vtree.leaf_bottomup() {
        let ti = t.idx();
        for i in 0..LEAF_WIDTH {
            let label = LeafLabel::from_idx(i);
            counts[ti][i] = match label {
                LeafLabel::Zero => sr.zero(),
                _ => sr.leaf(var, label),
            };
        }
    }
    for (t, left, right) in tdd.vtree.internal_bottomup() {
        let ti = t.idx();
        let li = left.idx();
        let ri = right.idx();
        let left_marg = tdd.levels[li].is_marginal();
        let right_marg = tdd.levels[ri].is_marginal();
        for (i, pairs) in tdd.levels[ti].internal_inputs_iter() {
            let mut total = sr.zero();
            for pair in pairs {
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("evaluate: a marginal level's inline ref (see the precondition)"),
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("evaluate: a marginal level's inline ref (see the precondition)"),
                };
                let prod = sr.mul(
                    &counts[li][l],
                    &counts[ri][r],
                );
                sr.add_assign(&mut total, &prod);
            }
            counts[ti][i] = total;
        }
        // The vtree is a tree: a node has exactly ONE parent, so its column has
        // exactly one consumer and is dead the moment that parent's column is
        // complete. Free it here rather than carrying every level's values to
        // the end of the walk — the live set becomes the frontier, not the
        // whole diagram. `out_t` is the one column read after the walk (it is
        // the root under the output-at-root invariant, hence never a child
        // here, but the walk does not rely on that).
        for c in [li, ri] {
            if c != out_t {
                counts[c] = Vec::new();
            }
        }
    }

    counts[out_t][out_i].clone()
}
