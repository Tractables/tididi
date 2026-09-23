//! Arithmetic for evaluating diagrams and storing weighted values.
//!
//! [`RationalWeights`] evaluates exact weighted sums. Implement [`EvalAlgebra`]
//! for another calculation, such as a minimum cost, and pass it to
//! [`Tdd::evaluate`](crate::Tdd::evaluate).
//!
//! Marginalized diagrams store weighted results as [`WeightValue`], using
//! exact rationals or the bounded-precision [`SignedLog`] representation.

mod rational;
mod weight;

pub use rational::{LiteralWeights, RationalWeights};
pub use weight::{SignedLog, WeightValue};
pub(crate) use weight::{weight_key, WeightKey};

use crate::diagram::LeafLabel;
use crate::vtree::VarId;

/// Define a value at each leaf and how to combine values at conjunctions and alternatives.
///
/// Pass an implementation to [`Tdd::evaluate`](crate::Tdd::evaluate), or retain
/// values under changing observations with [`Tdd::evaluator`](crate::Tdd::evaluator).
/// [`RationalWeights`] computes weighted sums; the
/// [cost example](crate::guide::examples::optimization) uses minimum and addition.
///
/// # Algebraic requirements
///
/// The operations must form a commutative semiring: addition is associative and
/// commutative with [`zero`](Self::zero) as identity; multiplication is
/// associative and commutative, distributes over addition, and has zero as an
/// absorbing element. The fold needs no explicit multiplicative-identity
/// method because every product starts with its two child values.
///
/// [`leaf`](Self::leaf) supplies each variable's positive, negative, and free
/// values. Its `One` value must equal the sum of its `Pos` and `Neg` values:
/// a free variable includes both assignments, so it is not generally the
/// multiplicative identity. `LeafLabel::Zero` is handled by `zero()` and is
/// never passed to `leaf`.
///
/// The library trusts these laws; violating them can make equivalent diagrams
/// evaluate differently. A table-based implementation may store weights in
/// `self`; a stateless algebra can be a unit struct.
///
/// Count the fewest true variables in any satisfying assignment with a min-plus algebra:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, Tdd, Vtree};
/// use tididi::diagram::{EvalAlgebra, LeafLabel};
///
/// use tididi::vtree::VarId;
///
/// struct FewestTrue;
/// impl EvalAlgebra for FewestTrue {
///     type Value = usize;
///     fn zero(&self) -> usize { usize::MAX } // no satisfying assignment
///     fn leaf(&self, _: VarId, label: LeafLabel) -> usize {
///         match label {
///             LeafLabel::Pos => 1,
///             LeafLabel::Neg | LeafLabel::One => 0,
///             LeafLabel::Zero => self.zero(),
///         }
///     }
///     fn add_assign(&self, a: &mut usize, b: &usize) { *a = (*a).min(*b); }
///     fn mul(&self, a: &usize, b: &usize) -> usize { a.saturating_add(*b) }
/// }
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, 2])? & literal(&vtree, 3)?;
/// assert_eq!(f.evaluate(&FewestTrue)?, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait EvalAlgebra {
    /// The value computed for a node.
    type Value: Clone;
    /// The additive identity.
    fn zero(&self) -> Self::Value;
    /// Value of leaf `label` for variable `var`. `LeafLabel::Zero` is never
    /// passed here; `evaluate` returns `zero()` for it.
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value;
    /// Accumulate `other` into `acc` (the semiring `+`).
    fn add_assign(&self, acc: &mut Self::Value, other: &Self::Value);
    /// The semiring product of `a` and `b`.
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
}
