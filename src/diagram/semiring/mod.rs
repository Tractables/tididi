//! The semiring a diagram's values are drawn from.
//!
//! A marginal level stores one value per node, and that value is a model
//! count, an exact rational weight, or a bounded-precision log-domain weight.
//! [`WeightValue`] is the weighted value type and [`SignedLog`] its log-domain
//! representation; [`EvalAlgebra`] is the trait a caller implements to fold a
//! whole diagram in an algebra of its own, and [`RationalWeights`] is the
//! exact-rational instance of it.
//!
//! The walk that consumes them is [`Tdd::evaluate`](crate::Tdd::evaluate).

mod rational;
mod weight;

pub use rational::{LiteralWeights, RationalWeights};
pub use weight::{SignedLog, WeightValue};
pub(crate) use weight::{weight_key, WeightKey};

use crate::diagram::LeafLabel;
use crate::vtree::VarId;

/// Arithmetic for evaluating a structural diagram with values at its leaves.
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
/// use tididi::{Tdd, Vtree};
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
/// let tree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&tree, [1, 2])? & Tdd::literal(&tree, 3)?;
/// assert_eq!(f.evaluate(&FewestTrue)?, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
