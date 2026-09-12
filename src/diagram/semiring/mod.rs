//! The semiring a diagram's values are drawn from.
//!
//! A marginal level stores one value per node, and that value is a model
//! count, an exact rational weight, or a bounded-precision log-domain weight.
//! [`WeightValue`] is the weighted value type and [`SignedLog`] its log-domain
//! representation; [`EvalAlgebra`] is the trait a caller implements to fold a
//! whole diagram in an algebra of its own, and [`RationalWeights`] is the
//! exact-rational instance of it.
//!
//! The walk that consumes them is [`query::evaluate`](crate::query::evaluate()).

mod rational;
mod weight;

pub use rational::RationalWeights;
pub use weight::{SignedLog, WeightValue};
pub(crate) use weight::{weight_key, WeightKey};

use crate::diagram::LeafLabel;
use crate::vtree::VarId;

/// Commutative semiring over `Value`, with leaf values keyed by
/// `(VarId, LeafLabel)` so weight-table semirings (weighted model counting,
/// say) can look up per-variable weights.
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
