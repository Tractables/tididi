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
pub(crate) use weight::{same_value, weight_key, WeightKey};

use crate::diagram::{LeafLabel, PairsIter};
use crate::vtree::{VarId, VtreeIdx};

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
    /// Accumulate the product of `a` and `b` into `acc`: `acc += a × b`.
    ///
    /// The default is [`mul`](Self::mul) followed by
    /// [`add_assign`](Self::add_assign). An algebra whose values own buffers
    /// may override it to accumulate the product without forming it first;
    /// the result must equal the default's.
    fn mul_add(&self, acc: &mut Self::Value, a: &Self::Value, b: &Self::Value) {
        let product = self.mul(a, b);
        self.add_assign(acc, &product);
    }
    /// The sum of `a × b` over `pairs`: a node's value from the values of
    /// its pairs' children, left then right.
    ///
    /// The default accumulates each product with [`mul_add`](Self::mul_add)
    /// in the order given. A node's pairs come in no meaningful order, and
    /// an algebra whose product costs more than its sum may override this
    /// to regroup them by distributivity, `a × b + a' × b = (a + a') × b`:
    /// every pair that names one child borrows that child's one value, so
    /// the values' addresses tell which pairs share a side. The result
    /// must equal the default's.
    fn sum_of_products<'v>(
        &self,
        pairs: impl ExactSizeIterator<Item = (&'v Self::Value, &'v Self::Value)>,
    ) -> Self::Value
    where
        Self::Value: 'v,
    {
        let mut acc = self.zero();
        for (a, b) in pairs {
            self.mul_add(&mut acc, a, b);
        }
        acc
    }
}

/// Evaluate a diagram into storage the algebra owns, one column per level.
///
/// [`EvalAlgebra`] gives every node an owned value, which the library keeps
/// in columns it allocates. A `ColumnAlgebra` owns the columns instead: one
/// per vtree node, holding one slot per node of that level, which the fold
/// writes in place. Each pair reaches [`fold`](Self::fold) as the slots of its
/// two children, so an algebra whose values have a fixed width can keep a level
/// in one flat buffer, and one whose representation differs between levels
/// can choose it per vtree node. Pass an implementation to
/// [`Tdd::evaluate_columns`](crate::Tdd::evaluate_columns).
///
/// # Algebraic requirements
///
/// The requirements are [`EvalAlgebra`]'s: a node's slot must hold the sum,
/// over its pairs, of the product of the left child's value and the right
/// child's, and a leaf's `One` slot must hold the sum of its `Pos` and `Neg`
/// slots. The library trusts these laws.
///
/// Count models, one flat buffer per level:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::diagram::{ColumnAlgebra, LeafLabel, SlotPairs};
/// use tididi::vtree::{VarId, VtreeIdx};
///
/// struct Counts;
/// impl ColumnAlgebra for Counts {
///     type Column = Vec<u128>;
///     type Value = u128;
///     fn zero(&self) -> u128 { 0 }
///     fn column(&self, _: VtreeIdx, width: usize) -> Vec<u128> { vec![0; width] }
///     fn leaf(&self, _: VtreeIdx, _: VarId, label: LeafLabel, col: &mut Vec<u128>) {
///         col[label as usize] = if label == LeafLabel::One { 2 } else { 1 };
///     }
///     fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<u128>, right: &Vec<u128>, out: &mut Vec<u128>) {
///         out[slot] = pairs.map(|(l, r)| left[l] * right[r]).sum();
///     }
///     fn read(&self, _: VtreeIdx, col: Vec<u128>, slot: usize) -> u128 { col[slot] }
/// }
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// assert_eq!(f.evaluate_columns(&Counts)?, 6);
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub trait ColumnAlgebra {
    /// One level's slots.
    type Column: Default;
    /// The value read at the output node.
    type Value;
    /// The value of the constant-false diagram, which has no output slot.
    fn zero(&self) -> Self::Value;
    /// A column of `width` slots for vtree node `t`, before any is written.
    fn column(&self, t: VtreeIdx, width: usize) -> Self::Column;
    /// Write slot `label as usize` of the column of vtree leaf `t`, whose
    /// variable is `var`. Called once for each of `One`, `Pos` and `Neg`;
    /// `LeafLabel::Zero` has no slot.
    fn leaf(&self, t: VtreeIdx, var: VarId, label: LeafLabel, col: &mut Self::Column);
    /// Write slot `slot` of `out`, the column of internal vtree node `t`: the
    /// sum over `pairs` of the product of the values at their slots in
    /// `left` and `right`, the columns of `t`'s children.
    fn fold(
        &self,
        t: VtreeIdx,
        slot: usize,
        pairs: SlotPairs<'_>,
        left: &Self::Column,
        right: &Self::Column,
        out: &mut Self::Column,
    );
    /// The output node's value, taken from slot `slot` of `col`, the column
    /// of vtree node `t`, which the evaluation no longer needs.
    fn read(&self, t: VtreeIdx, col: Self::Column, slot: usize) -> Self::Value;
}

/// The pairs of one node as the slots of their children:
/// `(left slot, right slot)`, in the node's pair order.
#[derive(Clone, Debug)]
pub struct SlotPairs<'a>(pub(crate) PairsIter<'a>);

impl Iterator for SlotPairs<'_> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        // Evaluation reads structural levels only, whose sides are node
        // indices.
        self.0.next().map(|p| (p.left.raw() as usize, p.right.raw() as usize))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for SlotPairs<'_> {}
