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

use num_bigint::BigUint;

use crate::diagram::{ChildDecoder, ChildRef, EncodedChildRef, LeafLabel, PairsIter, ValueRef};
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
/// # Marginal levels
///
/// A level that a marginalization summed out keeps, for each of its nodes,
/// only the number of assignments to its subtree's variables that the node
/// holds. An algebra that implements [`count`](Self::count) can evaluate such
/// a diagram: the fold reads `count(n)` where the structure would have
/// summed `n` products. The two agree exactly when the algebra values every
/// summed variable neutrally — `Pos` and `Neg` both the multiplicative
/// identity `count(1)`, so `One` is `count(2)` — because each of the `n`
/// assignments then contributes one identity. On those terms evaluating
/// [`Engine::and_marginalizing`](crate::Engine::and_marginalizing)'s result
/// equals evaluating the plain conjunction, without its summed structure.
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

    /// The value of `n` assignments to variables this algebra values
    /// neutrally: the multiplicative identity added to itself `n` times.
    ///
    /// Evaluation reads it at a marginal level, once per stored count and
    /// once per distinct count a reference carries inline; see *Marginal
    /// levels* above for when the result is exact. It must be additive and
    /// multiplicative in `n` (`count(m + n) = count(m) + count(n)`,
    /// `count(m × n) = count(m) × count(n)`), with `count(1)` the identity.
    /// `None` for a count the evaluation reads makes it return
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel)
    /// for that level; the default values no count, so a diagram with a
    /// marginal level is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{EvalAlgebra, LeafLabel};
    /// use tididi::vtree::VarId;
    ///
    /// // Model counts, which value every variable neutrally.
    /// struct Count;
    /// impl EvalAlgebra for Count {
    ///     type Value = u64;
    ///     fn zero(&self) -> u64 { 0 }
    ///     fn leaf(&self, _: VarId, label: LeafLabel) -> u64 { if label == LeafLabel::One { 2 } else { 1 } }
    ///     fn add_assign(&self, acc: &mut u64, other: &u64) { *acc += other; }
    ///     fn mul(&self, a: &u64, b: &u64) -> u64 { a * b }
    ///     fn count(&self, n: &BigUint) -> Option<u64> { u64::try_from(n).ok() }
    /// }
    ///
    /// // (x1 ∨ x2) ∧ (x3 ∨ x4), the subtree over x1 and x2 summed out.
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = vtree.children(vtree.root());
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// let counted = engine.and_marginalizing(f, g, &[left])?;
    /// assert!(counted.level(left).is_marginal());
    /// assert_eq!(counted.evaluate(&Count)?, 9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn count(&self, _n: &BigUint) -> Option<Self::Value> {
        None
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
/// A marginal level is read through [`count`](Self::count), on
/// [`EvalAlgebra::count`]'s terms. A column of a marginal level whose parent
/// carries counts inline in its references has a slot for each distinct such
/// count after its reference slots, so every pair still reaches
/// [`fold`](Self::fold) as two slots.
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

    /// Write slot `slot` of `col`, the column of vtree node `t`, with the
    /// value of `n` assignments to variables this algebra values neutrally
    /// ([`EvalAlgebra::count`]), and return `true`.
    ///
    /// Evaluation calls it for each slot of a marginal level and for each
    /// slot after the reference slots that stands for a count carried
    /// inline. `false` for a count makes the evaluation return
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel)
    /// for that level; the default writes nothing and returns `false`, so a
    /// diagram with a marginal level is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{ColumnAlgebra, LeafLabel, SlotPairs};
    /// use tididi::vtree::{VarId, VtreeIdx};
    ///
    /// // Model counts, one flat buffer per level.
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
    ///     fn count(&self, _: VtreeIdx, slot: usize, n: &BigUint, col: &mut Vec<u128>) -> bool {
    ///         u128::try_from(n).map(|n| col[slot] = n).is_ok()
    ///     }
    /// }
    ///
    /// // (x1 ∨ x2) ∧ (x3 ∨ x4), the subtree over x1 and x2 summed out.
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = vtree.children(vtree.root());
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// let counted = engine.and_marginalizing(f, g, &[left])?;
    /// assert_eq!(counted.evaluate_columns(&Counts)?, 9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn count(&self, _t: VtreeIdx, _slot: usize, _n: &BigUint, _col: &mut Self::Column) -> bool {
        false
    }
}

/// The pairs of one node as the slots of their children:
/// `(left slot, right slot)`, in the node's pair order.
///
/// A side naming a node, or a value a marginal child stores, is its index;
/// a side carrying a count inline is the slot after the child's reference
/// slots that holds that count's value. Evaluation hands them to
/// [`ColumnAlgebra::fold`]; the [`ColumnAlgebra`] example sums a product
/// over them.
#[derive(Clone, Debug)]
pub struct SlotPairs<'a> {
    pairs: PairsIter<'a>,
    left: InlineSlots<'a>,
    right: InlineSlots<'a>,
}

impl<'a> SlotPairs<'a> {
    /// `pairs`, each side read through the inline slots of its child.
    pub(crate) fn new(pairs: PairsIter<'a>, left: InlineSlots<'a>, right: InlineSlots<'a>) -> Self {
        SlotPairs { pairs, left, right }
    }
}

/// Where the counts one side carries inline sit in its child's column: after
/// the child's `base` reference slots, in the ascending order of `counts`.
/// Empty for a child whose references carry no count.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InlineSlots<'a> {
    pub(crate) base: usize,
    pub(crate) counts: &'a [u32],
}

impl InlineSlots<'_> {
    /// The column slot side `r` names.
    #[inline]
    fn slot(self, r: EncodedChildRef) -> usize {
        if self.counts.is_empty() {
            return r.raw() as usize;
        }
        match ChildDecoder::marginal().child(r) {
            ChildRef::Value(ValueRef::Inline(c)) => {
                self.base + self.counts.binary_search(&c).expect("every inline count has a slot")
            }
            _ => r.raw() as usize,
        }
    }
}

impl Iterator for SlotPairs<'_> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        let (left, right) = (self.left, self.right);
        self.pairs.next().map(|p| (left.slot(p.left), right.slot(p.right)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.pairs.size_hint()
    }
}

impl ExactSizeIterator for SlotPairs<'_> {}
