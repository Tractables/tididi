//! Exact weighted model counting over arbitrary-precision rationals.

use num_rational::BigRational;
use num_traits::{One, Zero};

use super::EvalAlgebra;
use crate::diagram::{LeafLabel, TddBuildError};
use crate::vtree::VarId;

// ── Exact rational weighted model counting ────────────────────────────────────

/// The weights of a variable's negative and positive literals.
///
/// `LiteralWeights<BigRational>` supplies one variable's weights to
/// [`RationalWeights::from_literals`]; `LiteralWeights<Vec<BigRational>>`
/// supplies the two polarity columns to [`RationalWeights::from_polarities`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiteralWeights<T> {
    /// Weight of the negative literal, or the column of negative-literal weights.
    pub negative: T,
    /// Weight of the positive literal, or the column of positive-literal weights.
    pub positive: T,
}

/// Exact weighted model counting over `num_rational::BigRational`.
///
/// Table entries are indexed by [`VarId`], not by leaf position. Supply
/// an entry for every named variable, including placeholders for gaps in a sparse
/// tree; `vtree.num_vars()` entries suffice. A missing entry can panic during evaluation.
/// A free variable contributes the sum of its positive and negative weights.
/// Unit weights reproduce the model count; nonnegative complementary weights
/// give independent-variable probabilities, as in [`Tdd::evaluate`](crate::Tdd::evaluate).
///
/// Weights may also be zero or negative. A zero result therefore does not imply
/// that the Boolean function is unsatisfiable, and no normalization is performed.
/// [`Tdd::evaluate`](crate::Tdd::evaluate) shows the dependency setup and
/// a complete probability calculation.
#[derive(Clone, PartialEq, Eq)]
pub struct RationalWeights {
    /// Positive-literal weight of each variable, indexed by `VarId`.
    w_pos: Vec<BigRational>,
    /// Negative-literal weight of each variable, indexed by `VarId`.
    w_neg: Vec<BigRational>,
}

impl std::fmt::Debug for RationalWeights {
    /// How many variables the table covers; the rationals themselves would
    /// print unboundedly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RationalWeights").field("variables", &self.w_pos.len()).finish()
    }
}

impl RationalWeights {
    /// The number of variables covered by the literal table.
    pub(crate) fn num_vars(&self) -> usize {
        self.w_pos.len()
    }

    /// Build from named literal weights in variable-index order; any rational is permitted.
    ///
    /// ```
    /// use tididi::diagram::{LiteralWeights, RationalWeights};
    /// use tididi::vtree::VarId;
    /// let weights = RationalWeights::from_literals(&[
    ///     LiteralWeights {
    ///         negative: num_rational::BigRational::from_integer((-2).into()),
    ///         positive: num_rational::BigRational::from_integer(3.into()),
    ///     },
    /// ]);
    /// assert_eq!(weights.neg_weight(VarId(1)).to_integer(), (-2).into());
    /// assert_eq!(weights.pos_weight(VarId(1)).to_integer(), 3.into());
    /// ```
    pub fn from_literals(weights: &[LiteralWeights<BigRational>]) -> Self {
        let mut w_neg = Vec::with_capacity(weights.len());
        let mut w_pos = Vec::with_capacity(weights.len());
        for weight in weights {
            w_neg.push(weight.negative.clone());
            w_pos.push(weight.positive.clone());
        }
        RationalWeights { w_pos, w_neg }
    }

    /// Build from named polarity columns indexed by variable, returning `None` for unequal lengths.
    ///
    /// ```
    /// use tididi::diagram::{LiteralWeights, RationalWeights};
    /// let half = num_rational::BigRational::new(1.into(), 2.into());
    /// let columns = LiteralWeights {
    ///     negative: vec![half.clone()],
    ///     positive: vec![half],
    /// };
    /// assert!(RationalWeights::from_polarities(columns).is_some());
    /// ```
    pub fn from_polarities(weights: LiteralWeights<Vec<BigRational>>) -> Option<Self> {
        let LiteralWeights { negative: w_neg, positive: w_pos } = weights;
        (w_pos.len() == w_neg.len()).then_some(RationalWeights { w_pos, w_neg })
    }

    /// The positive-literal weight of `var`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is past the table.
    pub fn pos_weight(&self, var: VarId) -> &BigRational {
        &self.w_pos[var.idx()]
    }

    /// The negative-literal weight of `var`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is past the table.
    pub fn neg_weight(&self, var: VarId) -> &BigRational {
        &self.w_neg[var.idx()]
    }

    /// All variables uniform with weight 1 on each polarity. Then
    /// `tdd.evaluate(&semiring)?` equals the (integer) model count of `tdd`,
    /// as an exact `BigRational` with denominator 1.
    pub fn unit(num_vars: usize) -> Self {
        RationalWeights {
            w_pos: vec![BigRational::one(); num_vars],
            w_neg: vec![BigRational::one(); num_vars],
        }
    }

    /// This table restated in another variable numbering: variable
    /// `VarId(i + 1)` of the result has the weights of `local_to_global[i]`
    /// here.
    ///
    /// Use it for a diagram compiled over some of the variables in a compact
    /// numbering of its own, such as one component of a formula. The result
    /// covers `local_to_global.len()` variables and owns its entries; a
    /// variable may appear more than once.
    ///
    /// ```
    /// use num_rational::BigRational;
    /// use tididi::diagram::{LiteralWeights, RationalWeights, TddBuildError};
    /// use tididi::vtree::VarId;
    /// let w = |n: i32| BigRational::from_integer(n.into());
    /// let global = RationalWeights::from_literals(&[
    ///     LiteralWeights { negative: w(1), positive: w(2) },
    ///     LiteralWeights { negative: w(3), positive: w(4) },
    ///     LiteralWeights { negative: w(5), positive: w(6) },
    /// ]);
    /// // A component over variables 3 and 1, numbered 1 and 2 within it.
    /// let local = global.restated(&[VarId(3), VarId(1)])?;
    /// assert_eq!(local.pos_weight(VarId(1)), &w(6));
    /// assert_eq!(local.neg_weight(VarId(2)), &w(1));
    /// assert_eq!(global.restated(&[VarId(4)]), Err(TddBuildError::MissingVariableWeight(VarId(4))));
    /// # Ok::<(), TddBuildError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`TddBuildError::MissingVariableWeight`] for a variable this table does
    /// not cover, or for `VarId(0)`, which names no variable.
    pub fn restated(&self, local_to_global: &[VarId]) -> Result<Self, TddBuildError> {
        if let Some(&var) = local_to_global.iter().find(|var| var.0 == 0 || var.idx() >= self.num_vars()) {
            return Err(TddBuildError::MissingVariableWeight(var));
        }
        Ok(RationalWeights {
            w_pos: local_to_global.iter().map(|&var| self.w_pos[var.idx()].clone()).collect(),
            w_neg: local_to_global.iter().map(|&var| self.w_neg[var.idx()].clone()).collect(),
        })
    }
}

impl EvalAlgebra for RationalWeights {
    type Value = BigRational;

    fn zero(&self) -> BigRational { BigRational::zero() }

    fn leaf(&self, v: VarId, label: LeafLabel) -> BigRational {
        let i = v.idx();
        match label {
            LeafLabel::Pos => self.w_pos[i].clone(),
            LeafLabel::Neg => self.w_neg[i].clone(),
            LeafLabel::One => &self.w_pos[i] + &self.w_neg[i],
            LeafLabel::Zero => BigRational::zero(),
        }
    }

    #[inline]
    fn add_assign(&self, acc: &mut BigRational, other: &BigRational) { *acc += other; }

    #[inline]
    fn mul(&self, a: &BigRational, b: &BigRational) -> BigRational { a * b }
}

#[cfg(test)]
#[path = "tests/rational.rs"]
mod tests;
