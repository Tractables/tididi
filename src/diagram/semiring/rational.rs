//! Exact weighted model counting over arbitrary-precision rationals.

use num_rational::BigRational;
use num_traits::{One, Zero};

use super::EvalAlgebra;
use crate::diagram::LeafLabel;
use crate::vtree::VarId;

// ── Exact rational weighted model counting ────────────────────────────────────

/// Exact weighted model counting over `num_rational::BigRational`.
///
/// `w_pos[v]` / `w_neg[v]` are the literal weights of variable `v`. A free
/// variable (`One` leaf) contributes `w_pos[v] + w_neg[v]`. Weights may be
/// zero or negative, so a satisfiable diagram can evaluate to 0; a caller
/// must not read that as unsatisfiable.
#[derive(Clone)]
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
    /// Build from per-variable `(w_neg, w_pos)` literal weights; any rational
    /// is permitted.
    pub fn from_weights(weights: &[(BigRational, BigRational)]) -> Self {
        let mut w_neg = Vec::with_capacity(weights.len());
        let mut w_pos = Vec::with_capacity(weights.len());
        for (wn, wp) in weights {
            w_neg.push(wn.clone());
            w_pos.push(wp.clone());
        }
        RationalWeights { w_pos, w_neg }
    }

    /// Build from the two per-polarity weight vectors, both indexed by
    /// `VarId`.
    ///
    /// `None` if the two vectors differ in length.
    pub fn new(w_pos: Vec<BigRational>, w_neg: Vec<BigRational>) -> Option<Self> {
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
    /// `evaluate(&tdd, &semiring)` equals the (integer) model count of `tdd`,
    /// as an exact `BigRational` with denominator 1.
    pub fn unit(num_vars: usize) -> Self {
        RationalWeights {
            w_pos: vec![BigRational::one(); num_vars],
            w_neg: vec![BigRational::one(); num_vars],
        }
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
