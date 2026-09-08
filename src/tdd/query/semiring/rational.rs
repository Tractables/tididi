//! Exact weighted model counting over arbitrary-precision rationals.

use num_rational::BigRational;
use num_traits::{One, Zero};

use super::Semiring;
use crate::tdd::types::LeafLabel;
use crate::vtree::VarId;

// ── Exact rational weighted model counting (Track 4 PWMC) ─────────────────────

/// Exact weighted model counting over `num_rational::BigRational`.
///
/// This evaluates the weighted sum in exact arbitrary-precision rational
/// arithmetic — the competition precision-category-A requirement for Track 4
/// (PWMC). It is the production on-ramp for weighted/algebraic counting:
/// compile the TDD without marginalization (Boolean structure intact), then
/// `evaluate` it under this semiring.
///
/// `w_pos[v]` / `w_neg[v]` are the literal weights of variable `v`. A free
/// variable (`One` leaf) contributes `w_pos[v] + w_neg[v]`. Weights may be
/// zero: a satisfiable instance can then have weighted value 0 (weight
/// cancellation) — this is NOT unsat, so callers must not treat a 0 result
/// as structural ⊥ (the zero-cancellation hazard lives only at the output
/// SAT/UNSAT label, never in this arithmetic).
#[derive(Clone)]
pub struct RationalSemiring {
    /// Positive-literal weight of each variable, indexed by `VarId`.
    pub w_pos: Vec<BigRational>,
    /// Negative-literal weight of each variable, indexed by `VarId`.
    pub w_neg: Vec<BigRational>,
}

impl RationalSemiring {
    /// Build from per-variable `(w_neg, w_pos)` literal weights. Zero and any
    /// nonneg/negative rational weight is permitted — exactness imposes no
    /// sign restriction.
    pub fn from_weights(weights: &[(BigRational, BigRational)]) -> Self {
        let mut w_neg = Vec::with_capacity(weights.len());
        let mut w_pos = Vec::with_capacity(weights.len());
        for (wn, wp) in weights {
            w_neg.push(wn.clone());
            w_pos.push(wp.clone());
        }
        RationalSemiring { w_pos, w_neg }
    }

    /// All variables uniform with weight 1 on each polarity. Then
    /// `evaluate(&tdd, &sr)` equals the (integer) model count of `tdd`,
    /// as an exact `BigRational` with denominator 1.
    pub fn unit(num_vars: usize) -> Self {
        RationalSemiring {
            w_pos: vec![BigRational::one(); num_vars],
            w_neg: vec![BigRational::one(); num_vars],
        }
    }
}

impl Semiring for RationalSemiring {
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
