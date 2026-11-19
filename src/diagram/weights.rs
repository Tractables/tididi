//! The literal weights and arithmetic of a weighted diagram.
//!
//! A weight-marginal level holds its own values
//! ([`TddLevel::weight_values`](crate::diagram::TddLevel::weight_values)),
//! each a [`WeightValue`] (exact `BigRational` or bounded-precision
//! `SignedLog` or `ScaledFloat`); leaf base values come from the
//! [`RationalWeights`] of [`Weights`], converted to the active mode by
//! `Weights::leaf_val`.

use std::sync::Arc;

use crate::diagram::{EvalAlgebra, RationalWeights, ScaledFloat, SignedLog, WeightValue};
use crate::diagram::LeafLabel;
use crate::vtree::VarId;

/// The numeric representation a weighted marginalization's values use: exact
/// `BigRational`, or one of two bounded-precision forms whose every operation
/// is O(1) `f64` work: the `SignedLog` domain, or the cheaper `ScaledFloat`.
/// They never mix among the diagrams of one weighted build; a caller decides once per weighted run
/// and passes it once to [`Weights::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Arithmetic {
    /// Exact rationals.
    ExactRational,
    /// Bounded-precision signed log domain.
    SignedLog,
    /// Bounded-precision `f64` mantissa with a binary exponent of its own;
    /// [`ScaledFloat`] states its error bounds.
    ScaledFloat,
}

/// The literal weights and arithmetic that every summed-out level of a
/// weighted diagram is computed under.
///
/// Give them to a home with [`Home::weighted`](crate::vtree::Home::weighted):
/// every diagram on that home is then weighted, and a sum-out on it stores
/// weight values. On a vtree that is not a weighted home, attach them to one
/// diagram with [`Tdd::set_weights`](crate::Tdd::set_weights). Query either
/// with [`Tdd::weighted_value`](crate::Tdd::weighted_value).
/// Choose [`Arithmetic::ExactRational`] for exact results, or
/// [`Arithmetic::ScaledFloat`] or [`Arithmetic::SignedLog`] for bounded
/// precision over a wider numeric range than ordinary floating-point weights.
///
/// A clone shares the table. Once structure has been replaced by values,
/// those levels cannot be evaluated under new weights. For repeated
/// evaluations with changing weights, retain a structural diagram and use
/// [`Tdd::evaluate`](crate::Tdd::evaluate).
#[derive(Clone, PartialEq, Eq)]
pub struct Weights {
    pub(super) table: Arc<RationalWeights>,
    arithmetic: Arithmetic,
}

/// The weights one diagram carries on a vtree that is not a weighted home.
pub type WeightStore = Weights;

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("arithmetic", &self.arithmetic)
            .finish()
    }
}

impl Weights {
    /// Whether columns computed under these weights and under `other` have the
    /// same interpretation.
    pub(crate) fn compatible(&self, other: &Self) -> bool {
        self == other
    }

    /// Require weights for every named variable before a reader indexes the table.
    pub(crate) fn check_variables(&self, variables: impl IntoIterator<Item = crate::vtree::VarId>) -> Result<(), crate::diagram::TddBuildError> {
        for var in variables {
            if var.idx() >= self.algebra().num_vars() {
                return Err(crate::diagram::TddBuildError::MissingVariableWeight(var));
            }
        }
        Ok(())
    }

    /// Compare arithmetic and literal weights under a rename whose variables both tables cover.
    pub(crate) fn compatible_after_rename(&self, other: &Self, variables: impl IntoIterator<Item = (crate::vtree::VarId, crate::vtree::VarId)>) -> bool {
        self.arithmetic() == other.arithmetic() && variables.into_iter().all(|(local, global)| {
            self.algebra().pos_weight(local) == other.algebra().pos_weight(global)
                && self.algebra().neg_weight(local) == other.algebra().neg_weight(global)
        })
    }

    /// Validate `self` against `levels` and install it in `slot`. Once a level
    /// holds weight-marginal values, a store that interprets the weights
    /// differently from the one they were computed under is refused.
    pub(crate) fn install(
        self,
        vtree: &crate::vtree::Vtree,
        levels: &[crate::diagram::TddLevel],
        slot: &mut Option<WeightStore>,
    ) -> Result<(), crate::diagram::TddBuildError> {
        self.check_levels(vtree, levels)?;
        if levels.iter().any(crate::diagram::TddLevel::is_weight_marginal)
            && slot.as_ref().is_some_and(|old| !old.compatible(&self))
        {
            return Err(crate::diagram::TddBuildError::IncompatibleWeights);
        }
        *slot = Some(self);
        Ok(())
    }

    /// Check that the table covers the vtree and every marginal level has its values.
    pub(crate) fn check_levels(&self, vtree: &crate::vtree::Vtree, levels: &[crate::diagram::TddLevel]) -> Result<(), crate::diagram::TddBuildError> {
        use crate::diagram::{TddBuildError, LEAF_WIDTH};
        self.check_variables(vtree.leaf_bottomup().map(|(_, var)| var))?;
        for (leaf, var) in vtree.leaf_bottomup() {
            if levels[leaf.idx()].is_weight_marginal() {
                let values = levels[leaf.idx()].weight_values().unwrap_or(&[]);
                if levels[leaf.idx()].slot_count() != LEAF_WIDTH || values.len() != LEAF_WIDTH {
                    return Err(TddBuildError::InvalidWeightColumn { level: leaf, reason: "must hold three leaf-label slots" });
                }
                for (slot, value) in values.iter().enumerate() {
                    let expected = self.leaf_val(var, LeafLabel::from_idx(slot));
                    if !crate::diagram::semiring::same_value(value, &expected) {
                        return Err(TddBuildError::InvalidWeightColumn { level: leaf, reason: "does not match its pinned leaf values" });
                    }
                }
            }
        }
        for t in vtree.bottomup() {
            let level = &levels[t.idx()];
            if !level.is_marginal() { continue; }
            if !level.is_weight_marginal() {
                return Err(TddBuildError::CountLevelWithWeights { level: t });
            }
            let values = level.weight_values().unwrap_or(&[]);
            if values.len() != level.slot_count() {
                return Err(TddBuildError::InvalidWeightColumn { level: t, reason: "does not match the level's slot count" });
            }
            if values.iter().any(|v| v.arithmetic() != self.arithmetic()) {
                return Err(TddBuildError::InvalidWeightColumn { level: t, reason: "uses a different arithmetic" });
            }
        }
        Ok(())
    }

    /// A store over `algebra` with no level marginal yet.
    pub fn new(algebra: RationalWeights, arithmetic: Arithmetic) -> Self {
        Self { table: Arc::new(algebra), arithmetic }
    }

    /// A clone, for another diagram of the same weighted build. The table is
    /// shared, not copied.
    pub fn empty_like(&self) -> Self {
        self.clone()
    }

    /// The algebra the values are folded in: the weight table and its
    /// operations.
    #[inline]
    pub fn algebra(&self) -> &RationalWeights {
        &self.table
    }

    /// The arithmetic, fixed at construction.
    #[inline]
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }

    /// True in a bounded-precision arithmetic: there value equality is not
    /// key equality, and addition depends on its order.
    #[inline]
    pub(crate) fn is_bounded(&self) -> bool {
        self.arithmetic != Arithmetic::ExactRational
    }

    /// The additive identity in the active mode.
    #[inline]
    pub(crate) fn wzero(&self) -> WeightValue {
        match self.arithmetic {
            Arithmetic::SignedLog => WeightValue::Log(SignedLog::zero()),
            Arithmetic::ScaledFloat => WeightValue::Scaled(ScaledFloat::ZERO),
            // The canonical exact zero: 0 always fits the small representation.
            Arithmetic::ExactRational => WeightValue::ExactSmall(0),
        }
    }

    /// Leaf base value for `(var, label)` in the active mode. The exact parsed
    /// weight from [`RationalWeights`] is authoritative; in a bounded mode it
    /// is converted exactly once here (per leaf read).
    #[inline]
    pub(crate) fn leaf_val(&self, var: VarId, label: LeafLabel) -> WeightValue {
        let r = self.table.leaf(var, label);
        match self.arithmetic {
            Arithmetic::SignedLog => WeightValue::Log(SignedLog::from_rational(&r)),
            Arithmetic::ScaledFloat => WeightValue::Scaled(ScaledFloat::from_rational(&r)),
            Arithmetic::ExactRational => WeightValue::exact(r),
        }
    }
}
