//! The public skip-form diagram, [`SkipTdd`], and the operations on it.

use std::sync::Arc;

use num_bigint::BigUint;

use super::conjoin;
use super::contract::contract;
use super::convert::{from_tdd, to_tdd};
use super::negate::negate;
use super::{Shape, SkipForm};
use crate::diagram::Tdd;
use crate::vtree::Vtree;
use crate::{Engine, LiteralInput, OperationError};

/// A Boolean diagram whose pairs may skip levels.
///
/// A [`Tdd`] stores a node at every level a function's paths cross. Where a
/// level's other child is a single variable that the function ignores, that
/// node is `true` on the variable and the rest of the function below, and a
/// `SkipTdd` does not store it: the pair side above names the node below
/// directly. On a right-linear vtree this is the reduced form of an ordered
/// BDD, where a [`Tdd`] is the quasi-reduced one, and it can hold many times
/// fewer pairs.
///
/// A `SkipTdd` is always canonical: two diagrams over the same vtree denote
/// the same function exactly when they are equal, and no operation on it
/// needs a reduction. It has no summed-out levels and no weights.
///
/// It is a type of its own because stored readers walk [`Tdd`] levels
/// directly, and a level a pair skips is not stored. The operations that
/// take a `SkipTdd` are named `skip_*` on [`Engine`], with
/// [`Engine::to_skip`] and [`Engine::to_stored`] converting between the
/// forms; every other operation takes the stored form.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Literal, Tdd, Vtree};
/// use tididi::vtree::VarId;
///
/// let vtree = Arc::new(Vtree::linear(3));
/// let eng = Engine::new();
/// let f = eng.to_skip(&Tdd::clause(&vtree, [Literal::pos(VarId(1)), Literal::pos(VarId(3))])?)?;
/// let g = eng.skip_and_clause(&f, &[Literal::neg(VarId(1)), Literal::pos(VarId(2))])?;
/// assert_eq!(eng.skip_model_count(&g)?, 4u32.into());
/// let stored = eng.to_stored(&g)?;
/// assert_eq!(eng.model_count(&stored)?, 4u32.into());
/// assert!(g.pair_count() < stored.pair_count());
/// # Ok::<(), tididi::OperationError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SkipTdd {
    vtree: Arc<Vtree>,
    pub(super) form: SkipForm,
}

impl SkipTdd {
    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`).
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// Whether the diagram is the constant false.
    pub fn is_zero(&self) -> bool {
        self.form.is_zero()
    }

    /// The pairs stored over all levels.
    pub fn pair_count(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].pairs.len()).sum()
    }

    /// The nodes stored over all levels.
    pub fn node_count(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].width()).sum()
    }

    /// The most nodes stored at one level.
    pub fn max_width(&self) -> usize {
        self.form.active.iter().map(|&t| self.form.levels[t as usize].width()).max().unwrap_or(0)
    }
}

impl Engine {
    /// The skip form of `f`.
    ///
    /// `f` need not be reduced. Converting a canonical `f` (a certified
    /// result) costs one pass over its pairs.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] when `f` has a summed-out level,
    /// [`OperationError::IncompatibleWeights`] when it carries weights,
    /// [`OperationError::OverBudget`] when the skip form does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn to_skip(&self, f: &Tdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let canonical = f.levels.is_canonical(f.output);
        let shape = Arc::new(Shape::of(f.vtree()));
        let form = from_tdd(f, shape, lim)?;
        let form = if canonical { form } else { contract(form, lim)? };
        Ok(SkipTdd { vtree: Arc::clone(f.vtree()), form })
    }

    /// The stored form of `f`: canonical, with a node at every level a pair
    /// of `f` skips.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] or [`OperationError::IndexOverflow`]
    /// when the stored levels do not fit.
    pub fn to_stored(&self, f: &SkipTdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        to_tdd(&f.form, self, &f.vtree)
    }

    /// `f ∧ g`.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the operands do not share a
    /// vtree allocation, [`OperationError::OverBudget`] when the result does
    /// not fit, and [`OperationError::Stopped`] for an armed stop.
    pub fn skip_and(&self, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if !Arc::ptr_eq(&f.vtree, &g.vtree) {
            return Err(OperationError::VtreeMismatch);
        }
        let raw = conjoin::and(&f.form, &g.form, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form: contract(raw, lim)? })
    }

    /// `f ∧ (l₁ ∨ … ∨ lₖ)` for the literals of `clause`, signed integers or
    /// typed literals as [`Engine::and_clause`] takes them; an empty clause
    /// gives false.
    ///
    /// # Errors
    ///
    /// As [`Engine::and_clause`].
    pub fn skip_and_clause<L: LiteralInput>(&self, f: &SkipTdd, clause: &[L]) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let c = self.and_clause(Tdd::one(&f.vtree), clause)?;
        let c = from_tdd(&c, Arc::clone(&f.form.shape), lim)?;
        let c = contract(c, lim)?;
        let raw = conjoin::and(&f.form, &c, lim)?;
        Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form: contract(raw, lim)? })
    }

    /// `¬f`.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] when the result does not fit, and
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn skip_negate(&self, f: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        Ok(SkipTdd { vtree: Arc::clone(&f.vtree), form: negate(&f.form, lim)? })
    }

    /// `f ∨ g`.
    ///
    /// # Errors
    ///
    /// As [`Engine::skip_and`].
    pub fn skip_or(&self, f: &SkipTdd, g: &SkipTdd) -> Result<SkipTdd, OperationError> {
        let _op = self.limits().enter()?;
        let both = self.skip_and(&self.skip_negate(f)?, &self.skip_negate(g)?)?;
        self.skip_negate(&both)
    }

    /// The number of assignments to the vtree's variables that satisfy `f`.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] where the counts do not fit.
    pub fn skip_model_count(&self, f: &SkipTdd) -> Result<BigUint, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        f.form.model_count(lim)
    }
}
