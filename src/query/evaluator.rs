//! Retained algebra values for repeated evaluations under changing evidence.

use std::borrow::Borrow;

use crate::{Engine, Tdd, OperationError, LiteralInput};
use crate::diagram::EvalAlgebra;
use crate::vtree::VarId;
use crate::value::Retention;
use super::cache::{Observations, BoundState, refresh_columns};
use super::evaluate::Evaluate;
use super::fold::LevelFold;

/// Evaluate a circuit repeatedly under changing observations.
///
/// Create one with [`Tdd::evaluator`]. It owns the algebra and caches one value
/// per node. Observing a variable refreshes only its ancestors on the next
/// [`value`](Self::value) call. The diagram remains unchanged.
///
/// Evidence excludes assignments with the opposite literal; an observed
/// variable keeps its literal weight. With probability weights, the result is
/// the joint probability of the circuit and evidence, not a conditional
/// probability. Divide by the evidence probability to normalize it.
///
/// ```
/// use std::sync::Arc;
/// use num_rational::BigRational;
/// use tididi::{Tdd, Vtree};
/// use tididi::diagram::{LiteralWeights, RationalWeights};
/// let vtree = Arc::new(Vtree::balanced(2));
/// let either = Tdd::clause(&vtree, [1, 2])?;
/// let half = BigRational::new(1.into(), 2.into());
/// let weights = RationalWeights::from_literals(&[
///     LiteralWeights { negative: half.clone(), positive: half.clone() },
///     LiteralWeights { negative: half.clone(), positive: half },
/// ]);
/// let mut evaluator = either.evaluator(weights)?;
/// assert_eq!(evaluator.value()?, BigRational::new(3.into(), 4.into()));
/// evaluator.observe([-1])?;
/// assert_eq!(evaluator.value()?, BigRational::new(1.into(), 4.into()));
/// evaluator.clear_pins();
/// assert_eq!(evaluator.value()?, BigRational::new(3.into(), 4.into()));
/// # tididi::test_helpers::assert_canonical(&either);
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// The algebra must keep the same interpretation between reads, including if
/// it uses interior mutability. Treat returned values as immutable if their
/// clones share storage. Use [`replace_algebra`](Self::replace_algebra)
/// to change weights and invalidate all cached values. Custom algebras obey
/// the laws of [`EvalAlgebra`]; the circuit must retain its Boolean structure.
/// Attached diagram weights do not override the supplied algebra.
pub struct Evaluation<S: EvalAlgebra, D: Borrow<Tdd>> {
    tdd: D,
    state: EvaluationState<S>,
}

/// An evaluator borrowing its circuit. Created by [`Tdd::evaluator`].
pub type Evaluator<'a, S> = Evaluation<S, &'a Tdd>;
/// An evaluator owning its circuit. Created by [`Tdd::into_evaluator`].
pub type OwnedEvaluator<S> = Evaluation<S, Tdd>;

struct EvaluationState<S: EvalAlgebra> {
    algebra: S,
    cols: Vec<Vec<S::Value>>,
    observations: Observations,
}

impl<S: EvalAlgebra, D: Borrow<Tdd>> std::fmt::Debug for Evaluation<S, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator").field("observations", &self.state.observations).finish_non_exhaustive()
    }
}

impl Tdd {
    /// Retain algebra values for repeated queries under changing evidence.
    ///
    /// See [`Evaluator`] for observations, weight semantics and an example.
    /// Evaluation is deferred until its first [`value`](Evaluator::value) call.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] if structure was summed out,
    /// [`OperationError::OverBudget`] if cache metadata cannot be reserved,
    /// or [`OperationError::Stopped`] for an armed stop.
    pub fn evaluator<S: EvalAlgebra>(&self, algebra: S) -> Result<Evaluator<'_, S>, OperationError> {
        self.vtree().context().run(|eng| Evaluator::new(eng, self, algebra))
    }

    /// Move this circuit into an evaluator without copying it.
    ///
    /// Evaluation and errors follow [`Self::evaluator`]. The result owns both
    /// circuit and algebra; [`Evaluation::into_inner`] recovers the circuit.
    /// On construction failure, the consumed circuit is dropped.
    pub fn into_evaluator<S: EvalAlgebra>(self, algebra: S) -> Result<OwnedEvaluator<S>, OperationError> {
        let context = self.context().clone();
        context.run(|eng| Evaluation::new(eng, self, algebra))
    }

}

impl<S: EvalAlgebra, D: Borrow<Tdd>> Evaluation<S, D> {
    fn new(eng: &Engine, tdd: D, algebra: S) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        tdd.borrow().require_structure()?;
        lim.check_stop()?;
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.borrow().vtree.num_nodes())?;
        cols.resize_with(tdd.borrow().vtree.num_nodes(), Vec::new);
        let observations = Observations::new(eng, tdd.borrow().vtree.num_leaves() as usize, Retention::All)?;
        lim.check_stop()?;
        Ok(Self { tdd, state: EvaluationState { algebra, cols, observations } })
    }

    /// Set or clear a variable's observation; updates take effect on the next read.
    ///
    /// An absent variable returns [`OperationError::VariableNotInVtree`] without
    /// changing observations or cached values. Repeating an observation does no work.
    pub fn set_pin(&mut self, var: VarId, value: Option<bool>) -> Result<(), OperationError> {
        self.state.observations.set_pin(self.tdd.borrow(), var, value)
    }

    /// Observe signed integers or named literals. The last value of a variable wins.
    ///
    /// All inputs are validated first. Invalid literals or absent variables
    /// return the corresponding [`OperationError`] without changing any observation.
    pub fn observe<L: LiteralInput>(&mut self, literals: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.state.observations.observe(self.tdd.borrow(), literals)
    }

    /// Apply observations and removals together, validating all variables first.
    ///
    /// Errors and repeated variables follow [`Self::set_pin`] and [`Self::observe`].
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.state.observations.set_pins(self.tdd.borrow(), pins)
    }

    /// Clear all observations, retaining allocated columns for the next read.
    pub fn clear_pins(&mut self) { self.state.observations.clear_pins(); }

    /// Replace the algebra, invalidate every cached value and return the previous one.
    /// Observations are retained. Evaluation remains deferred until the next read.
    pub fn replace_algebra(&mut self, algebra: S) -> S {
        self.state.observations.invalidate();
        std::mem::replace(&mut self.state.algebra, algebra)
    }

    /// Refresh pending observations and return the circuit's value under evidence.
    ///
    /// Returns [`OperationError::OverBudget`] on a refused column or worklist
    /// allocation and [`OperationError::Stopped`] on cancellation, including for
    /// a cached answer. A failed refresh is retried in full on the next read;
    /// observations are preserved. Algebra panics propagate and invalidate an
    /// unfinished refresh. Allocations inside algebra values are not metered.
    pub fn value(&mut self) -> Result<S::Value, OperationError> {
        let tdd = self.tdd.borrow();
        tdd.context().run(|eng| self.state.value_on(eng, tdd))
    }

    fn value_on(&mut self, eng: &Engine) -> Result<S::Value, OperationError> {
        self.state.value_on(eng, self.tdd.borrow())
    }

    /// The unchanged circuit, available for read-only queries.
    pub fn circuit(&self) -> &Tdd { self.tdd.borrow() }

    /// Discard cached values and observations and recover the circuit owner or borrow.
    pub fn into_inner(self) -> D { self.tdd }

    /// Use an engine's limits for reads until the binding is dropped.
    /// Observations and cached values remain available on this evaluator afterwards.
    pub fn bind<'b>(&'b mut self, engine: &'b Engine) -> BoundEvaluation<'b, S, D> {
        BoundEvaluation { state: BoundState::Borrowed(self), engine }
    }
}

/// An evaluator whose reads use a borrowed engine's limits.
///
/// Create it with [`Engine::evaluator`] or [`Evaluator::bind`]. It cannot
/// outlive that engine checkout; pins and failure recovery follow [`Evaluator`].
pub struct BoundEvaluation<'b, S: EvalAlgebra, D: Borrow<Tdd>> {
    state: BoundState<'b, Evaluation<S, D>>,
    engine: &'b Engine,
}

/// A borrowed diagram evaluator bound to an engine.
pub type BoundEvaluator<'a, 'b, S> = BoundEvaluation<'b, S, &'a Tdd>;

impl<S: EvalAlgebra, D: Borrow<Tdd>> std::fmt::Debug for BoundEvaluation<'_, S, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match &self.state { BoundState::Owned(value) => value, BoundState::Borrowed(value) => value };
        f.debug_struct("BoundEvaluator").field("evaluator", state).finish_non_exhaustive()
    }
}

impl Engine {
    /// Create an [`Evaluator`] whose reads use this batch's resource limits.
    /// Construction returns the errors documented on [`Tdd::evaluator`].
    pub fn evaluator<'a, 'b, S: EvalAlgebra>(&'b self, tdd: &'a Tdd, algebra: S) -> Result<BoundEvaluator<'a, 'b, S>, OperationError> {
        Ok(BoundEvaluation { state: BoundState::Owned(Evaluator::new(self, tdd, algebra)?), engine: self })
    }
}

impl<S: EvalAlgebra, D: Borrow<Tdd>> BoundEvaluation<'_, S, D> {
    /// [`Evaluator::set_pin`] within this binding.
    pub fn set_pin(&mut self, var: VarId, value: Option<bool>) -> Result<(), OperationError> {
        self.state.get_mut().set_pin(var, value)
    }
    /// [`Evaluator::observe`] within this binding.
    pub fn observe<L: LiteralInput>(&mut self, literals: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.state.get_mut().observe(literals)
    }
    /// [`Evaluator::set_pins`] within this binding.
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.state.get_mut().set_pins(pins)
    }
    /// [`Evaluator::clear_pins`] within this binding.
    pub fn clear_pins(&mut self) { self.state.get_mut().clear_pins(); }
    /// [`Evaluator::replace_algebra`] within this binding.
    pub fn replace_algebra(&mut self, algebra: S) -> S { self.state.get_mut().replace_algebra(algebra) }
    /// [`Evaluator::value`] under this engine's limits, including cached reads.
    pub fn value(&mut self) -> Result<S::Value, OperationError> { self.state.get_mut().value_on(self.engine) }
}

impl<S: EvalAlgebra> EvaluationState<S> {
    fn value_on(&mut self, eng: &Engine, tdd: &Tdd) -> Result<S::Value, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        let result = (|| {
            lim.check_stop()?;
            let mut gate = lim.gate();
            let value = if tdd.is_zero() { self.algebra.zero() } else {
                let (tdd, algebra, cols) = (tdd, &self.algebra, &mut self.cols);
                self.observations.refresh(eng, tdd, &mut gate, |pins, changed, gate| {
                    let fold = Evaluate::new(algebra, pins);
                    refresh_columns(&fold, eng, tdd, cols, changed, gate, |fold, col, width| {
                        if col.len() != width { *col = fold.alloc(eng, width)?; }
                        Ok(())
                    })
                })?;
                let output = tdd.output();
                self.cols[output.vtree.idx()][output.local.idx()].clone()
            };
            gate.poll(1)?;
            gate.flush()?;
            Ok(value)
        })();
        if result.is_err() { self.observations.invalidate(); }
        result
    }

}

#[cfg(test)]
#[path = "tests/evaluator.rs"]
mod tests;
