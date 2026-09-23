//! Python handles for owned cached queries.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use num_bigint::BigUint;
use tididi::query::OwnedModelCounter;
use tididi::Tdd;
use crate::circuit::PyCircuit;
use crate::domain::{self, PyLimits};
use crate::operations;


/// Reuse cached counts while observations change. Created by Circuit.counter(), which consumes it.
/// observe() replaces pins for the named variables; other pins remain. finish() returns the original circuit.
#[pyclass(name = "Counter", module = "tididi")]
pub struct PyCounter {
    cell: Option<OwnedModelCounter>,
}

impl PyCounter {
    pub fn new(py: Python<'_>, circuit: Tdd) -> PyResult<Self> {
        let cell = py.detach(|| circuit.into_counter()).map_err(crate::operation_error)?;
        Ok(Self { cell: Some(cell) })
    }
    fn cell(&mut self) -> PyResult<&mut OwnedModelCounter> {
        self.cell.as_mut().ok_or_else(|| PyRuntimeError::new_err("Counter has been finished"))
    }
}

#[pymethods]
impl PyCounter {
    /// Set observations from signed integers or Literal values without changing the circuit.
    /// A later observation for a variable replaces its previous value.
    fn observe(&mut self, literals: &Bound<'_, PyAny>) -> PyResult<()> {
        let literals = domain::read_literals(literals)?;
        let cell = self.cell()?;
        domain::check_variables(cell.circuit().vtree(), literals.iter().map(|l| l.var))?;
        cell.observe(literals).map_err(crate::operation_error)
    }

    /// Remove one observation, leaving that variable free again.
    fn clear(&mut self, variable: u32) -> PyResult<()> {
        let var = domain::variable_id(variable)?;
        let cell = self.cell()?;
        domain::check_variables(cell.circuit().vtree(), [var])?;
        cell.set_pin(var, None).map_err(crate::operation_error)
    }

    /// Remove all observations.
    fn clear_observations(&mut self) -> PyResult<()> {
        self.cell()?.clear_pins();
        Ok(())
    }

    /// Count assignments consistent with the observations. Borrows the owned circuit.
    #[pyo3(signature = (*, limits=None))]
    fn model_count(&mut self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<BigUint> {
        let limits = domain::config(limits)?;
        let cell = self.cell()?;
        let vtree = std::sync::Arc::clone(cell.circuit().vtree());
        operations::run(py, &vtree, limits, |engine| {
            cell.bind(engine).model_count()
        })
    }

    /// Discard cached counts and observations, returning the original circuit. Closes this counter.
    fn finish(&mut self) -> PyResult<PyCircuit> {
        self.cell.take().map(|cell| PyCircuit::new(cell.into_inner()))
            .ok_or_else(|| PyRuntimeError::new_err("Counter has been finished"))
    }

    fn __repr__(&self) -> &'static str {
        if self.cell.is_some() { "Counter(open)" } else { "Counter(finished)" }
    }
}


type NativeEvaluator = tididi::query::OwnedEvaluator<tididi::diagram::RationalWeights>;

/// Cache exact weighted sums under changing evidence. Circuit.evaluator(weights) consumes the circuit.
/// value() returns the joint weight of the circuit and observations. finish() returns the circuit.
#[pyclass(name = "Evaluator", module = "tididi")]
pub struct PyEvaluator { cell: Option<NativeEvaluator> }

impl PyEvaluator {
    pub fn new(py: Python<'_>, circuit: Tdd, weights: tididi::diagram::RationalWeights) -> PyResult<Self> {
        let cell = py.detach(|| circuit.into_evaluator(weights)).map_err(crate::operation_error)?;
        Ok(Self { cell: Some(cell) })
    }
    fn cell(&mut self) -> PyResult<&mut NativeEvaluator> {
        self.cell.as_mut().ok_or_else(|| PyRuntimeError::new_err("Evaluator has been finished"))
    }
}

#[pymethods]
impl PyEvaluator {
    /// Observe signed integers or Literal values. Invalid input preserves all observations.
    fn observe(&mut self, literals: &Bound<'_, PyAny>) -> PyResult<()> {
        let literals = domain::read_literals(literals)?;
        self.cell()?.observe(literals).map_err(crate::operation_error)
    }
    /// Clear one observation.
    fn clear(&mut self, variable: u32) -> PyResult<()> {
        let variable = domain::variable_id(variable)?;
        self.cell()?.set_pin(variable, None).map_err(crate::operation_error)
    }
    /// Clear all observations, retaining cached storage.
    fn clear_observations(&mut self) -> PyResult<()> {
        self.cell()?.clear_pins();
        Ok(())
    }
    /// Replace every variable's (negative, positive) weights, retaining observations.
    /// Weights are int or Fraction values. Invalid input leaves previous weights in place.
    fn set_weights(&mut self, py: Python<'_>, weights: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        let cell = self.cell()?;
        let weights = crate::evaluation::weights(py, cell.circuit(), weights)?;
        cell.replace_algebra(weights);
        Ok(())
    }
    /// Return the exact weighted sum under observations, refreshing only affected ancestors.
    /// Probability weights give joint probability; no normalization is performed.
    #[pyo3(signature = (*, limits=None))]
    fn value(&mut self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<num_rational::BigRational> {
        let config = domain::config(limits)?;
        let cell = self.cell()?;
        let vtree = std::sync::Arc::clone(cell.circuit().vtree());
        operations::run(py, &vtree, config, |engine| cell.bind(engine).value())
    }
    /// Close this evaluator and return the original circuit without its observations.
    fn finish(&mut self) -> PyResult<PyCircuit> {
        self.cell.take().map(|cell| PyCircuit::new(cell.into_inner()))
            .ok_or_else(|| PyRuntimeError::new_err("Evaluator has been finished"))
    }
    fn __repr__(&self) -> &'static str {
        if self.cell.is_some() { "Evaluator(open)" } else { "Evaluator(finished)" }
    }
}
