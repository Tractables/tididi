//! Own the diagram and its borrowing counter without exposing Rust lifetimes to Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use num_bigint::BigUint;
use tididi::query::ModelCounter;
use tididi::Tdd;
use crate::circuit::PyCircuit;
use crate::domain::{self, PyLimits};
use crate::operations;

self_cell::self_cell!(
    struct CounterCell {
        owner: Tdd,
        #[covariant]
        dependent: ModelCounter,
    }
);

/// Reuse cached counts while observations change. Created by Circuit.counter(), which consumes it.
/// observe() replaces pins for the named variables; other pins remain. finish() returns the original circuit.
#[pyclass(name = "Counter", module = "tididi")]
pub struct PyCounter {
    cell: Option<CounterCell>,
}

impl PyCounter {
    pub fn new(py: Python<'_>, circuit: Tdd) -> PyResult<Self> {
        let cell = py.detach(|| CounterCell::try_new(circuit, |f| f.counter())).map_err(crate::operation_error)?;
        Ok(Self { cell: Some(cell) })
    }
    fn cell(&mut self) -> PyResult<&mut CounterCell> {
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
        domain::check_variables(cell.borrow_owner().vtree(), literals.iter().map(|l| l.var))?;
        cell.with_dependent_mut(|_, counter| counter.observe(literals)).map_err(crate::operation_error)
    }

    /// Remove one observation, leaving that variable free again.
    fn clear(&mut self, variable: u32) -> PyResult<()> {
        let var = domain::variable_id(variable)?;
        let cell = self.cell()?;
        domain::check_variables(cell.borrow_owner().vtree(), [var])?;
        cell.with_dependent_mut(|_, counter| counter.set_pin(var, None)).map_err(crate::operation_error)
    }

    /// Remove all observations.
    fn clear_observations(&mut self) -> PyResult<()> {
        self.cell()?.with_dependent_mut(|_, counter| counter.clear_pins());
        Ok(())
    }

    /// Count assignments consistent with the observations. Borrows the owned circuit.
    #[pyo3(signature = (*, limits=None))]
    fn model_count(&mut self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<BigUint> {
        let limits = domain::config(limits)?;
        let cell = self.cell()?;
        let vtree = std::sync::Arc::clone(cell.borrow_owner().vtree());
        operations::run(py, &vtree, limits, |engine| {
            cell.with_dependent_mut(|_, counter| counter.bind(engine).model_count())
        })
    }

    /// Discard cached counts and observations, returning the original circuit. Closes this counter.
    fn finish(&mut self) -> PyResult<PyCircuit> {
        self.cell.take().map(|cell| PyCircuit::new(cell.into_owner()))
            .ok_or_else(|| PyRuntimeError::new_err("Counter has been finished"))
    }

    fn __repr__(&self) -> &'static str {
        if self.cell.is_some() { "Counter(open)" } else { "Counter(finished)" }
    }
}
