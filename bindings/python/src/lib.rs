//! Python ownership, conversion and error boundaries for tididi.

use pyo3::prelude::*;
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError};
use tididi::OperationError;

mod domain;
mod operations;
mod circuit;
mod counter;
mod evaluation;

pyo3::create_exception!(tididi, ConsumedCircuitError, PyRuntimeError,
    "A circuit was already consumed. Copy it before a consuming operation to retain it.");
pyo3::create_exception!(tididi, ResourceLimitError, PyRuntimeError,
    "An operation exceeded its output-node cap or reached its deadline.");

fn operation_error(error: OperationError) -> PyErr {
    match error {
        OperationError::OverBudget => PyMemoryError::new_err(error.to_string()),
        OperationError::IndexOverflow => PyOverflowError::new_err(error.to_string()),
        OperationError::Stopped | OperationError::OutputCap => ResourceLimitError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pymodule]
fn _tididi(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<domain::PyVtree>()?;
    module.add_class::<domain::PyLiteral>()?;
    module.add_class::<domain::PyLimits>()?;
    module.add_class::<circuit::PyCircuit>()?;
    module.add_class::<counter::PyCounter>()?;
    module.add_class::<counter::PyEvaluator>()?;
    module.add("ConsumedCircuitError", module.py().get_type::<ConsumedCircuitError>())?;
    module.add("ResourceLimitError", module.py().get_type::<ResourceLimitError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(operations::literal, module)?)?;
    module.add_function(wrap_pyfunction!(operations::cube, module)?)?;
    module.add_function(wrap_pyfunction!(operations::clause, module)?)?;
    module.add_function(wrap_pyfunction!(operations::one, module)?)?;
    module.add_function(wrap_pyfunction!(operations::zero, module)?)?;
    module.add_function(wrap_pyfunction!(operations::from_models, module)?)?;
    module.add_function(wrap_pyfunction!(operations::and_, module)?)?;
    module.add_function(wrap_pyfunction!(operations::or_, module)?)?;
    module.add_function(wrap_pyfunction!(operations::xor, module)?)?;
    module.add_function(wrap_pyfunction!(operations::ite, module)?)?;
    module.add_function(wrap_pyfunction!(operations::or_many, module)?)?;
    module.add_function(wrap_pyfunction!(operations::and_exists, module)?)?;
    Ok(())
}
