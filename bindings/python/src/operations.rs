//! Construction and consuming operations share one checked ownership boundary.

use std::collections::HashSet;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::limits::LimitConfig;
use crate::circuit::PyCircuit;
use crate::domain::{self, PyLimits, PyVtree};

pub fn run<T: Send>(py: Python<'_>, vtree: &Arc<Vtree>, limits: LimitConfig,
    operation: impl FnOnce(&Engine) -> Result<T, OperationError> + Send) -> PyResult<T> {
    let context = Arc::clone(vtree.context());
    py.detach(move || context.with_limits(limits, operation)).map_err(crate::operation_error)
}

/// Validate all handles while holding their mutable borrows, then move every operand.
/// Duplicate handles are rejected before anything is taken, including aliases in a list.
pub fn take_operands(py: Python<'_>, operands: &[Py<PyCircuit>],
    validate: impl FnOnce(&Vtree) -> PyResult<()>) -> PyResult<(Arc<Vtree>, Vec<Tdd>)> {
    if operands.is_empty() { return Err(PyValueError::new_err("at least one circuit is required")); }
    let mut seen = HashSet::new();
    for operand in operands {
        if !seen.insert(operand.as_ptr()) {
            return Err(PyValueError::new_err("the same circuit occurs twice; copy one occurrence before this operation"));
        }
    }
    let mut borrowed = operands.iter().map(|f| f.try_borrow_mut(py)).collect::<Result<Vec<_>, _>>()?;
    let vtree = Arc::clone(borrowed[0].get()?.vtree());
    for f in &borrowed {
        if !Arc::ptr_eq(f.get()?.vtree(), &vtree) {
            return Err(PyValueError::new_err("circuits must share the same Vtree object"));
        }
    }
    validate(&vtree)?;
    let owned = borrowed.iter_mut().map(|f| f.take()).collect::<PyResult<Vec<_>>>()?;
    Ok((vtree, owned))
}

pub fn binary(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, limits: Option<&PyLimits>,
    operation: fn(&Engine, Tdd, Tdd) -> Result<Tdd, OperationError>) -> PyResult<PyCircuit> {
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[left, right], |_| Ok(()))?;
    let right = inputs.pop().unwrap();
    let left = inputs.pop().unwrap();
    run(py, &vtree, limits, move |engine| operation(engine, left, right)).map(PyCircuit::new)
}

/// Construct a literal circuit from a signed integer or an immutable Literal value.
#[pyfunction]
#[pyo3(signature = (vtree, value, *, limits=None))]
pub fn literal(py: Python<'_>, vtree: &PyVtree, value: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let value = domain::read_literal(value)?;
    run(py, &vtree.0, domain::config(limits)?, |e| e.literal(&vtree.0, value)).map(PyCircuit::new)
}

/// Construct a conjunction of signed integers or Literal values. Omitted variables are free.
/// Each variable may occur once; an empty iterable constructs true.
#[pyfunction]
#[pyo3(signature = (vtree, literals, *, limits=None))]
pub fn cube(py: Python<'_>, vtree: &PyVtree, literals: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let literals = domain::read_literals(literals)?;
    run(py, &vtree.0, domain::config(limits)?, |e| e.cube(&vtree.0, literals)).map(PyCircuit::new)
}

/// Construct a disjunction of literals. An empty iterable constructs false.
#[pyfunction]
#[pyo3(signature = (vtree, literals, *, limits=None))]
pub fn clause(py: Python<'_>, vtree: &PyVtree, literals: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let literals = domain::read_literals(literals)?;
    run(py, &vtree.0, domain::config(limits)?, |e| e.clause(&vtree.0, literals)).map(PyCircuit::new)
}

/// Construct true over every variable of vtree.
#[pyfunction]
pub fn one(vtree: &PyVtree) -> PyCircuit { PyCircuit::new(Tdd::one(&vtree.0)) }

/// Construct false over every variable of vtree.
#[pyfunction]
pub fn zero(vtree: &PyVtree) -> PyCircuit { PyCircuit::new(Tdd::zero(&vtree.0)) }

/// Construct the set of Boolean rows over variables, leaving other variables free.
/// Every row must contain one bool per variable. Repeated rows count once.
/// Empty rows denote false; one empty row over no variables denotes true.
#[pyfunction]
#[pyo3(signature = (vtree, variables, rows, *, limits=None))]
pub fn from_models(py: Python<'_>, vtree: &PyVtree, variables: Vec<u32>, rows: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let variables = domain::variable_ids(&variables)?;
    let words = variables.len().div_ceil(64).max(1);
    let mut packed = Vec::new();
    for row in rows.try_iter()? {
        let row = row?.extract::<Vec<bool>>()?;
        if row.len() != variables.len() { return Err(PyValueError::new_err("each row must have one bool per variable")); }
        let start = packed.len();
        packed.resize(start + words, 0u64);
        for (i, value) in row.into_iter().enumerate() {
            if value { packed[start + i / 64] |= 1 << (i % 64); }
        }
    }
    run(py, &vtree.0, domain::config(limits)?, |e| e.from_models(&vtree.0, &variables, &packed)).map(PyCircuit::new)
}

/// Conjoin two circuits, consuming both. Equivalent to left & right, with optional limits.
/// Copy operands first if they must remain usable, including for retry after a resource error.
#[pyfunction]
#[pyo3(signature = (left, right, *, limits=None))]
pub fn and_(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, limits, Engine::and)
}

/// Disjoin two circuits, consuming both. Equivalent to left | right, with optional limits.
#[pyfunction]
#[pyo3(signature = (left, right, *, limits=None))]
pub fn or_(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, limits, Engine::or)
}

/// Exclusive-or two circuits, consuming both. Also available as left ^ right.
#[pyfunction]
#[pyo3(signature = (left, right, *, limits=None))]
pub fn xor(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, limits, Engine::xor)
}

/// Build (condition & then_branch) | (~condition & else_branch), consuming all three inputs.
#[pyfunction]
#[pyo3(signature = (condition, then_branch, else_branch, *, limits=None))]
pub fn ite(py: Python<'_>, condition: Py<PyCircuit>, then_branch: Py<PyCircuit>, else_branch: Py<PyCircuit>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[condition, then_branch, else_branch], |_| Ok(()))?;
    let no = inputs.pop().unwrap();
    let yes = inputs.pop().unwrap();
    let condition = inputs.pop().unwrap();
    run(py, &vtree, limits, move |e| e.ite(condition, yes, no)).map(PyCircuit::new)
}

/// Union a nonempty iterable of circuits, consuming every element.
/// Duplicate handles are rejected; pass explicit copies when reusing a circuit.
#[pyfunction]
#[pyo3(signature = (circuits, *, limits=None))]
pub fn or_many(py: Python<'_>, circuits: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let circuits = circuits.try_iter()?.map(|v| Ok(v?.extract::<Py<PyCircuit>>()?)).collect::<PyResult<Vec<_>>>()?;
    let limits = domain::config(limits)?;
    let (vtree, inputs) = take_operands(py, &circuits, |_| Ok(()))?;
    run(py, &vtree, limits, move |e| e.or_many(inputs)).map(PyCircuit::new)
}

/// Conjoin, then existentially quantify variables in one call, consuming both circuits.
/// This computes the same function as (left & right).exists(variables).
#[pyfunction]
#[pyo3(signature = (left, right, variables, *, limits=None))]
pub fn and_exists(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, variables: Vec<u32>, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let vars = domain::variable_ids(&variables)?;
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[left, right], |v| domain::check_variables(v, vars.iter().copied()))?;
    let right = inputs.pop().unwrap();
    let left = inputs.pop().unwrap();
    run(py, &vtree, limits, move |e| e.and_exists(left, right, &vars)).map(PyCircuit::new)
}
