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

/// Run an operation that returns a circuit; canonical minimizes the result under the same limits.
pub fn produce(py: Python<'_>, vtree: &Arc<Vtree>, limits: LimitConfig, canonical: bool,
    operation: impl FnOnce(&Engine) -> Result<Tdd, OperationError> + Send) -> PyResult<PyCircuit> {
    run(py, vtree, limits, move |engine| {
        let mut f = operation(engine)?;
        if canonical { engine.minimize(&mut f)?; }
        Ok(f)
    }).map(PyCircuit::new)
}

/// Validate all handles while holding their mutable borrows, then move every operand.
/// Duplicate handles are rejected before anything is taken, including aliases in a list.
pub fn take_operands(py: Python<'_>, operands: &[Py<PyCircuit>],
    validate: impl FnOnce(&Vtree) -> PyResult<()>) -> PyResult<(tididi::vtree::Home, Vec<Tdd>)> {
    if operands.is_empty() { return Err(PyValueError::new_err("at least one circuit is required")); }
    let mut seen = HashSet::new();
    for operand in operands {
        if !seen.insert(operand.as_ptr()) {
            return Err(PyValueError::new_err("the same circuit occurs twice; copy one occurrence before this operation"));
        }
    }
    let mut borrowed = operands.iter().map(|f| f.try_borrow_mut(py)).collect::<Result<Vec<_>, _>>()?;
    let vtree = domain::domain(borrowed[0].get()?);
    for f in &borrowed {
        if !Arc::ptr_eq(&domain::domain(f.get()?), &vtree) {
            return Err(PyValueError::new_err("circuits must share the same Vtree object"));
        }
    }
    validate(&vtree)?;
    let owned = borrowed.iter_mut().map(|f| f.take()).collect::<PyResult<Vec<_>>>()?;
    Ok((vtree, owned))
}

pub fn binary(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, canonical: bool, limits: Option<&PyLimits>,
    operation: fn(&Engine, Tdd, Tdd) -> Result<Tdd, OperationError>) -> PyResult<PyCircuit> {
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[left, right], |_| Ok(()))?;
    let right = inputs.pop().unwrap();
    let left = inputs.pop().unwrap();
    produce(py, &vtree, limits, canonical, move |engine| operation(engine, left, right))
}

/// Construct a literal circuit from a signed integer or an immutable Literal value.
/// The result is minimized.
#[pyfunction]
#[pyo3(signature = (vtree, value, *, canonical=false, limits=None))]
pub fn literal(py: Python<'_>, vtree: &PyVtree, value: &Bound<'_, PyAny>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let value = domain::read_literal(value)?;
    produce(py, &vtree.0, domain::config(limits)?, canonical, |e| e.literal(&vtree.0, value))
}

/// Construct a conjunction of signed integers or Literal values. Omitted variables are free.
/// Each variable may occur once; an empty iterable constructs true.
/// The result is minimized.
#[pyfunction]
#[pyo3(signature = (vtree, literals, *, canonical=false, limits=None))]
pub fn cube(py: Python<'_>, vtree: &PyVtree, literals: &Bound<'_, PyAny>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let literals = domain::read_literals(literals)?;
    produce(py, &vtree.0, domain::config(limits)?, canonical, |e| e.cube(&vtree.0, literals))
}

/// Construct a disjunction of literals. An empty iterable constructs false.
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (vtree, literals, *, canonical=false, limits=None))]
pub fn clause(py: Python<'_>, vtree: &PyVtree, literals: &Bound<'_, PyAny>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let literals = domain::read_literals(literals)?;
    produce(py, &vtree.0, domain::config(limits)?, canonical, |e| e.clause(&vtree.0, literals))
}

/// Construct true over every variable of vtree.
/// The result is minimized.
#[pyfunction]
pub fn one(vtree: &PyVtree) -> PyCircuit { PyCircuit::new(Tdd::one(&vtree.0)) }

/// Construct false over every variable of vtree.
/// The result is minimized.
#[pyfunction]
pub fn zero(vtree: &PyVtree) -> PyCircuit { PyCircuit::new(Tdd::zero(&vtree.0)) }

/// Construct the set of Boolean rows over variables, leaving other variables free.
/// Every row must contain one bool per variable. Repeated rows count once.
/// Empty rows denote false; one empty row over no variables denotes true.
/// The result is minimized.
#[pyfunction]
#[pyo3(signature = (vtree, variables, rows, *, canonical=false, limits=None))]
pub fn from_models(py: Python<'_>, vtree: &PyVtree, variables: Vec<u32>, rows: &Bound<'_, PyAny>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
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
    produce(py, &vtree.0, domain::config(limits)?, canonical, |e| e.from_models(&vtree.0, &variables, &packed))
}

/// Conjoin two circuits, consuming both. Equivalent to left & right, with optional limits.
/// Copy operands first if they must remain usable, including for retry after a resource error.
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (left, right, *, canonical=false, limits=None))]
pub fn and_(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, canonical, limits, Engine::and)
}

/// Disjoin two circuits, consuming both. Equivalent to left | right, with optional limits.
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (left, right, *, canonical=false, limits=None))]
pub fn or_(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, canonical, limits, Engine::or)
}

/// Exclusive-or two circuits, consuming both. Also available as left ^ right.
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (left, right, *, canonical=false, limits=None))]
pub fn xor(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    binary(py, left, right, canonical, limits, Engine::xor)
}

/// Build (condition & then_branch) | (~condition & else_branch), consuming all three inputs.
/// The result is not minimized unless both branches hold the same levels, as copies do; canonical=True
/// or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (condition, then_branch, else_branch, *, canonical=false, limits=None))]
pub fn ite(py: Python<'_>, condition: Py<PyCircuit>, then_branch: Py<PyCircuit>, else_branch: Py<PyCircuit>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[condition, then_branch, else_branch], |_| Ok(()))?;
    let no = inputs.pop().unwrap();
    let yes = inputs.pop().unwrap();
    let condition = inputs.pop().unwrap();
    produce(py, &vtree, limits, canonical, move |e| e.ite(condition, yes, no))
}

/// Union a nonempty iterable of circuits, consuming every element.
/// Duplicate handles are rejected; pass explicit copies when reusing a circuit.
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (circuits, *, canonical=false, limits=None))]
pub fn or_many(py: Python<'_>, circuits: &Bound<'_, PyAny>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let circuits = circuits.try_iter()?.map(|v| Ok(v?.extract::<Py<PyCircuit>>()?)).collect::<PyResult<Vec<_>>>()?;
    let limits = domain::config(limits)?;
    let (vtree, inputs) = take_operands(py, &circuits, |_| Ok(()))?;
    produce(py, &vtree, limits, canonical, move |e| e.or_many(inputs))
}

/// Conjoin, then existentially quantify variables in one call, consuming both circuits.
/// This computes the same function as (left & right).exists(variables).
/// The result is not minimized; canonical=True or minimize() makes it canonical.
#[pyfunction]
#[pyo3(signature = (left, right, variables, *, canonical=false, limits=None))]
pub fn and_exists(py: Python<'_>, left: Py<PyCircuit>, right: Py<PyCircuit>, variables: Vec<u32>, canonical: bool, limits: Option<&PyLimits>) -> PyResult<PyCircuit> {
    let vars = domain::variable_ids(&variables)?;
    let limits = domain::config(limits)?;
    let (vtree, mut inputs) = take_operands(py, &[left, right], |v| domain::check_variables(v, vars.iter().copied()))?;
    let right = inputs.pop().unwrap();
    let left = inputs.pop().unwrap();
    produce(py, &vtree, limits, canonical, move |e| e.and_exists(left, right, &vars))
}
