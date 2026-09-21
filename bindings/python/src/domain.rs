//! Vtrees, literal values and per-operation limits.

use std::sync::Arc;
use std::time::{Duration, Instant};
use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyBool;
use tididi::{Literal, Vtree};
use tididi::limits::LimitConfig;
use tididi::vtree::VarId;

/// A shared variable decomposition. Reuse one vtree for circuits you will combine.
#[pyclass(name = "Vtree", module = "tididi", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct PyVtree(pub Arc<Vtree>);

#[pymethods]
impl PyVtree {
    /// Build a balanced vtree over variables 1 through n. n must be positive.
    #[staticmethod]
    fn balanced(n: u32) -> PyResult<Self> {
        if n == 0 { return Err(PyValueError::new_err("a vtree needs at least one variable")); }
        Ok(Self(Arc::new(Vtree::balanced(n))))
    }

    /// Build a balanced vtree with leaves in the supplied order of distinct positive IDs.
    #[staticmethod]
    fn balanced_over(variables: Vec<u32>) -> PyResult<Self> {
        let variables = variable_ids(&variables)?;
        Vtree::balanced_over(&variables).map(|v| Self(Arc::new(v))).map_err(value_error)
    }

    /// Build a right-linear vtree with leaves in the supplied order.
    #[staticmethod]
    fn linear(variables: Vec<u32>) -> PyResult<Self> {
        Vtree::linear_from_order(&variable_ids(&variables)?).map(|v| Self(Arc::new(v))).map_err(value_error)
    }

    /// Build a one-variable vtree.
    #[staticmethod]
    fn leaf(variable: u32) -> PyResult<Self> {
        Ok(Self(Arc::new(Vtree::leaf(variable_id(variable)?))))
    }

    /// Join disjoint vtrees under a new root, leaving both inputs usable.
    #[staticmethod]
    fn join(left: &Self, right: &Self) -> PyResult<Self> {
        Vtree::join(&left.0, &right.0).map(|v| Self(Arc::new(v))).map_err(value_error)
    }

    /// Parse .vtree text. Reuse the returned object when loading related circuits.
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        Vtree::from_text(text).map(|v| Self(Arc::new(v))).map_err(value_error)
    }

    /// Serialize the shape and variable IDs as .vtree text.
    fn to_text(&self) -> String { self.0.to_text() }

    /// Return Graphviz DOT describing this vtree.
    fn to_dot(&self) -> String { tididi::io::vtree_to_dot(&self.0) }

    /// Variable IDs in left-to-right leaf order.
    #[getter]
    fn variables(&self) -> Vec<u32> {
        let mut pending = vec![self.0.root()];
        let mut variables = Vec::new();
        while let Some(node) = pending.pop() {
            match self.0.node(node) {
                tididi::vtree::VtreeNode::Leaf { var, .. } => variables.push(var.0),
                tididi::vtree::VtreeNode::Internal { left, right, .. } => {
                    pending.push(*right);
                    pending.push(*left);
                }
            }
        }
        variables
    }

    /// Release idle operation scratch without changing any circuit.
    fn clear_scratch(&self) { self.0.context().clear_scratch(); }

    fn __repr__(&self) -> String { format!("Vtree(variables={:?})", self.variables()) }
}

/// An immutable named literal value. Positive integers mean true, negative integers false.
/// Unlike a Circuit, a Literal is never consumed.
#[pyclass(name = "Literal", module = "tididi", skip_from_py_object, frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyLiteral(pub Literal);

#[pymethods]
impl PyLiteral {
    #[new]
    #[pyo3(signature = (variable, sign=true))]
    fn new(variable: u32, sign: bool) -> PyResult<Self> {
        Ok(Self(Literal::new(variable_id(variable)?, sign)))
    }
    /// The positive, one-based variable ID.
    #[getter]
    fn variable(&self) -> u32 { self.0.var.0 }
    /// True for a positive literal, false for a negated one.
    #[getter]
    fn sign(&self) -> bool { self.0.sign }
    fn __invert__(&self) -> Self { Self(self.0.negated()) }
    fn __int__(&self) -> i64 {
        i64::from(self.0.var.0) * if self.0.sign { 1 } else { -1 }
    }
    fn __repr__(&self) -> String {
        format!("Literal({}, {})", self.0.var.0, if self.0.sign { "True" } else { "False" })
    }
}

/// Optional resource bounds for one operation. Unspecified bounds are unlimited.
/// memory_bytes bounds charged operation storage, not the process's total memory.
/// timeout is in seconds and is checked at cooperative polling points.
#[pyclass(name = "Limits", module = "tididi", skip_from_py_object, frozen)]
#[derive(Clone, Default)]
pub struct PyLimits {
    memory_bytes: Option<u64>,
    output_nodes: Option<u64>,
    timeout: Option<f64>,
}

#[pymethods]
impl PyLimits {
    #[new]
    #[pyo3(signature = (*, memory_bytes=None, output_nodes=None, timeout=None))]
    fn new(memory_bytes: Option<u64>, output_nodes: Option<u64>, timeout: Option<f64>) -> PyResult<Self> {
        let result = Self { memory_bytes, output_nodes, timeout };
        result.config()?;
        Ok(result)
    }
    fn __repr__(&self) -> String {
        format!("Limits(memory_bytes={}, output_nodes={}, timeout={})",
            self.memory_bytes.map_or("None".into(), |v| v.to_string()),
            self.output_nodes.map_or("None".into(), |v| v.to_string()),
            self.timeout.map_or("None".into(), |v| v.to_string()))
    }
}

impl PyLimits {
    pub fn config(&self) -> PyResult<LimitConfig> {
        let deadline = self.timeout.map(|seconds| {
            let duration = Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err("timeout must be finite and nonnegative"))?;
            Instant::now().checked_add(duration).ok_or_else(|| PyValueError::new_err("timeout is too large"))
        }).transpose()?;
        Ok(LimitConfig::none().with_memory_budget_bytes(self.memory_bytes)
            .with_output_node_cap(self.output_nodes).with_deadline(deadline))
    }
}

pub fn config(limits: Option<&PyLimits>) -> PyResult<LimitConfig> {
    limits.map(PyLimits::config).unwrap_or_else(|| Ok(LimitConfig::none()))
}

pub fn value_error(error: impl std::fmt::Display) -> PyErr { PyValueError::new_err(error.to_string()) }

pub fn variable_id(value: u32) -> PyResult<VarId> {
    if value == 0 { Err(PyValueError::new_err("variable IDs start at 1")) } else { Ok(VarId(value)) }
}

pub fn variable_ids(values: &[u32]) -> PyResult<Vec<VarId>> {
    values.iter().map(|&v| variable_id(v)).collect()
}

pub fn check_variables(vtree: &Vtree, vars: impl IntoIterator<Item = VarId>) -> PyResult<()> {
    for var in vars {
        if vtree.leaf_of(var).is_none() {
            return Err(PyValueError::new_err(format!("variable {} is not in this vtree", var.0)));
        }
    }
    Ok(())
}

pub fn read_literal(value: &Bound<'_, PyAny>) -> PyResult<Literal> {
    if let Ok(literal) = value.extract::<PyRef<'_, PyLiteral>>() { return Ok(literal.0); }
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("a literal must be a signed integer or Literal, not bool"));
    }
    let signed = value.extract::<i64>()?;
    let var = u32::try_from(signed.unsigned_abs()).map_err(|_| PyValueError::new_err("literal ID exceeds u32"))?;
    Ok(Literal::new(variable_id(var)?, signed > 0))
}

pub fn read_literals(values: &Bound<'_, PyAny>) -> PyResult<Vec<Literal>> {
    values.try_iter()?.map(|v| read_literal(&v?)).collect()
}
