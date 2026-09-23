//! Exact native weights and a fallible Python callback adapter.

use std::cell::RefCell;
use std::collections::HashSet;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};
use pyo3::exceptions::{PyTypeError, PyValueError};
use num_rational::BigRational;
use num_traits::One;
use tididi::diagram::{EvalAlgebra, LeafLabel, LiteralWeights, RationalWeights};
use tididi::limits::LimitConfig;
use tididi::vtree::VarId;
use tididi::Tdd;
use crate::{domain, operations};

pub(crate) fn weights(py: Python<'_>, f: &Tdd, values: &Bound<'_, PyDict>) -> PyResult<RationalWeights> {
    let fraction = py.import("fractions")?.getattr("Fraction")?;
    let rational = |value: Bound<'_, PyAny>| -> PyResult<BigRational> {
        if value.is_instance_of::<PyInt>() {
            return Ok(BigRational::from_integer(value.extract()?));
        }
        if !value.is_instance(&fraction)? { return Err(PyTypeError::new_err("weights must be int or fractions.Fraction")); }
        value.extract()
    };
    let mut weights = vec![LiteralWeights { negative: BigRational::one(), positive: BigRational::one() }; f.vtree().num_vars() as usize];
    let mut supplied = HashSet::new();
    for (key, value) in values.iter() {
        let var = domain::variable_id(key.extract()?)?;
        domain::check_variables(f.vtree(), [var])?;
        let (negative, positive) = value.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        weights[var.idx()] = LiteralWeights { negative: rational(negative)?, positive: rational(positive)? };
        supplied.insert(var);
    }
    for t in f.vtree().bottomup() {
        if let tididi::vtree::VtreeNode::Leaf { var, .. } = f.vtree().node(t)
            && !supplied.contains(var) {
            return Err(PyValueError::new_err(format!("missing weights for variable {}", var.0)));
        }
    }
    Ok(RationalWeights::from_literals(&weights))
}

pub fn weighted_count(py: Python<'_>, f: &Tdd, values: &Bound<'_, PyDict>, limits: LimitConfig) -> PyResult<BigRational> {
    let weights = weights(py, f, values)?;
    operations::run(py, f.vtree(), limits, |engine| engine.evaluate(f, &weights))
}

struct PythonAlgebra<'py> {
    py: Python<'py>,
    object: Bound<'py, PyAny>,
    error: RefCell<Option<PyErr>>,
}

impl<'py> PythonAlgebra<'py> {
    fn call(&self, callback: impl FnOnce() -> PyResult<Bound<'py, PyAny>>) -> Bound<'py, PyAny> {
        if self.error.borrow().is_none() {
            match callback() {
                Ok(value) => return value,
                Err(error) => *self.error.borrow_mut() = Some(error),
            }
        }
        self.py.None().into_bound(self.py)
    }
}

impl<'py> EvalAlgebra for PythonAlgebra<'py> {
    type Value = Bound<'py, PyAny>;
    fn zero(&self) -> Self::Value { self.call(|| self.object.call_method0("zero")) }
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value {
        let sign = match label {
            LeafLabel::Pos => Some(true),
            LeafLabel::Neg => Some(false),
            LeafLabel::One => None,
            LeafLabel::Zero => return self.zero(),
        };
        self.call(|| self.object.call_method1("leaf", (var.0, sign)))
    }
    fn add_assign(&self, acc: &mut Self::Value, other: &Self::Value) {
        *acc = self.call(|| self.object.call_method1("add", (&*acc, other)));
    }
    fn mul(&self, left: &Self::Value, right: &Self::Value) -> Self::Value {
        self.call(|| self.object.call_method1("mul", (left, right)))
    }
}

pub fn evaluate<'py>(py: Python<'py>, f: &Tdd, object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    for method in ["zero", "leaf", "add", "mul"] {
        if !object.getattr(method)?.is_callable() {
            return Err(PyTypeError::new_err(format!("algebra.{method} must be callable")));
        }
    }
    let algebra = PythonAlgebra { py, object: object.clone(), error: RefCell::new(None) };
    let result = f.evaluate(&algebra);
    if let Some(error) = algebra.error.into_inner() { return Err(error); }
    result.map_err(crate::operation_error)
}
