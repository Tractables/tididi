//! Circuit objects expose borrowing queries and explicitly consuming transformations.

use std::path::PathBuf;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyBytes, PyDict};
use num_bigint::BigUint;
use num_rational::BigRational;
use tididi::Tdd;
use crate::domain::{self, PyLimits, PyLiteral, PyVtree};
use crate::operations;

/// An owned Boolean circuit. Construct one with literal, cube, clause, from_models, one or zero.
///
/// ``&``, ``|``, ``^``, ``~`` and transformation methods consume their circuit operands.
/// Queries borrow. copy() explicitly duplicates storage while sharing the vtree.
/// Reusing a consumed object raises ConsumedCircuitError, including through an alias.
#[pyclass(name = "Circuit", module = "tididi")]
pub struct PyCircuit {
    circuit: Option<Tdd>,
}

impl PyCircuit {
    pub fn new(circuit: Tdd) -> Self { Self { circuit: Some(circuit) } }
    pub fn get(&self) -> PyResult<&Tdd> {
        self.circuit.as_ref().ok_or_else(consumed)
    }
    pub fn take(&mut self) -> PyResult<Tdd> { self.circuit.take().ok_or_else(consumed) }
}

fn consumed() -> PyErr {
    crate::ConsumedCircuitError::new_err("Circuit has been consumed. Copy it before passing it to a consuming operation.")
}

#[pymethods]
impl PyCircuit {
    /// Whether this wrapper has given up its circuit. This property remains readable afterward.
    #[getter]
    fn is_consumed(&self) -> bool { self.circuit.is_none() }

    /// The shared vtree. Circuits built on this value can be combined with this circuit.
    #[getter]
    fn vtree(&self) -> PyResult<PyVtree> { Ok(PyVtree(Arc::clone(self.get()?.vtree()))) }

    /// Duplicate diagram storage, leaving the original usable. The vtree is shared.
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let f = self.get()?;
        Ok(Self::new(py.detach(|| f.clone())))
    }
    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> { self.copy(py) }
    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyDict>) -> PyResult<Self> { self.copy(py) }
    fn __repr__(&self) -> String {
        match &self.circuit {
            Some(f) => format!("Circuit(nodes={}, pairs={})", f.node_count(), f.pair_count()),
            None => "Circuit(consumed)".to_owned(),
        }
    }
    fn __bool__(&self) -> PyResult<bool> {
        self.get()?;
        Err(PyTypeError::new_err("a Circuit has no Python truth value; use is_sat(), and use &, |, ~ for Boolean operations"))
    }
    fn __and__(slf: Py<Self>, other: Py<Self>, py: Python<'_>) -> PyResult<Self> {
        operations::and_(py, slf, other, None)
    }
    fn __or__(slf: Py<Self>, other: Py<Self>, py: Python<'_>) -> PyResult<Self> {
        operations::or_(py, slf, other, None)
    }
    fn __xor__(slf: Py<Self>, other: Py<Self>, py: Python<'_>) -> PyResult<Self> {
        operations::xor(py, slf, other, None)
    }
    fn __invert__(&mut self, py: Python<'_>) -> PyResult<Self> { self.negate(py, None) }

    /// Complement this circuit, consuming it. Equivalent to ~circuit, with optional limits.
    #[pyo3(signature = (*, limits=None))]
    fn negate(&mut self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let limits = domain::config(limits)?;
        let f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| e.negate(f)).map(Self::new)
    }

    /// Substitute the given literal values, consuming this circuit.
    /// Repeated literals are ignored; opposite signs for one variable produce false.
    /// Assigned variables become free in the resulting function's counting universe.
    /// To count assignments consistent with evidence, use counter() instead.
    #[pyo3(signature = (literals, *, limits=None))]
    fn condition(&mut self, py: Python<'_>, literals: &Bound<'_, PyAny>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let literals = domain::read_literals(literals)?;
        domain::check_variables(self.get()?.vtree(), literals.iter().map(|l| l.var))?;
        let limits = domain::config(limits)?;
        let f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| e.condition(f, literals)).map(Self::new)
    }

    /// Existentially quantify variables, consuming this circuit.
    /// Quantified variables remain free in the vtree; use projected_model_count to count distinct projections.
    #[pyo3(signature = (variables, *, limits=None))]
    fn exists(&mut self, py: Python<'_>, variables: Vec<u32>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let vars = domain::variable_ids(&variables)?;
        domain::check_variables(self.get()?.vtree(), vars.iter().copied())?;
        let limits = domain::config(limits)?;
        let f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| e.exists_vars(f, &vars)).map(Self::new)
    }

    /// Simultaneously rename variables using a dict {source: target}, consuming this circuit.
    /// Swaps and cycles are simultaneous; several sources may map to the same target.
    /// The vtree and its counting universe stay unchanged.
    #[pyo3(signature = (mapping, *, limits=None))]
    fn rename(&mut self, py: Python<'_>, mapping: &Bound<'_, PyDict>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let pairs = mapping.iter().map(|(a, b)| Ok((domain::variable_id(a.extract()?)?, domain::variable_id(b.extract()?)?)))
            .collect::<PyResult<Vec<_>>>()?;
        domain::check_variables(self.get()?.vtree(), pairs.iter().flat_map(|&(a, b)| [a, b]))?;
        let limits = domain::config(limits)?;
        let f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| e.rename_vars(f, &pairs)).map(Self::new)
    }

    /// Minimize and return this circuit, consuming the old wrapper.
    /// The result is canonical for its vtree; its Boolean function is unchanged.
    #[pyo3(signature = (*, limits=None))]
    fn minimize(&mut self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let limits = domain::config(limits)?;
        let mut f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| { e.minimize(&mut f)?; Ok(f) }).map(Self::new)
    }

    /// Insert cubes, then remove cubes, returning a minimized circuit and consuming this one.
    /// Each cube is an iterable of signed integers or Literal values. Omitted variables
    /// are free; an empty cube matches every assignment. Contradictory cubes change nothing.
    /// Limits apply to each update and to the final minimization separately.
    #[pyo3(signature = (*, insert=None, remove=None, limits=None))]
    fn update(&mut self, py: Python<'_>, insert: Option<&Bound<'_, PyAny>>, remove: Option<&Bound<'_, PyAny>>, limits: Option<&PyLimits>) -> PyResult<Self> {
        let read = |rows: Option<&Bound<'_, PyAny>>| -> PyResult<Vec<Vec<tididi::Literal>>> {
            rows.map(|rows| rows.try_iter()?.map(|r| domain::read_literals(&r?)).collect()).unwrap_or_else(|| Ok(Vec::new()))
        };
        let insert = read(insert)?;
        let remove = read(remove)?;
        domain::check_variables(self.get()?.vtree(), insert.iter().chain(&remove).flatten().map(|l| l.var))?;
        let limits = domain::config(limits)?;
        let mut f = self.take()?;
        let vtree = Arc::clone(f.vtree());
        operations::run(py, &vtree, limits, move |e| {
            {
                let mut batch = e.maintain(&mut f)?;
                for row in insert { batch.insert_model(row)?; }
                for row in remove { batch.remove_model(row)?; }
            }
            e.minimize(&mut f)?;
            Ok(f)
        }).map(Self::new)
    }

    /// Count satisfying assignments over all vtree variables. Returns an arbitrary-precision int.
    #[pyo3(signature = (*, limits=None))]
    fn model_count(&self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<BigUint> {
        let f = self.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.model_count(f))
    }

    /// Count distinct assignments to variables that extend to a satisfying assignment.
    #[pyo3(signature = (variables, *, limits=None))]
    fn projected_model_count(&self, py: Python<'_>, variables: Vec<u32>, limits: Option<&PyLimits>) -> PyResult<BigUint> {
        let f = self.get()?;
        let vars = domain::variable_ids(&variables)?;
        domain::check_variables(f.vtree(), vars.iter().copied())?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.projected_model_count(f, &vars))
    }

    /// Whether at least one assignment satisfies this circuit. Borrows the circuit.
    #[pyo3(signature = (*, limits=None))]
    fn is_sat(&self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<bool> {
        let f = self.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.is_sat(f))
    }

    /// Compare Boolean functions without consuming either circuit. Both must share a vtree.
    #[pyo3(signature = (other, *, limits=None))]
    fn equivalent(&self, py: Python<'_>, other: &Self, limits: Option<&PyLimits>) -> PyResult<bool> {
        let f = self.get()?;
        let g = other.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.equivalent(f, g))
    }

    /// Whether every model of this circuit satisfies other. Borrows both circuits.
    #[pyo3(signature = (other, *, limits=None))]
    fn implies(&self, py: Python<'_>, other: &Self, limits: Option<&PyLimits>) -> PyResult<bool> {
        let f = self.get()?;
        let g = other.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.implies(f, g))
    }

    /// Variable IDs on which the function depends, without consuming the circuit.
    #[pyo3(signature = (*, limits=None))]
    fn support(&self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<Vec<u32>> {
        let f = self.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.support(f)).map(|vs| vs.into_iter().map(|v| v.0).collect())
    }

    /// Literals true in every model. False implies both signs of every vtree variable.
    #[pyo3(signature = (*, limits=None))]
    fn implied_literals(&self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<Vec<PyLiteral>> {
        let f = self.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.implied_literals(f)).map(|ls| ls.into_iter().map(PyLiteral).collect())
    }

    /// One complete satisfying assignment as Literal values, or None if unsatisfiable.
    #[pyo3(signature = (*, limits=None))]
    fn satisfying_assignment(&self, py: Python<'_>, limits: Option<&PyLimits>) -> PyResult<Option<Vec<PyLiteral>>> {
        let f = self.get()?;
        operations::run(py, f.vtree(), domain::config(limits)?, |e| e.satisfying_assignment(f)).map(|ls| ls.map(|ls| ls.into_iter().map(PyLiteral).collect()))
    }

    /// Exact weighted sum as fractions.Fraction. Supply {variable: (negative, positive)}
    /// weights, each an int or Fraction, for every vtree variable. Borrows the circuit.
    #[pyo3(signature = (weights, *, limits=None))]
    fn weighted_count(&self, py: Python<'_>, weights: &Bound<'_, PyDict>, limits: Option<&PyLimits>) -> PyResult<BigRational> {
        crate::evaluation::weighted_count(py, self.get()?, weights, domain::config(limits)?)
    }

    /// Evaluate a Python algebra with zero(), leaf(variable, sign), add(a, b), mul(a, b).
    /// sign is True, False, or None for a free variable. Values must be immutable;
    /// add and mul return new values. Borrows the circuit and propagates callback errors.
    /// Python callbacks run attached to the interpreter; use weighted_count for a native exact sum.
    fn evaluate<'py>(&self, py: Python<'py>, algebra: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::evaluation::evaluate(py, self.get()?, algebra)
    }

    /// Move this circuit into a reusable evidence counter. Call counter.finish() to recover it.
    /// Copy first to retain an independently usable circuit while the counter is open.
    fn counter(&mut self, py: Python<'_>) -> PyResult<crate::counter::PyCounter> {
        crate::counter::PyCounter::new(py, self.take()?)
    }

    /// Number of stored circuit nodes, excluding implicit leaf nodes.
    fn node_count(&self) -> PyResult<usize> { Ok(self.get()?.node_count()) }
    /// Number of stored child pairs.
    fn pair_count(&self) -> PyResult<usize> { Ok(self.get()?.pair_count()) }
    /// Snapshot of (vtree_node, local_node, pair_count) for every stored internal node.
    /// Storage IDs are local to this circuit and may change after transformation.
    fn node_sizes(&self) -> PyResult<Vec<(u32, usize, usize)>> {
        let f = self.get()?;
        Ok(f.vtree().bottomup().flat_map(|v| f.level(v).internal_inputs_iter()
            .map(move |(i, pairs)| (v.0, i, pairs.len()))).collect())
    }

    /// Serialize to bytes. Save the vtree too; related circuits must reload onto one shared vtree.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes = Vec::new();
        tididi::io::write_tdd(&mut bytes, self.get()?).map_err(domain::value_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    /// Load serialized circuit bytes onto an existing vtree.
    #[staticmethod]
    fn from_bytes(vtree: &PyVtree, mut data: &[u8]) -> PyResult<Self> {
        tididi::io::read_tdd(&mut data, &vtree.0).map(Self::new).map_err(domain::value_error)
    }
    /// Write this circuit to a path, leaving it usable. Save its vtree separately.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        tididi::io::save_tdd(self.get()?, path).map_err(io_error)
    }
    /// Read a circuit file onto an existing vtree.
    #[staticmethod]
    fn load(vtree: &PyVtree, path: PathBuf) -> PyResult<Self> {
        tididi::io::load_tdd(path, &vtree.0).map(Self::new).map_err(io_error)
    }
    /// Return Graphviz DOT for inspecting this circuit.
    fn to_dot(&self) -> PyResult<String> { tididi::io::tdd_to_dot(self.get()?).map_err(domain::value_error) }
}

fn io_error(error: tididi::io::IoError) -> PyErr {
    match error {
        tididi::io::IoError::Io(e) => e.into(),
        other => domain::value_error(other),
    }
}
