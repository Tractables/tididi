use std::ffi::c_char;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tididi::{Engine, Literal, Tdd, Vtree};
use tididi::limits::LimitConfig;
use tididi::vtree::{Home, VarId};
use crate::*;

/// A shared vtree handle. Circuits retain the vtree after this handle is freed.
pub struct TididiVtree(Home);
/// A handle on `vtree` made a home: a circuit built on it is held on the home's projection on the
/// variables it uses, so it stores no level for any other variable.
fn domain_handle(vtree: Vtree) -> *mut TididiVtree { Box::into_raw(Box::new(TididiVtree(Home::new(vtree)))) }
/// The vtree a circuit was built on: the home of its projection.
pub(crate) fn domain(f: &Tdd) -> Home { f.home() }
/// Per-operation limits and options. Pass NULL for unlimited work and no options, or initialize with
/// tididi_limits_default.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TididiLimits {
    /// Charged operation storage in bytes; UINT64_MAX means unlimited.
    pub memory_bytes: u64,
    /// Maximum emitted circuit nodes; UINT64_MAX means unlimited.
    pub output_nodes: u64,
    /// Cooperative timeout in seconds; -1 means unlimited. Other values must be finite and nonnegative.
    pub timeout_seconds: f64,
    /// Whether a call that returns a circuit minimizes it, within the same limits, so that the result is
    /// canonical. False leaves the result as the operation built it. Calls that return no circuit
    /// reject true.
    pub canonical: bool,
}
/// Return unlimited limits. Modify fields before passing their address to an operation.
#[unsafe(no_mangle)]
pub extern "C" fn tididi_limits_default() -> TididiLimits {
    TididiLimits { memory_bytes: u64::MAX, output_nodes: u64::MAX, timeout_seconds: -1.0, canonical: false }
}
pub(crate) unsafe fn limits(value: *const TididiLimits) -> Result<LimitConfig> {
    let options = unsafe { options(value)? };
    if options.canonical { return Err(invalid("canonical applies only to calls that return a circuit")); }
    Ok(options.config)
}
pub(crate) struct Options { pub config: LimitConfig, pub canonical: bool }
pub(crate) unsafe fn options(value: *const TididiLimits) -> Result<Options> {
    let Some(value) = (unsafe { value.as_ref() }) else { return Ok(Options { config: LimitConfig::none(), canonical: false }); };
    let deadline = if value.timeout_seconds == -1.0 { None } else {
        let duration = Duration::try_from_secs_f64(value.timeout_seconds).map_err(invalid)?;
        Some(Instant::now().checked_add(duration).ok_or_else(|| invalid("timeout is too large"))?)
    };
    let config = LimitConfig::none().with_memory_budget_bytes((value.memory_bytes != u64::MAX).then_some(value.memory_bytes))
        .with_output_node_cap((value.output_nodes != u64::MAX).then_some(value.output_nodes)).with_deadline(deadline);
    Ok(Options { config, canonical: value.canonical })
}
pub(crate) fn run<T>(vtree: &Arc<Vtree>, config: LimitConfig,
    f: impl FnOnce(&Engine) -> std::result::Result<T, tididi::OperationError>) -> Result<T> {
    vtree.context().with_limits(config, f).map_err(Into::into)
}
pub(crate) fn produce(vtree: &Arc<Vtree>, options: Options,
    f: impl FnOnce(&Engine) -> std::result::Result<Tdd, tididi::OperationError>) -> Result<Tdd> {
    run(vtree, options.config, |e| {
        let mut result = f(e)?;
        if options.canonical { e.minimize(&mut result)?; }
        Ok(result)
    })
}
pub(crate) fn variable(value: u32) -> Result<VarId> {
    if value == 0 { Err(invalid("variable IDs begin at 1")) } else { Ok(VarId(value)) }
}
pub(crate) fn variables(values: &[u32]) -> Result<Vec<VarId>> { values.iter().map(|&v| variable(v)).collect() }
pub(crate) fn literal_value(value: i64) -> Result<Literal> {
    let id = u32::try_from(value.unsigned_abs()).map_err(invalid)?;
    Ok(Literal::new(variable(id)?, value > 0))
}
pub(crate) fn literals(values: &[i64]) -> Result<Vec<Literal>> { values.iter().map(|&v| literal_value(v)).collect() }
pub(crate) fn check_variables(vtree: &Vtree, vars: impl IntoIterator<Item = VarId>) -> Result<()> {
    for var in vars { if vtree.leaf_of(var).is_none() { return Err(invalid(format!("variable {} is absent from the vtree", var.0))); } }
    Ok(())
}
pub(crate) unsafe fn vtree<'a>(value: *const TididiVtree) -> Result<&'a Home> { Ok(&unsafe { required(value)? }.0) }

/// Build a balanced vtree over variables 1 through n; n must be positive. out must point to NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_balanced(n: u32, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; if n == 0 { return Err(invalid("a vtree needs at least one variable")); }
        *out = domain_handle(Vtree::balanced(n)); Ok(()) })
}
/// Build a balanced vtree over distinct positive variable IDs in the given leaf order.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_balanced_over(order: *const u32, len: usize, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vars = variables(unsafe { array(order, len)? })?;
        *out = domain_handle(Vtree::balanced_over(&vars).map_err(invalid)?); Ok(()) })
}
/// Build a right-linear vtree over distinct positive IDs in the supplied order.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_linear(order: *const u32, len: usize, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vars = variables(unsafe { array(order, len)? })?;
        *out = domain_handle(Vtree::linear_from_order(&vars).map_err(invalid)?); Ok(()) })
}
/// Join disjoint vtrees under a new root. Borrows both inputs; the result is a new domain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_join(left: *const TididiVtree, right: *const TididiVtree, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? };
        let joined = Vtree::join(unsafe { vtree(left)? }, unsafe { vtree(right)? }).map_err(invalid)?;
        *out = domain_handle(joined); Ok(()) })
}
/// Serialize the vtree as an owned UTF-8 string. Free it with tididi_string_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_to_text(value: *const TididiVtree, out: *mut *mut c_char) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = string(unsafe { vtree(value)? }.to_text())?; Ok(()) })
}
/// Parse vtree text. Load related circuits onto this one returned domain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_from_text(value: *const c_char, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let parsed = Vtree::from_text(unsafe { text(value)? }).map_err(invalid)?;
        *out = domain_handle(parsed); Ok(()) })
}
/// Release idle operation buffers without changing circuits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_clear_scratch(value: *const TididiVtree) -> *mut TididiError {
    boundary(|| { unsafe { vtree(value)? }.context().clear_scratch(); Ok(()) })
}
/// Free this vtree handle; NULL is accepted. Existing circuits retain their shared vtree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_free(value: *mut TididiVtree) {
    if !value.is_null() { drop(unsafe { Box::from_raw(value) }); }
}
/// Return another handle to a live circuit's shared vtree. Free the returned handle normally.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_circuit_vtree(value: *const TididiCircuit, out: *mut *mut TididiVtree) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        *out = Box::into_raw(Box::new(TididiVtree(domain(&f)))); Ok(()) })
}
