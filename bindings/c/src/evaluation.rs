// scenario: docs/scenarios.md#minimum-cost
use std::ffi::{c_char, c_void};
use std::collections::HashSet;
use std::sync::Arc;
use num_rational::BigRational;
use num_traits::{One, Zero};
use tididi::diagram::{EvalAlgebra, LeafLabel, LiteralWeights, RationalWeights};
use crate::*;

/// Exact negative/positive literal weights, written as integers or fractions such as "3/5".
#[repr(C)]
pub struct TididiWeight { pub variable: u32, pub negative: *const c_char, pub positive: *const c_char }
unsafe fn weights(f: &tididi::Tdd, values: *const TididiWeight, len: usize) -> Result<RationalWeights> {
    let mut rows = vec![LiteralWeights { negative: BigRational::one(), positive: BigRational::one() }; f.vtree().num_vars() as usize];
    let mut seen = HashSet::new();
    for value in unsafe { array(values, len)? } {
        let var = variable(value.variable)?; check_variables(f.vtree(), [var])?;
        if !seen.insert(var) { return Err(invalid("duplicate weight variable")); }
        rows[var.idx()] = LiteralWeights { negative: unsafe { text(value.negative)? }.parse().map_err(invalid)?, positive: unsafe { text(value.positive)? }.parse().map_err(invalid)? };
    }
    for level in f.vtree().bottomup() {
        if let tididi::vtree::VtreeNode::Leaf { var, .. } = f.vtree().node(level) && !seen.contains(var) { return Err(invalid(format!("missing weights for variable {}", var.0))); }
    }
    Ok(RationalWeights::from_literals(&rows))
}
/// Exact weighted sum, returned as an owned integer/fraction string. Supply weights for every vtree variable.
/// Borrows the circuit. The string is freed with tididi_string_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_weighted_count(value: *const TididiCircuit, values: *const TididiWeight, len: usize, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? }; let weights = unsafe { weights(&f, values, len)? };
        *out = string(run(f.vtree(), unsafe { limits(config)? }, |e| e.evaluate(&f, &weights))?)?; Ok(()) })
}
/// Divide two exact weighted sums, borrowing both circuits on one shared vtree.
/// For P(query|evidence), numerator must already represent query AND evidence.
/// A zero denominator returns InvalidArgument; out is an owned integer/fraction string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_weighted_ratio(numerator: *const TididiCircuit, denominator: *const TididiCircuit, values: *const TididiWeight, len: usize, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(numerator)? }; let g = unsafe { borrow(denominator)? };
        if !Arc::ptr_eq(f.vtree(), g.vtree()) { return Err(invalid("circuits must share one vtree")); }
        let weights = unsafe { weights(&f, values, len)? }; let config = unsafe { limits(config)? };
        let denominator = run(f.vtree(), config.clone(), |e| e.evaluate(&g, &weights))?;
        if denominator.is_zero() { return Err(invalid("denominator has zero weight")); }
        *out = string(run(f.vtree(), config, |e| e.evaluate(&f, &weights))? / denominator)?; Ok(()) })
}
/// Numeric evaluation callbacks. Every callback is required; userdata is borrowed for the call.
/// leaf sign is 1 (true), 0 (false), or -1 (free). Callbacks must not unwind or longjmp.
/// add combines disjoint alternatives; mul combines independent variable groups.
/// Both operations must be associative and commutative, mul must distribute over add,
/// and zero must be the identity for add and absorbing for mul.
/// leaf(v, -1) must equal add(leaf(v, 0), leaf(v, 1)): a free variable includes both signs.
/// Equivalent circuits need not evaluate equally if these laws are violated.
/// Double arithmetic approximates these laws; use weighted_count for exact rational sums.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TididiAlgebra {
    pub userdata: *mut c_void,
    pub zero: Option<unsafe extern "C" fn(*mut c_void) -> f64>,
    pub leaf: Option<unsafe extern "C" fn(*mut c_void, u32, i8) -> f64>,
    pub add: Option<unsafe extern "C" fn(*mut c_void, f64, f64) -> f64>,
    pub mul: Option<unsafe extern "C" fn(*mut c_void, f64, f64) -> f64>,
}
impl EvalAlgebra for TididiAlgebra {
    type Value = f64;
    fn zero(&self) -> f64 { unsafe { self.zero.unwrap()(self.userdata) } }
    fn leaf(&self, var: tididi::vtree::VarId, label: LeafLabel) -> f64 {
        let sign = match label { LeafLabel::Pos => 1, LeafLabel::Neg => 0, LeafLabel::One => -1, LeafLabel::Zero => return self.zero() };
        unsafe { self.leaf.unwrap()(self.userdata, var.0, sign) }
    }
    fn add_assign(&self, acc: &mut f64, value: &f64) { *acc = unsafe { self.add.unwrap()(self.userdata, *acc, *value) }; }
    fn mul(&self, a: &f64, b: &f64) -> f64 { unsafe { self.mul.unwrap()(self.userdata, *a, *b) } }
}
/// Evaluate a numeric algebra while borrowing the circuit. Callback values use double precision.
/// Reentrant attempts to consume or free this circuit return BorrowConflict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluate_f64(value: *const TididiCircuit, algebra: *const TididiAlgebra, out: *mut f64, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let algebra = *unsafe { required(algebra)? };
        if algebra.zero.is_none() || algebra.leaf.is_none() || algebra.add.is_none() || algebra.mul.is_none() { return Err(invalid("all algebra callbacks are required")); }
        let f = unsafe { borrow(value)? }; *out = run(f.vtree(), unsafe { limits(config)? }, |e| e.evaluate(&f, &algebra))?; Ok(()) })
}
