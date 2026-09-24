use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::sync::Arc;
use tididi::{Engine, Tdd, Vtree};
use tididi::limits::LimitConfig;
use crate::*;

/// An owned circuit handle. Transformations consume its payload, not this handle.
/// Free every handle, including consumed ones, with tididi_circuit_free.
pub struct TididiCircuit(RefCell<Option<Tdd>>);
pub(crate) fn circuit(value: Tdd) -> *mut TididiCircuit { Box::into_raw(Box::new(TididiCircuit(RefCell::new(Some(value))))) }
pub(crate) unsafe fn borrow<'a>(value: *const TididiCircuit) -> Result<Ref<'a, Tdd>> {
    let value = unsafe { required(value)? }.0.try_borrow().map_err(|_| busy())?;
    Ref::filter_map(value, Option::as_ref).map_err(|_| consumed())
}
/// Borrow and validate every operand before taking any payload.
/// Holding the guards also prevents reentrant mutation during validation.
pub(crate) unsafe fn take_many(values: &[*mut TididiCircuit], validate: impl FnOnce(&Vtree) -> Result<()>)
    -> Result<(Arc<Vtree>, Vec<Tdd>)> {
    if values.is_empty() { return Err(invalid("at least one circuit is required")); }
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(*value) {
            return Err(invalid("the same circuit occurs twice; copy one operand"));
        }
    }
    let mut handles = Vec::new();
    for &value in values {
        handles.push(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
    }
    let vtree = Arc::clone(handles[0].as_ref().ok_or_else(consumed)?.vtree());
    for handle in &handles {
        if !Arc::ptr_eq(handle.as_ref().ok_or_else(consumed)?.vtree(), &vtree) {
            return Err(invalid("circuits must share one vtree"));
        }
    }
    validate(&vtree)?;
    let circuits = handles.iter_mut().map(|handle| handle.take().unwrap()).collect();
    Ok((vtree, circuits))
}

unsafe fn unary(value: *mut TididiCircuit, config: LimitConfig, validate: impl FnOnce(&Vtree) -> Result<()>,
    operation: impl FnOnce(&Engine, Tdd) -> std::result::Result<Tdd, tididi::OperationError>) -> Result<Tdd> {
    let (vtree, mut inputs) = unsafe { take_many(&[value], validate)? };
    run(&vtree, config, |e| operation(e, inputs.pop().unwrap()))
}
unsafe fn binary(left: *mut TididiCircuit, right: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits,
    operation: fn(&Engine, Tdd, Tdd) -> std::result::Result<Tdd, tididi::OperationError>) -> Result<()> {
    let out = unsafe { vacant(out)? }; let config = unsafe { limits(config)? };
    let (vtree, mut inputs) = unsafe { take_many(&[left, right], |_| Ok(()))? };
    let right = inputs.pop().unwrap(); let left = inputs.pop().unwrap();
    *out = circuit(run(&vtree, config, |e| operation(e, left, right))?); Ok(())
}

/// Copy diagram storage while sharing the vtree. The input remains usable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_copy(value: *const TididiCircuit, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = circuit(unsafe { borrow(value)? }.clone()); Ok(()) })
}
/// Report whether an operation consumed this handle's payload. The handle itself must still be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_is_consumed(value: *const TididiCircuit, out: *mut bool) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; *out = unsafe { required(value)? }.0.try_borrow().map_err(|_| busy())?.is_none(); Ok(()) })
}
/// Free a circuit handle, including a consumed one. NULL is accepted.
/// Returns BorrowConflict without freeing if a callback attempts to free an active handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_circuit_free(value: *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { if !value.is_null() {
        drop(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
        drop(unsafe { Box::from_raw(value) });
    } Ok(()) })
}
/// Construct a signed, one-based literal; zero is invalid. out must point to NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_literal(tree: *const TididiVtree, value: i64, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vtree = unsafe { vtree(tree)? }; let value = literal_value(value)?;
        *out = circuit(run(vtree, unsafe { limits(config)? }, |e| e.literal(vtree, value))?); Ok(()) })
}
/// Construct the constant true function over all vtree variables.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_one(tree: *const TididiVtree, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = circuit(Tdd::one(unsafe { vtree(tree)? })); Ok(()) })
}
/// Construct the constant false function over all vtree variables.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_zero(tree: *const TididiVtree, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = circuit(Tdd::zero(unsafe { vtree(tree)? })); Ok(()) })
}
/// Conjoin signed literals, each variable occurring once. An empty cube is true; omitted variables are free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_cube(tree: *const TididiVtree, values: *const i64, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vtree = unsafe { vtree(tree)? }; let values = literals(unsafe { array(values, len)? })?;
        *out = circuit(run(vtree, unsafe { limits(config)? }, |e| e.cube(vtree, values))?); Ok(()) })
}
/// Disjoin signed literals. An empty clause is false.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_clause(tree: *const TididiVtree, values: *const i64, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vtree = unsafe { vtree(tree)? }; let values = literals(unsafe { array(values, len)? })?;
        *out = circuit(run(vtree, unsafe { limits(config)? }, |e| e.clause(vtree, values))?); Ok(()) })
}
/// Conjoin two circuits, consuming both. Preflight errors preserve inputs; execution failures consume them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_and(left: *mut TididiCircuit, right: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| unsafe { binary(left, right, out, config, Engine::and) })
}
/// Disjoin two circuits, consuming both. Operands must be distinct handles sharing one vtree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_or(left: *mut TididiCircuit, right: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| unsafe { binary(left, right, out, config, Engine::or) })
}
/// Exclusive-or two circuits, consuming both.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_xor(left: *mut TididiCircuit, right: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| unsafe { binary(left, right, out, config, Engine::xor) })
}
/// Complement a circuit, consuming it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_negate(value: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = circuit(unsafe { unary(value, limits(config)?, |_| Ok(()), Engine::negate)? }); Ok(()) })
}
/// Minimize without changing the function. Consumes the old payload and returns a canonical circuit for its vtree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_minimize(value: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; *out = circuit(unsafe { unary(value, limits(config)?, |_| Ok(()), |e, mut f| { e.minimize(&mut f)?; Ok(f) })? }); Ok(()) })
}
/// Substitute literal values, consuming the circuit. Repeats are ignored; opposite signs produce false.
/// Substituted variables remain free in the counting universe; use a counter to count under observations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_condition(value: *mut TididiCircuit, assignments: *const i64, len: usize,
    out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| {
        let out = unsafe { vacant(out)? };
        let assignments = literals(unsafe { array(assignments, len)? })?;
        let vars: Vec<_> = assignments.iter().map(|literal| literal.var).collect();
        *out = circuit(unsafe {
            unary(value, limits(config)?, |vtree| check_variables(vtree, vars),
                |engine, circuit| engine.condition(circuit, assignments))?
        });
        Ok(())
    })
}
/// Existentially quantify variables, consuming the circuit. They remain free in the vtree's counting universe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_exists(value: *mut TididiCircuit, vars: *const u32, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vars = variables(unsafe { array(vars, len)? })?;
        *out = circuit(unsafe { unary(value, limits(config)?, |v| check_variables(v, vars.iter().copied()), |e, f| e.exists_vars(f, &vars))? }); Ok(()) })
}
/// Conjoin then quantify in one call, consuming both inputs. Equivalent to and followed by exists.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_and_exists(left: *mut TididiCircuit, right: *mut TididiCircuit, vars: *const u32, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vars = variables(unsafe { array(vars, len)? })?; let config = unsafe { limits(config)? };
        let (vtree, mut inputs) = unsafe { take_many(&[left, right], |v| check_variables(v, vars.iter().copied()))? };
        let right = inputs.pop().unwrap(); let left = inputs.pop().unwrap();
        *out = circuit(run(&vtree, config, |e| e.and_exists(left, right, &vars))?); Ok(()) })
}
/// Rename from[i] to to[i] simultaneously, consuming the circuit. Swaps and cycles are simultaneous.
/// Each source occurs once; several sources may share a target. The vtree stays unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_rename(value: *mut TididiCircuit, from: *const u32, to: *const u32, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? };
        let from = variables(unsafe { array(from, len)? })?; let to = variables(unsafe { array(to, len)? })?;
        let mut seen = HashSet::new(); if from.iter().any(|v| !seen.insert(*v)) { return Err(invalid("duplicate rename source")); }
        let pairs: Vec<_> = from.into_iter().zip(to).collect();
        *out = circuit(unsafe { unary(value, limits(config)?, |v| check_variables(v, pairs.iter().flat_map(|&(a,b)| [a,b])), |e,f| e.rename_vars(f, &pairs))? }); Ok(()) })
}
/// Union a nonempty array of distinct circuit handles, consuming every payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_or_many(values: *const *mut TididiCircuit, len: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let config = unsafe { limits(config)? };
        let (vtree, values) = unsafe { take_many(array(values, len)?, |_| Ok(()))? };
        *out = circuit(run(&vtree, config, |e| e.or_many(values))?); Ok(()) })
}
/// Build if-then-else, consuming three distinct circuit handles sharing one vtree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_ite(condition: *mut TididiCircuit, yes: *mut TididiCircuit, no: *mut TididiCircuit, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let config = unsafe { limits(config)? };
        let (vtree, mut values) = unsafe { take_many(&[condition, yes, no], |_| Ok(()))? };
        let no = values.pop().unwrap(); let yes = values.pop().unwrap(); let condition = values.pop().unwrap();
        *out = circuit(run(&vtree, config, |e| e.ite(condition, yes, no))?); Ok(()) })
}
/// Build a set of row-major Boolean rows. Every cell is 0 or 1; duplicates count once.
/// rows contains nrows*nvars bytes. Other vtree variables are free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_from_models(tree: *const TididiVtree, vars: *const u32, nvars: usize, rows: *const u8, nrows: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vtree = unsafe { vtree(tree)? };
        let vars = variables(unsafe { array(vars, nvars)? })?;
        let cells = nrows.checked_mul(nvars).ok_or_else(|| invalid("row dimensions overflow"))?;
        let rows = unsafe { array(rows, cells)? }; if rows.iter().any(|&v| v > 1) { return Err(invalid("row values must be 0 or 1")); }
        let words = nvars.div_ceil(64).max(1); let len = nrows.checked_mul(words).ok_or_else(|| invalid("row dimensions overflow"))?;
        let mut packed = vec![0; len];
        for row in 0..nrows { for col in 0..nvars { if rows[row*nvars+col] != 0 { packed[row*words+col/64] |= 1u64 << (col%64); } } }
        *out = circuit(run(vtree, unsafe { limits(config)? }, |e| e.from_models(vtree, &vars, &packed))?); Ok(()) })
}
/// One partial assignment used in an update. Omitted variables are free; an empty cube matches all assignments.
#[repr(C)]
pub struct TididiCube { pub literals: *const i64, pub len: usize }
/// Insert cubes, then remove cubes; consume the old circuit and return its minimized replacement.
/// Contradictory cubes change nothing. Limits apply to individual updates and final minimization.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_update(value: *mut TididiCircuit, insert: *const TididiCube, ninsert: usize, remove: *const TididiCube, nremove: usize, out: *mut *mut TididiCircuit, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? };
        let read = |cubes: &[TididiCube]| cubes.iter().map(|c| literals(unsafe { array(c.literals, c.len)? })).collect::<Result<Vec<_>>>();
        let insert = read(unsafe { array(insert, ninsert)? })?; let remove = read(unsafe { array(remove, nremove)? })?;
        let vars: Vec<_> = insert.iter().chain(&remove).flatten().map(|l| l.var).collect();
        *out = circuit(unsafe { unary(value, limits(config)?, |v| check_variables(v, vars), |e, mut f| {
            { let mut batch = e.maintain(&mut f)?;
              for row in insert { batch.insert_model(row)?; } for row in remove { batch.remove_model(row)?; } }
            e.minimize(&mut f)?; Ok(f)
        })? }); Ok(()) })
}
