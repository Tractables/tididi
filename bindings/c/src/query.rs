use std::ffi::c_char;
use num_traits::ToPrimitive;
use crate::*;

pub(crate) fn integer(value: impl ToPrimitive) -> Result<u64> {
    value.to_u64().ok_or_else(|| TididiError::new(TididiErrorCode::Overflow, "count does not fit uint64_t; use the decimal query"))
}
/// Count models over every vtree variable, borrowing the circuit. Overflow leaves out unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_model_count(value: *const TididiCircuit, out: *mut u64, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let f = unsafe { borrow(value)? };
        *out = integer(run(f.vtree(), unsafe { limits(config)? }, |e| e.model_count(&f))?)?; Ok(()) })
}
/// Count exactly, returning an owned decimal string of arbitrary length. Free it with tididi_string_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_model_count_decimal(value: *const TididiCircuit, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        *out = string(run(f.vtree(), unsafe { limits(config)? }, |e| e.model_count(&f))?)?; Ok(()) })
}
/// Count distinct assignments to the selected variables that extend to a model. Borrows the circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_projected_model_count(value: *const TididiCircuit, vars: *const u32, len: usize, out: *mut u64, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let vars = variables(unsafe { array(vars, len)? })?; let f = unsafe { borrow(value)? };
        *out = integer(run(f.vtree(), unsafe { limits(config)? }, |e| e.projected_model_count(&f, &vars))?)?; Ok(()) })
}
/// Count distinct projections exactly, returning an owned decimal string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_projected_model_count_decimal(value: *const TididiCircuit, vars: *const u32, len: usize, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let vars = variables(unsafe { array(vars, len)? })?; let f = unsafe { borrow(value)? };
        *out = string(run(f.vtree(), unsafe { limits(config)? }, |e| e.projected_model_count(&f, &vars))?)?; Ok(()) })
}
/// Whether at least one assignment satisfies the circuit. Borrows it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_is_sat(value: *const TididiCircuit, out: *mut bool, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let f = unsafe { borrow(value)? };
        *out = run(f.vtree(), unsafe { limits(config)? }, |e| e.is_sat(&f))?; Ok(()) })
}
/// Compare Boolean functions without consuming either handle. Aliases are allowed; vtrees must match.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_equivalent(left: *const TididiCircuit, right: *const TididiCircuit, out: *mut bool, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let f = unsafe { borrow(left)? }; let g = unsafe { borrow(right)? };
        *out = run(f.vtree(), unsafe { limits(config)? }, |e| e.equivalent(&f, &g))?; Ok(()) })
}
/// Whether every model of left satisfies right. Borrows both handles; vtrees must match.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_implies(left: *const TididiCircuit, right: *const TididiCircuit, out: *mut bool, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let f = unsafe { borrow(left)? }; let g = unsafe { borrow(right)? };
        *out = run(f.vtree(), unsafe { limits(config)? }, |e| e.implies(&f, &g))?; Ok(()) })
}
/// An owned list of signed, one-based literals. Its data remains valid until the list is freed.
pub struct TididiLiterals(Vec<i64>);
fn literal_list(values: impl IntoIterator<Item = tididi::Literal>) -> *mut TididiLiterals {
    Box::into_raw(Box::new(TididiLiterals(values.into_iter().map(|l| i64::from(l.var.0) * if l.sign { 1 } else { -1 }).collect())))
}
/// Return the size of a live, nonnull literal list.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_literals_len(value: *const TididiLiterals) -> usize { unsafe { (*value).0.len() } }
/// Borrow the signed literal array. Do not write to or free it separately.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_literals_data(value: *const TididiLiterals) -> *const i64 { unsafe { (*value).0.as_ptr() } }
/// Free a literal list; NULL is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_literals_free(value: *mut TididiLiterals) {
    if !value.is_null() { drop(unsafe { Box::from_raw(value) }); }
}
/// Return literals implied by the function. False implies both signs of every vtree variable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_implied_literals(value: *const TididiCircuit, out: *mut *mut TididiLiterals, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        *out = literal_list(run(f.vtree(), unsafe { limits(config)? }, |e| e.implied_literals(&f))?); Ok(()) })
}
/// Return one complete satisfying assignment, or an empty list when unsatisfiable. Borrows the circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_satisfying_assignment(value: *const TididiCircuit, out: *mut *mut TididiLiterals, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        *out = literal_list(run(f.vtree(), unsafe { limits(config)? }, |e| e.satisfying_assignment(&f))?.unwrap_or_default()); Ok(()) })
}
/// Return the function's support as positive variable IDs in a literal list. Borrows the circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_support(value: *const TididiCircuit, out: *mut *mut TididiLiterals, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        *out = literal_list(run(f.vtree(), unsafe { limits(config)? }, |e| e.support(&f))?.into_iter().map(tididi::Literal::pos)); Ok(()) })
}
/// Report stored nodes and child pairs, not models. Implicit leaves are excluded from the node count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_size(value: *const TididiCircuit, nodes: *mut usize, pairs: *mut usize) -> *mut TididiError {
    boundary(|| { if nodes == pairs { return Err(invalid("size output slots must be distinct")); }
        let nodes = unsafe { output(nodes)? }; let pairs = unsafe { output(pairs)? }; let f = unsafe { borrow(value)? };
        *nodes = f.node_count(); *pairs = f.pair_count(); Ok(()) })
}
/// One internal node's storage size. IDs are local and may change after transformations.
#[repr(C)]
pub struct TididiNodeSize { pub vtree_node: u32, pub local_node: usize, pub pairs: usize }
/// An owned snapshot of internal-node sizes, independent of the circuit's lifetime.
pub struct TididiNodeSizes(Vec<TididiNodeSize>);
/// Snapshot the size of every stored internal node, borrowing the circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_node_sizes(value: *const TididiCircuit, out: *mut *mut TididiNodeSizes) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let f = unsafe { borrow(value)? };
        let rows = f.vtree().bottomup().flat_map(|v| f.level(v).internal_inputs_iter()
            .map(move |(i, pairs)| TididiNodeSize { vtree_node: v.0, local_node: i, pairs: pairs.len() })).collect();
        *out = Box::into_raw(Box::new(TididiNodeSizes(rows))); Ok(()) })
}
/// Return the length of a live, nonnull node-size snapshot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_node_sizes_len(value: *const TididiNodeSizes) -> usize { unsafe { (*value).0.len() } }
/// Borrow the snapshot's rows until it is freed. Do not modify or free the array separately.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_node_sizes_data(value: *const TididiNodeSizes) -> *const TididiNodeSize { unsafe { (*value).0.as_ptr() } }
/// Free a snapshot; NULL is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_node_sizes_free(value: *mut TididiNodeSizes) {
    if !value.is_null() { drop(unsafe { Box::from_raw(value) }); }
}
