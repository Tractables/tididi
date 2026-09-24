use std::cell::{RefCell, RefMut};
use std::ffi::c_char;
use std::sync::Arc;
use tididi::query::OwnedModelCounter;
use crate::*;

/// A reusable evidence counter that owns its circuit until finish. Free its handle even after finish.
pub struct TididiCounter(RefCell<Option<OwnedModelCounter>>);
unsafe fn counter<'a>(value: *mut TididiCounter) -> Result<RefMut<'a, OwnedModelCounter>> {
    let value = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
    RefMut::filter_map(value, Option::as_mut).map_err(|_| invalid("counter has been finished"))
}
/// Move a circuit into an evidence counter. Copy first if the original must remain usable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter(value: *mut TididiCircuit, out: *mut *mut TididiCounter) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let (_, mut values) = unsafe { take_many(&[value], |_| Ok(()))? };
        let cell = values.pop().unwrap().into_counter()?;
        *out = Box::into_raw(Box::new(TididiCounter(RefCell::new(Some(cell))))); Ok(()) })
}
/// Observe signed literals. Later observations replace pins for named variables and retain the other pins.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_observe(value: *mut TididiCounter, values: *const i64, len: usize) -> *mut TididiError {
    boundary(|| { let values = literals(unsafe { array(values, len)? })?; let mut cell = unsafe { counter(value)? };
        cell.observe(values)?; Ok(()) })
}
/// Remove the observation for one variable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_clear(value: *mut TididiCounter, var: u32) -> *mut TididiError {
    boundary(|| { let var = variable(var)?; let mut cell = unsafe { counter(value)? };
        cell.set_pin(var, None)?; Ok(()) })
}
/// Remove all observations, retaining the circuit and reusable counter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_clear_all(value: *mut TididiCounter) -> *mut TididiError {
    boundary(|| { unsafe { counter(value)? }.clear_pins(); Ok(()) })
}
/// Count assignments consistent with observations. Does not consume the counter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_model_count(value: *mut TididiCounter, out: *mut u64, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let mut cell = unsafe { counter(value)? };
        let vtree = Arc::clone(cell.circuit().vtree());
        *out = integer(run(&vtree, unsafe { limits(config)? }, |e| cell.bind(e).model_count())?)?; Ok(()) })
}
/// Count under observations exactly, returning an owned decimal string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_model_count_decimal(value: *mut TididiCounter, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { counter(value)? }; let vtree = Arc::clone(cell.circuit().vtree());
        *out = string(run(&vtree, unsafe { limits(config)? }, |e| cell.bind(e).model_count())?)?; Ok(()) })
}
/// Discard observations/cache and return the original circuit. Closes the counter; its handle still needs freeing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_finish(value: *mut TididiCounter, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
        *out = circuit(cell.take().ok_or_else(|| invalid("counter has been finished"))?.into_inner()); Ok(()) })
}
/// Free a counter handle, whether open or finished. NULL is accepted; active handles return BorrowConflict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_free(value: *mut TididiCounter) -> *mut TididiError {
    boundary(|| { if !value.is_null() { drop(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
        drop(unsafe { Box::from_raw(value) }); } Ok(()) })
}


type NativeEvaluator = tididi::query::OwnedEvaluator<tididi::diagram::RationalWeights>;
/// A reusable exact weighted evaluator owning its circuit until finish. Free even after finish.
pub struct TididiEvaluator(RefCell<Option<NativeEvaluator>>);
unsafe fn evaluator<'a>(value: *mut TididiEvaluator) -> Result<RefMut<'a, NativeEvaluator>> {
    let value = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
    RefMut::filter_map(value, Option::as_mut).map_err(|_| invalid("evaluator has been finished"))
}
/// Move a circuit into an evaluator. Supply weights for every variable; they are copied.
/// Invalid weights leave the circuit usable. Copy the circuit first to retain it after success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator(value: *mut TididiCircuit, values: *const TididiWeight, len: usize, out: *mut *mut TididiEvaluator) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? };
        let weights = { let f = unsafe { borrow(value)? }; unsafe { evaluation::weights(&f, values, len)? } };
        let (_, mut values) = unsafe { take_many(&[value], |_| Ok(()))? };
        let cell = values.pop().unwrap().into_evaluator(weights)?;
        *out = Box::into_raw(Box::new(TididiEvaluator(RefCell::new(Some(cell))))); Ok(()) })
}
/// Observe signed literals; later values replace earlier observations of the same variable.
/// Invalid input preserves all observations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_observe(value: *mut TididiEvaluator, values: *const i64, len: usize) -> *mut TididiError {
    boundary(|| { let values = literals(unsafe { array(values, len)? })?;
        unsafe { evaluator(value)? }.observe(values)?; Ok(()) })
}
/// Clear one variable's observation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_clear(value: *mut TididiEvaluator, var: u32) -> *mut TididiError {
    boundary(|| { let var = variable(var)?; unsafe { evaluator(value)? }.set_pin(var, None)?; Ok(()) })
}
/// Clear all observations, retaining cached storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_clear_all(value: *mut TididiEvaluator) -> *mut TididiError {
    boundary(|| { unsafe { evaluator(value)? }.clear_pins(); Ok(()) })
}
/// Replace every variable's weights, retaining observations and invalidating cached values.
/// Invalid weights leave the previous values in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_set_weights(value: *mut TididiEvaluator, values: *const TididiWeight, len: usize) -> *mut TididiError {
    boundary(|| { let mut cell = unsafe { evaluator(value)? };
        let weights = unsafe { evaluation::weights(cell.circuit(), values, len)? };
        cell.replace_algebra(weights); Ok(()) })
}
/// Return the exact weighted sum under observations as an owned integer/fraction string.
/// Probability weights give joint probability, without normalization. Free with tididi_string_free.
/// Reads obey limits, including cached reads; refused work retains observations for retry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_value(value: *mut TididiEvaluator, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { evaluator(value)? };
        let vtree = Arc::clone(cell.circuit().vtree());
        *out = string(run(&vtree, unsafe { limits(config)? }, |eng| cell.bind(eng).value())?)?; Ok(()) })
}
/// Close an evaluator and return its original circuit, discarding observations. The handle still needs freeing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_finish(value: *mut TididiEvaluator, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
        *out = circuit(cell.take().ok_or_else(|| invalid("evaluator has been finished"))?.into_inner()); Ok(()) })
}
/// Free an evaluator handle. NULL is accepted; active handles return BorrowConflict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_free(value: *mut TididiEvaluator) -> *mut TididiError {
    boundary(|| { if !value.is_null() { drop(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
        drop(unsafe { Box::from_raw(value) }); } Ok(()) })
}
