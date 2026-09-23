use std::cell::{RefCell, RefMut};
use std::ffi::c_char;
use std::sync::Arc;
use tididi::{Tdd, query::ModelCounter};
use crate::*;

self_cell::self_cell!(
    struct CounterCell {
        owner: Tdd,
        #[covariant]
        dependent: ModelCounter,
    }
);
/// A reusable evidence counter that owns its circuit until finish. Free its handle even after finish.
pub struct TididiCounter(RefCell<Option<CounterCell>>);
unsafe fn counter<'a>(value: *mut TididiCounter) -> Result<RefMut<'a, CounterCell>> {
    let value = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
    RefMut::filter_map(value, Option::as_mut).map_err(|_| invalid("counter has been finished"))
}
/// Move a circuit into an evidence counter. Copy first if the original must remain usable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter(value: *mut TididiCircuit, out: *mut *mut TididiCounter) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let (_, mut values) = unsafe { take_many(&[value], |_| Ok(()))? };
        let cell = CounterCell::try_new(values.pop().unwrap(), |f| f.counter())?;
        *out = Box::into_raw(Box::new(TididiCounter(RefCell::new(Some(cell))))); Ok(()) })
}
/// Observe signed literals. Later observations replace pins for named variables and retain the other pins.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_observe(value: *mut TididiCounter, values: *const i64, len: usize) -> *mut TididiError {
    boundary(|| { let values = literals(unsafe { array(values, len)? })?; let mut cell = unsafe { counter(value)? };
        check_variables(cell.borrow_owner().vtree(), values.iter().map(|l| l.var))?;
        cell.with_dependent_mut(|_, c| c.observe(values))?; Ok(()) })
}
/// Remove the observation for one variable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_clear(value: *mut TididiCounter, var: u32) -> *mut TididiError {
    boundary(|| { let var = variable(var)?; let mut cell = unsafe { counter(value)? };
        check_variables(cell.borrow_owner().vtree(), [var])?;
        cell.with_dependent_mut(|_, c| c.set_pin(var, None))?; Ok(()) })
}
/// Remove all observations, retaining the circuit and reusable counter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_clear_all(value: *mut TididiCounter) -> *mut TididiError {
    boundary(|| { unsafe { counter(value)? }.with_dependent_mut(|_, c| c.clear_pins()); Ok(()) })
}
/// Count assignments consistent with observations. Does not consume the counter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_model_count(value: *mut TididiCounter, out: *mut u64, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { output(out)? }; let mut cell = unsafe { counter(value)? };
        let vtree = Arc::clone(cell.borrow_owner().vtree());
        *out = integer(run(&vtree, unsafe { limits(config)? }, |e| cell.with_dependent_mut(|_, c| c.bind(e).model_count()))?)?; Ok(()) })
}
/// Count under observations exactly, returning an owned decimal string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_model_count_decimal(value: *mut TididiCounter, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { counter(value)? }; let vtree = Arc::clone(cell.borrow_owner().vtree());
        *out = string(run(&vtree, unsafe { limits(config)? }, |e| cell.with_dependent_mut(|_, c| c.bind(e).model_count()))?)?; Ok(()) })
}
/// Discard observations/cache and return the original circuit. Closes the counter; its handle still needs freeing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_finish(value: *mut TididiCounter, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
        *out = circuit(cell.take().ok_or_else(|| invalid("counter has been finished"))?.into_owner()); Ok(()) })
}
/// Free a counter handle, whether open or finished. NULL is accepted; active handles return BorrowConflict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_counter_free(value: *mut TididiCounter) -> *mut TididiError {
    boundary(|| { if !value.is_null() { drop(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
        drop(unsafe { Box::from_raw(value) }); } Ok(()) })
}


type NativeEvaluator<'a> = tididi::query::Evaluator<'a, tididi::diagram::RationalWeights>;
self_cell::self_cell!(
    struct EvaluatorCell {
        owner: Tdd,
        #[covariant]
        dependent: NativeEvaluator,
    }
);
/// A reusable exact weighted evaluator owning its circuit until finish. Free even after finish.
pub struct TididiEvaluator(RefCell<Option<EvaluatorCell>>);
unsafe fn evaluator<'a>(value: *mut TididiEvaluator) -> Result<RefMut<'a, EvaluatorCell>> {
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
        let cell = EvaluatorCell::try_new(values.pop().unwrap(), |f| f.evaluator(weights))?;
        *out = Box::into_raw(Box::new(TididiEvaluator(RefCell::new(Some(cell))))); Ok(()) })
}
/// Observe signed literals; later values replace earlier observations of the same variable.
/// Invalid input preserves all observations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_observe(value: *mut TididiEvaluator, values: *const i64, len: usize) -> *mut TididiError {
    boundary(|| { let values = literals(unsafe { array(values, len)? })?;
        unsafe { evaluator(value)? }.with_dependent_mut(|_, e| e.observe(values))?; Ok(()) })
}
/// Clear one variable's observation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_clear(value: *mut TididiEvaluator, var: u32) -> *mut TididiError {
    boundary(|| { let var = variable(var)?; unsafe { evaluator(value)? }.with_dependent_mut(|_, e| e.set_pin(var, None))?; Ok(()) })
}
/// Clear all observations, retaining cached storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_clear_all(value: *mut TididiEvaluator) -> *mut TididiError {
    boundary(|| { unsafe { evaluator(value)? }.with_dependent_mut(|_, e| e.clear_pins()); Ok(()) })
}
/// Replace every variable's weights, retaining observations and invalidating cached values.
/// Invalid weights leave the previous values in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_set_weights(value: *mut TididiEvaluator, values: *const TididiWeight, len: usize) -> *mut TididiError {
    boundary(|| { let mut cell = unsafe { evaluator(value)? };
        let weights = unsafe { evaluation::weights(cell.borrow_owner(), values, len)? };
        cell.with_dependent_mut(|_, e| { e.replace_algebra(weights); }); Ok(()) })
}
/// Return the exact weighted sum under observations as an owned integer/fraction string.
/// Probability weights give joint probability, without normalization. Free with tididi_string_free.
/// Reads obey limits, including cached reads; refused work retains observations for retry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_value(value: *mut TididiEvaluator, out: *mut *mut c_char, config: *const TididiLimits) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { evaluator(value)? };
        let vtree = Arc::clone(cell.borrow_owner().vtree());
        *out = string(run(&vtree, unsafe { limits(config)? }, |eng| cell.with_dependent_mut(|_, e| e.bind(eng).value()))?)?; Ok(()) })
}
/// Close an evaluator and return its original circuit, discarding observations. The handle still needs freeing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_finish(value: *mut TididiEvaluator, out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| { let out = unsafe { vacant(out)? }; let mut cell = unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?;
        *out = circuit(cell.take().ok_or_else(|| invalid("evaluator has been finished"))?.into_owner()); Ok(()) })
}
/// Free an evaluator handle. NULL is accepted; active handles return BorrowConflict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_evaluator_free(value: *mut TididiEvaluator) -> *mut TididiError {
    boundary(|| { if !value.is_null() { drop(unsafe { required(value)? }.0.try_borrow_mut().map_err(|_| busy())?);
        drop(unsafe { Box::from_raw(value) }); } Ok(()) })
}
