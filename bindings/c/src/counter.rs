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
