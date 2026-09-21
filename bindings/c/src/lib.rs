//! C ownership and allocation boundaries. Algorithms live in the tididi crate.
//!
//! All pointers crossing this interface must refer to valid, aligned storage of
//! the declared type for the duration of the call. Output slots must not overlap
//! input storage. A null array is accepted only with length zero. Handles must
//! originate from this library; synchronize access to each handle across threads.
#![allow(clippy::missing_safety_doc)] // The common pointer contract applies to every entry point.

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tididi::OperationError;

mod domain;
mod circuit;
mod query;
mod counter;
mod evaluation;
mod storage;
pub use domain::*;
pub use circuit::*;
pub use query::*;
pub use counter::*;
pub use evaluation::*;
pub use storage::*;

/// Classification of an operation failure.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TididiErrorCode {
    /// Invalid input, incompatible domains, or a finished counter.
    InvalidArgument = 1,
    /// A live circuit handle no longer contains a circuit.
    ConsumedCircuit = 2,
    /// A callback tried to mutate or free an actively borrowed handle.
    BorrowConflict = 3,
    /// The operation exceeded its charged-storage budget.
    MemoryLimit = 4,
    /// The operation reached its timeout or output-node cap.
    ResourceLimit = 5,
    /// A count or index cannot fit the requested integer representation.
    Overflow = 6,
    /// An unexpected Rust panic was caught at the boundary.
    InternalPanic = 7,
}

/// An owned error. Read its code/message, then call tididi_error_free.
#[derive(Debug)]
pub struct TididiError { code: TididiErrorCode, message: CString }
impl TididiError {
    fn new(code: TididiErrorCode, message: impl ToString) -> Self {
        Self { code, message: CString::new(message.to_string().replace('\0', "\\0")).unwrap() }
    }
}
type Result<T> = std::result::Result<T, TididiError>;
fn invalid(message: impl ToString) -> TididiError { TididiError::new(TididiErrorCode::InvalidArgument, message) }
fn busy() -> TididiError { TididiError::new(TididiErrorCode::BorrowConflict, "handle is already borrowed by an operation") }
fn consumed() -> TididiError { TididiError::new(TididiErrorCode::ConsumedCircuit, "circuit has been consumed; copy it before a consuming operation to retain it") }
impl From<OperationError> for TididiError {
    fn from(error: OperationError) -> Self {
        let code = match error {
            OperationError::OverBudget => TididiErrorCode::MemoryLimit,
            OperationError::Stopped | OperationError::OutputCap => TididiErrorCode::ResourceLimit,
            OperationError::IndexOverflow => TididiErrorCode::Overflow,
            _ => TididiErrorCode::InvalidArgument,
        };
        Self::new(code, error)
    }
}
fn boundary(operation: impl FnOnce() -> Result<()>) -> *mut TididiError {
    let result = catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|_| {
        Err(TididiError::new(TididiErrorCode::InternalPanic, "unexpected Rust panic"))
    });
    match result { Ok(()) => std::ptr::null_mut(), Err(error) => Box::into_raw(Box::new(error)) }
}
unsafe fn required<'a, T>(value: *const T) -> Result<&'a T> {
    unsafe { value.as_ref() }.ok_or_else(|| invalid("required pointer is null"))
}
unsafe fn output<'a, T>(value: *mut T) -> Result<&'a mut T> {
    unsafe { value.as_mut() }.ok_or_else(|| invalid("output pointer is null"))
}
unsafe fn vacant<'a, T>(value: *mut *mut T) -> Result<&'a mut *mut T> {
    let slot = unsafe { output(value)? };
    if !slot.is_null() { return Err(invalid("initialize an owned output pointer to NULL before calling")); }
    Ok(slot)
}
unsafe fn array<'a, T>(data: *const T, len: usize) -> Result<&'a [T]> {
    if len == 0 { return Ok(&[]); }
    if data.is_null() { return Err(invalid("nonempty array has a null pointer")); }
    if len > isize::MAX as usize / std::mem::size_of::<T>().max(1) { return Err(invalid("array is too large")); }
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}
unsafe fn text<'a>(value: *const c_char) -> Result<&'a str> {
    if value.is_null() { return Err(invalid("string pointer is null")); }
    unsafe { CStr::from_ptr(value) }.to_str().map_err(invalid)
}
fn string(value: impl ToString) -> Result<*mut c_char> {
    CString::new(value.to_string()).map(CString::into_raw).map_err(invalid)
}

/// Return an error's code. error must be nonnull and live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_error_code(error: *const TididiError) -> TididiErrorCode { unsafe { (*error).code } }
/// Borrow a UTF-8 message until error is freed. error must be nonnull and live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_error_message(error: *const TididiError) -> *const c_char { unsafe { (*error).message.as_ptr() } }
/// Free an error; NULL is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_error_free(error: *mut TididiError) {
    if !error.is_null() { drop(unsafe { Box::from_raw(error) }); }
}
/// Free an unchanged string returned by this library; NULL is accepted. Never use free().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_string_free(value: *mut c_char) {
    if !value.is_null() { drop(unsafe { CString::from_raw(value) }); }
}
/// Return the binding version as a borrowed static string. Do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn tididi_version() -> *const c_char { concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast() }

#[cfg(test)]
#[path = "../tests/boundary.rs"]
mod tests;
