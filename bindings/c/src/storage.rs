use std::ffi::c_char;
use crate::*;

/// An owned byte buffer. Read its data/length and free it with tididi_bytes_free.
pub struct TididiBytes(Vec<u8>);

/// Return the size of a live, nonnull buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_bytes_len(value: *const TididiBytes) -> usize {
    unsafe { (*value).0.len() }
}

/// Borrow a buffer's data until it is freed. Never modify it or free it separately.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_bytes_data(value: *const TididiBytes) -> *const u8 {
    unsafe { (*value).0.as_ptr() }
}

/// Free an owned byte buffer; NULL is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_bytes_free(value: *mut TididiBytes) {
    if !value.is_null() { drop(unsafe { Box::from_raw(value) }); }
}

/// Serialize a circuit without consuming it. Save its vtree separately.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_to_bytes(value: *const TididiCircuit, out: *mut *mut TididiBytes) -> *mut TididiError {
    boundary(|| {
        let out = unsafe { vacant(out)? };
        let f = unsafe { borrow(value)? };
        let mut bytes = Vec::new();
        tididi::io::write_tdd(&mut bytes, &f).map_err(invalid)?;
        *out = Box::into_raw(Box::new(TididiBytes(bytes)));
        Ok(())
    })
}

/// Load circuit bytes onto an existing vtree. Use one shared vtree for related circuits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_from_bytes(tree: *const TididiVtree, data: *const u8, len: usize,
    out: *mut *mut TididiCircuit) -> *mut TididiError {
    boundary(|| {
        let out = unsafe { vacant(out)? };
        let mut bytes = unsafe { array(data, len)? };
        *out = circuit(tididi::io::read_tdd(&mut bytes, unsafe { vtree(tree)? }).map_err(invalid)?);
        Ok(())
    })
}

/// Export a circuit as owned Graphviz text. Does not consume it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_to_dot(value: *const TididiCircuit, out: *mut *mut c_char) -> *mut TididiError {
    boundary(|| {
        let out = unsafe { vacant(out)? };
        let f = unsafe { borrow(value)? };
        *out = string(tididi::io::tdd_to_dot(&f).map_err(invalid)?)?;
        Ok(())
    })
}

/// Export a vtree as owned Graphviz text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tididi_vtree_to_dot(value: *const TididiVtree, out: *mut *mut c_char) -> *mut TididiError {
    boundary(|| {
        let out = unsafe { vacant(out)? };
        *out = string(tididi::io::vtree_to_dot(unsafe { vtree(value)? }))?;
        Ok(())
    })
}
