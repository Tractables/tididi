use super::*;

#[test]
fn rust_panics_become_owned_errors() {
    let error = boundary(|| panic!("test panic at the boundary"));
    assert!(!error.is_null());
    unsafe {
        assert_eq!(tididi_error_code(error), TididiErrorCode::InternalPanic);
        assert_eq!(CStr::from_ptr(tididi_error_message(error)).to_str().unwrap(), "unexpected Rust panic");
        tididi_error_free(error);
    }
}

#[test]
fn success_and_embedded_nul_error_text_are_safe() {
    assert!(boundary(|| Ok(())).is_null());
    let error = boundary(|| Err(invalid("bad\0argument")));
    unsafe {
        assert_eq!(CStr::from_ptr(tididi_error_message(error)).to_str().unwrap(), "bad\\0argument");
        tididi_error_free(error);
        tididi_error_free(std::ptr::null_mut());
        tididi_string_free(std::ptr::null_mut());
    }
}

#[test]
fn ffi_transformation_results_satisfy_diagram_invariants() {
    unsafe {
        let mut vtree = std::ptr::null_mut();
        assert!(tididi_vtree_balanced(4, &mut vtree).is_null());
        let mut a = std::ptr::null_mut();
        let mut b = std::ptr::null_mut();
        let mut result = std::ptr::null_mut();
        assert!(tididi_literal(vtree, 1, &mut a, std::ptr::null()).is_null());
        assert!(tididi_literal(vtree, 3, &mut b, std::ptr::null()).is_null());
        tididi::test_helpers::assert_canonical(&borrow(a).unwrap());
        tididi::test_helpers::assert_canonical(&borrow(b).unwrap());
        assert!(tididi_xor(a, b, &mut result, std::ptr::null()).is_null());
        tididi::test_helpers::assert_canonical(&borrow(result).unwrap());
        let mut projected = std::ptr::null_mut();
        assert!(tididi_exists(result, [1].as_ptr(), 1, &mut projected, std::ptr::null()).is_null());
        tididi::test_helpers::assert_canonical(&borrow(projected).unwrap());
        for handle in [a, b, result, projected] { assert!(tididi_circuit_free(handle).is_null()); }
        tididi_vtree_free(vtree);
    }
}
