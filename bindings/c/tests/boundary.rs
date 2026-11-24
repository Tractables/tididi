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

/// Every circuit a C entry returns holds every diagram invariant. A C vtree
/// is a plain root, so a circuit stores a level for each of its variables;
/// after `minimize` the leaves it names `One` are exactly the variables
/// outside its support. The formulas are random clause sets over a few
/// variables scattered across a random vtree of many.
#[test]
fn every_c_producer_stores_only_used_levels() {
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as usize % n
        }
    }
    unsafe fn ok(error: *mut TididiError) {
        if !error.is_null() {
            panic!("{}", unsafe { CStr::from_ptr(tididi_error_message(error)) }.to_string_lossy());
        }
    }
    unsafe fn random_vtree(rng: &mut Rng, vars: &[u32]) -> *mut TididiVtree {
        let mut out = std::ptr::null_mut();
        if vars.len() == 1 {
            unsafe { ok(tididi_vtree_balanced_over(vars.as_ptr(), 1, &mut out)) };
            return out;
        }
        let split = 1 + rng.below(vars.len() - 1);
        let (left, right) = unsafe { (random_vtree(rng, &vars[..split]), random_vtree(rng, &vars[split..])) };
        unsafe {
            ok(tididi_vtree_join(left, right, &mut out));
            tididi_vtree_free(left);
            tididi_vtree_free(right);
        }
        out
    }
    unsafe fn copy(c: *mut TididiCircuit) -> *mut TididiCircuit {
        let mut out = std::ptr::null_mut();
        unsafe { ok(tididi_copy(c, &mut out)) };
        out
    }
    unsafe fn check(c: *mut TididiCircuit, what: &str) {
        let Ok(f) = (unsafe { borrow(c) }) else { panic!("{what}: no circuit") };
        let checked = std::panic::catch_unwind(AssertUnwindSafe(|| tididi::test_helpers::assert_canonical(&f)));
        assert!(checked.is_ok(), "{what}: the result breaks an invariant");
        let mut minimized = f.clone();
        minimized.minimize().unwrap();
        if let Err(e) = tididi::test_helpers::check::check_padded_support(&minimized) {
            panic!("{what}: minimized: {e}");
        }
        drop(f);
        assert!(unsafe { tididi_circuit_free(c) }.is_null());
    }
    let none = std::ptr::null();
    let mut rng = Rng(1124);
    for case in 0..60 {
        let n = 4 + rng.below(13) as u32;
        let mut order: Vec<u32> = (1..=n).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let vtree = unsafe { random_vtree(&mut rng, &order) };
        let used: Vec<u32> = order[..2 + rng.below(4.min(n as usize - 1))].to_vec();
        let lit = |rng: &mut Rng, v: u32| if rng.below(2) == 0 { i64::from(v) } else { -i64::from(v) };
        let formula = |rng: &mut Rng| unsafe {
            let mut f = std::ptr::null_mut();
            ok(tididi_one(vtree, &mut f));
            for _ in 0..1 + rng.below(4) {
                let width = 1 + rng.below(3.min(used.len()));
                let clause: Vec<i64> = (0..width).map(|_| {
                    let v = used[rng.below(used.len())];
                    lit(rng, v)
                }).collect();
                let (mut c, mut g) = (std::ptr::null_mut(), std::ptr::null_mut());
                ok(tididi_clause(vtree, clause.as_ptr(), clause.len(), &mut c, none));
                ok(tididi_and(f, c, &mut g, none));
                assert!(tididi_circuit_free(f).is_null() && tididi_circuit_free(c).is_null());
                f = g;
            }
            f
        };
        let (f, g, h) = (formula(&mut rng), formula(&mut rng), formula(&mut rng));
        let (x, y) = (used[0], used[1]);
        let some = &used[..1 + rng.below(used.len())];
        let what = |name: &str| format!("case {case} on {n} variables, formulas over {used:?}: {name}");
        let mut out = std::ptr::null_mut();
        unsafe {
            let produce = |name: &str, error: *mut TididiError, out: &mut *mut TididiCircuit| {
                ok(error);
                check(*out, &what(name));
                *out = std::ptr::null_mut();
            };
            let (xi, yi) = (i64::from(x), i64::from(y));
            produce("copy", tididi_copy(f, &mut out), &mut out);
            produce("literal", tididi_literal(vtree, -xi, &mut out, none), &mut out);
            produce("cube", tididi_cube(vtree, [xi, -yi].as_ptr(), 2, &mut out, none), &mut out);
            produce("clause", tididi_clause(vtree, [-xi, yi].as_ptr(), 2, &mut out, none), &mut out);
            produce("one", tididi_one(vtree, &mut out), &mut out);
            produce("zero", tididi_zero(vtree, &mut out), &mut out);
            produce("from_models", tididi_from_models(vtree, [x, y].as_ptr(), 2, [1u8, 0, 0, 1].as_ptr(), 2, &mut out, none), &mut out);
            produce("and", tididi_and(copy(f), copy(g), &mut out, none), &mut out);
            produce("or", tididi_or(copy(f), copy(g), &mut out, none), &mut out);
            produce("xor", tididi_xor(copy(f), copy(g), &mut out, none), &mut out);
            produce("negate", tididi_negate(copy(f), &mut out, none), &mut out);
            produce("minimize", tididi_minimize(copy(f), &mut out, none), &mut out);
            produce("condition", tididi_condition(copy(f), [xi, -yi].as_ptr(), 2, &mut out, none), &mut out);
            produce("exists", tididi_exists(copy(f), some.as_ptr(), some.len(), &mut out, none), &mut out);
            produce("forall", tididi_forall(copy(f), some.as_ptr(), some.len(), &mut out, none), &mut out);
            produce("and_exists", tididi_and_exists(copy(f), copy(g), some.as_ptr(), some.len(), &mut out, none), &mut out);
            produce("rename", tididi_rename(copy(f), [x, y].as_ptr(), [y, x].as_ptr(), 2, &mut out, none), &mut out);
            produce("or_many", tididi_or_many([copy(f), copy(g), copy(h)].as_ptr(), 3, &mut out, none), &mut out);
            produce("ite", tididi_ite(copy(f), copy(g), copy(h), &mut out, none), &mut out);
            let (insert, remove) = ([xi, yi], [-xi, -yi]);
            let cube = |c: &[i64; 2]| TididiCube { literals: c.as_ptr(), len: 2 };
            produce("update", tididi_update(copy(f), &cube(&insert), 1, &cube(&remove), 1, &mut out, none), &mut out);
            let mut bytes = std::ptr::null_mut();
            ok(tididi_to_bytes(f, &mut bytes));
            produce("from_bytes", tididi_from_bytes(vtree, tididi_bytes_data(bytes), tididi_bytes_len(bytes), &mut out), &mut out);
            tididi_bytes_free(bytes);
            let mut counter = std::ptr::null_mut();
            ok(tididi_counter(copy(f), &mut counter));
            produce("counter_finish", tididi_counter_finish(counter, &mut out), &mut out);
            assert!(tididi_counter_free(counter).is_null());
            let mut evaluator = std::ptr::null_mut();
            let (half, one) = (c"1/2", c"1");
            let weights: Vec<TididiWeight> = order.iter().map(|&variable| TididiWeight { variable, negative: half.as_ptr(), positive: one.as_ptr() }).collect();
            ok(tididi_evaluator(copy(f), weights.as_ptr(), weights.len(), &mut evaluator));
            produce("evaluator_finish", tididi_evaluator_finish(evaluator, &mut out), &mut out);
            assert!(tididi_evaluator_free(evaluator).is_null());
            for c in [f, g, h] {
                assert!(tididi_circuit_free(c).is_null());
            }
            tididi_vtree_free(vtree);
        }
    }
}
