use crate::{Engine, Tdd, Vtree};
use crate::test_helpers::assert_canonical;
use std::sync::Arc;

#[test]
fn conjunction_returns_workspace_after_refusal_and_reuses_it() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let left = Tdd::clause(&vtree, [1, 3, 5]).unwrap();
    let right = Tdd::clause(&vtree, [2, 4, 6]).unwrap();
    assert_canonical(&left);
    assert_canonical(&right);
    let mut f = eng.and(left.clone(), right.clone()).unwrap();
    eng.minimize(&mut f).unwrap();
    assert_canonical(&f);
    let allocation = eng.scratch.apply.workspace.checkout(eng.limits()).f_widths.as_ptr();
    let mut refusals = 0;
    for cut in 0..20 {
        eng.limits().refuse_nth_reserve(cut);
        let result = eng.and(left.clone(), right.clone());
        eng.limits().grant_every_reserve();
        if result.is_err() { refusals += 1; }
        assert_eq!(eng.scratch.apply.workspace.checkout(eng.limits()).f_widths.as_ptr(), allocation);
        let mut recovered = eng.and(left.clone(), right.clone()).unwrap();
        eng.minimize(&mut recovered).unwrap();
        assert_canonical(&recovered);
        assert!(recovered.equivalent(&f).unwrap());
    }
    assert!(refusals > 0);
}

#[test]
fn ordinary_conjunctions_reuse_the_context_workspace() {
    let vtree = Arc::new(Vtree::balanced(4));
    let left = Tdd::clause(&vtree, [1, 3]).unwrap();
    let right = Tdd::clause(&vtree, [2, 4]).unwrap();
    assert_canonical(&left);
    assert_canonical(&right);
    let mut warm = crate::and(left.clone(), right.clone()).unwrap();
    warm.minimize().unwrap();
    assert_canonical(&warm);
    let allocation = vtree.context().run(|eng| {
        eng.scratch.apply.workspace.checkout(eng.limits()).f_widths.as_ptr()
    });
    for operator in [false, true] {
        let mut result = if operator { left.clone() & right.clone() }
            else { crate::and(left.clone(), right.clone()).unwrap() };
        vtree.context().run(|eng| {
            assert_eq!(eng.scratch.apply.workspace.checkout(eng.limits()).f_widths.as_ptr(), allocation);
        });
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), 9u32.into());
    }
}

#[test]
fn returning_the_workspace_trims_the_width_arrays() {
    use crate::execution::pool::SCRATCH_RETAIN_BYTES;
    let eng = Engine::new();
    {
        let mut ws = eng.scratch.apply.workspace.checkout(eng.limits());
        // `reserve` does not touch the pages, so the test stays small in memory.
        ws.f_widths.reserve(SCRATCH_RETAIN_BYTES / std::mem::size_of::<usize>() + 1);
        ws.g_widths.reserve(8);
    }
    let ws = eng.scratch.apply.workspace.checkout(eng.limits());
    assert_eq!(ws.f_widths.capacity(), 0, "an over-cap width array is released on return");
    assert!(ws.g_widths.capacity() >= 8, "an under-cap width array stays warm");
}
