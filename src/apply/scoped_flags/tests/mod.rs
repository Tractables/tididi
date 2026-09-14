use super::*;
use crate::Engine;

#[test]
fn warmed_flags_reuse_the_rollback_log_after_unwind() {
    let eng = Engine::new();
    let pool = Pool::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut flags = ScopedFlags::take(eng.limits(), &pool, 100).unwrap();
        flags.set(VtreeIdx(7));
        flags.set(VtreeIdx(7));
        panic!("marking stopped");
    }));
    assert!(result.is_err());
    let _limit = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    let flags = ScopedFlags::take(eng.limits(), &pool, 100).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
}

#[test]
fn set_reports_first_visit_and_nested_scopes_keep_independent_marks() {
    let eng = Engine::new();
    let pool = Pool::default();
    let mut outer = ScopedFlags::take(eng.limits(), &pool, 8).unwrap();
    assert!(outer.set(VtreeIdx(3)));
    assert!(!outer.set(VtreeIdx(3)));
    {
        let mut inner = ScopedFlags::take(eng.limits(), &pool, 8).unwrap();
        assert!(inner.set(VtreeIdx(3)));
        assert!(inner.set(VtreeIdx(5)));
    }
    assert!(outer[3]);
    assert!(!outer[5]);
    drop(outer);
    let reused = ScopedFlags::take(eng.limits(), &pool, 8).unwrap();
    assert!(reused.iter().all(|&flag| !flag));
}
