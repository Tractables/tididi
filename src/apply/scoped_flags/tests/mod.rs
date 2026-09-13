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
