use super::*;

/// Within a compound operation the dense preflight compares the predicted
/// cells with the budget left after the earlier steps' charges, not with the
/// whole budget, so it refuses before the conjunction starts growing.
#[test]
fn dense_preflight_reads_the_budget_left_in_the_operation() {
    use crate::limits::LimitConfig;
    let engine = Engine::new();
    let lim = engine.limits();
    let _limit = lim.scope(LimitConfig::none().with_memory_budget_bytes(Some(10 * APPLY_BYTES_PER_CELL)));
    let _op = lim.begin_operation();
    assert_eq!(preflight_dense_budget(lim, 10), Ok(()));
    lim.charge_in_flight(APPLY_BYTES_PER_CELL);
    assert_eq!(preflight_dense_budget(lim, 10), Err(OperationError::OverBudget));
    assert_eq!(preflight_dense_budget(lim, 9), Ok(()));
}
