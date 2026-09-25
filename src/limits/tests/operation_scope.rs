//! The meters are zeroed when an operation starts, and only by the outermost
//! one on an engine.

use std::sync::Arc;

use crate::reduce::{ReductionPlan};
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

use super::*;

#[test]
fn the_outermost_operation_zeroes_the_meters_and_a_nested_one_keeps_them() {
    let lim = Limits::new();
    lim.charge_in_flight(7);
    let outer = lim.begin_operation();
    assert_eq!(lim.meters().in_flight_bytes, 0);
    lim.charge_in_flight(5);
    let inner = lim.begin_operation();
    assert_eq!(lim.meters().in_flight_bytes, 5, "a nested operation keeps the outer meter");
    drop(inner);
    assert_eq!(lim.meters().in_flight_bytes, 5);
    drop(outer);
    let _next = lim.begin_operation();
    assert_eq!(lim.meters().in_flight_bytes, 0);
}

#[test]
fn a_reduction_does_not_inherit_an_earlier_operations_charge() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = Tdd::clause(&vtree, [1, -2, 3]).unwrap() & Tdd::clause(&vtree, [-1, 4, -5]).unwrap() & Tdd::clause(&vtree, [2, 5, 6]).unwrap();
    let before = f.model_count().unwrap();
    // What an operation that ended mid-way leaves behind, above the budget.
    eng.limits().charge_in_flight(1 << 30);
    let _armed = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(1 << 20)));
    eng.reduce(&mut f, ReductionPlan::default())
        .expect("a reduction meters only what it charges itself");
    assert_eq!(f.model_count().unwrap(), before);
}

#[test]
fn compound_operations_refuse_their_own_work_and_release_the_scope() {
    use crate::test_helpers::assert_canonical;
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    let care = Tdd::clause(&tree, [-1, 3]).unwrap();
    assert_canonical(&f);
    assert_canonical(&care);
    let eng = Engine::new();
    {
        let _budget = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert!(matches!(eng.negate(f.clone()), Err(OperationError::OverBudget)));
        assert!(matches!(eng.or(f.clone(), care.clone()), Err(OperationError::OverBudget)));
        assert!(matches!(eng.restrict_to_care(f.clone(), care.clone()), Err(OperationError::OverBudget)));
    }
    let g = eng.negate(f).unwrap();
    assert_canonical(&g);
    assert_eq!(g.model_count().unwrap(), 4u32.into());
    assert!(eng.limits().meters().in_flight_bytes > 0);
    let zero = Tdd::zero(&tree);
    assert_canonical(&zero);
    let one = eng.negate(zero).unwrap();
    assert_canonical(&one);
    assert_eq!(one.model_count().unwrap(), 16u32.into());
}

#[test]
fn an_armed_stop_comes_before_every_input_error() {
    use crate::vtree::{VarId, VtreeIdx};
    use crate::OperationError::Stopped;
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, -2]).unwrap();
    // The same shape on another vtree: every binary operation rejects the pair.
    let other = Tdd::clause(&Arc::new(Vtree::balanced(4)), [1, -2]).unwrap();
    let absent = [VarId(99)];
    let wide: Vec<VarId> = (1..=65).map(VarId).collect();
    let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
        StopCallback::new(|_, _| StopDecision::Stop))));
    let g = || other.clone();
    assert_eq!(eng.equivalent(&f, &other).err(), Some(Stopped));
    assert_eq!(eng.implies(&f, &other).err(), Some(Stopped));
    assert_eq!(eng.and(f.clone(), g()).err(), Some(Stopped));
    assert_eq!(eng.and_marginalizing(f.clone(), g(), &[]).err(), Some(Stopped));
    assert_eq!(eng.and_filter_products(f.clone(), g(), |_, _, _| true).err(), Some(Stopped));
    assert_eq!(eng.and_exists(f.clone(), g(), &absent).err(), Some(Stopped));
    assert_eq!(eng.or(f.clone(), g()).err(), Some(Stopped));
    assert_eq!(eng.or_many(vec![f.clone(), g()]).err(), Some(Stopped));
    assert_eq!(eng.nor_many(vec![f.clone(), g()]).err(), Some(Stopped));
    assert_eq!(eng.xor(f.clone(), g()).err(), Some(Stopped));
    assert_eq!(eng.ite(f.clone(), g(), f.clone()).err(), Some(Stopped));
    assert_eq!(eng.restrict_to_care(f.clone(), g()).err(), Some(Stopped));
    assert_eq!(eng.restrict_to_care_bounded(f.clone(), &other, 10).err(), Some(Stopped));
    assert_eq!(eng.exists_vars(f.clone(), &absent).err(), Some(Stopped));
    assert_eq!(eng.projected_model_count(&f, &absent).err(), Some(Stopped));
    assert_eq!(eng.condition(f.clone(), [99]).err(), Some(Stopped));
    assert_eq!(eng.cube(&vtree, [99]).err(), Some(Stopped));
    assert_eq!(eng.clause(&vtree, [0]).err(), Some(Stopped));
    assert_eq!(eng.from_models(&vtree, &wide, &[0]).err(), Some(Stopped));
    assert_eq!(eng.rotate_marginal_cluster(&mut f.clone(), VtreeIdx(999), 1, &mut Vec::new()).err(), Some(Stopped));
    assert_eq!(eng.marginalize_levels(&mut f.clone(), &[VtreeIdx(999)]).err(), Some(Stopped));
    // An unweighted diagram has no weighted value, but the stop is tested first.
    assert_eq!(eng.weighted_value(&f).err(), Some(Stopped));
}
