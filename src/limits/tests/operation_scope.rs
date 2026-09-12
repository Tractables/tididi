//! The meters are zeroed when an operation starts, and only by the outermost
//! one on an engine.

use std::sync::Arc;

use crate::reduce::{try_minimize, MinimizeOptions};
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
    let mut f = Tdd::clause(&vtree, [1, -2, 3]) & Tdd::clause(&vtree, [-1, 4, -5]) & Tdd::clause(&vtree, [2, 5, 6]);
    let before = f.model_count();
    // What an operation that ended mid-way leaves behind, above the budget.
    eng.limits().charge_in_flight(1 << 30);
    let _armed = eng.limits().scope(LimitSet::none().budget(Some(1 << 20)));
    try_minimize(&eng, &mut f, MinimizeOptions::default())
        .expect("a reduction meters only what it charges itself");
    assert_eq!(f.model_count(), before);
}
