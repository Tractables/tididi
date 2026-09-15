//! The clause rebuild under the engine's limits: a stop decision and the
//! output cap both cut it between two spine levels.

use std::sync::Arc;

use crate::limits::LimitConfig;
use crate::test_helpers::stopping_engine;
use crate::vtree::Vtree;
use crate::{OperationError, Engine, Tdd};

#[test]
fn a_stop_decision_cuts_the_rebuild() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = stopping_engine();
    let f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap();
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(3).unwrap(), crate::Literal::try_from(4).unwrap()]).err(), Some(OperationError::Stopped));
}

#[test]
fn the_output_cap_counts_the_rebuilt_levels() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap();
    let _armed = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(3).unwrap(), crate::Literal::try_from(4).unwrap()]).err(), Some(OperationError::OutputCap));
}
