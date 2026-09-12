//! The two clauses with no spine: the empty clause is ⊥, a tautology is ⊤.

use std::sync::Arc;

use num_bigint::BigUint;

use crate::vtree::Vtree;
use crate::{Engine, Tdd};

#[test]
fn the_empty_clause_is_false() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [] as [i32; 0]);
    assert!(f.is_zero());
    assert_eq!(f.model_count(), BigUint::from(0u32));

    let g = Tdd::clause(&vtree, [1, -2]);
    let h = Engine::new().and_clause(g, &[]).expect("no limit is armed");
    assert!(h.is_zero());
    assert_eq!(h.model_count(), BigUint::from(0u32));
}

#[test]
fn a_tautological_clause_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, -1]);
    assert_eq!(f.model_count(), BigUint::from(8u32));
    let g = Engine::new().and_clause(Tdd::clause(&vtree, [2, 3]), &[1.into(), (-1).into()]).expect("no limit is armed");
    assert_eq!(g.model_count(), BigUint::from(6u32));
}
