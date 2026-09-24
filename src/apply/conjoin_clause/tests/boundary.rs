//! The two clauses with no spine: the empty clause is ⊥, a tautology is ⊤.

use std::sync::Arc;

use num_bigint::BigUint;

use crate::vtree::Vtree;
use crate::{Engine, Tdd};

#[test]
fn the_empty_clause_is_false() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [] as [i32; 0]).unwrap();
    assert!(f.is_zero());
    assert_eq!(f.model_count().unwrap(), BigUint::from(0u32));

    let g = Tdd::clause(&vtree, [1, -2]).unwrap();
    let h = Engine::new().and_clause(g, &[] as &[crate::Literal]).expect("no limit is armed");
    assert!(h.is_zero());
    assert_eq!(h.model_count().unwrap(), BigUint::from(0u32));
}

#[test]
fn a_tautological_clause_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, -1]).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(8u32));
    let g = Engine::new().and_clause(Tdd::clause(&vtree, [2, 3]).unwrap(), &[crate::Literal::try_from(1).unwrap(), crate::Literal::try_from(-1).unwrap()]).expect("no limit is armed");
    assert_eq!(g.model_count().unwrap(), BigUint::from(6u32));
}

/// The two identity exits hand the accumulator back as it stands: a
/// minimized operand owes the reduction passes nothing afterwards, where a
/// rebuilt one would owe them every level.
#[test]
fn the_identity_exits_leave_nothing_to_contract() {
    let vtree = Arc::new(Vtree::balanced(3));
    let eng = Engine::new();
    let tautology = [crate::Literal::try_from(1).unwrap(), crate::Literal::try_from(-1).unwrap()];
    let mut f = Tdd::clause(&vtree, [2, 3]).unwrap();
    eng.minimize(&mut f).unwrap();
    let g = eng.and_clause(f, &tautology).unwrap();
    assert!(g.dirty.is_empty(), "a tautology conjoined leaves the minimized operand as it was");
    assert_eq!(g.model_count().unwrap(), BigUint::from(6u32));

    let mut zero = Tdd::clause(&vtree, [] as [i32; 0]).unwrap();
    eng.minimize(&mut zero).unwrap();
    let still_zero = eng.and_clause(zero, &[crate::Literal::try_from(2).unwrap()]).unwrap();
    assert!(still_zero.is_zero());
    assert!(still_zero.dirty.is_empty(), "a false operand is handed back as it stands");
}
