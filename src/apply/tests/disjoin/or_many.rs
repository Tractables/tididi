//! `or_many` against a fold of `or`: the same function, fewer complements.

use std::sync::Arc;
use crate::test_helpers::assert_canonical;

use crate::build::constant_zero;
use crate::vtree::Vtree;
use crate::{or, or_many, OperationError, Tdd};

fn vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

/// Every operand set a caller is likely to hand over: overlapping cubes,
/// disjoint cubes, a repeated operand, and single-literal operands.
fn operand_sets(v: &Arc<Vtree>) -> Vec<Vec<Tdd>> {
    let cube = |lits: &[i32]| Tdd::cube(v, lits.iter().copied()).expect("a cube");
    vec![
        vec![cube(&[1, 2]), cube(&[3]), cube(&[-4])],
        vec![cube(&[1]), cube(&[2]), cube(&[3]), cube(&[4])],
        vec![cube(&[1, 2, 3, 4]), cube(&[1, 2, 3, 4])],
        vec![cube(&[1, -2]), cube(&[-1, 2]), cube(&[1, 2]), cube(&[-1, -2]), cube(&[3, 4])],
        vec![cube(&[1])],
    ]
}

#[test]
fn or_many_equals_a_fold_of_or() {
    let v = vtree(4);
    for operands in operand_sets(&v) {
        let folded = operands
            .iter()
            .cloned()
            .reduce(|a, b| or(a, b).expect("a disjunction"))
            .expect("a nonempty set");
        let many = or_many(operands).expect("a disjunction");
        assert_canonical(&many);
        assert!(many.equivalent(&folded).expect("comparable"), "{} against {}", many.model_count().unwrap(), folded.model_count().unwrap());
    }
}

/// The point of the operation: `3(n - 1)` complements become `n + 1`, and the
/// result is no larger for it.
#[test]
fn or_many_is_no_larger_than_a_fold() {
    let v = vtree(4);
    for operands in operand_sets(&v) {
        let folded = operands
            .iter()
            .cloned()
            .reduce(|a, b| or(a, b).expect("a disjunction"))
            .expect("a nonempty set");
        let many = or_many(operands).expect("a disjunction");
        assert_canonical(&many);
        assert_eq!(many.pair_count(), folded.pair_count(), "both results are minimized, so both are canonical");
    }
}

#[test]
fn a_false_operand_drops_out() {
    let eng = &crate::Engine::new();
    let v = vtree(4);
    let f = Tdd::cube(&v, [1, 2]).expect("a cube");
    let with_zero = or_many([f.clone(), constant_zero(eng, &v)]).expect("a disjunction");
    assert_canonical(&with_zero);
    assert!(with_zero.equivalent(&f).expect("comparable"));
}

#[test]
fn every_operand_false_is_false() {
    let eng = &crate::Engine::new();
    let v = vtree(4);
    let result = or_many([constant_zero(eng, &v), constant_zero(eng, &v)]).expect("a disjunction");
    assert_canonical(&result);
    assert!(result.is_zero());
}

#[test]
fn one_operand_is_that_operand() {
    let v = vtree(4);
    let f = Tdd::cube(&v, [1, -3]).expect("a cube");
    let result = or_many([f.clone()]).expect("a disjunction");
    assert_canonical(&result);
    assert!(result.equivalent(&f).expect("comparable"));
}

#[test]
fn no_operand_has_no_vtree_to_answer_over() {
    assert!(matches!(or_many([]), Err(OperationError::EmptyOperands)));
}

#[test]
fn operands_must_share_a_vtree() {
    let one = Tdd::cube(&vtree(4), [1]).expect("a cube");
    let other = Tdd::cube(&vtree(4), [2]).expect("a cube");
    assert!(matches!(or_many([one, other]), Err(OperationError::VtreeMismatch)));
}
