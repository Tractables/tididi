//! Balanced and sequential disjunctions produce the same canonical result.

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

/// Both groupings minimize their result, so their canonical sizes agree.
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

/// A false operand does not short-circuit the two-operand form either: the
/// other operand comes back minimized whichever side it is on, and two false
/// operands give a minimized false.
#[test]
fn a_false_operand_leaves_the_other_minimized() {
    let eng = &crate::Engine::new();
    let v = vtree(4);
    let zero = constant_zero(eng, &v);
    let raw = crate::apply::apply_and(
        Tdd::clause(&v, [1, 2]).expect("a clause"),
        Tdd::clause(&v, [1, -3]).expect("a clause"),
    );
    assert!(!raw.levels.is_canonical(raw.output), "the fixture still owes its reduction");
    for (f, g) in [(zero.clone(), raw.clone()), (raw.clone(), zero.clone())] {
        let result = eng.or(f, g).expect("a disjunction");
        assert_canonical(&result);
        assert!(result.levels.is_canonical(result.output));
        assert!(result.equivalent(&raw).expect("comparable"));
    }
    let both = eng.or(zero.clone(), zero).expect("a disjunction");
    assert!(both.is_zero());
    assert!(both.levels.is_canonical(both.output));
}

#[test]
fn two_operands_must_share_a_vtree() {
    let eng = &crate::Engine::new();
    let one = Tdd::cube(&vtree(4), [1]).expect("a cube");
    let other = Tdd::cube(&vtree(4), [2]).expect("a cube");
    assert!(matches!(eng.or(one, other), Err(OperationError::VtreeMismatch)));
}

/// `or_many` and `nor_many` refused at each reserve in turn: each refusal
/// answers `OverBudget` with the operands' model counts as given, the
/// engine disjoins after it, and every granted attempt is the disjunction.
#[test]
fn every_refused_reserve_of_a_disjunction_leaves_the_operands() {
    use crate::test_helpers::{compile_clauses_on, rand_cnf, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x0d15_7017);
    let vtree = vtree(6);
    let operands: Vec<Tdd> =
        (0..5).map(|_| compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 6, CnfShape { clauses: 4, width: 3 }))).collect();
    let counts = || operands.iter().map(|t| t.model_count().unwrap()).collect::<Vec<_>>();
    let given = counts();
    let mut refused = 0;
    for nor in [false, true] {
        let disjoin = |eng: &crate::Engine| if nor { eng.nor_many(operands.clone()) } else { eng.or_many(operands.clone()) };
        let want = disjoin(&eng).unwrap();
        refused += eng
            .sweep_reserves(
                &OperationError::OverBudget,
                || {
                    let out = disjoin(&eng);
                    assert_eq!(counts(), given, "nor {nor}");
                    out
                },
                |got, at| {
                    assert_canonical(&got);
                    assert!(got.equivalent(&want).unwrap(), "nor {nor}, {at}");
                },
            )
            .refused;
    }
    assert!(refused > 0, "no disjunction refused");
}
