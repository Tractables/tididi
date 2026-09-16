use std::sync::Arc;

use tididi::{Engine, Literal, OperationError, Tdd, Vtree};
use tididi::limits::LimitConfig;
use tididi::test_helpers::assert_canonical;
use tididi::vtree::VarId;

/// Minimize a clause result and compare it to the independently built clause.
fn assert_clause(mut result: Tdd, expected: &Tdd) {
    result.minimize().unwrap();
    assert_canonical(&result);
    assert!(result.equivalent(expected).unwrap());
}

#[test]
fn clause_conjunction_accepts_integer_and_typed_collections() {
    let tree = Arc::new(Vtree::balanced(3));
    let signed = vec![1, -2];
    let typed = vec![Literal::pos(VarId(0)), Literal::neg(VarId(1))];
    let expected = Tdd::clause(&tree, &signed).unwrap();
    let one = Tdd::one(&tree);
    assert_canonical(&expected);
    assert_canonical(&one);
    let engine = Engine::new();
    assert_clause(one.clone().and_clause([1, -2]).unwrap(), &expected);
    assert_clause(one.clone().and_clause(signed.as_slice()).unwrap(), &expected);
    assert_clause(one.clone().and_clause(&signed).unwrap(), &expected);
    assert_clause(one.clone().and_clause(signed).unwrap(), &expected);
    assert_clause(engine.and_clause(one.clone(), &[typed[0], typed[1]]).unwrap(), &expected);
    assert_clause(engine.and_clause(one.clone(), typed.as_slice()).unwrap(), &expected);
    assert_clause(engine.and_clause(one.clone(), &typed).unwrap(), &expected);
    assert_clause(engine.and_clause(one, &typed).unwrap(), &expected);
}

#[test]
fn integer_clause_boundaries_preserve_boolean_meaning() {
    let tree = Arc::new(Vtree::balanced(3));
    let input = tididi::literal(&tree, 3).unwrap();
    let zero = Tdd::zero(&tree);
    let expected = Tdd::cube(&tree, [1, 3]).unwrap();
    assert_canonical(&input);
    assert_canonical(&zero);
    assert_canonical(&expected);
    assert_clause(input.clone().and_clause([] as [i32; 0]).unwrap(), &zero);
    assert_clause(input.clone().and_clause([1, -1]).unwrap(), &input);
    assert_clause(input.and_clause([1, 1]).unwrap(), &expected);
    assert_clause(zero.clone().and_clause([1, -2]).unwrap(), &zero);
}

#[test]
fn integer_clauses_validate_even_after_tautology_or_on_false() {
    let tree = Arc::new(Vtree::balanced(3));
    for input in [Tdd::one(&tree), Tdd::zero(&tree)] {
        assert_canonical(&input);
        assert_eq!(input.clone().and_clause([1, -1, 0]).unwrap_err(), OperationError::InvalidLiteral(0));
        assert_eq!(input.clone().and_clause([1, -1, 4]).unwrap_err(), OperationError::VariableNotInVtree(VarId(3)));
        assert_eq!(input.and_clause([i32::MIN]).unwrap_err(), OperationError::VariableNotInVtree(VarId(i32::MAX as u32)));
    }
}

#[test]
fn integer_conversion_obeys_the_batch_allocation_budget() {
    let tree = Arc::new(Vtree::balanced(3));
    let input = Tdd::one(&tree);
    assert_canonical(&input);
    let engine = Engine::new();
    {
        let _limits = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.and_clause(input.clone(), &[1, -1]).unwrap_err(), OperationError::OverBudget);
    }
    let expected = tididi::literal(&tree, 1).unwrap();
    assert_canonical(&expected);
    assert_clause(engine.and_clause(input, &[1]).unwrap(), &expected);
}
