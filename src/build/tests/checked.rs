use std::sync::Arc;
use crate::{Engine, Literal, OperationError, Tdd};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree};

#[test]
fn checked_constructors_report_invalid_variables() {
    let tree = Arc::new(Vtree::balanced(3));
    let eng = Engine::new();
    let missing = Literal::pos(VarId(u32::MAX));
    assert_eq!(eng.cube(&tree, [missing]).err(), Some(OperationError::VariableNotInVtree(missing.var)));
    assert_eq!(eng.clause(&tree, [Literal::try_from(1).unwrap(), Literal::try_from(-1).unwrap(), missing]).err(), Some(OperationError::VariableNotInVtree(missing.var)));
    for literals in [[1, 1], [1, -1]] {
        assert_eq!(eng.cube(&tree, literals).err(), Some(OperationError::DuplicateVariable(VarId(0))));
    }
    assert_eq!(OperationError::VariableNotInVtree(missing.var).to_string(), "variable x4294967296 is not in the vtree");
}

#[test]
fn checked_constructors_obey_limits_and_leave_the_engine_reusable() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    for (config, error) in [
        (LimitConfig::none().with_memory_budget_bytes(Some(0)), OperationError::OverBudget),
        (LimitConfig::none().with_output_node_cap(Some(0)), OperationError::OutputCap),
        (LimitConfig::none().with_stop_rules(StopRules { unconditional: Some(StopAt::WorkUnits(0)), ..StopRules::default() }), OperationError::Stopped),
    ] {
        {
            let _scope = eng.limits().scope(config);
            assert_eq!(eng.cube(&tree, [1, -2]).err(), Some(error));
            assert_eq!(eng.clause(&tree, [1, -2]).err(), Some(error));
        }
        let cube = eng.cube(&tree, [1, -2]).unwrap();
        let clause = eng.clause(&tree, [1, -2]).unwrap();
        assert_canonical(&cube);
        assert_canonical(&clause);
        assert_eq!(cube.model_count().unwrap(), 4u32.into());
        assert_eq!(clause.model_count().unwrap(), 12u32.into());
    }
}

#[test]
fn checked_constructors_poll_while_reading_literals() {
    let tree = Arc::new(Vtree::balanced(8));
    for cube in [false, true] {
        let eng = Engine::new();
        eng.limits().pin_reduce_poll_stride(Some(1));
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(3)), ..StopRules::default()
        }));
        let input = (1..=8).inspect(|&i| assert!(i <= 3, "construction must stop reading the iterator"));
        let result = if cube { eng.cube(&tree, input) } else { eng.clause(&tree, input) };
        assert_eq!(result.err(), Some(OperationError::Stopped));
    }
}

#[test]
fn clause_constants_and_duplicate_literals_keep_their_semantics() {
    let tree = Arc::new(Vtree::balanced(3));
    for (literals, count) in [(vec![], 0u32), (vec![1, -1], 8), (vec![1, 1], 4)] {
        let result = Engine::new().clause(&tree, &literals).unwrap();
        let convenience = Tdd::clause(&tree, &literals).unwrap();
        assert_canonical(&result);
        assert_canonical(&convenience);
        assert_eq!(result.model_count().unwrap(), count.into());
        assert_eq!(result.model_count().unwrap(), convenience.model_count().unwrap());
    }
}

#[test]
fn an_empty_cube_polls_during_node_construction() {
    let tree = Arc::new(Vtree::balanced(8));
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(2));
    {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(2)), ..StopRules::default()
        }));
        assert_eq!(eng.cube(&tree, std::iter::empty::<Literal>()).err(), Some(OperationError::Stopped));
        assert_eq!(eng.limits().work_units(), 2);
    }
    let result = eng.cube(&tree, std::iter::empty::<Literal>()).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 256u32.into());
}

#[test]
fn zero_integer_literals_return_errors_at_checked_entry_points() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    assert_eq!(Literal::try_from(0), Err(OperationError::InvalidLiteral(0)));
    assert_eq!(Literal::try_from(&0), Err(OperationError::InvalidLiteral(0)));
    assert_eq!(engine.literal(&tree, 0).err(), Some(OperationError::InvalidLiteral(0)));
    assert_eq!(engine.cube(&tree, [1, 0]).err(), Some(OperationError::InvalidLiteral(0)));
    for literals in [vec![0], vec![1, -1, 0]] {
        assert_eq!(engine.clause(&tree, &literals).err(), Some(OperationError::InvalidLiteral(0)));
    }
    for literals in [vec![0], vec![1, -1, 0]] {
        for f in [engine.one(&tree), engine.zero(&tree)] {
            assert_canonical(&f);
            assert_eq!(engine.condition(f, &literals).err(), Some(OperationError::InvalidLiteral(0)));
        }
    }
    let valid = engine.clause(&tree, [1, -2]).unwrap();
    assert_canonical(&valid);
    assert_eq!(valid.model_count().unwrap(), 6u32.into());
    assert_eq!(OperationError::InvalidLiteral(0).to_string(), "0 is not a literal; use a nonzero signed integer");
}

#[test]
fn literal_conversion_handles_signed_endpoints_and_borrowed_inputs() {
    assert_eq!(Literal::try_from(i32::MIN), Ok(Literal::neg(VarId(2147483647))));
    assert_eq!(Literal::try_from(i32::MAX), Ok(Literal::pos(VarId(2147483646))));
    let tree = Arc::new(Vtree::balanced(2));
    let engine = Engine::new();
    let integers = [1, -2];
    let typed = [Literal::pos(VarId(0)), Literal::neg(VarId(1))];
    for result in [engine.cube(&tree, integers), engine.cube(&tree, integers.iter()),
        engine.cube(&tree, typed), engine.cube(&tree, typed.iter())] {
        let f = result.unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), 1u32.into());
    }
}
