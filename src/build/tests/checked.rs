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
    assert_eq!(eng.clause(&tree, [Literal::from(1), Literal::from(-1), missing]).err(), Some(OperationError::VariableNotInVtree(missing.var)));
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
        assert_eq!(cube.model_count(), 4u32.into());
        assert_eq!(clause.model_count(), 12u32.into());
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
        let convenience = Tdd::clause(&tree, &literals);
        assert_canonical(&result);
        assert_canonical(&convenience);
        assert_eq!(result.model_count(), count.into());
        assert_eq!(result.model_count(), convenience.model_count());
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
    assert_eq!(result.model_count(), 256u32.into());
}
