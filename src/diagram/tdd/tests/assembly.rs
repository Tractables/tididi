use super::*;
use crate::{Engine, OperationError};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;

#[test]
fn checked_assembly_charges_worklists_even_when_levels_are_already_allocated() {
    let tree = Arc::new(Vtree::balanced(4));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let eng = Engine::new();
    let output = source.output;
    let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(Tdd::try_from_levels_on(&eng, tree, source.levels, output).err(), Some(OperationError::OverBudget));
}

#[test]
fn checked_assembly_polls_while_seeding_worklists() {
    let tree = Arc::new(Vtree::linear(16));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let output = source.output;
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(3)), ..StopRules::default()
    }));
    assert_eq!(Tdd::try_from_levels_on(&eng, tree, source.levels, output).err(), Some(OperationError::Stopped));
    assert_eq!(eng.limits().meters().work_units, 3);
}
