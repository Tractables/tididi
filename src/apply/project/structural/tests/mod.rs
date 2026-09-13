use super::*;
use std::sync::Arc;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree};

#[test]
fn regrouping_checks_its_allocations_before_reduction() {
    let tree = Arc::new(Vtree::balanced(4));
    let original = Tdd::clause(&tree, [1, -2, 3]);
    assert_canonical(&original);
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    {
        let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        let mut work = Rewrite { eng: &eng, gate: PollGate::new(1), emitted: 0 };
        assert_eq!(regroup_leaf_parent(&mut work, &mut original.clone(), parent, tree.children(parent).0 == leaf).err(), Some(OperationError::OverBudget));
    }
    let result = exists_var_structural(&eng, original, leaf).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count(), 16u32.into());
}

#[test]
fn regrouping_polls_inside_a_level() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&tree, [1, -2, 3]);
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(2)), ..StopRules::default()
    }));
    let mut work = Rewrite { eng: &eng, gate: PollGate::new(1), emitted: 0 };
    assert_eq!(regroup_leaf_parent(&mut work, &mut f, parent, tree.children(parent).0 == leaf).err(), Some(OperationError::Stopped));
    assert_eq!(work.emitted, 0);
}

#[test]
fn structural_projection_recovers_after_each_refused_reservation() {
    let tree = Arc::new(Vtree::balanced(5));
    let f = Tdd::clause(&tree, [1, -2, 3]) & Tdd::clause(&tree, [-1, 4, 5]);
    let mut f = f;
    crate::reduce::minimize(&mut f);
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    let mut reached_success = false;
    for nth in 0..512 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        match exists_var_structural(&eng, f.clone(), leaf) {
            Ok(result) => { assert_canonical(&result); reached_success = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        eng.limits().grant_every_reserve();
        let result = exists_var_structural(&eng, f.clone(), leaf).unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count(), 30u32.into());
    }
    assert!(reached_success, "the sweep must cover every reservation");
}
