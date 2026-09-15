use std::sync::Arc;

use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::limits::LimitConfig;
use tididi::test_helpers::assert_canonical;

#[test]
fn per_node_counts_preserve_exact_zero_and_saturate_overflow() {
    let tree = Arc::new(Vtree::balanced(129));
    let one = Tdd::one(&tree);
    let zero = Tdd::zero(&tree);
    assert_canonical(&one);
    assert_canonical(&zero);
    let counts = one.node_counts_u128().unwrap();
    assert_eq!(counts[tree.root().idx()][one.output().local.idx()], u128::MAX);
    let counts = zero.node_counts_u128().unwrap();
    assert!(counts[tree.root().idx()].is_empty());
}

#[test]
fn per_node_count_refusal_leaves_the_diagram_available() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    {
        let _limits = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.node_counts_u128(&f), Err(OperationError::OverBudget));
    }
    let counts = engine.node_counts_u128(&f).unwrap();
    assert_eq!(counts[tree.root().idx()][f.output().local.idx()], 6);
    assert_canonical(&f);
}
