use tididi::vtree::Home;

use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::limits::LimitConfig;
use tididi::test_helpers::assert_canonical;

#[test]
fn per_node_counts_preserve_exact_zero_and_saturate_overflow() {
    let tree = Home::new(Vtree::balanced(129));
    // On a home the constants lie on the empty projection; a clause of
    // every variable keeps the 129 leaves.
    let all = Tdd::clause(&tree, 1..=129).unwrap();
    let zero = Tdd::zero(&tree);
    assert_canonical(&all);
    assert_canonical(&zero);
    let counts = all.node_counts_u128().unwrap();
    assert_eq!(counts[all.vtree().root().idx()][all.output().local.idx()], u128::MAX);
    let counts = zero.node_counts_u128().unwrap();
    assert!(counts.iter().all(Vec::is_empty));
}

#[test]
fn per_node_count_refusal_leaves_the_diagram_available() {
    let tree = Home::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    {
        let _limits = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.node_counts_u128(&f), Err(OperationError::OverBudget));
    }
    let counts = engine.node_counts_u128(&f).unwrap();
    // Per node over the leaves of the projection on x1 and x2; the model
    // count is over the home's three.
    assert_eq!(counts[f.vtree().root().idx()][f.output().local.idx()], 3);
    assert_eq!(f.model_count().unwrap(), 6u32.into());
    assert_canonical(&f);
}
