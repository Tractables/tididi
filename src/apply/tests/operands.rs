use super::*;
use crate::OperationError;

/// Separately allocated trees are rejected even when an operand would give an immediate answer.
#[test]
fn mismatched_vtrees_are_rejected_before_shortcuts_or_allocation() {
    let engine = Engine::new();
    let a = Arc::new(Vtree::balanced(2));
    let b = Arc::new(Vtree::balanced(2));
    let _limits = engine.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    for zero in [false, true] {
        let f = if zero { Tdd::zero(&a) } else { Tdd::one(&a) };
        let g = Tdd::one(&b);
        assert_canonical(&f);
        assert_canonical(&g);
        assert_eq!(engine.and(f.clone(), g.clone()).unwrap_err(), OperationError::VtreeMismatch);
        assert_eq!(engine.or(f.clone(), g.clone()).unwrap_err(), OperationError::VtreeMismatch);
        assert_eq!(engine.restrict_to_care(f.clone(), g.clone()).unwrap_err(), OperationError::VtreeMismatch);
        assert_eq!(engine.and_marginalizing(f, g, &[a.root()]).unwrap_err(), OperationError::VtreeMismatch);
    }
}

/// Invalid output roots are rejected before the conjunction can consume either diagram.
#[test]
fn mismatched_output_roots_are_rejected() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::one(&tree);
    let mut g = Tdd::zero(&tree);
    assert_canonical(&f);
    assert_canonical(&g);
    // Deliberately violate the output convention after checking the fixture.
    g.output.vtree = tree.leaf_of(VarId(0)).unwrap();
    assert_eq!(engine.and(f.clone(), g.clone()).unwrap_err(), OperationError::RootMismatch);
    assert_eq!(engine.and_marginalizing(f, g, &[]).unwrap_err(), OperationError::RootMismatch);
}
