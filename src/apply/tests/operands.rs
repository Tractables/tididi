use super::*;
use crate::OperationError;

/// Boolean operators combine diagrams after their independent engines are dropped.
#[test]
fn operators_accept_diagrams_from_dropped_independent_engines() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = {
        let engine = Engine::new();
        engine.clause(&tree, [1, 2]).unwrap()
    };
    let g = {
        let engine = Engine::new();
        engine.clause(&tree, [-1, 3]).unwrap()
    };
    assert_canonical(&f);
    assert_canonical(&g);

    for (mut result, expected) in [
        (f.clone() & g.clone(), 4u32),
        (f.clone() | g.clone(), 8u32),
        (!(f & g), 4u32),
    ] {
        assert_eq!(result.model_count(), expected.into());
        crate::reduce::minimize(&mut result);
        assert_canonical(&result);
        assert_eq!(result.model_count(), expected.into());
    }
}

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
