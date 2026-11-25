use crate::vtree::Home;
use super::*;
use crate::limits::{LimitConfig, OperationError};

use crate::vtree::VtreeIdx;

#[test]
fn negation_and_disjunction_reject_summed_out_structure() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 3]).unwrap();
    crate::test_helpers::marginalize_under(&eng, &mut f, &[tree.children(tree.root()).0]).unwrap();
    assert_canonical(&f);
    let first = f.levels.iter().position(|level| level.is_marginal()).unwrap();
    let error = OperationError::MarginalLevel(VtreeIdx(first as u32));
    let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(eng.negate(f.clone()).unwrap_err(), error);
    for other in [Tdd::zero(&crate::vtree::Home::unchecked(&tree)), Tdd::one(&crate::vtree::Home::unchecked(&tree))] {
        assert_canonical(&other);
        assert_eq!(eng.or(f.clone(), other.clone()).unwrap_err(), error);
        assert_eq!(eng.or(other, f.clone()).unwrap_err(), error);
    }
}

#[test]
fn a_clause_rejects_unknown_variables_even_on_false_input() {
    let tree = Home::new(Vtree::balanced(2));
    let eng = Engine::new();
    for f in [Tdd::zero(&tree), Tdd::one(&tree)] {
        assert_canonical(&f);
        assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(3).unwrap()]).unwrap_err(), OperationError::VariableNotInVtree(VarId(3)));
    }
}

#[test]
fn a_clause_rejects_marginal_leaf_labels() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 2]).unwrap();
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    crate::test_helpers::marginalize_under(&eng, &mut f, &[leaf]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(1).unwrap()]).unwrap_err(), OperationError::MarginalLevel(leaf));
}

#[test]
fn projection_rejects_a_marginal_ancestor() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 3]).unwrap();
    let left = tree.children(tree.root()).0;
    crate::test_helpers::marginalize_under(&eng, &mut f, &[left]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.exists_var(f, VarId(1)).unwrap_err(), OperationError::MarginalLevel(left));
}

#[test]
fn projection_rejects_a_marginal_target_leaf() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 2]).unwrap();
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    crate::test_helpers::marginalize_under(&eng, &mut f, &[leaf]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.exists_var(f, VarId(1)).unwrap_err(), OperationError::MarginalLevel(leaf));
}

#[test]
fn projection_rejects_a_rewritten_ancestors_marginal_grandchild() {
    let tree = Arc::new(Vtree::balanced(8));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 5]).unwrap();
    let left = tree.children(tree.root()).0;
    let grandchild = tree.children(left).0;
    crate::test_helpers::marginalize_under(&eng, &mut f, &[grandchild]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    assert_eq!(eng.exists_var(f, VarId(5)).unwrap_err(), OperationError::MarginalLevel(grandchild));
}

#[test]
fn a_clause_rejects_a_marginal_spine() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, 3]).unwrap();
    let left = tree.children(tree.root()).0;
    crate::test_helpers::marginalize_under(&eng, &mut f, &[left]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(1).unwrap()]).unwrap_err(), OperationError::MarginalLevel(left));
}
