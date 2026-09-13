use super::*;
use crate::limits::{LimitConfig, OperationError};
use crate::marginal::marginalize_levels;
use crate::vtree::VtreeIdx;

#[test]
fn invalid_targets_are_rejected_before_work_or_mutation() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let original = Tdd::clause(&tree, [1, 3]);
    assert_canonical(&original);
    let before = normalized_levels(&original);
    for bad in [VtreeIdx(tree.num_nodes() as u32), VtreeIdx(u32::MAX)] {
        let targets = [tree.children(tree.root()).0, bad];
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        let mut f = original.clone();
        assert_eq!(marginalize_levels(&eng, &mut f, &targets), Err(OperationError::LevelNotInVtree(bad)));
        assert_canonical(&f);
        assert_eq!(normalized_levels(&f), before);
        assert!(!f.has_marginal_level());
        assert_eq!(eng.and_marginalizing(f, original.clone(), &targets).unwrap_err(), OperationError::LevelNotInVtree(bad));
    }
}

#[test]
fn negation_and_disjunction_reject_summed_out_structure() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 3]);
    marginalize_levels(&eng, &mut f, &[tree.children(tree.root()).0]).unwrap();
    assert_canonical(&f);
    let first = f.levels.iter().position(|level| level.is_marginal()).unwrap();
    let error = OperationError::MarginalLevel(VtreeIdx(first as u32));
    let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(eng.negate(f.clone()).unwrap_err(), error);
    for other in [Tdd::zero(&tree), Tdd::one(&tree)] {
        assert_canonical(&other);
        assert_eq!(eng.or(f.clone(), other.clone()).unwrap_err(), error);
        assert_eq!(eng.or(other, f.clone()).unwrap_err(), error);
    }
}

#[test]
fn a_clause_rejects_unknown_variables_even_on_false_input() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    for f in [Tdd::zero(&tree), Tdd::one(&tree)] {
        assert_canonical(&f);
        assert_eq!(eng.and_clause(f, &[3.into()]).unwrap_err(), OperationError::VariableNotInVtree(VarId(2)));
    }
}

#[test]
fn a_clause_rejects_marginal_leaf_labels() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 2]);
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    marginalize_levels(&eng, &mut f, &[leaf]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.and_clause(f, &[1.into()]).unwrap_err(), OperationError::MarginalLevel(leaf));
}

#[test]
fn projection_rejects_a_marginal_ancestor() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 3]);
    let left = tree.children(tree.root()).0;
    marginalize_levels(&eng, &mut f, &[left]).unwrap();
    assert_canonical(&f);
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        assert_eq!(eng.exists_var(f.clone(), VarId(0), how).unwrap_err(), OperationError::MarginalLevel(left));
    }
}

#[test]
fn projection_rejects_a_marginal_target_leaf() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 2]);
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    marginalize_levels(&eng, &mut f, &[leaf]).unwrap();
    assert_canonical(&f);
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        assert_eq!(eng.exists_var(f.clone(), VarId(0), how).unwrap_err(), OperationError::MarginalLevel(leaf));
    }
}

#[test]
fn projection_rejects_a_rewritten_ancestors_marginal_grandchild() {
    let tree = Arc::new(Vtree::balanced(8));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 5]);
    let left = tree.children(tree.root()).0;
    let grandchild = tree.children(left).0;
    marginalize_levels(&eng, &mut f, &[grandchild]).unwrap();
    crate::reduce::minimize(&mut f);
    assert_canonical(&f);
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        assert_eq!(eng.exists_var(f.clone(), VarId(4), how).unwrap_err(), OperationError::MarginalLevel(grandchild));
    }
}

#[test]
fn a_clause_rejects_a_marginal_spine() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1, 3]);
    let left = tree.children(tree.root()).0;
    marginalize_levels(&eng, &mut f, &[left]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.and_clause(f, &[1.into()]).unwrap_err(), OperationError::MarginalLevel(left));
}
