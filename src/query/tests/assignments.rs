use super::*;
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;
use crate::Literal;
use crate::OperationError;

fn function(tree: &Arc<Vtree>, bits: u16) -> Tdd {
    let eng = Engine::new();
    let n = tree.num_leaves() as usize;
    let mut result = Tdd::zero(tree);
    for row in 0..(1usize << n) {
        if bits & (1 << row) != 0 {
            let cube = eng
                .cube(
                    tree,
                    (0..n).map(|v| Literal::new(VarId(v as u32 + 1), row & (1 << v) != 0)),
                )
                .unwrap();
            result = eng.or(result, cube).unwrap();
        }
    }
    result
}

fn row_of(model: &[Literal]) -> usize {
    model.iter().fold(0usize, |a, lit| a | ((lit.sign as usize) << lit.var.idx()))
}

#[test]
fn a_false_function_lists_no_assignments() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::zero(&tree);
    assert_canonical(&f);
    assert!(f.satisfying_assignments(8).unwrap().is_empty());
    assert!(f.satisfying_assignment().unwrap().is_none());
}

#[test]
fn a_zero_limit_returns_empty_after_the_structure_check() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    assert!(f.satisfying_assignments(0).unwrap().is_empty());
    let mut marginal = f.clone();
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    Engine::new().marginalize_levels(&mut marginal, &[leaf]).unwrap();
    assert_eq!(
        marginal.satisfying_assignments(0).unwrap_err(),
        OperationError::MarginalLevel(leaf)
    );
}

#[test]
fn the_first_listed_assignment_is_the_witness() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let witness = f.satisfying_assignment().unwrap().unwrap();
    let models = f.satisfying_assignments(8).unwrap();
    assert_eq!(models[0], witness);
    assert_eq!(models.len(), 6);
}

#[test]
fn listed_assignments_match_every_two_variable_truth_table() {
    let eng = Engine::new();
    for tree in [Vtree::balanced(2), Vtree::linear(2), Vtree::reverse_linear(2)] {
        let tree = Arc::new(tree);
        for bits in 0..16u16 {
            let f = function(&tree, bits);
            assert_canonical(&f);
            let models = f.satisfying_assignments(16).unwrap();
            assert_eq!(eng.satisfying_assignments(&f, 16).unwrap(), models);
            let mut seen = [false; 4];
            for model in &models {
                assert_eq!(model.len(), 2);
                assert_eq!(
                    model.iter().map(|lit| lit.var).collect::<Vec<_>>(),
                    vec![VarId(1), VarId(2)]
                );
                let row = row_of(model);
                assert!(bits & (1 << row) != 0, "bits {bits} extra row {row}");
                assert!(!seen[row], "bits {bits} repeated row {row}");
                seen[row] = true;
                assert!(Tdd::cube(&tree, model.clone()).unwrap().implies(&f).unwrap());
            }
            for (row, hit) in seen.iter().enumerate() {
                assert_eq!(*hit, bits & (1 << row) != 0, "bits {bits} missing row {row}");
            }
            match f.satisfying_assignment().unwrap() {
                None => assert!(models.is_empty()),
                Some(first) => assert_eq!(models[0], first),
            }
            let capped = f.satisfying_assignments(2).unwrap();
            assert!(capped.len() <= 2);
            assert_eq!(capped, models.iter().take(2).cloned().collect::<Vec<_>>());
        }
    }
}

#[test]
fn free_variables_take_both_values() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let models = f.satisfying_assignments(16).unwrap();
    assert_eq!(models.len(), 8);
    let mut seen = [false; 8];
    for model in models {
        seen[row_of(&model)] = true;
    }
    assert!(seen.iter().all(|&hit| hit));
}

#[test]
fn listed_assignments_ignore_weights() {
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1]).unwrap();
    assert_canonical(&f);
    let unweighted = f.satisfying_assignments(8).unwrap();
    f.set_weights(WeightStore::new(RationalWeights::unit(2), Arithmetic::ExactRational)).unwrap();
    assert_eq!(f.satisfying_assignments(8).unwrap(), unweighted);
}

#[test]
fn a_deep_vtree_lists_the_witness_without_recursion() {
    let tree = Arc::new(Vtree::linear(8192));
    let eng = Engine::new();
    let f = eng.literal(&tree, 8192).unwrap();
    let models = eng.satisfying_assignments(&f, 1).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0], eng.satisfying_assignment(&f).unwrap().unwrap());
    assert_eq!(models[0].len(), 8192);
    assert!(models[0][..8191].iter().all(|lit| !lit.sign));
    assert_eq!(models[0][8191], Literal::pos(VarId(8192)));
}

#[test]
fn listing_assignments_honors_stops_and_recovers() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(0)),
            after_pairs: None,
        }));
        assert_eq!(eng.satisfying_assignments(&f, 4), Err(OperationError::Stopped));
    }
    assert_eq!(eng.satisfying_assignments(&f, 4).unwrap().len(), 3);
}

#[test]
fn listing_assignments_refuses_a_zero_memory_budget() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(eng.satisfying_assignments(&f, 4), Err(OperationError::OverBudget));
}
