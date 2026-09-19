use super::*;
use std::sync::Arc;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree};

/// The map a quantified leaf hands its parent: every label becomes ⊤.
fn freed_leaf(eng: &Engine) -> Remap {
    Runs::all_to_first(eng.limits(), crate::diagram::LEAF_WIDTH, 0u32).unwrap()
}

#[test]
fn regrouping_checks_its_allocations_before_reduction() {
    let tree = Arc::new(Vtree::balanced(4));
    let original = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    assert_canonical(&original);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    {
        let freed = freed_leaf(&eng);
        let (left, _) = tree.children(parent);
        let (below_left, below_right) =
            if left == leaf { (Some(&freed), None) } else { (None, Some(&freed)) };
        let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        let mut work = Rewrite { eng: &eng, gate: eng.limits().gate_with(1), emitted: 0 };
        assert_eq!(regroup(&mut work, &mut original.clone(), parent, below_left, below_right).err(), Some(OperationError::OverBudget));
    }
    let result = exists_leaves_structural(&eng, original, &[leaf]).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 16u32.into());
}

#[test]
fn regrouping_polls_inside_a_level() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let parent = tree.node(leaf).parent().unwrap();
    let eng = Engine::new();
    let freed = freed_leaf(&eng);
    let (left, _) = tree.children(parent);
    let (below_left, below_right) =
        if left == leaf { (Some(&freed), None) } else { (None, Some(&freed)) };
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(2)), ..StopRules::default()
    }));
    let mut work = Rewrite { eng: &eng, gate: eng.limits().gate_with(1), emitted: 0 };
    assert_eq!(regroup(&mut work, &mut f, parent, below_left, below_right).err(), Some(OperationError::Stopped));
    assert_eq!(work.emitted, 0);
}

#[test]
fn freeing_a_whole_subtree_matches_one_variable_at_a_time() {
    let tree = Arc::new(Vtree::balanced(6));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap() & Tdd::clause(&tree, [-4, 5, 6]).unwrap();
    let mut f = f;
    f.minimize().unwrap();
    assert_canonical(&f);
    // The balanced vtree over six variables groups 1..=3 under one subtree.
    let block: Vec<VtreeIdx> = [1, 2, 3].iter().map(|&v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let at_once = exists_leaves_structural(&eng, f.clone(), &block).unwrap();
    assert_canonical(&at_once);
    let mut one_at_a_time = f;
    for &leaf in &block {
        one_at_a_time = exists_leaves_structural(&eng, one_at_a_time, &[leaf]).unwrap();
    }
    assert_canonical(&one_at_a_time);
    assert!(at_once.equivalent(&one_at_a_time).unwrap());
    assert_eq!(at_once.node_count(), one_at_a_time.node_count());
    assert_eq!(at_once.pair_count(), one_at_a_time.pair_count());
    assert_eq!(at_once.model_count().unwrap(), one_at_a_time.model_count().unwrap());
}

#[test]
fn quantifying_every_variable_leaves_the_constant() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    let all: Vec<VtreeIdx> = (1..=4).map(|v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let result = exists_leaves_structural(&eng, f, &all).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 16u32.into());
}

#[test]
fn structural_projection_recovers_after_each_refused_reservation() {
    let tree = Arc::new(Vtree::balanced(5));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap() & Tdd::clause(&tree, [-1, 4, 5]).unwrap();
    let mut f = f;
    f.minimize().unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(VarId(1)).unwrap();
    let mut reached_success = false;
    for nth in 0..512 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        match exists_leaves_structural(&eng, f.clone(), &[leaf]) {
            Ok(result) => { assert_canonical(&result); reached_success = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        eng.limits().grant_every_reserve();
        let result = exists_leaves_structural(&eng, f.clone(), &[leaf]).unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), 30u32.into());
    }
    assert!(reached_success, "the sweep must cover every reservation");
}

#[test]
fn quantifying_an_unreduced_product_matches_quantifying_a_reduced_one() {
    // `and` leaves its result for the caller to reduce, so the product holds
    // nodes nothing reaches. The sweep prunes them first; what comes out is
    // what comes out of the same quantification over the reduced product.
    let tree = Arc::new(Vtree::balanced(6));
    let product = Tdd::clause(&tree, [1, -2, 3]).unwrap()
        & Tdd::clause(&tree, [-1, 4, 5]).unwrap()
        & Tdd::clause(&tree, [2, -4, 6]).unwrap();
    assert!(!product.dirty.is_empty(), "the product still owes the passes");
    let mut reduced = product.clone();
    reduced.minimize().unwrap();
    assert!(reduced.dirty.is_empty(), "minimize discharges them");

    let leaves: Vec<VtreeIdx> = [1, 4].iter().map(|&v| tree.leaf_of(VarId(v)).unwrap()).collect();
    let eng = Engine::new();
    let from_product = exists_leaves_structural(&eng, product, &leaves).unwrap();
    let from_reduced = exists_leaves_structural(&eng, reduced, &leaves).unwrap();
    assert_canonical(&from_product);
    assert_eq!(from_product.node_count(), from_reduced.node_count());
    assert_eq!(from_product.pair_count(), from_reduced.pair_count());
    assert_eq!(from_product.model_count().unwrap(), from_reduced.model_count().unwrap());
}
