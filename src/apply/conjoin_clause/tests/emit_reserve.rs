use super::*;
use crate::engine::Engine;

use crate::test_helpers::{assert_canonical, brute_force_count};
use crate::vtree::Vtree;
use num_bigint::BigUint;

/// Fold a formula into a diagram over `vtree`, one clause at a time through
/// `apply_and_clause` — the rebuild path under test.
fn fold_cnf(_eng: &Engine, vtree: &Arc<Vtree>, cnf: &[Vec<i32>]) -> Tdd {
    let mut acc = Tdd::one(vtree);
    for clause in cnf {
        let literals: Vec<Literal> = clause.iter().map(|&l| l.try_into().unwrap()).collect();
        acc = acc.and_clause(&literals).unwrap();
    }
    acc
}

/// The canonical form of a fold result, which the rebuild leaves un-minimized.
fn assert_fold_is_valid(acc: &Tdd) {
    let mut canonical = acc.clone();
    canonical.minimize().unwrap();
    assert_canonical(&canonical);
}

/// Per-level emit sizing is demand-driven, so a rebuild whose real output
/// lands far below the four-fold worst-case bound must still emit every pair.
/// The three trailing unit clauses are the collapse: each rebuilt level
/// keeps at most one pair per surviving node where the bound allows four, so
/// the level grows past the initial slab on the early clauses and stays far
/// under it on the late ones — both regimes of the top-up.
#[test]
fn clause_rebuild_exact_when_output_far_below_worst_case() {
    let eng = Engine::new();
    let n: u32 = 7;
    let cnf: Vec<Vec<i32>> = vec![
        vec![1, -2, 3],
        vec![-3, 4],
        vec![2, 5, -6],
        vec![4, -5, 7],
        vec![-1, 6, -7],
        vec![3, -4, 5, -6],
        vec![1],
        vec![-2],
        vec![7],
    ];
    let acc = fold_cnf(&eng, &Arc::new(Vtree::random(n, 7)), &cnf);
    let expected = BigUint::from(brute_force_count(n, &cnf));
    assert!(expected > BigUint::from(0u32), "fixture must stay satisfiable");
    assert_eq!(acc.model_count().unwrap(), expected);
    assert_fold_is_valid(&acc);
}

/// The same conjunction, one clause at a time against a wide accumulator
/// whose levels are rebuilt with both children on the spine (the
/// three-pairs-per-input-pair case) and with the complement lane live — the
/// widest per-node top-up. Counts must match the oracle exactly.
#[test]
fn both_relevant_rebuild_exact_under_demand_reserve() {
    let eng = Engine::new();
    let n: u32 = 6;
    // Every clause spans variables from both halves of the vtree, so the
    // meet levels take the both-relevant path.
    let cnf: Vec<Vec<i32>> = vec![
        vec![1, 4],
        vec![-1, 5],
        vec![2, -4, 6],
        vec![-2, -5, 3],
        vec![3, -6],
        vec![-3, 4, -5],
    ];
    let acc = fold_cnf(&eng, &Arc::new(Vtree::random(n, 3)), &cnf);
    assert_eq!(acc.model_count().unwrap(), BigUint::from(brute_force_count(n, &cnf)));
    assert_fold_is_valid(&acc);
}

#[test]
fn a_clause_reuses_the_single_node_input_arena() {
    let tree = Arc::new(Vtree::balanced(8));
    let input = Tdd::one(&tree);
    assert_canonical(&input);
    let nodes = input.level(tree.root()).nodes().as_ptr();
    let result = Engine::new().and_clause(input, [crate::Literal::try_from(1).unwrap()]).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 128u32.into());
    assert_eq!(result.level(tree.root()).nodes().as_ptr(), nodes);
}

#[test]
fn a_clause_reuses_empty_input_pair_capacity() {
    let tree = Arc::new(Vtree::balanced(8));
    let eng = Engine::new();
    let one = Tdd::one(&tree);
    assert_canonical(&one);
    let input = eng.and_clause(one, [crate::Literal::try_from(1).unwrap()]).unwrap();
    assert_canonical(&input);
    let level = input.level(tree.root());
    assert!(level.pairs.is_empty());
    assert!(level.pairs.capacity() > 0);
    let pairs = level.pairs.as_ptr();
    let result = eng.and_clause(input, [crate::Literal::try_from(2).unwrap()]).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 64u32.into());
    assert_eq!(result.level(tree.root()).pairs.as_ptr(), pairs);
}
