//! The reduction-size metrics.
//!
//! Sibling of `query_tests.rs`, which holds the fixtures these read.

use super::*;
use crate::apply::conjoin::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};
use std::sync::Arc;


#[test]
fn test_reduced_size_constant_one_all_reducible() {
    let eng = &crate::engine::Engine::new();
    // constant_one TDD: every internal node has E = {(one_{t1}, one_{t2})} → all reducible.
    // A 3-var balanced vtree has 2 internal vtree nodes.
    // The root has 1 internal tdd-node with 1 pair; its child also has 1 with 1 pair.
    // Both are reducible → reduced_size == 0.
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    assert_eq!(reduced_size(&tdd, ReductionRule::R1Sdd), 0);
}

#[test]
fn test_reduced_size_irrelevant_variable() {
    let eng = &crate::engine::Engine::new();
    // Clause (x0) over a 2-var balanced vtree with x0 and x1.
    // After building and minimizing, the root node has E = {(c_{x0}, one_{t2})} →
    // exactly 1 reducible node. reduced_size == tdd.size() - 1.
    let vtree = Arc::new(Vtree::balanced(2));
    let clause = vec![Literal::pos(VarId(0))];
    let mut tdd = clause_to_tdd(eng, &vtree, &clause);
    minimize(&mut tdd);
    let size = tdd.size();
    assert!(size >= 1);
    assert_eq!(reduced_size(&tdd, ReductionRule::R1Sdd), size - 1);
}

#[test]
fn test_reduced_size_no_reducible_nodes() {
    let eng = &crate::engine::Engine::new();
    // Pigeonhole PHP(2,1): x0 ∧ x1. Both variables are relevant on both sides
    // at the root, so no node has an unconstrained child subtree.
    let vtree = Arc::new(Vtree::balanced(2));
    let t0 = clause_to_tdd(eng, &vtree, &[Literal::pos(VarId(0))]);
    let t1 = clause_to_tdd(eng, &vtree, &[Literal::pos(VarId(1))]);
    let mut tdd = apply_and(t0, t1);
    minimize(&mut tdd);
    assert_eq!(reduced_size(&tdd, ReductionRule::R1Sdd), tdd.size());
}

#[test]
fn test_reduced_size_multi_pair_generalisation() {
    let eng = &crate::engine::Engine::new();
    // apply_and of (x0 ∨ x1) and (¬x0 ∨ x1) on a linear(3) vtree.
    //
    // After minimize, twin contraction merges Pos_x0 and Neg_x0 into One_x0 at
    // the leaf level. The root node ends up with a single pair {(One_x0, c_t1)}
    // where One_x0 has model count 2 = 2^|vars(x0)|.
    let vtree = Arc::new(Vtree::linear(3));
    let f = vec![Literal::pos(VarId(0)), Literal::pos(VarId(1))];
    let g = vec![Literal::neg(VarId(0)), Literal::pos(VarId(1))];
    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut tdd = apply_and(t1, t2);
    minimize(&mut tdd);
    // The root node is reducible: reduced size must be strictly smaller.
    assert!(reduced_size(&tdd, ReductionRule::R1Sdd) < tdd.size(),
        "expected reduction: size={}, rtdd={}",
        tdd.size(), reduced_size(&tdd, ReductionRule::R1Sdd));
}
