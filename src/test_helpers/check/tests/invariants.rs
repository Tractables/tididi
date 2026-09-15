//! Unit tests for diagram structural invariants.
//!
//! Checker functions (`validate_vtree_structure`, `check_canonicity`, etc.) are
//! defined in `super` (reachable from integration tests too).
//! This file contains small-formula unit tests that exercise those checkers by
//! building fixtures directly from `Tdd::clause` / `apply_and`.

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};

use crate::apply::apply_and;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;
use super::*;
use super::projective::check_canonicity_projective;
use crate::test_helpers::check::structure::check_no_false_nodes_in_levels;
use crate::test_helpers::check_minimize_soundness;

use crate::test_helpers::{compile_clauses, test_cases, vtree_shapes};


// ── Local test helpers ───────────────────────────────────────────────────────

const CANONICITY_ROUNDS: u32 = 3;

/// Every check there is, including minimize soundness, which mutates `tdd` by
/// one minimize.
fn check_all_deep(tdd: &mut Tdd, label: &str) {
    check_all_fast(tdd, label);
    require(label, "minimize_soundness", check_minimize_soundness(tdd, 3));
}

// ==================== Fixtures every checker accepts ====================
//
// `constant_one` and a single-clause diagram are the two shapes a builder hands
// back before any conjunction has happened. Each one goes past every checker
// the crate has, at every variable count and vtree shape, in one loop — a
// checker that failed names itself, so the report is as precise as a test per
// checker would be.

#[test]
fn every_checker_accepts_constant_one() {
    let eng = &crate::engine::Engine::new();
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let what = format!("constant_one({num_vars} vars, {name})");
            let tdd = constant_one(eng, &vtree);
            check_all_fast(&tdd, &what);
            check_determinism(&tdd).unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }
}

#[test]
fn every_checker_accepts_a_clause_diagram() {
    let eng = &crate::engine::Engine::new();
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (4, vec![-1, -2, -3, -4]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = crate::test_helpers::literals(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let what = format!("clause {lits:?} ({name})");
            let tdd = clause_to_tdd(eng, &vtree, &clause);
            check_all_fast(&tdd, &what);
        }
    }
}

/// The determinism checker over the same clause diagrams. Its own case list:
/// the all-negative clause the other checkers take is not one of its cases.
#[test]
fn test_determinism_clause_tdd() {
    let eng = &crate::engine::Engine::new();
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (2, vec![1]),
    ];
    for (num_vars, literals) in &clauses {
        let clause = crate::test_helpers::literals(literals);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(eng, &vtree, &clause);
            check_determinism(&tdd).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", literals, name, e)
            });
        }
    }
}

// ==================== A raw product, before minimize ====================

#[test]
fn test_structure_after_apply() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[1, 2]));
    let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-3, 4]));
    let product = apply_and(f, g);
    validate_vtree_structure(&product).unwrap_or_else(|e| panic!("after apply: {}", e));
}

#[test]
fn test_determinism_after_apply() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[1, 2]));
    let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-3, 4]));
    // Product before minimize is not necessarily deterministic (it has width left_width*right_width),
    // but after minimize it should be.
    let mut minimized = apply_and(f, g);
    minimized.minimize().unwrap();
    check_determinism(&minimized).unwrap_or_else(|e| panic!("after apply+minimize: {}", e));
}

#[test]
fn test_determinism_after_minimize() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[1, 2]));
    let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-1, 3]));
    let c3 = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-2, -3, 4]));
    let mut tdd = apply_and(f, g);
    tdd.minimize().unwrap();
    tdd = apply_and(tdd, c3);
    tdd.minimize().unwrap();
    check_determinism(&tdd).unwrap_or_else(|e| panic!("after multi-apply+minimize: {}", e));
}

// ==================== Projective (ray) canonicity ====================

/// A vanilla Boolean diagram has no marginal levels: val is 0/1-valued, so
/// proportional is the same relation as equal and the projective (ray) classes
/// coincide with the exact Inv-3 classes. `check_canonicity_projective` degrades
/// to `check_canonicity`, and both pass.
#[test]
fn test_projective_boolean_ray_equals_exact() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[1, 2]));
    let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-1, 3]));
    let c3 = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(&[-2, -3, 4]));
    let mut tdd = apply_and(f, g);
    tdd.minimize().unwrap();
    tdd = apply_and(tdd, c3);
    tdd.minimize().unwrap();

    check_canonicity(&tdd, CANONICITY_ROUNDS).expect("Boolean exact canonicity");
    check_canonicity_projective(&tdd, CANONICITY_ROUNDS)
        .expect("Boolean projective canonicity == Inv 3");
}

/// Two proportional-but-unequal marginal nodes (scalar counts 2 and 3) share one
/// ray class but two exact classes: `check_canonicity` passes (distinct
/// signatures) while `check_canonicity_projective` errs (proportional nodes).
/// Hand-built at the level layer — this crate cannot depend on the compile
/// pipeline that produces marginal diagrams, and this shape is exactly a
/// marginalized boundary level.
#[test]
fn test_projective_marginal_ray_below_exact() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut levels = take_levels(eng, vtree.num_nodes());
    // Root (an internal vtree node) marginalized to two scalar count-nodes.
    levels[root.idx()].set_counts_state(vec![2, 3], None);
    let tdd = Tdd::from_levels_unchecked(
        Arc::clone(&vtree),
        levels,
        TddNodeId { vtree: root, local: NodeIdx(0) },
    );

    // Exact Inv-3 holds (2 is not 3); projective Inv-3 does not.
    check_canonicity(&tdd, CANONICITY_ROUNDS)
        .expect("distinct counts pass exact canonicity");
    let err = check_canonicity_projective(&tdd, CANONICITY_ROUNDS)
        .expect_err("proportional marginal nodes must fail projective canonicity");
    assert!(err.contains("ray-equivalent"), "unexpected error: {err}");
}

/// An unreachable node must be excluded from the ray classification. Mid- and
/// post-compile levels accumulate such orphans; classifying them would report a
/// collision that no live node has. Here node B at level 3 is an exact copy of
/// the live node A and is referenced by nothing, so the level is projectively
/// canonical despite holding two identical nodes.
#[test]
fn test_ray_classification_excludes_unreachable_node() {
    let eng = &crate::engine::Engine::new();
    // balanced(3): leaves 0/1/2; level 3 = parent of leaves 0,1; level 4 = root (3,2).
    let vtree = Arc::new(Vtree::balanced(3));
    let pos = NodeIdx(LeafLabel::Pos as u32);

    let mut levels = take_levels(eng, vtree.num_nodes());
    // A (index 0): live — the root will reference it.
    let a = levels[3].push_internal_node(&[ChildPair::new(pos, pos)]);
    // B (index 1): the same node again, referenced by no parent. It carries a
    // nonzero signature, so only reachability can exclude it.
    let _b = levels[3].push_internal_node(&[ChildPair::new(pos, pos)]);
    // Root references only A (plus a literal on leaf 2).
    let root = levels[4].push_internal_node(&[ChildPair::new(a, pos)]);

    let tdd = Tdd::from_levels_unchecked(
        Arc::clone(&vtree),
        levels,
        TddNodeId { vtree: VtreeIdx(4), local: root },
    );

    check_canonicity_projective(&tdd, CANONICITY_ROUNDS)
        .expect("the orphan copy of A is not a live ray collision");
}

// ==================== Minimize soundness ====================

#[test]
fn test_minimize_soundness_single_clause() {
    let eng = &crate::engine::Engine::new();
    // Clause diagrams are inherently deterministic (width 2, exclusive c/d nodes)
    for num_vars in 2..=5 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let literals: Vec<i32> = (1..=num_vars as i32).collect();
            let clause = crate::test_helpers::literals(&literals);
            let mut tdd = clause_to_tdd(eng, &vtree, &clause);
            check_minimize_soundness(&mut tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", literals, name, e)
            });
        }
    }
}

#[test]
fn test_minimize_soundness_raw_product() {
    let eng = &crate::engine::Engine::new();
    // Product of deterministic diagrams is deterministic, so semiring eval is sound
    let formulas: Vec<(u32, Vec<Vec<i32>>)> = vec![
        (3, vec![vec![1, 2], vec![-2, 3], vec![-1, -3]]),
        // Parity — caught a mod_mul bug (PRIME-1 mask cleared bit 0)
        (3, vec![vec![1, 2, 3], vec![-1, -2, 3], vec![-1, 2, -3], vec![1, -2, -3]]),
        (4, vec![vec![1, 2], vec![3, 4], vec![-1, -3], vec![-2, -4]]),
        (2, vec![vec![1], vec![-1]]),
    ];
    for (num_vars, clauses) in &formulas {
        for (name, vtree) in vtree_shapes(*num_vars) {
            let mut tdd = constant_one(eng, &vtree);
            for literals in clauses {
                let c_tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(literals));
                tdd = apply_and(tdd, c_tdd);
            }
            check_minimize_soundness(&mut tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("raw product {:?} ({}): {}", clauses, name, e)
            });
        }
    }
}

// ==================== No false nodes, before minimize ====================
//
// No real node computes the constant-false function. The false function is
// represented exclusively by the ZERO sentinel (NodeIdx(u32::MAX)), which never
// appears in any level's nodes Vec. The checks above take the invariant on
// finished diagrams; these two take it on raw products.

#[test]
fn test_no_false_nodes_after_apply_before_minimize() {
    let eng = &crate::engine::Engine::new();
    // The invariant holds for apply_and output even before minimize:
    // apply_and's compacting construction never creates false nodes in levels.
    let vtree = Arc::new(Vtree::balanced(4));
    let cases: Vec<(Vec<i32>, Vec<i32>)> = vec![
        (vec![1, 2], vec![-3, 4]),
        (vec![1], vec![-1]),              // contradictory → ZERO output
        (vec![1, 2, 3, 4], vec![-1, -2]), // overlapping
        (vec![1], vec![2]),               // disjoint
    ];
    for (lits1, lits2) in &cases {
        let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(lits1));
        let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(lits2));
        let product = apply_and(f, g);
        // No minimize! Check the raw product has no false nodes in levels.
        check_no_false_nodes_in_levels(&product).unwrap_or_else(|e| {
            panic!("apply_and({:?}, {:?}) before minimize: {}", lits1, lits2, e)
        });
    }
}

#[test]
fn test_no_false_nodes_multi_apply_before_minimize() {
    let eng = &crate::engine::Engine::new();
    // Chain of apply_and calls without intermediate minimize.
    let vtree = Arc::new(Vtree::balanced(5));
    let clauses: Vec<Vec<i32>> = vec![
        vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5],
    ];
    let mut acc = constant_one(eng, &vtree);
    for literals in &clauses {
        let c = clause_to_tdd(eng, &vtree, &crate::test_helpers::literals(literals));
        acc = apply_and(acc, c);
        // Check after each conjunction, before any minimize
        check_no_false_nodes_in_levels(&acc).unwrap_or_else(|e| {
            panic!("after conjoining {:?} (no minimize): {}", literals, e)
        });
    }
}

/// A leaf label stored in an internal level is rejected by the structural check.
#[test]
fn test_validate_vtree_structure_internal_has_leaf_node() {
    use crate::diagram::{LeafLabel, NodeIdx, Tdd, TddLevel, EncodedNode, TddNodeId};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    levels[vtree.root().idx()].nodes.push(EncodedNode::leaf(LeafLabel::One));
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: NodeIdx(0) },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "internal level holds a non-internal node");
    assert!(result.unwrap_err().contains("non-Internal"));
}

// ── Structural checks on diagrams no builder would hand back ────────────────
//
// These reach past the checked construction door on purpose: each names one
// level shape a diagram must never have, and asks the runtime check to say so.
// They live here rather than beside the checks' other callers because the
// unchecked constructor is the crate's own.

#[test]
fn test_validate_vtree_structure_output_vtree_mismatch() {
    use crate::diagram::{Tdd, TddNodeId, NodeIdx, TddLevel};
    use crate::vtree::VtreeIdx;

    let vtree = Arc::new(Vtree::balanced(2));
    // Create a TDD with output at wrong vtree node (not root)
    let levels = vec![TddLevel::new(); vtree.num_nodes()];
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId {
            vtree: VtreeIdx(0), // not the root
            local: NodeIdx(0),
        },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "should fail: output vtree != root");
    assert!(result.unwrap_err().contains("output vtree"));
}

#[test]
fn test_validate_vtree_structure_output_local_oob() {
    use crate::diagram::{Tdd, TddNodeId, NodeIdx, TddLevel};

    let vtree = Arc::new(Vtree::balanced(2));
    let levels = vec![TddLevel::new(); vtree.num_nodes()];
    // Output local index beyond root level width (which is 0)
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId {
            vtree: vtree.root(),
            local: NodeIdx(5), // way out of bounds
        },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "should fail: output local index OOB");
    assert!(result.unwrap_err().contains("output local index"));
}

#[test]
fn test_validate_vtree_structure_leaf_has_internal_node() {
    use crate::diagram::{Tdd, TddNodeId, NodeIdx, TddLevel, ChildPair};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let (leaf_idx, _) = vtree.leaf_bottomup().next().unwrap();
    levels[leaf_idx.idx()].push_internal_node(&[ChildPair::new(NodeIdx(0), NodeIdx(0))]);
    let root_idx = vtree.root().idx();
    levels[root_idx].push_internal_node(&[ChildPair::new(NodeIdx(0), NodeIdx(0))]);
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: NodeIdx(0) },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "should fail: leaf has non-empty nodes vec");
    assert!(result.unwrap_err().contains("non-empty nodes vec"));
}

#[test]
fn test_validate_vtree_structure_child_index_oob() {
    use crate::diagram::{Tdd, TddNodeId, NodeIdx, TddLevel, ChildPair};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let root_idx = vtree.root().idx();
    levels[root_idx].push_internal_node(&[ChildPair::new(NodeIdx(99), NodeIdx(0))]);
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: NodeIdx(0) },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "should fail: left child index OOB");
    assert!(result.unwrap_err().contains("left index"));
}

#[test]
fn test_validate_vtree_structure_right_child_oob() {
    use crate::diagram::{Tdd, TddNodeId, NodeIdx, TddLevel, ChildPair};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let root_idx = vtree.root().idx();
    levels[root_idx].push_internal_node(&[ChildPair::new(NodeIdx(0), NodeIdx(99))]);
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: NodeIdx(0) },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "should fail: right child index OOB");
    assert!(result.unwrap_err().contains("right index"));
}

#[test]
fn test_check_no_false_nodes_empty_internal() {
    use crate::diagram::{Tdd, TddNodeId, TddLevel};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    // Put an Internal node with empty inputs — violates invariant
    let root_idx = vtree.root().idx();
    levels[root_idx].push_internal_node(&[]);
    let tdd = Tdd::from_levels_unchecked(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: crate::diagram::ZERO },
    );
    let result = check_no_false_nodes_in_levels(&tdd);
    assert!(result.is_err(), "should fail: empty Internal");
    assert!(result.unwrap_err().contains("empty inputs"));
}

// ── Whole compiled formulas ──────────────────────────────────────────────────
//
// The tests above build their fixtures a node at a time, so each one names the
// exact shape it puts in front of a checker. These run the whole checker set
// over whole formulas instead, where the shape is whatever the fold arrives at.
//
// Operands come from `compile_clauses`, which conjoins the clauses one at a
// time against a fixed vtree. A driver that preprocesses the formula first —
// splitting it into components, grafting them together, choosing a vtree per
// component — reaches these diagrams by other routes; that variety belongs to
// the driver's own tests, not here.

/// Every invariant the crate checks, over every shared case and every vtree
/// shape. `check_all_deep` names the checker that failed, so one loop reports
/// as precisely as one test per checker would.
#[test]
fn every_compiled_diagram_satisfies_every_invariant() {
    for (num_vars, clauses) in test_cases() {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let what = format!("{num_vars} vars, {} clauses, {shape} vtree", clauses.len());
            let mut tdd = compile_clauses(&vtree, &clauses);
            check_all_deep(&mut tdd, &what);
        }
    }
}

/// A formula whose variables are scattered across the vtree leaves the fold
/// with identity pass-through levels between the ones it touches. Nodes left
/// unreachable there are invisible to a checker but not to the next
/// conjunction, which reads them and counts wrong — so the statement is that
/// the count does not depend on the vtree.
#[test]
fn a_scattered_variable_formula_counts_the_same_on_every_vtree() {


    let clauses = vec![
        vec![1, 3],
        vec![-3, 5],
        vec![5, -7],
        vec![-1, 7],
        vec![3, -5, 9],
        vec![-7, 9],
        vec![1, -9],
        vec![-3, -9, 11],
    ];
    let reference = (compile_clauses(&Arc::new(Vtree::balanced(12)), &clauses)).model_count().unwrap();
    for (shape, vtree) in vtree_shapes(12) {
        assert_eq!(
            (compile_clauses(&vtree, &clauses)).model_count().unwrap(),
            reference,
            "{shape} vtree counts a different number than the balanced one",
        );
    }
}

/// The canonicity checker earns its keep by failing: two nodes on one level
/// computing the same function is exactly what it is looking for, so a diagram
/// rebuilt with one node duplicated has to be rejected.
#[test]
fn canonicity_rejects_a_duplicated_node() {
    use crate::diagram::ChildPair;

    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::random(4, 42));
    let tdd = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![-3, 4]]);

    let target = vtree
        .internal_bottomup()
        .map(|(t, _, _)| t)
        .find(|&t| !tdd.level(t).nodes().is_empty())
        .expect("the diagram holds at least one internal node");

    let mut b = Tdd::builder(eng, &vtree);
    for (t, _, _) in vtree.internal_bottomup() {
        b.replace_level(t, tdd.level_view(t)).unwrap();
        if t == target {
            let pairs: Vec<ChildPair> = tdd.level(t).pairs_of_idx(0).to_vec();
            b.push(t, &pairs);
        }
    }
    let corrupted = b.finish(tdd.output()).expect("a duplicated node is still well-formed");

    assert!(
        check_canonicity(&corrupted, CANONICITY_ROUNDS).is_err(),
        "two nodes on one level computing the same function went undetected",
    );
}

/// A node's input pairs are the equivalence classes it accepts, so dropping
/// one narrows the function the diagram holds and the count has to move. The
/// statement guards an operation that rebuilds a node and forgets to put a
/// pair back.
#[test]
fn dropping_an_input_pair_changes_the_model_count() {

    use num_bigint::BigUint;

    let eng = &crate::engine::Engine::new();
    let clauses = vec![
        vec![1, 2],
        vec![-2, 3],
        vec![3, 4],
        vec![-4, 5],
        vec![-1, -5],
        vec![2, -3, 4],
    ];
    for (shape, vtree) in vtree_shapes(5) {
        let tdd = compile_clauses(&vtree, &clauses);
        let original = tdd.model_count().unwrap();
        assert_ne!(original, BigUint::ZERO, "{shape}: this formula is satisfiable");

        // A node with more than one pair: dropping one leaves it non-empty.
        let Some((target_level, target_node)) = vtree.internal_bottomup().find_map(|(t, _, _)| {
            let level = tdd.level(t);
            (0..level.nodes().len())
                .find(|&i| level.pairs_of_idx(i).len() >= 2)
                .map(|i| (t, i))
        }) else {
            continue;
        };

        let mut b = Tdd::builder(eng, &vtree);
        for (t, _, _) in vtree.internal_bottomup() {
            if t != target_level {
                b.replace_level(t, tdd.level_view(t)).unwrap();
                continue;
            }
            for i in 0..tdd.level(t).nodes().len() {
                let mut pairs = tdd.level(t).pairs_of_idx(i).to_vec();
                if i == target_node {
                    pairs.pop();
                }
                b.push(t, &pairs);
            }
        }
        let corrupted = b
            .finish(tdd.output())
            .expect("dropping one pair of a multi-pair node leaves it non-empty");

        assert_ne!(
            original,
            corrupted.model_count().unwrap(),
            "{shape}: dropping an input pair left the model count where it was",
        );
    }
}
