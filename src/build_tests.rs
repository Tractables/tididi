use std::sync::Arc;

use super::*;
use crate::reduce::minimize;
use crate::query::model_count;
use crate::diagram::ZERO;
use crate::vtree::{Vtree, VtreeIdx};

// ── Raw clause TDD construction (test-only, pre-implicit-leaves) ─────────
//
// The old clause_to_tdd_raw function built explicit leaf nodes including
// Leaf(Zero). With implicit leaf representation, this is no longer valid.
// Tests that validated the raw/unpruned structure have been removed.
// The public clause_to_tdd (with prune) is the only construction path.

// Retained for test helpers below (c_t=0 at relevant, d_t=1 at relevant).
#[allow(dead_code)]
const C: LocalNodeIdx = LocalNodeIdx(0);
#[allow(dead_code)]
const D: LocalNodeIdx = LocalNodeIdx(1);

// ── Removed: clause_to_tdd_raw and helpers ──────────────────────────────
// build_c_pairs, build_d_inputs, clause_to_tdd_raw removed — they created
// explicit leaf nodes (including Zero) which no longer exist.



#[test]
fn test_constant_one() {
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(&vtree);

    // Internal levels have width 1 (one node with pair (One, One))
    for (t, _left, _right) in vtree.internal_bottomup() {
        assert_eq!(tdd.level(t).width(), 1, "Internal width at {:?}", t);
    }
    // Leaf levels are marginal (empty nodes vec, virtual width 3)
    for (t, _var) in vtree.leaf_bottomup() {
        assert_eq!(tdd.level(t).width(), 0, "Leaf level should have no stored nodes");
        assert_eq!(tdd.effective_width(t), LEAF_WIDTH, "Leaf effective width should be LEAF_WIDTH");
    }
    // Output: for multi-var vtrees, root is internal → output at index 0.
    // For single-var vtrees, root is leaf → output at implicit One index 0.
    assert_eq!(tdd.output.local, LocalNodeIdx(0));
}

// test_clause_tdd_width and test_clause_tdd_leaf_labels removed:
// they tested the old raw (unpruned) representation with explicit leaf nodes.
// With implicit leaves, clause_to_tdd is the only construction path.

// ==================== Clause TDD structural invariants ====================

/// Helper: build a Clause from DIMACS-style signed integers (1-indexed).
/// All vtree shapes for a given variable count.
fn vtree_shapes(num_vars: u32) -> Vec<(&'static str, Arc<Vtree>)> {
    vec![
        ("balanced", Arc::new(Vtree::balanced(num_vars))),
        ("linear", Arc::new(Vtree::linear(num_vars))),
        ("random", Arc::new(Vtree::random(num_vars, 42))),
    ]
}


/// Validate that a TDD has no dead input pairs: no pair references a child
/// node that computes the constant-false function.
///
/// With implicit leaves, leaf children are never false (Pos/Neg/One are all non-zero).
/// Only internal children can be false (empty pairs).
fn validate_no_dead_pairs(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    for (t, left, right) in vtree.internal_bottomup() {
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();
        let level = tdd.level(t);

        for (i, node) in level.nodes.iter().enumerate() {
            let pairs = level.pairs_of(node);
            for (j, pair) in pairs.iter().enumerate() {
                // Leaf children are never false (implicit Pos/Neg/One).
                if !left_is_leaf {
                    let left_node = &tdd.level(left).nodes[pair.left.idx()];
                    if left_node.is_internal() && tdd.level(left).pairs_of(left_node).is_empty() {
                        return Err(format!(
                            "vtree {:?} node {} input {}: left={:?} references a zero child",
                            t, i, j, pair.left
                        ));
                    }
                }
                if !right_is_leaf {
                    let right_node = &tdd.level(right).nodes[pair.right.idx()];
                    if right_node.is_internal() && tdd.level(right).pairs_of(right_node).is_empty() {
                        return Err(format!(
                            "vtree {:?} node {} input {}: right={:?} references a zero child",
                            t, i, j, pair.right
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate no duplicate nodes exist at any internal level.
/// Leaf levels are marginal (no stored nodes) — duplicates impossible.
fn validate_no_duplicate_nodes(tdd: &Tdd) -> Result<(), String> {
    for (t, _left, _right) in tdd.vtree.internal_bottomup() {
        let level = tdd.level(t);
        for i in 0..level.nodes.len() {
            for j in (i + 1)..level.nodes.len() {
                if level.pairs_of_idx(i) == level.pairs_of_idx(j) {
                    return Err(format!(
                        "vtree {:?}: nodes {} and {} are identical — {:?}",
                        t, i, j, level.nodes[i]
                    ));
                }
            }
        }
    }
    Ok(())
}

// validate_clause_tdd_raw, validate_input_pair_counts, test_clause_tdd_no_dead_pairs,
// test_clause_tdd_canonical_no_duplicates removed — they tested the old raw
// (unpruned) representation with explicit leaf nodes.

#[test]
fn test_clause_tdd_minimize_preserves_function() {
    // Minimize changes clause TDD structure (prune removes the unreachable d_root
    // at the root, and for sparse clauses cascades further), but must preserve
    // the Boolean function. Verify via model count.
    let cases: Vec<(u32, Vec<i32>)> = vec![
        (2, vec![1]),
        (3, vec![1, 2]),
        (3, vec![1, 2, 3]),
        (4, vec![1, -3]),
        (4, vec![-1, -2, -3, -4]),
        (6, vec![1]),
        (6, vec![3, 5]),
        (8, vec![1, 4, 8]),
    ];
    for (num_vars, lits) in &cases {
        let clause = crate::test_helpers::lits(lits);
        for (shape_name, vtree) in vtree_shapes(*num_vars) {
            let tdd_before = clause_to_tdd(&vtree, &clause);
            let count_before = model_count(&tdd_before);

            let mut tdd_after = tdd_before.clone();
            minimize(&mut tdd_after);
            let count_after = model_count(&tdd_after);

            assert_eq!(
                count_before, count_after,
                "clause {:?} ({} vars, {}): model count changed by minimize ({} → {})",
                lits, num_vars, shape_name, count_before, count_after
            );

            // After minimize, all our structural invariants should still hold
            // (on the minimized TDD, which may have fewer nodes).
            validate_no_dead_pairs(&tdd_after).unwrap_or_else(|e| {
                panic!("clause {:?} ({} vars, {}) post-minimize: {}", lits, num_vars, shape_name, e)
            });
            validate_no_duplicate_nodes(&tdd_after).unwrap_or_else(|e| {
                panic!("clause {:?} ({} vars, {}) post-minimize: {}", lits, num_vars, shape_name, e)
            });
        }
    }
}

/// Check that every node in a TDD is reachable from the output.
fn validate_all_nodes_reachable(tdd: &Tdd) -> Result<(), String> {
    // ZERO sentinel: UNSAT TDD has no real nodes to check.
    if tdd.output.local == ZERO {
        return Ok(());
    }

    let vtree = &tdd.vtree;
    let num_nodes = vtree.num_nodes();

    // Track reachability per (vtree_level, local_index)
    let mut reachable: Vec<Vec<bool>> = (0..num_nodes)
        .map(|i| vec![false; tdd.effective_width(VtreeIdx(i as u32))])
        .collect();

    // Mark output
    reachable[tdd.output.vtree.idx()][tdd.output.local.idx()] = true;

    // Top-down propagation
    for t_idx in (0..num_nodes).rev() {
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(VtreeIdx(t_idx as u32));
        for i in 0..tdd.levels[t_idx].width() {
            if !reachable[t_idx][i] {
                continue;
            }
            let pairs = tdd.levels[t_idx].pairs_of_idx(i);
            for pair in pairs {
                reachable[left.idx()][pair.left.idx()] = true;
                reachable[right.idx()][pair.right.idx()] = true;
            }
        }
    }

    // Verify all stored nodes are reachable (skip leaf levels — marginal nodes are always "reachable")
    for t_idx in 0..num_nodes {
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() { continue; }
        for i in 0..tdd.levels[t_idx].width() {
            if !reachable[t_idx][i] {
                return Err(format!(
                    "vtree {:?} node {}: unreachable from output",
                    VtreeIdx(t_idx as u32), i
                ));
            }
        }
    }
    Ok(())
}

/// Validate that no node computes the zero (false) function:
/// - No leaf with LeafLabel::Zero
/// - No internal node with empty inputs
fn validate_no_zero_nodes(tdd: &Tdd) -> Result<(), String> {
    for t in tdd.vtree.bottomup() {
        let level = tdd.level(t);
        for (i, node) in level.nodes.iter().enumerate() {
            if node.is_leaf() && node.leaf_label() == LeafLabel::Zero {
                return Err(format!(
                    "vtree {:?} node {}: leaf computes Zero (false)",
                    t, i
                ));
            } else if node.is_internal() && level.pairs_of(node).is_empty() {
                return Err(format!(
                    "vtree {:?} node {}: internal node has empty inputs (computes false)",
                    t, i
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn test_clause_to_tdd_is_minimal() {
    // clause_to_tdd should return a minimal, canonical TDD with:
    // - no unreachable nodes
    // - no dead pairs
    // - no duplicate nodes
    // - no zero (false-function) nodes
    let cases: Vec<(u32, Vec<i32>)> = vec![
        (2, vec![1]),
        (3, vec![1, 2]),
        (3, vec![1, 2, 3]),
        (4, vec![1, -3]),
        (4, vec![-1, -2, -3, -4]),
        (6, vec![1]),
        (6, vec![3, 5]),
        (8, vec![1, 4, 8]),
    ];
    for (num_vars, lits) in &cases {
        let clause = crate::test_helpers::lits(lits);
        for (shape_name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            let label = format!("clause {:?} ({} vars, {})", lits, num_vars, shape_name);

            validate_all_nodes_reachable(&tdd)
                .unwrap_or_else(|e| panic!("{}: {}", label, e));
            validate_no_dead_pairs(&tdd)
                .unwrap_or_else(|e| panic!("{}: {}", label, e));
            validate_no_duplicate_nodes(&tdd)
                .unwrap_or_else(|e| panic!("{}: {}", label, e));
            validate_no_zero_nodes(&tdd)
                .unwrap_or_else(|e| panic!("{}: {}", label, e));
        }
    }
}
