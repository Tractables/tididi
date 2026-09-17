
use crate::diagram::ChildDecoder;
use std::sync::Arc;

use super::*;

use crate::apply::conjoin_clause::conjoin_clause_owned;
use crate::test_helpers::clause_to_tdd;


use crate::diagram::ZERO;
use crate::test_helpers::{assert_canonical, literals, vtree_shapes};
use crate::vtree::{Vtree, VtreeIdx};

#[test]
fn test_constant_one() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);

    // Internal levels have width 1 (one node with pair (One, One))
    for (t, _left, _right) in vtree.internal_bottomup() {
        assert_eq!(tdd.level(t).slot_count(), 1, "Internal width at {:?}", t);
    }
    // Leaf levels are marginal (empty nodes vec, virtual width 3)
    for (t, _var) in vtree.leaf_bottomup() {
        assert_eq!(tdd.level(t).slot_count(), 0, "Leaf level should have no stored nodes");
        assert_eq!(tdd.reference_slot_count(t), LEAF_WIDTH, "Leaf effective width should be LEAF_WIDTH");
    }
    // Output: for multi-var vtrees, root is internal → output at index 0.
    // For single-var vtrees, root is leaf → output at implicit One index 0.
    assert_eq!(tdd.output.local, NodeIdx(0));
}

// ==================== Clause diagram structural invariants ====================

#[test]
fn test_clause_tdd_minimize_preserves_function() {
    let eng = &crate::Engine::new();
    // Minimize changes clause diagram structure (prune removes the unreachable d_root
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
        let clause = literals(lits);
        for (shape_name, vtree) in vtree_shapes(*num_vars) {
            let tdd_before = clause_to_tdd(eng, &vtree, &clause);
            let count_before = tdd_before.model_count().unwrap();

            let mut tdd_after = tdd_before.clone();
            tdd_after.minimize().unwrap();
            let count_after = tdd_after.model_count().unwrap();

            assert_eq!(
                count_before, count_after,
                "clause {:?} ({} vars, {}): model count changed by ({} → {}).minimize().unwrap()",
                lits, num_vars, shape_name, count_before, count_after
            );

            // After minimize, every invariant the crate checks still holds
            // (on the minimized diagram, which may have fewer nodes).
            assert_canonical(&tdd_after);
        }
    }
}

/// Check that every node in a diagram is reachable from the output.
fn validate_all_nodes_reachable(tdd: &Tdd) -> Result<(), String> {
    // ZERO sentinel: UNSAT diagram has no real nodes to check.
    if tdd.output.local == ZERO {
        return Ok(());
    }

    let vtree = &tdd.vtree;
    let num_nodes = vtree.num_nodes();

    // Track reachability per (vtree_level, local_index)
    let mut reachable: Vec<Vec<bool>> = (0..num_nodes)
        .map(|i| vec![false; tdd.reference_slot_count(VtreeIdx(i as u32))])
        .collect();

    // Mark output
    reachable[tdd.output.vtree.idx()][tdd.output.local.idx()] = true;

    // Top-down propagation
    for t_idx in (0..num_nodes).rev() {
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(VtreeIdx(t_idx as u32));
        for i in 0..tdd.levels[t_idx].slot_count() {
            if !reachable[t_idx][i] {
                continue;
            }
            let pairs = tdd.levels[t_idx].pairs_of_idx(i);
            for pair in pairs {
                reachable[left.idx()][ChildDecoder::structural().node(pair.left).idx()] = true;
                reachable[right.idx()][ChildDecoder::structural().node(pair.right).idx()] = true;
            }
        }
    }

    // Verify all stored nodes are reachable (skip leaf levels — marginal nodes are always "reachable")
    // Indexes the vtree, `tdd.levels` and `reachable` at the same position.
    #[allow(clippy::needless_range_loop)]
    for t_idx in 0..num_nodes {
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() { continue; }
        for i in 0..tdd.levels[t_idx].slot_count() {
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

#[test]
fn test_clause_to_tdd_is_minimal() {
    let eng = &crate::Engine::new();
    // `clause_to_tdd` should return a minimal, canonical diagram with no
    // unreachable nodes.
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
        let clause = literals(lits);
        for (shape_name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(eng, &vtree, &clause);
            let label = format!("clause {:?} ({} vars, {})", lits, num_vars, shape_name);

            assert_canonical(&tdd);
            validate_all_nodes_reachable(&tdd)
                .unwrap_or_else(|e| panic!("{}: {}", label, e));
        }
    }
}

/// Every clause shape the tests above cover, built directly and by conjoining
/// the clause into the constant-true diagram, stores the same nodes and pairs
/// in the same order at every level.
#[test]
fn a_clause_builds_the_levels_its_conjunction_into_one_emits() {
    let eng = &Engine::new();
    let cases: Vec<(u32, Vec<i32>)> = vec![
        (1, vec![1]),
        (1, vec![-1]),
        (2, vec![1]),
        (3, vec![1, 2]),
        (3, vec![1, 2, 3]),
        (3, vec![1, -1]),
        (3, vec![-1, 1, 2]),
        (3, vec![2, 1, -2]),
        (3, vec![1, 2, 1]),
        (4, vec![1, -3]),
        (4, vec![-1, -2, -3, -4]),
        (6, vec![1]),
        (6, vec![3, 5]),
        (8, vec![1, 4, 8]),
    ];
    for (num_vars, lits) in &cases {
        let clause = literals(lits);
        for (shape_name, vtree) in vtree_shapes(*num_vars) {
            let label = format!("clause {lits:?} ({num_vars} vars, {shape_name})");
            let built = clause_to_tdd(eng, &vtree, &clause);
            let conjoined =
                conjoin_clause_owned(eng, constant_one(eng, &vtree), &clause)
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(built.output, conjoined.output, "{label}: output");
            assert_eq!(built.levels.len(), conjoined.levels.len(), "{label}: level count");
            for (t, (a, b)) in built.levels.iter().zip(&conjoined.levels).enumerate() {
                assert!(
                    a.nodes == b.nodes && a.pairs == b.pairs && a.multi_pairs == b.multi_pairs,
                    "{label}: level {t} stores different nodes or pairs"
                );
                assert!(!a.is_marginal() && !b.is_marginal(), "{label}: level {t} is structural");
                assert_eq!(a.inlined_sides, b.inlined_sides, "{label}: level {t} inline markers");
                assert_eq!(a.n_tombstones, b.n_tombstones, "{label}: level {t} tombstones");
                assert_eq!(a.dead_pairs, b.dead_pairs, "{label}: level {t} dead pairs");
            }
        }
    }
}

/// A clause's literals are a set: a variable in both polarities makes the
/// disjunction true everywhere, and a variable repeated in one polarity builds
/// what the deduplicated literals spell. Both clause entry points answer the
/// same function, over every vtree shape.
#[test]
fn a_clause_reads_its_literals_as_a_set() {
    for (name, vtree) in vtree_shapes(3) {
        let all = (constant_one(&Engine::new(), &vtree)).model_count().unwrap();
        for clause in [vec![1, -1], vec![-1, 1], vec![1, -1, 2], vec![2, 1, -2]] {
            let built = Tdd::clause(&vtree, literals(&clause)).unwrap();
            assert_canonical(&built);
            assert_eq!(built.model_count().unwrap(), all, "{name}: {clause:?} is satisfied everywhere");

            let conjoined =
                (Tdd::clause(&vtree, [3]).unwrap()).and_clause(literals(&clause)).unwrap();
            assert_eq!(
                conjoined.model_count().unwrap(),
                (Tdd::clause(&vtree, [3]).unwrap()).model_count().unwrap(),
                "{name}: conjoining {clause:?} is the identity"
            );
        }
        let repeated = Tdd::clause(&vtree, literals(&[1, 2, 1])).unwrap();
        assert_canonical(&repeated);
        assert_eq!(
            repeated.model_count().unwrap(),
            (Tdd::clause(&vtree, literals(&[1, 2])).unwrap()).model_count().unwrap(),
            "{name}: a repeated literal says nothing new"
        );
    }
}
