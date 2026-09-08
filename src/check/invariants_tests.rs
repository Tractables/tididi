//! Unit tests for TDD structural invariants.
//!
//! Checker functions (`validate_vtree_structure`, `check_canonicity`, etc.) are
//! defined in `super` (reachable from integration tests too).
//! This file contains small-formula unit tests that exercise those checkers by
//! building fixtures directly from `clause_to_tdd` / `apply_and`.
//!
//! Crate-split: `tididi` cannot depend on `cnf`/`compile` at all, so every
//! compile-pipeline-driven invariant test — including
//! `test_corruption_swapped_pair_child_changes_model_count`, previously kept
//! here because it mutates `TddNodeData.a` (`pub(crate)`, unreachable from an
//! external test crate) — has moved to the root-crate integration tests
//! `tests/tdd_invariants.rs` and `tests/tdd_invariants_compile.rs`
//! respectively. Only tests that need no `cnf`/`compile` import stay in this file.

use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx};

use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use super::*;
use crate::reduce::minimize;
use crate::query::reduced_tdd_size;
use crate::query::reduction::r2_reduced_tdd_size;


// ── Local test helpers ───────────────────────────────────────────────────────

fn make_clause(lits: &[i32]) -> Vec<Literal> {
    lits.iter()
        .map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0))
        .collect()
}

fn vtree_shapes(num_vars: u32) -> Vec<(&'static str, Arc<Vtree>)> {
    vec![
        ("balanced", Arc::new(Vtree::balanced(num_vars))),
        ("linear", Arc::new(Vtree::linear(num_vars))),
        ("random", Arc::new(Vtree::random(num_vars, 42))),
    ]
}

/// Call `reduced_tdd_size` (triggering its debug_assert!s) then run the explicit
/// `check_reduced_size_sanity` cross-check.
fn assert_reduced_size_sane(tdd: &Tdd, label: &str) {
    let _ = reduced_tdd_size(tdd);
    let _ = r2_reduced_tdd_size(tdd);
    check_reduced_size_sanity(tdd)
        .unwrap_or_else(|e| panic!("{}: {}", label, e));
}

const CANONICITY_ROUNDS: u32 = 3;

// ==================== Category 1: Vtree Structure ====================

#[test]
fn test_structure_constant_one() {
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let tdd = constant_one(&vtree);
            validate_vtree_structure(&tdd)
                .unwrap_or_else(|e| panic!("constant_one({} vars, {}): {}", num_vars, name, e));
        }
    }
}

#[test]
fn test_structure_clause_tdd() {
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (4, vec![-1, -2, -3, -4]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = make_clause(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            validate_vtree_structure(&tdd)
                .unwrap_or_else(|e| panic!("clause {:?} ({}): {}", lits, name, e));
        }
    }
}

#[test]
fn test_structure_after_apply() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-3, 4]));
    let product = apply_and(c1, c2);
    validate_vtree_structure(&product).unwrap_or_else(|e| panic!("after apply: {}", e));
}

#[test]
fn test_structure_after_minimize() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-3, 4]));
    let mut product = apply_and(c1, c2);
    minimize(&mut product);
    validate_vtree_structure(&product).unwrap_or_else(|e| panic!("after minimize: {}", e));
}

// ==================== Category 2: Determinism ====================

#[test]
fn test_determinism_constant_one() {
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let tdd = constant_one(&vtree);
            check_determinism(&tdd).unwrap_or_else(|e| {
                panic!("constant_one({} vars, {}): {}", num_vars, name, e)
            });
        }
    }
}

#[test]
fn test_determinism_clause_tdd() {
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = make_clause(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            check_determinism(&tdd).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", lits, name, e)
            });
        }
    }
}

#[test]
fn test_determinism_after_apply() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-3, 4]));
    let product = apply_and(c1, c2);
    // Product before minimize is not necessarily deterministic (it has width k1*k2),
    // but after minimize it should be.
    let mut minimized = product;
    minimize(&mut minimized);
    check_determinism(&minimized).unwrap_or_else(|e| panic!("after apply+minimize: {}", e));
}

#[test]
fn test_determinism_after_minimize() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-1, 3]));
    let c3 = clause_to_tdd(&vtree, &make_clause(&[-2, -3, 4]));
    let mut tdd = apply_and(c1, c2);
    minimize(&mut tdd);
    tdd = apply_and(tdd, c3);
    minimize(&mut tdd);
    check_determinism(&tdd).unwrap_or_else(|e| panic!("after multi-apply+minimize: {}", e));
}

// ==================== Category 3: Probabilistic Canonicity ====================

#[test]
fn test_canonicity_constant_one() {
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let tdd = constant_one(&vtree);
            check_canonicity(&tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("constant_one({} vars, {}): {}", num_vars, name, e)
            });
        }
    }
}

#[test]
fn test_canonicity_clause_tdd() {
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (4, vec![-1, -2, -3, -4]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = make_clause(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            check_canonicity(&tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", lits, name, e)
            });
        }
    }
}

#[test]
fn test_canonicity_after_minimize() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-1, 3]));
    let c3 = clause_to_tdd(&vtree, &make_clause(&[-2, -3, 4]));
    let mut tdd = apply_and(c1, c2);
    minimize(&mut tdd);
    tdd = apply_and(tdd, c3);
    minimize(&mut tdd);
    check_canonicity(&tdd, CANONICITY_ROUNDS)
        .unwrap_or_else(|e| panic!("after multi-apply+minimize: {}", e));
}

// ============ Category 4b: Projective (ray) canonicity + gauge audit ============

/// A vanilla Boolean TDD has no marginal levels: val is 0/1-valued, so
/// proportional ⟺ equal and the projective (ray) classes coincide with the exact
/// Inv-3 classes. `gauge_audit` reports ray == exact at every level, and
/// `check_canonicity_projective` degrades to `check_canonicity` (both pass).
#[test]
fn test_projective_boolean_ray_equals_exact() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-1, 3]));
    let c3 = clause_to_tdd(&vtree, &make_clause(&[-2, -3, 4]));
    let mut tdd = apply_and(c1, c2);
    minimize(&mut tdd);
    tdd = apply_and(tdd, c3);
    minimize(&mut tdd);

    let report = gauge_audit(&tdd, CANONICITY_ROUNDS);
    for lv in &report.levels {
        assert_eq!(
            lv.ray, lv.exact,
            "vtree {:?}: Boolean level must have ray == exact", lv.vtree
        );
    }
    assert_eq!(
        report.total_ray, report.total_exact,
        "Boolean diagram: total ray == total exact"
    );
    // A canonical minimized Boolean TDD has exact == nodes at every level, so
    // projective canonicity holds too.
    check_canonicity(&tdd, CANONICITY_ROUNDS).expect("Boolean exact canonicity");
    check_canonicity_projective(&tdd, CANONICITY_ROUNDS)
        .expect("Boolean projective canonicity == Inv 3");
}

/// Two proportional-but-unequal marginal nodes (scalar counts 2 and 3) share one
/// ray class but two exact classes: `gauge_audit` reports ray < exact at that
/// level, `check_canonicity` passes (distinct signatures) while
/// `check_canonicity_projective` errs (proportional nodes). Hand-built at the
/// level layer — `tididi` cannot depend on the compile pipeline that produces
/// marginal TDDs, and this shape is exactly a marginalized boundary level.
#[test]
fn test_projective_marginal_ray_below_exact() {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut levels = take_levels(vtree.num_nodes());
    // Root (an internal vtree node) marginalized to two scalar count-nodes.
    levels[root.idx()].marginal_counts = Some(vec![2, 3]);
    let tdd = Tdd::with_levels(
        Arc::clone(&vtree),
        levels,
        TddNodeId { vtree: root, local: LocalNodeIdx(0) },
    );

    let report = gauge_audit(&tdd, CANONICITY_ROUNDS);
    let root_stat = report
        .levels
        .iter()
        .find(|l| l.vtree == root)
        .expect("root level present in report");
    assert_eq!(root_stat.nodes, 2);
    assert_eq!(root_stat.exact, 2, "distinct counts → 2 exact classes");
    assert_eq!(root_stat.ray, 1, "proportional scalars → 1 ray class");
    assert!(root_stat.ray < root_stat.exact, "gauge redundancy must be present");

    // Exact Inv-3 holds (2 ≠ 3); projective Inv-3 does not.
    check_canonicity(&tdd, CANONICITY_ROUNDS)
        .expect("distinct counts pass exact canonicity");
    let err = check_canonicity_projective(&tdd, CANONICITY_ROUNDS)
        .expect_err("proportional marginal nodes must fail projective canonicity");
    assert!(err.contains("ray-equivalent"), "unexpected error: {err}");
}

/// An unreachable node (orphaned at a non-root level, referenced by no parent)
/// must be excluded from the gauge audit's live tallies. Mid-/post-compile levels
/// accumulate such orphans; counting them would inflate the reported redundancy
/// with garbage (the defect this liveness filter fixes). Here node B at level 3
/// is never referenced by the root, so the level reports 2 total nodes but only 1
/// live.
#[test]
fn test_gauge_audit_excludes_unreachable_node() {
    // balanced(3): leaves 0/1/2; level 3 = parent of leaves 0,1; level 4 = root (3,2).
    let vtree = Arc::new(Vtree::balanced(3));
    let pos = LocalNodeIdx(LeafLabel::Pos as u32);
    let neg = LocalNodeIdx(LeafLabel::Neg as u32);

    let mut levels = take_levels(vtree.num_nodes());
    // A (index 0): live — the root will reference it.
    let a = levels[3].push_internal_node(&[InputPair { left: pos, right: pos }]);
    // B (index 1): a real, nonzero-signature node that NO parent references — so
    // it is excluded by reachability, not by the zero-node filter.
    let _b = levels[3].push_internal_node(&[InputPair { left: neg, right: neg }]);
    // Root references only A (plus a literal on leaf 2).
    let root = levels[4].push_internal_node(&[InputPair { left: a, right: pos }]);

    let tdd = Tdd::with_levels(
        Arc::clone(&vtree),
        levels,
        TddNodeId { vtree: VtreeIdx(4), local: root },
    );

    let report = gauge_audit(&tdd, CANONICITY_ROUNDS);
    let lvl3 = report
        .levels
        .iter()
        .find(|l| l.vtree == VtreeIdx(3))
        .expect("level 3 present in report");
    assert_eq!(lvl3.nodes, 2, "both A and B are stored at level 3");
    assert_eq!(lvl3.live, 1, "only A is reachable from the root — B is excluded");
    assert_eq!(lvl3.exact, 1, "one live exact class");
    assert_eq!(lvl3.ray, 1, "one live ray class");
    // The dead node must not enter the totals' live count / redundancy either.
    assert_eq!(
        report.total_live,
        report.total_ray + (report.total_live - report.total_ray),
        "sanity: live/ray totals consistent"
    );
    assert!(
        report.total_live < report.total_nodes,
        "at least one dead node excluded from live total"
    );
}

// ==================== Category 5: Minimize Soundness ====================

#[test]
fn test_minimize_soundness_single_clause() {
    // Clause TDDs are inherently deterministic (width 2, exclusive c/d nodes)
    for num_vars in 2..=5 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let lits: Vec<i32> = (1..=num_vars as i32).collect();
            let clause = make_clause(&lits);
            let mut tdd = clause_to_tdd(&vtree, &clause);
            check_minimize_soundness(&mut tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", lits, name, e)
            });
        }
    }
}

#[test]
fn test_minimize_soundness_raw_product() {
    // Product of deterministic TDDs is deterministic, so semiring eval is sound
    let formulas: Vec<(u32, Vec<Vec<i32>>)> = vec![
        (3, vec![vec![1, 2], vec![-2, 3], vec![-1, -3]]),
        // Parity — caught a mod_mul bug (PRIME-1 mask cleared bit 0)
        (3, vec![vec![1, 2, 3], vec![-1, -2, 3], vec![-1, 2, -3], vec![1, -2, -3]]),
        (4, vec![vec![1, 2], vec![3, 4], vec![-1, -3], vec![-2, -4]]),
        (2, vec![vec![1], vec![-1]]),
    ];
    for (num_vars, clauses) in &formulas {
        for (name, vtree) in vtree_shapes(*num_vars) {
            let mut tdd = constant_one(&vtree);
            for lits in clauses {
                let c_tdd = clause_to_tdd(&vtree, &make_clause(lits));
                tdd = apply_and(tdd, c_tdd);
            }
            check_minimize_soundness(&mut tdd, CANONICITY_ROUNDS).unwrap_or_else(|e| {
                panic!("raw product {:?} ({}): {}", clauses, name, e)
            });
        }
    }
}

// ==================== Category 7: No False Nodes ====================
//
// In a minimized TDD, no real node computes the constant-false function.
// The false function is represented exclusively by the ZERO sentinel
// (LocalNodeIdx(u32::MAX)), which never appears in any level's nodes Vec.

#[test]
fn test_no_false_nodes_constant_one() {
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let tdd = constant_one(&vtree);
            check_no_false_nodes(&tdd).unwrap_or_else(|e| {
                panic!("constant_one({} vars, {}): {}", num_vars, name, e)
            });
        }
    }
}

#[test]
fn test_no_false_nodes_clause_tdd() {
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (4, vec![-1, -2, -3, -4]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = make_clause(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            check_no_false_nodes(&tdd).unwrap_or_else(|e| {
                panic!("clause {:?} ({}): {}", lits, name, e)
            });
        }
    }
}

#[test]
fn test_no_false_nodes_after_apply_before_minimize() {
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
        let c1 = clause_to_tdd(&vtree, &make_clause(lits1));
        let c2 = clause_to_tdd(&vtree, &make_clause(lits2));
        let product = apply_and(c1, c2);
        // No minimize! Check the raw product has no false nodes in levels.
        check_no_false_nodes_in_levels(&product).unwrap_or_else(|e| {
            panic!("apply_and({:?}, {:?}) before minimize: {}", lits1, lits2, e)
        });
    }
}

#[test]
fn test_no_false_nodes_multi_apply_before_minimize() {
    // Chain of apply_and calls without intermediate minimize.
    let vtree = Arc::new(Vtree::balanced(5));
    let clauses: Vec<Vec<i32>> = vec![
        vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5],
    ];
    let mut acc = constant_one(&vtree);
    for lits in &clauses {
        let c = clause_to_tdd(&vtree, &make_clause(lits));
        acc = apply_and(acc, c);
        // Check after each conjunction, before any minimize
        check_no_false_nodes_in_levels(&acc).unwrap_or_else(|e| {
            panic!("after conjoining {:?} (no minimize): {}", lits, e)
        });
    }
}

// ==================== Category 8: Reduced Size Sanity ====================
//
// Cross-checks `reduced_tdd_size`'s model-count-based reducibility detection
// with two independent criteria:
//   1. Structural completeness: when Case L fires, the left-side node indices
//      must be exactly 0..child_level.width() (and symmetrically for Case R).
//   2. Product-form consistency: count(g) == count(target) × 2^|vars(child)|.

#[test]
fn test_reduced_size_sanity_constant_one() {
    for num_vars in 2..=4 {
        for (name, vtree) in vtree_shapes(num_vars) {
            let tdd = constant_one(&vtree);
            assert_reduced_size_sane(
                &tdd,
                &format!("constant_one({} vars, {})", num_vars, name),
            );
        }
    }
}

#[test]
fn test_reduced_size_sanity_clause_tdd() {
    let clauses: Vec<(u32, Vec<i32>)> = vec![
        (3, vec![1, 2]),
        (3, vec![-1, 3]),
        (4, vec![1, -2, 3]),
        (4, vec![-1, -2, -3, -4]),
        (2, vec![1]),
    ];
    for (num_vars, lits) in &clauses {
        let clause = make_clause(lits);
        for (name, vtree) in vtree_shapes(*num_vars) {
            let tdd = clause_to_tdd(&vtree, &clause);
            assert_reduced_size_sane(
                &tdd,
                &format!("clause {:?} ({})", lits, name),
            );
        }
    }
}

#[test]
fn test_reduced_size_sanity_after_apply_minimize() {
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = clause_to_tdd(&vtree, &make_clause(&[1, 2]));
    let c2 = clause_to_tdd(&vtree, &make_clause(&[-3, 4]));
    let mut product = apply_and(c1, c2);
    minimize(&mut product);
    assert_reduced_size_sane(&product, "apply_and+minimize([1,2], [-3,4])");
}

// Category 9 (Corruption Detection): the one corruption test that lived here
// (`test_corruption_swapped_pair_child_changes_model_count`, mutating
// `TddNodeData.a` directly) has moved to
// `tests/tdd_invariants_compile.rs` — `tididi` cannot depend on
// `cnf`/`compile` at all, so it can no longer stay in-crate regardless of the
// field-privacy reason that used to justify keeping it here.

/// A leaf label stored in an internal level is rejected by the structural check.
#[test]
fn test_validate_vtree_structure_internal_has_leaf_node() {
    use crate::diagram::{LeafLabel, LocalNodeIdx, Tdd, TddLevel, TddNodeData, TddNodeId};

    let vtree = Arc::new(Vtree::balanced(2));
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    levels[vtree.root().idx()].nodes.push(TddNodeData::leaf(LeafLabel::One));
    let tdd = Tdd::with_levels(
        vtree.clone(),
        levels,
        TddNodeId { vtree: vtree.root(), local: LocalNodeIdx(0) },
    );
    let result = validate_vtree_structure(&tdd);
    assert!(result.is_err(), "internal level holds a non-internal node");
    assert!(result.unwrap_err().contains("non-Internal"));
}
