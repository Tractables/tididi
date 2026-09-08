//! Regressions in the sparse product path, forced on (`min_grid=1`,
//! `sparsity_factor=1`) so small formulas take the code paths that normally
//! only large grids reach: the leaf-alive table's label order, duplicate
//! output pairs from the scatter phase, chunked emission, workspace residue
//! across compiles, and the sparse/dense seam within one apply.
use crate::engine::Limits;
use std::sync::Arc;

use num_bigint::BigUint;

use super::{with_sparse_chunk_bytes, with_sparse_config};
use crate::apply::conjoin::{apply_and, try_apply_and};
use crate::reduce::minimize;
use crate::query::model_count;
use crate::test_helpers::{brute_force_count, compile_clauses, normalized_levels, test_cases};
use crate::diagram::Tdd;
use crate::check::{check_all_fast, check_minimize_soundness};
use crate::vtree::Vtree;

fn with_sparse<F: FnOnce()>(f: F) {
    with_sparse_config(1, 1, f);
}

fn with_dense<F: FnOnce()>(f: F) {
    with_sparse_config(usize::MAX, 64, f);
}

/// Conjunction of two clause-fold operands (each half of `clauses`).
fn two_operand_apply(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mid = clauses.len() / 2;
    let c1 = compile_clauses(vtree, &clauses[..mid]);
    let c2 = compile_clauses(vtree, &clauses[mid..]);
    let mut result = apply_and(c1, c2);
    minimize(&mut result);
    result
}

const WIDE: [[i32; 3]; 24] = [
    [1, 2, 3], [-1, 4, 5], [2, -3, 6], [-4, -5, 7],
    [1, -6, 8], [-2, 5, -7], [3, 4, -8], [-3, 6, 9],
    [1, -8, 10], [-4, 7, -9], [2, -10, 11], [-5, 8, -11],
    [3, -7, 12], [-6, 9, -12], [4, -11, 13], [-1, 10, -13],
    [5, -9, 14], [-2, 11, -14], [1, -12, 13], [-3, 8, -14],
    [2, 7, -10], [-5, -8, 12], [6, -11, -13], [-4, 9, 14],
];

fn wide() -> Vec<Vec<i32>> {
    WIDE.iter().map(|c| c.to_vec()).collect()
}

const STICK: [[i32; 2]; 8] = [
    [1, -3], [-2, 4], [5, -6], [-7, 8],
    [-1, 2], [3, 5], [-4, -6], [7, -8],
];

fn stick() -> Vec<Vec<i32>> {
    STICK.iter().map(|c| c.to_vec()).collect()
}

/// A stick vtree has a leaf left child at every internal level, so every
/// sparse level goes through the leaf-alive table.
#[test]
fn leaf_alive_stick_vtree() {
    with_sparse(|| {
        let clauses = vec![
            vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5],
            vec![-1, -5], vec![2, -3, 4],
        ];
        let vtree = Arc::new(Vtree::linear(5));
        let tdd = compile_clauses(&vtree, &clauses);
        assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(5, &clauses)));
    });
}

#[test]
fn leaf_alive_two_operand_apply_stick() {
    with_sparse(|| {
        let vtree = Arc::new(Vtree::linear(8));
        let mut result = two_operand_apply(&vtree, &stick());
        assert_eq!(model_count(&result), BigUint::from(brute_force_count(8, &stick())));
        check_minimize_soundness(&mut result, 3).expect("minimize soundness");
    });
}

/// Several product entries can map to one output pair; without dedup the
/// duplicates corrupted the node encoding and contraction panicked.
#[test]
fn dedup_balanced_vtree() {
    with_sparse(|| {
        let clauses = vec![
            vec![1, 2, 3], vec![-1, -2, 4], vec![2, -3, 5],
            vec![-2, 3, -4], vec![1, -4, -5], vec![-1, 3, 4, 5],
            vec![2, -5, 6], vec![-3, 4, -6], vec![1, -2, -6],
            vec![-1, 5, 6], vec![3, -4, -5, 6],
        ];
        let vtree = Arc::new(Vtree::balanced(6));
        let mut tdd = compile_clauses(&vtree, &clauses);
        assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(6, &clauses)));
        check_all_fast(&tdd, "dedup_balanced_vtree");
        check_minimize_soundness(&mut tdd, 3).expect("minimize soundness");
    });
}

#[test]
fn dedup_dense_formula() {
    with_sparse(|| {
        let clauses = vec![
            vec![1, 2], vec![1, 3], vec![1, 4], vec![2, 3],
            vec![2, 4], vec![3, 4], vec![-1, -2], vec![-1, -3],
            vec![-2, -4], vec![-3, -4], vec![1, -4], vec![-1, 4],
        ];
        let vtree = Arc::new(Vtree::balanced(4));
        let mut tdd = compile_clauses(&vtree, &clauses);
        assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(4, &clauses)));
        check_all_fast(&tdd, "dedup_dense_formula");
        check_minimize_soundness(&mut tdd, 3).expect("minimize soundness");
    });
}

#[test]
fn all_test_cases_sparse() {
    with_sparse(|| {
        for (num_vars, clauses) in test_cases() {
            let expected = BigUint::from(brute_force_count(num_vars, &clauses));
            for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
                let vtree = Arc::new(vtree);
                assert_eq!(
                    model_count(&compile_clauses(&vtree, &clauses)),
                    expected,
                    "n={num_vars} clauses={clauses:?}"
                );
            }
        }
    });
}

/// A chunk budget so small that every non-trivial sparse level splits must
/// agree with chunking disabled.
#[test]
fn chunked_equivalence_all_cases() {
    with_sparse(|| {
        for (num_vars, clauses) in test_cases() {
            let expected = BigUint::from(brute_force_count(num_vars, &clauses));
            for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
                let vtree = Arc::new(vtree);
                let chunked = with_sparse_chunk_bytes(256, || compile_clauses(&vtree, &clauses));
                let unchunked = with_sparse_chunk_bytes(usize::MAX, || compile_clauses(&vtree, &clauses));
                assert_eq!(model_count(&chunked), expected, "chunked: n={num_vars} clauses={clauses:?}");
                assert_eq!(model_count(&unchunked), expected, "unchunked: n={num_vars} clauses={clauses:?}");
            }
        }
    });
}

/// Wide enough that the chunk budget forces multi-chunk splits.
#[test]
fn chunked_wide_formula() {
    with_sparse(|| {
        let clauses = wide();
        let expected = BigUint::from(brute_force_count(14, &clauses));
        let vtree = Arc::new(Vtree::balanced(14));
        let chunked = with_sparse_chunk_bytes(256, || compile_clauses(&vtree, &clauses));
        let unchunked = with_sparse_chunk_bytes(usize::MAX, || compile_clauses(&vtree, &clauses));
        assert_eq!(model_count(&chunked), expected);
        assert_eq!(model_count(&unchunked), expected);
    });
}

/// Alternating narrow and wide compiles on one thread reuse the sparse
/// workspace with overlapping index ranges; a stale map entry would undercount
/// the later compile.
#[test]
fn workspace_has_no_stale_residue() {
    with_sparse(|| {
        let wide_clauses = wide();
        let wide_expected = BigUint::from(brute_force_count(14, &wide_clauses));
        let wide_vtree = Arc::new(Vtree::balanced(14));
        for (num_vars, clauses) in test_cases() {
            let vtree = Arc::new(Vtree::balanced(num_vars));
            assert_eq!(
                model_count(&compile_clauses(&vtree, &clauses)),
                BigUint::from(brute_force_count(num_vars, &clauses)),
                "narrow: n={num_vars} clauses={clauses:?}"
            );
            assert_eq!(
                model_count(&compile_clauses(&wide_vtree, &wide_clauses)),
                wide_expected,
                "wide formula after narrow compile: n={num_vars}"
            );
        }
    });
}

/// Forcing every internal level into the streaming-emit branch must match the
/// dense path.
#[test]
fn streaming_implicit_equivalence_all_cases() {
    let lim = Limits::new();
    with_dense(|| {
        for (num_vars, clauses) in test_cases() {
            let expected = BigUint::from(brute_force_count(num_vars, &clauses));
            for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
                let vtree = Arc::new(vtree);
                let mid = clauses.len() / 2;
                let c1 = compile_clauses(&vtree, &clauses[..mid]);
                let c2 = compile_clauses(&vtree, &clauses[mid..]);
                let normal = apply_and(c1.clone(), c2.clone());
                let targets = vec![true; vtree.num_nodes()];
                let streamed = try_apply_and(&lim, c1, c2, Some(&targets))
                    .expect("streaming apply within budget");
                assert_eq!(model_count(&normal), expected, "dense: n={num_vars} clauses={clauses:?}");
                assert_eq!(model_count(&streamed), expected, "streamed: n={num_vars} clauses={clauses:?}");
            }
        }
    });
}

/// Small `min_grid` thresholds move the sparse/dense split to different
/// levels of one apply; the canonical result must not depend on where it
/// lands.
#[test]
fn mixed_sparse_dense_min_grid_sweep() {
    let clauses = wide();
    let expected = BigUint::from(brute_force_count(14, &clauses));
    let vtree = Arc::new(Vtree::balanced(14));
    let reference = with_sparse_config(1, 1, || compile_clauses(&vtree, &clauses));
    assert_eq!(model_count(&reference), expected);
    for min_grid in [2usize, 4, 8, 16] {
        let tdd = with_sparse_config(min_grid, 1, || compile_clauses(&vtree, &clauses));
        assert_eq!(model_count(&tdd), expected, "min_grid={min_grid}");
        assert_eq!(
            normalized_levels(&tdd),
            normalized_levels(&reference),
            "min_grid={min_grid} differs structurally from min_grid=1"
        );
    }
}
