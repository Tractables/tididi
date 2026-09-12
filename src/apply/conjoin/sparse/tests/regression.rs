//! Regressions in the sparse product path, forced on (`min_grid=1`,
//! `sparsity_factor=1`) so small formulas take the code paths that normally
//! only large grids reach: the leaf-alive table's label order, duplicate
//! output pairs from the scatter phase, chunked emission, workspace residue
//! across compiles, and the sparse/dense seam within one apply.
use crate::engine::Engine;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::apply::conjoin::conjoin_owned;

use super::{ForcedThresholds, SparseThresholds};
use crate::apply::conjoin_clause::clause_to_tdd;
use crate::build::constant_one;
use crate::reduce::{try_reduce, ReductionPlan};
use crate::query::model_count;
use crate::test_helpers::{brute_force_count, literals, normalized_levels, test_cases};
use crate::diagram::Tdd;
use crate::test_helpers::check::check_all_fast;
use crate::test_helpers::check_minimize_soundness;
use crate::vtree::Vtree;

/// The sparse-route thresholds a test's applies decide by, and the engine
/// they run on.
struct Sparse {
    eng: Engine,
    thresholds: SparseThresholds,
}

impl Sparse {
    /// Every level of every apply takes the sparse route.
    fn always() -> Sparse {
        Sparse {
            eng: Engine::new(),
            thresholds: SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION },
        }
    }

    /// No level of any apply takes the sparse route.
    fn never() -> Sparse {
        Sparse {
            eng: Engine::new(),
            thresholds: SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION },
        }
    }

    /// The same thresholds with a different chunk budget, on a fresh engine.
    fn rechunked(&self, chunk_bytes: usize) -> Sparse {
        Sparse { eng: Engine::new(), thresholds: SparseThresholds { chunk_bytes, ..self.thresholds } }
    }

    /// `Engine::and` under these thresholds.
    fn and(&self, f: Tdd, g: Tdd, targets: Option<&[bool]>) -> Tdd {
        let _forced = ForcedThresholds::install(self.thresholds);
        conjoin_owned(&self.eng, f, g, targets).expect("an unarmed engine refuses nothing")
    }

    /// Clause-by-clause fold with a minimize after each clause, as
    /// `test_helpers::compile_clauses_on`, under these thresholds.
    fn compile(&self, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
        let mut acc = constant_one(&self.eng, vtree);
        for clause in clauses {
            let cl = clause_to_tdd(&self.eng, vtree, &literals(clause));
            acc = self.and(acc, cl, None);
            try_reduce(&self.eng, &mut acc, ReductionPlan::default())
                .expect("an unarmed engine refuses nothing");
        }
        acc
    }
}

/// Conjunction of two clause-fold operands (each half of `clauses`).
fn two_operand_apply(sparse: &Sparse, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mid = clauses.len() / 2;
    let f = sparse.compile(vtree, &clauses[..mid]);
    let g = sparse.compile(vtree, &clauses[mid..]);
    let mut result = sparse.and(f, g, None);
    try_reduce(&sparse.eng, &mut result, ReductionPlan::default()).expect("minimize");
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
    let sparse = Sparse::always();
    let clauses = vec![
        vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5],
        vec![-1, -5], vec![2, -3, 4],
    ];
    let vtree = Arc::new(Vtree::linear(5));
    let tdd = sparse.compile(&vtree, &clauses);
    assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(5, &clauses)));
}

#[test]
fn leaf_alive_two_operand_apply_stick() {
    let sparse = Sparse::always();
    let vtree = Arc::new(Vtree::linear(8));
    let mut result = two_operand_apply(&sparse, &vtree, &stick());
    assert_eq!(model_count(&result), BigUint::from(brute_force_count(8, &stick())));
    check_minimize_soundness(&mut result, 3).expect("minimize soundness");
}

/// Several product entries can map to one output pair; without dedup the
/// duplicates corrupted the node encoding and contraction panicked.
#[test]
fn dedup_balanced_vtree() {
    let sparse = Sparse::always();
    let clauses = vec![
        vec![1, 2, 3], vec![-1, -2, 4], vec![2, -3, 5],
        vec![-2, 3, -4], vec![1, -4, -5], vec![-1, 3, 4, 5],
        vec![2, -5, 6], vec![-3, 4, -6], vec![1, -2, -6],
        vec![-1, 5, 6], vec![3, -4, -5, 6],
    ];
    let vtree = Arc::new(Vtree::balanced(6));
    let mut tdd = sparse.compile(&vtree, &clauses);
    assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(6, &clauses)));
    check_all_fast(&tdd, "dedup_balanced_vtree");
    check_minimize_soundness(&mut tdd, 3).expect("minimize soundness");
}

#[test]
fn dedup_dense_formula() {
    let sparse = Sparse::always();
    let clauses = vec![
        vec![1, 2], vec![1, 3], vec![1, 4], vec![2, 3],
        vec![2, 4], vec![3, 4], vec![-1, -2], vec![-1, -3],
        vec![-2, -4], vec![-3, -4], vec![1, -4], vec![-1, 4],
    ];
    let vtree = Arc::new(Vtree::balanced(4));
    let mut tdd = sparse.compile(&vtree, &clauses);
    assert_eq!(model_count(&tdd), BigUint::from(brute_force_count(4, &clauses)));
    check_all_fast(&tdd, "dedup_dense_formula");
    check_minimize_soundness(&mut tdd, 3).expect("minimize soundness");
}

#[test]
fn all_test_cases_sparse() {
    let sparse = Sparse::always();
    for (num_vars, clauses) in test_cases() {
        let expected = BigUint::from(brute_force_count(num_vars, &clauses));
        for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
            let vtree = Arc::new(vtree);
            assert_eq!(
                model_count(&sparse.compile(&vtree, &clauses)),
                expected,
                "n={num_vars} clauses={clauses:?}"
            );
        }
    }
}

/// A chunk budget so small that every non-trivial sparse level splits must
/// agree with chunking disabled.
#[test]
fn chunked_equivalence_all_cases() {
    let sparse = Sparse::always();
    let chunking = sparse.rechunked(256);
    let whole = sparse.rechunked(usize::MAX);
    for (num_vars, clauses) in test_cases() {
        let expected = BigUint::from(brute_force_count(num_vars, &clauses));
        for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
            let vtree = Arc::new(vtree);
            let chunked = chunking.compile(&vtree, &clauses);
            let unchunked = whole.compile(&vtree, &clauses);
            assert_eq!(model_count(&chunked), expected, "chunked: n={num_vars} clauses={clauses:?}");
            assert_eq!(model_count(&unchunked), expected, "unchunked: n={num_vars} clauses={clauses:?}");
        }
    }
}

/// Wide enough that the chunk budget forces multi-chunk splits.
#[test]
fn chunked_wide_formula() {
    let sparse = Sparse::always();
    let clauses = wide();
    let expected = BigUint::from(brute_force_count(14, &clauses));
    let vtree = Arc::new(Vtree::balanced(14));
    let chunked = sparse.rechunked(256).compile(&vtree, &clauses);
    let unchunked = sparse.rechunked(usize::MAX).compile(&vtree, &clauses);
    assert_eq!(model_count(&chunked), expected);
    assert_eq!(model_count(&unchunked), expected);
}

/// Alternating narrow and wide compiles on one thread reuse the sparse
/// workspace with overlapping index ranges; a stale map entry would undercount
/// the later compile.
#[test]
fn workspace_has_no_stale_residue() {
    let sparse = Sparse::always();
    let wide_clauses = wide();
    let wide_expected = BigUint::from(brute_force_count(14, &wide_clauses));
    let wide_vtree = Arc::new(Vtree::balanced(14));
    for (num_vars, clauses) in test_cases() {
        let vtree = Arc::new(Vtree::balanced(num_vars));
        assert_eq!(
            model_count(&sparse.compile(&vtree, &clauses)),
            BigUint::from(brute_force_count(num_vars, &clauses)),
            "narrow: n={num_vars} clauses={clauses:?}"
        );
        assert_eq!(
            model_count(&sparse.compile(&wide_vtree, &wide_clauses)),
            wide_expected,
            "wide formula after narrow compile: n={num_vars}"
        );
    }
}

/// Forcing every internal level into the streaming-emit branch must match the
/// dense path.
#[test]
fn streaming_implicit_equivalence_all_cases() {
    let dense = Sparse::never();
    for (num_vars, clauses) in test_cases() {
        let expected = BigUint::from(brute_force_count(num_vars, &clauses));
        for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars)] {
            let vtree = Arc::new(vtree);
            let mid = clauses.len() / 2;
            let f = dense.compile(&vtree, &clauses[..mid]);
            let g = dense.compile(&vtree, &clauses[mid..]);
            let normal = dense.and(f.clone(), g.clone(), None);
            let targets = vec![true; vtree.num_nodes()];
            let streamed = dense.and(f, g, Some(&targets));
            assert_eq!(model_count(&normal), expected, "dense: n={num_vars} clauses={clauses:?}");
            assert_eq!(model_count(&streamed), expected, "streamed: n={num_vars} clauses={clauses:?}");
        }
    }
}

/// Small `min_grid` thresholds move the sparse/dense split to different
/// levels of one apply; the canonical result must not depend on where it
/// lands.
#[test]
fn mixed_sparse_dense_min_grid_sweep() {
    let clauses = wide();
    let expected = BigUint::from(brute_force_count(14, &clauses));
    let vtree = Arc::new(Vtree::balanced(14));
    let reference = Sparse::always().compile(&vtree, &clauses);
    assert_eq!(model_count(&reference), expected);
    for min_grid in [2usize, 4, 8, 16] {
        let mut sparse = Sparse::always();
        sparse.thresholds.min_grid = min_grid;
        let tdd = sparse.compile(&vtree, &clauses);
        assert_eq!(model_count(&tdd), expected, "min_grid={min_grid}");
        assert_eq!(
            normalized_levels(&tdd),
            normalized_levels(&reference),
            "min_grid={min_grid} differs structurally from min_grid=1"
        );
    }
}
