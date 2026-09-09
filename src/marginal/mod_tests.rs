//! Marginalization primitives exercised in isolation on hand-compiled diagrams:
//! the weighted cluster closure, the reclaim of subsumed marginal data, and
//! single-leaf inlining.

use crate::engine::Engine;
use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;

use super::{marginalize, marginalize_closure, marginalize_leaf_inline, weighted_value};
use crate::reduce::try_minimize;
use crate::diagram::{RationalWeights, WeightVal};
use crate::query::{evaluate, model_count};
use crate::test_helpers::compile_clauses;
use crate::diagram::Tdd;
use crate::check::marg::subsumed_marginal_data_violations;
use crate::diagram::{Precision, WeightStore};
use crate::vtree::{VarId, Vtree, VtreeIdx};

fn exact(v: &WeightVal) -> BigRational {
    assert!(!matches!(v, WeightVal::Log(_)), "test expected exact-mode WeightVal");
    v.as_rational().into_owned()
}

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

/// Clauses spanning both halves of a 4-var balanced vtree, so the root is a
/// genuine structural node over two internal subtrees.
fn closure_cluster_clauses() -> Vec<Vec<i32>> {
    vec![vec![1, 2], vec![2, -3], vec![3, 4]]
}

/// Marginalize only the root's two child subtrees through the weighted
/// dispatch, leaving the root structural over two weight-marginal children,
/// then close the cluster and read the root weight.
fn weighted_closure_root(eng: &Engine, clauses: &[Vec<i32>], vtree: &Arc<Vtree>, sr: RationalWeights) -> BigRational {
    let (a, b) = vtree.children(vtree.root());
    let mut tdd = compile_clauses(vtree, clauses);
    tdd.attach_weights(WeightStore::new(sr, Precision::Exact));
    marginalize(eng, &mut tdd, &[a, b]).expect("no wall is installed in a test");
    check_marginal_invariants(&tdd, "weighted_closure_root");
    assert!(
        tdd.levels[a.idx()].is_weight_marginal() && tdd.levels[b.idx()].is_weight_marginal(),
        "both child subtrees must be weight-marginal before the closure"
    );
    assert!(
        !tdd.levels[vtree.root().idx()].is_marginal(),
        "root must still be structural (a closure target) before the closure"
    );
    marginalize_closure(eng, &mut tdd, vtree).expect("no wall is installed in a test");
    assert!(
        tdd.levels[vtree.root().idx()].is_weight_marginal(),
        "closure must collapse the root through the weighted path"
    );
    exact(&weighted_value(&tdd).expect("a weighted diagram has a value"))
}

#[test]
fn weighted_closure_unit_weights_matches_model_count() {
    let eng = Engine::new();
    let clauses = closure_cluster_clauses();
    let vtree = Arc::new(Vtree::balanced(4));
    let mc = BigRational::from(BigInt::from(model_count(&compile_clauses(&vtree, &clauses))));
    let got = weighted_closure_root(&eng, &clauses, &vtree, RationalWeights::unit(4));
    assert_eq!(got, mc, "weighted closure unit count != model count");
}

#[test]
fn weighted_closure_nonunit_weights_matches_evaluate() {
    let eng = Engine::new();
    let clauses = closure_cluster_clauses();
    let vtree = Arc::new(Vtree::balanced(4));
    let weights = vec![
        (rat(1, 2), rat(1, 3)),
        (rat(2, 5), rat(3, 7)),
        (rat(5, 11), rat(2, 9)),
        (rat(1, 1), rat(4, 9)),
    ];
    let oracle = evaluate(&compile_clauses(&vtree, &clauses), &RationalWeights::from_weights(&weights));
    let got = weighted_closure_root(&eng, &clauses, &vtree, RationalWeights::from_weights(&weights));
    assert_eq!(got, oracle, "weighted closure non-unit count != evaluate oracle");
}

fn nested_region_clauses() -> Vec<Vec<i32>> {
    vec![vec![1, 2], vec![-2, 3], vec![3, -4], vec![-4, 5], vec![1, -5]]
}

/// Marginal levels under a marginal parent (counted here so the assertion is
/// known to have bitten).
fn nested_marginal_levels(tdd: &Tdd) -> usize {
    let vtree = &tdd.vtree;
    (0..vtree.num_nodes())
        .filter(|&i| {
            tdd.levels[i].is_marginal()
                && vtree
                    .node(VtreeIdx(i as u32))
                    .parent()
                    .is_some_and(|p| tdd.levels[p.idx()].is_marginal())
        })
        .count()
}

/// Marginalizing every internal level except the root leaves the root
/// structural over fully marginalized subtrees; their interior levels must end
/// up data-free and the count unchanged.
#[test]
fn integer_marginalize_leaves_no_subsumed_data() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(5));
    let mut tdd = compile_clauses(&vtree, &nested_region_clauses());
    let mc_before = model_count(&tdd);
    let targets: Vec<_> = vtree.bottomup_slice().iter().copied().filter(|&t| t != vtree.root()).collect();
    marginalize(&eng, &mut tdd, &targets).expect("no wall is installed in a test");
    check_marginal_invariants(&tdd, "integer_marginalize_leaves_no_subsumed_data");

    let viol = subsumed_marginal_data_violations(&tdd);
    assert!(viol.is_empty(), "subsumed marginal levels still hold data: {viol:?}");
    assert_eq!(model_count(&tdd), mc_before, "marginalization changed #F");
    assert!(nested_marginal_levels(&tdd) >= 1, "test setup: expected nested marginal regions");
}

/// The weighted pipeline frees subsumed levels too, including their
/// `WeightStore` entries; unit weights confirm the surviving root value.
#[test]
fn weighted_marginalize_leaves_no_subsumed_data() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(5));
    let mut tdd = compile_clauses(&vtree, &nested_region_clauses());
    let mc = BigRational::from(BigInt::from(model_count(&tdd)));

    let targets: Vec<_> = vtree.bottomup_slice().to_vec();
    tdd.attach_weights(WeightStore::new(RationalWeights::unit(5), Precision::Exact));
    marginalize(&eng, &mut tdd, &targets).expect("no wall is installed in a test");
    check_marginal_invariants(&tdd, "weighted_marginalize_leaves_no_subsumed_data");

    assert_eq!(
        exact(&weighted_value(&tdd).expect("a weighted diagram has a value")),
        mc,
        "freeing subsumed children corrupted the weighted root value"
    );
    let viol = subsumed_marginal_data_violations(&tdd);
    assert!(viol.is_empty(), "subsumed weighted levels hold data: {viol:?}");

    // Vtree leaves keep their pinned 3-slot column; every other subsumed level
    // must have released its store entry.
    let ws = tdd.weights().expect("attached above");
    let mut nested = 0;
    for i in 0..vtree.num_nodes() {
        if !tdd.levels[i].is_marginal() || vtree.node(VtreeIdx(i as u32)).is_leaf() {
            continue;
        }
        let Some(p) = vtree.node(VtreeIdx(i as u32)).parent() else { continue };
        if !tdd.levels[p.idx()].is_marginal() {
            continue;
        }
        nested += 1;
        assert!(
            ws.level(i).map(|v| v.is_empty()).unwrap_or(true),
            "subsumed weighted level {i} still has WeightStore values"
        );
    }
    assert!(nested >= 1, "test setup: expected nested marginal regions");
}

fn leaf_inline_clauses() -> Vec<Vec<i32>> {
    vec![vec![1, 2], vec![-3, -4], vec![5, 6], vec![-1, 3], vec![4, -5]]
}

/// Inlining a leaf's fixed 0/1/2 count at its parent is count-neutral and flips
/// the leaf marginal.
#[test]
fn leaf_inline_preserves_count() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    for v in 0..6u32 {
        let mut t = compile_clauses(&vtree, &leaf_inline_clauses());
        let baseline = model_count(&t);
        let leaf = vtree.leaf_of(VarId(v)).expect("the vtree carries this variable");
        marginalize_leaf_inline(&eng, &mut t, leaf, &vtree);
        try_minimize(&eng, &mut t, Default::default()).unwrap();
        assert!(t.levels[leaf.idx()].is_marginal(), "leaf {v} not marginal");
        assert_eq!(model_count(&t), baseline, "count changed marginalizing leaf {v}");
    }
}

/// Inlining every leaf in sequence still reduces to the baseline count.
#[test]
fn all_leaves_inline_preserve_count() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let mut t = compile_clauses(&vtree, &leaf_inline_clauses());
    let baseline = model_count(&t);
    for v in 0..6u32 {
        marginalize_leaf_inline(&eng, &mut t, vtree.leaf_of(VarId(v)).expect("the vtree carries this variable"), &vtree);
    }
    try_minimize(&eng, &mut t, Default::default()).unwrap();
    assert_eq!(model_count(&t), baseline);
}

/// Every invariant a marginalized diagram must satisfy.
///
/// Two of `check_all_deep`'s checks are left out because they describe a
/// structural diagram, not this one. `reduced_size` and its validator read a
/// child level's per-node counts by slot, which a marginal level does not keep.
/// And canonicity is not a post-condition of `marginalize`: the fusion sweep in
/// its epilogue merges slots with equal values, which can make two parent nodes
/// content-equal — minimize's twin contraction is what removes those, and it
/// runs later.
fn check_marginal_invariants(tdd: &Tdd, label: &str) {
    crate::check::validate_vtree_structure(tdd)
        .unwrap_or_else(|e| panic!("{label}: vtree structure: {e}"));
    crate::check::check_no_false_nodes(tdd)
        .unwrap_or_else(|e| panic!("{label}: no_false_nodes: {e}"));
    crate::check::marg::check_no_fusion_redexes(tdd)
        .unwrap_or_else(|e| panic!("{label}: no_fusion_redexes: {e}"));
    crate::check::marg::check_slot_count_uniqueness(tdd)
        .unwrap_or_else(|e| panic!("{label}: slot_count_uniqueness: {e}"));
    crate::check::marg::check_no_orphan_slots(tdd)
        .unwrap_or_else(|e| panic!("{label}: no_orphan_slots: {e}"));
}
