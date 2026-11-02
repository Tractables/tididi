use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::restructure::DistinctError;
use crate::test_helpers::{assert_canonical, compile_clauses, eval, random_diagrams, test_cases, under, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};

/// Every vtree node a count may be keyed on: all but the root.
fn keys(vtree: &Vtree) -> Vec<VtreeIdx> {
    vtree.bottomup().filter(|&t| t != vtree.root()).collect()
}

fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    vtree.leaf_bottomup().filter(|&(leaf, _)| under(vtree, leaf, t)).map(|(_, v)| v).collect()
}

/// Every count of `f` keyed on each node [`keys`] gives, for `m` up to
/// four, against enumerating its truth table.
fn check(eng: &Engine, f: &Tdd) {
    let vtree = f.vtree();
    let n = vtree.num_vars() as usize;
    let truth: Vec<Vec<bool>> = (0..1u64 << n)
        .map(|bits| (0..n).map(|i| bits >> i & 1 == 1).collect::<Vec<bool>>())
        .filter(|asn| eval(f, asn))
        .collect();
    for key in keys(vtree) {
        let counted = vtree.sibling(key);
        let (kv, sv) = (vars_under(vtree, key), vars_under(vtree, counted));
        let mut distinct: HashMap<Vec<bool>, HashSet<Vec<bool>>> = HashMap::new();
        for asn in &truth {
            let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
            let s: Vec<bool> = sv.iter().map(|v| asn[v.idx()]).collect();
            distinct.entry(k).or_default().insert(s);
        }
        for m in 0..=4u64 {
            let g = eng.at_least_distinct(f, key, m).unwrap();
            assert_canonical(&g);
            assert_eq!(g.vtree().num_leaves() as usize, kv.len());
            for bits in 0..1u64 << kv.len() {
                let mut asn = vec![false; n];
                for (i, v) in kv.iter().enumerate() {
                    asn[v.idx()] = bits >> i & 1 == 1;
                }
                let k: Vec<bool> = kv.iter().map(|v| asn[v.idx()]).collect();
                let count = distinct.get(&k).map_or(0, HashSet::len) as u64;
                assert_eq!(eval(&g, &asn), count >= m, "key {key:?} m {m} at {k:?}: {count} distinct");
            }
        }
    }
}

#[test]
fn the_threshold_counts_the_distinct_values_of_the_sibling() {
    for (num_vars, clauses) in test_cases() {
        for (_, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            check(&Engine::new(), &f);
        }
    }
}

#[test]
fn unreduced_diagrams_count_as_their_functions_do() {
    for f in random_diagrams(11, 24, 3..9) {
        check(&Engine::new(), &f);
    }
}

#[test]
fn a_projection_onto_a_subtree_is_the_quantified_diagram() {
    let eng = Engine::new();
    for f in random_diagrams(5, 16, 3..9) {
        let vtree = f.vtree();
        for key in keys(vtree) {
            let kept = vars_under(vtree, key);
            let dropped: Vec<VarId> = vtree.leaf_bottomup().map(|(_, v)| v).filter(|v| !kept.contains(v)).collect();
            let g = eng.project_to_subtree(&f, key).unwrap();
            assert_canonical(&g);
            let (back, _) = eng.embed(&g, vtree, |v| v).unwrap();
            let expected = eng.exists_vars(f.clone(), &dropped).unwrap();
            assert!(eng.equivalent(&back, &expected).unwrap(), "key {key:?}");
        }
    }
}

#[test]
fn false_counts_nothing_and_zero_is_true() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1], vec![-1]]);
    let left = vtree.children(vtree.root()).0;
    assert!(eng.at_least_distinct(&f, left, 1).unwrap().is_zero());
    let all = eng.at_least_distinct(&f, left, 0).unwrap();
    assert_eq!(eng.model_count(&all).unwrap(), 4u32.into());
}

#[test]
fn the_root_is_refused_as_a_key() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::linear(4));
    let f = compile_clauses(&vtree, &[vec![1, 2, 3, 4]]);
    assert!(matches!(eng.at_least_distinct(&f, vtree.root(), 1), Err(DistinctError::Root)));
    // x4 sits three levels down a linear vtree of four variables: a key
    // there reads the levels on its path only.
    let deep = vtree.leaf_of(VarId(4)).unwrap();
    let g = eng.project_to_subtree(&f, deep).unwrap();
    assert_eq!(eng.model_count(&g).unwrap(), 2u32.into());
}
