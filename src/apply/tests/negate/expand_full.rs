use super::*;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;


use crate::vtree::Vtree;
use num_bigint::BigUint;

fn balanced_vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

// ── Explicit expand_full tests ─────────────────────────────────────────

#[test]
fn expand_full_constant_one() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = constant_one(eng, &vtree);
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();
}

#[test]
fn expand_full_single_clause() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true), (2, false)]));
    let count_before = tdd.model_count().unwrap();
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();
    assert_eq!(count_before, tdd.model_count().unwrap());
}

#[test]
fn expand_full_preserves_determinism() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true), (3, false)]));
    tdd.minimize().unwrap();
    let widths_before: Vec<usize> = tdd.levels.iter().map(|l| l.slot_count()).collect();
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();

    let counts = crate::test_helpers::node_counts(&tdd);
    let mut subtree_vars = vec![0u32; vtree.num_nodes()];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }
    for (t, _left, _right) in vtree.internal_bottomup() {
        let ti = t.idx();
        if tdd.levels[ti].slot_count() <= widths_before[ti] { continue; }
        let expected = BigUint::from(1u32) << subtree_vars[ti] as usize;
        let total: BigUint = counts[ti].iter().sum();
        assert_eq!(total, expected);
    }
}
