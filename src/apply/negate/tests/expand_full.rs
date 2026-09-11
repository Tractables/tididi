use super::*;
use crate::apply::conjoin_clause::clause_to_tdd;
use crate::build::constant_one;
use crate::reduce::minimize;
use crate::query::model_count;
use crate::vtree::Vtree;
use num_bigint::BigUint;

fn balanced_vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

// ── Explicit expand_full tests ─────────────────────────────────────────

#[test]
fn expand_full_constant_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = constant_one(eng, &vtree);
    expand_full(&mut tdd);
}

#[test]
fn expand_full_single_clause() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, false)]));
    let count_before = model_count(&tdd);
    expand_full(&mut tdd);
    assert_eq!(count_before, model_count(&tdd));
}

#[test]
fn expand_full_preserves_determinism() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (2, false)]));
    minimize(&mut tdd);
    let widths_before: Vec<usize> = tdd.levels.iter().map(|l| l.width()).collect();
    expand_full(&mut tdd);

    let counts = crate::query::node_counts(&tdd);
    let mut subtree_vars = vec![0u32; vtree.num_nodes()];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }
    for (t, _left, _right) in vtree.internal_bottomup() {
        let ti = t.idx();
        if tdd.levels[ti].width() <= widths_before[ti] { continue; }
        let expected = BigUint::from(1u32) << subtree_vars[ti] as usize;
        let total: BigUint = counts[ti].iter().sum();
        assert_eq!(total, expected);
    }
}
