use super::*;
use std::sync::Arc;
use crate::Vtree;
use crate::test_helpers::assert_canonical;

#[test]
fn pair_marks_union_contexts_across_word_boundaries() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let eng = Engine::new();
    let (v, node) = (f.output.vtree, f.output.local);
    for count in [1, 63, 64, 65, 127, 128, 129, 1025] {
        let mut marks = PairMarks::new(&eng, &f).unwrap();
        for k in (0..count).step_by(2) { marks.mark(&eng, v, node, k, count).unwrap(); }
        for k in 0..count { assert_eq!(marks.contains(v, node, k, count), k % 2 == 0); }
        assert_eq!(marks.complete(v, node, count), count == 1);
        for k in (1..count).step_by(2) { marks.mark(&eng, v, node, k, count).unwrap(); }
        assert!(marks.complete(v, node, count));
    }
}

#[test]
fn wide_marks_recover_after_allocation_refusal() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let eng = Engine::new();
    let (v, node) = (f.output.vtree, f.output.local);
    for nth in 0..2 {
        let mut marks = PairMarks::new(&eng, &f).unwrap();
        eng.limits().refuse_nth_reserve(nth);
        assert_eq!(marks.mark(&eng, v, node, 64, 65), Err(OperationError::OverBudget));
        eng.limits().grant_every_reserve();
        assert!(!marks.contains(v, node, 64, 65));
        marks.mark(&eng, v, node, 64, 65).unwrap();
        assert!(marks.contains(v, node, 64, 65));
        assert!(!marks.contains(v, node, 0, 65));
    }
}

#[test]
fn restriction_of_a_wide_relation_preserves_exact_models() {
    // A bijection across the root cut needs a distinct pair for each row.
    let tree = Arc::new(Vtree::balanced(14));
    let vars: Vec<_> = (1..=14).map(crate::vtree::VarId).collect();
    let rows: Vec<u64> = (0..128).map(|x| x | (((x * 37 + 11) % 128) << 7)).collect();
    let f = Tdd::from_models(&tree, &vars, &rows).unwrap();
    assert_canonical(&f);
    assert!(f.levels[f.output.vtree.idx()].pairs_of_idx(f.output.local.idx()).len() > 64);
    let care_rows: Vec<_> = rows.iter().copied().step_by(3).collect();
    let c = Tdd::from_models(&tree, &vars, &care_rows).unwrap();
    assert_canonical(&c);
    let eng = Engine::new();
    let expected = eng.and(f.clone(), c.clone()).unwrap();
    assert_canonical(&expected);
    let mut g = eng.restrict_to_care(f.clone(), c.clone()).unwrap().into_tdd();
    eng.minimize(&mut g).unwrap();
    assert_canonical(&g);
    assert!(g.pair_count() < f.pair_count());
    let joined = eng.and(g, c).unwrap();
    assert_canonical(&joined);
    assert!(joined.equivalent(&expected).unwrap());
    assert_eq!(joined.model_count().unwrap(), care_rows.len().into());
}
