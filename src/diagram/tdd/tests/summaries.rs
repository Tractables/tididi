use super::*;
use crate::test_helpers::assert_canonical;

fn check(f: &Tdd) {
    assert_canonical(f);
    let nodes: usize = f.levels().iter().map(TddLevel::slot_count).sum();
    let width = f.levels().iter().map(TddLevel::slot_count).max().unwrap_or(0);
    let marginal = f.levels().iter().any(TddLevel::is_marginal);
    let pairs: usize = f.levels().iter().map(|l| (0..l.nodes().len()).map(|i| l.pair_count_at(i)).sum::<usize>()).sum();
    for _ in 0..2 {
        assert_eq!((f.node_count(), f.max_width(), f.has_marginal_level(), f.pair_count()), (nodes, width, marginal, pairs));
    }
}

#[test]
fn summaries_follow_consuming_edits_clones_and_marginalization() {
    let tree = Arc::new(Vtree::balanced(8));
    let eng = crate::Engine::new();
    let mut f = eng.clause(&tree, [1, 2, 3]).unwrap();
    check(&f);
    let original = f.clone();
    f = eng.and_clause(f, &[-1, 4]).unwrap();
    check(&f);
    check(&original);
    f = eng.condition_var(f, crate::vtree::VarId(4), false).unwrap();
    check(&f);
    let root = tree.root();
    eng.marginalize_levels(&mut f, &[root]).unwrap();
    eng.minimize(&mut f).unwrap();
    check(&f);
    assert!(f.has_marginal_level());
    check(&eng.zero(&tree));
}

#[test]
fn mutable_iteration_and_indexing_invalidate_cached_summaries() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&tree);
    check(&f);
    // Mutable access alone must forget a snapshot, even if a caller restores
    // identical levels before returning the diagram.
    let root = tree.root().idx();
    let level = f.levels[root].clone();
    f.levels[root] = level;
    check(&f);
    for level in &mut f.levels { let _ = level.slot_count(); }
    check(&f);
}
