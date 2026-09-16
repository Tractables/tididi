use super::*;
use crate::limits::pool::Pool;

#[test]
fn interrupted_ancestor_walk_returns_clean_flags_for_retry() {
    let eng = Engine::new();
    let tree = Vtree::balanced(4);
    let pool = Pool::default();
    let clause = [Literal::pos(crate::vtree::VarId(0)), Literal::pos(crate::vtree::VarId(3))];
    {
        let mut flags = ScopedFlags::take(eng.limits(), &pool, tree.num_nodes()).unwrap();
        let mut visited = 0;
        let result = mark_clause_levels_with(&tree, &clause, |t| flags.set(t), || {
            visited += 1;
            if visited == 3 { Err(OperationError::Stopped) } else { Ok(()) }
        });
        assert_eq!(result, Err(OperationError::Stopped));
        assert!(flags.iter().any(|&flag| flag));
    }
    let mut flags = ScopedFlags::take(eng.limits(), &pool, tree.num_nodes()).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
    mark_clause_levels_with(&tree, &clause, |t| flags.set(t), || Ok(())).unwrap();
    let mut expected = vec![false; tree.num_nodes()];
    for literal in clause {
        let mut current = Some(tree.leaf_of(literal.var).unwrap());
        while let Some(t) = current {
            expected[t.idx()] = true;
            current = tree.node(t).parent();
        }
    }
    assert_eq!(&*flags, &expected);
}
