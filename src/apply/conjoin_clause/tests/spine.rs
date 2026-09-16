use super::*;
use crate::limits::pool::Pool;

#[test]
fn interrupted_ancestor_walk_returns_clean_flags_for_retry() {
    let eng = Engine::new();
    let vtree = Vtree::balanced(4);
    let pool = Pool::default();
    let mut internal = Vec::new();
    let mut stack = Vec::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let clause = [Literal::pos(crate::vtree::VarId(0)), Literal::pos(crate::vtree::VarId(3))];
    {
        let mut flags = SpineMarks::take(eng.limits(), &pool, vtree.num_nodes()).unwrap();
        let _scope = eng.limits().scope(crate::limits::LimitConfig::none().with_stop_rules(crate::limits::StopRules {
            unconditional: Some(crate::limits::StopAt::WorkUnits(3)),
            after_pairs: None,
        }));
        let result = build_clause_spine(eng.limits(), &vtree, &clause, &mut flags, &mut internal, &mut stack);
        assert_eq!(result, Err(OperationError::Stopped));
        assert!(flags.iter().any(|&flag| flag));
    }
    let mut flags = SpineMarks::take(eng.limits(), &pool, vtree.num_nodes()).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
    build_clause_spine(eng.limits(), &vtree, &clause, &mut flags, &mut internal, &mut stack).unwrap();
    let mut expected = vec![false; vtree.num_nodes()];
    for literal in clause {
        let mut current = Some(vtree.leaf_of(literal.var).unwrap());
        while let Some(t) = current {
            expected[t.idx()] = true;
            current = vtree.node(t).parent();
        }
    }
    assert_eq!(&*flags, &expected);
}


#[test]
fn warmed_flags_reuse_the_rollback_log_after_unwind() {
    let eng = Engine::new();
    let pool = Pool::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut flags = SpineMarks::take(eng.limits(), &pool, 100).unwrap();
        flags.set(VtreeIdx(7));
        flags.set(VtreeIdx(7));
        panic!("marking stopped");
    }));
    assert!(result.is_err());
    let _limit = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    let flags = SpineMarks::take(eng.limits(), &pool, 100).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
}

#[test]
fn set_reports_first_visit_and_nested_scopes_keep_independent_marks() {
    let eng = Engine::new();
    let pool = Pool::default();
    let mut outer = SpineMarks::take(eng.limits(), &pool, 8).unwrap();
    assert!(outer.set(VtreeIdx(3)));
    assert!(!outer.set(VtreeIdx(3)));
    {
        let mut inner = SpineMarks::take(eng.limits(), &pool, 8).unwrap();
        assert!(inner.set(VtreeIdx(3)));
        assert!(inner.set(VtreeIdx(5)));
    }
    assert!(outer[3]);
    assert!(!outer[5]);
    drop(outer);
    let reused = SpineMarks::take(eng.limits(), &pool, 8).unwrap();
    assert!(reused.iter().all(|&flag| !flag));
}
