use super::*;
use crate::execution::pool::{Pool, PoolGuard};
use crate::limits::Limits;

/// A mark buffer from `pool`, covering `num_nodes` levels.
fn take<'a>(lim: &'a Limits, pool: &'a Pool<MarkBuffer>, num_nodes: usize) -> Result<PoolGuard<'a, MarkBuffer>, OperationError> {
    let mut marks = pool.checkout(lim);
    marks.cover(lim, num_nodes)?;
    Ok(marks)
}

#[test]
fn interrupted_ancestor_walk_returns_clean_flags_for_retry() {
    let eng = Engine::new();
    let vtree = Vtree::balanced(4);
    let pool = Pool::default();
    let mut internal = Vec::new();
    let mut stack = Vec::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let clause = [crate::vtree::VarId(1), crate::vtree::VarId(4)].map(|var| (Literal::pos(var), vtree.leaf_of(var).unwrap()));
    {
        let mut flags = take(eng.limits(), &pool, vtree.num_nodes()).unwrap();
        let _scope = eng.limits().scope(crate::limits::LimitConfig::none().with_stop_rules(crate::limits::StopRules {
            unconditional: Some(crate::limits::StopAt::WorkUnits(3)),
            after_pairs: None,
        }));
        let result = build_clause_spine(eng.limits(), &vtree, &clause, &mut flags, &mut internal, &mut stack);
        assert_eq!(result, Err(OperationError::Stopped));
        assert!(flags.iter().any(|&flag| flag));
    }
    let mut flags = take(eng.limits(), &pool, vtree.num_nodes()).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
    build_clause_spine(eng.limits(), &vtree, &clause, &mut flags, &mut internal, &mut stack).unwrap();
    let mut expected = vec![false; vtree.num_nodes()];
    for (_, leaf) in clause {
        let mut current = Some(leaf);
        while let Some(t) = current {
            expected[t.idx()] = true;
            current = vtree.node(t).parent();
        }
    }
    assert_eq!(&**flags, &expected);
}

#[test]
fn a_stopped_walk_hands_back_flags_the_next_checkout_reuses() {
    let eng = Engine::new();
    let pool = Pool::default();
    let allocation = {
        let mut flags = take(eng.limits(), &pool, 100).unwrap();
        flags.mark(VtreeIdx(7));
        flags.mark(VtreeIdx(7));
        flags.as_ptr()
    };
    let _limit = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    let flags = take(eng.limits(), &pool, 100).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
    assert_eq!(flags.as_ptr(), allocation, "the warmed buffer is reused, not reallocated");
}

#[test]
fn mark_reports_first_visit_and_nested_checkouts_keep_independent_marks() {
    let eng = Engine::new();
    let pool = Pool::default();
    let mut outer = take(eng.limits(), &pool, 8).unwrap();
    assert!(outer.mark(VtreeIdx(3)));
    assert!(!outer.mark(VtreeIdx(3)));
    {
        let mut inner = take(eng.limits(), &pool, 8).unwrap();
        assert!(inner.mark(VtreeIdx(3)));
        assert!(inner.mark(VtreeIdx(5)));
    }
    assert!(outer[3]);
    assert!(!outer[5]);
    drop(outer);
    let reused = take(eng.limits(), &pool, 8).unwrap();
    assert!(reused.iter().all(|&flag| !flag));
}
