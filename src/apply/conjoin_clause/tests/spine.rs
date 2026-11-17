use super::*;
use crate::execution::pool::{Pool, PoolGuard};

/// A working set from `pool`, its spine marks covering `num_nodes` levels.
fn take<'a>(eng: &'a Engine, pool: &'a Pool<ClauseWork>, num_nodes: usize) -> Result<PoolGuard<'a, ClauseWork>, OperationError> {
    let mut work = pool.checkout(eng);
    work.on_spine.cover(eng.limits(), num_nodes)?;
    Ok(work)
}

#[test]
fn interrupted_ancestor_walk_returns_clean_flags_for_retry() {
    let eng = Engine::new();
    let vtree = Vtree::balanced(4);
    let pool = Pool::default();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let clause = [crate::vtree::VarId(1), crate::vtree::VarId(4)].map(|var| (Literal::pos(var), vtree.leaf_of(var).unwrap()));
    {
        let mut work = take(&eng, &pool, vtree.num_nodes()).unwrap();
        let w = &mut *work;
        let _scope = eng.limits().scope(crate::limits::LimitConfig::none().with_stop_rules(crate::limits::StopRules {
            unconditional: Some(crate::limits::StopAt::WorkUnits(3)),
            after_pairs: None,
        }));
        let result = build_clause_spine(eng.limits(), &vtree, &clause, &mut w.on_spine, &mut w.spine_internal);
        assert_eq!(result, Err(OperationError::Stopped));
        assert!(w.on_spine.iter().any(|&flag| flag));
    }
    let mut work = take(&eng, &pool, vtree.num_nodes()).unwrap();
    let w = &mut *work;
    assert!(w.on_spine.iter().all(|&flag| !flag));
    build_clause_spine(eng.limits(), &vtree, &clause, &mut w.on_spine, &mut w.spine_internal).unwrap();
    let mut expected = vec![false; vtree.num_nodes()];
    for (_, leaf) in clause {
        let mut current = Some(leaf);
        while let Some(t) = current {
            expected[t.idx()] = true;
            current = vtree.node(t).parent();
        }
    }
    assert_eq!(&w.on_spine[..], &expected[..]);
}

#[test]
fn a_stopped_walk_hands_back_flags_the_next_checkout_reuses() {
    let eng = Engine::new();
    let pool = Pool::default();
    let allocation = {
        let mut work = take(&eng, &pool, 100).unwrap();
        work.on_spine.mark(VtreeIdx(7));
        work.on_spine.mark(VtreeIdx(7));
        work.on_spine.as_ptr()
    };
    let _limit = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    let work = take(&eng, &pool, 100).unwrap();
    assert!(work.on_spine.iter().all(|&flag| !flag));
    assert_eq!(work.on_spine.as_ptr(), allocation, "the warmed buffer is reused, not reallocated");
}

#[test]
fn mark_reports_first_visit_and_nested_checkouts_keep_independent_marks() {
    let eng = Engine::new();
    let pool = Pool::default();
    let mut outer = take(&eng, &pool, 8).unwrap();
    assert!(outer.on_spine.mark(VtreeIdx(3)));
    assert!(!outer.on_spine.mark(VtreeIdx(3)));
    {
        let mut inner = take(&eng, &pool, 8).unwrap();
        assert!(inner.on_spine.mark(VtreeIdx(3)));
        assert!(inner.on_spine.mark(VtreeIdx(5)));
    }
    assert!(outer.on_spine[3]);
    assert!(!outer.on_spine[5]);
    drop(outer);
    let reused = take(&eng, &pool, 8).unwrap();
    assert!(reused.on_spine.iter().all(|&flag| !flag));
}
