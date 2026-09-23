use super::*;
use crate::{Engine, Vtree};

#[test]
fn canonical_guarantee_follows_storage_and_output() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    f.minimize().unwrap();
    assert!(f.levels.is_canonical(f.output));
    assert!(f.clone().levels.is_canonical(f.output));
    assert!(f.try_clone_on(&Engine::new()).unwrap().levels.is_canonical(f.output));
    let output = f.output;
    f.output.local = ZERO;
    assert!(!f.levels.is_canonical(f.output));
    f.output = output;
    assert!(f.levels.is_canonical(f.output));
    let _ = &mut f.levels[output.vtree.idx()];
    assert!(!f.levels.is_canonical(f.output));
    f.minimize().unwrap();
    assert!(f.levels.is_canonical(f.output));
    let same = vtree.clone();
    unsafe { f.reseat_vtree_unchecked(&same); }
    assert!(!f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn builder_and_marginal_storage_are_not_certified() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    f.minimize().unwrap();
    let mut rebuilt = Tdd::from_levels_unchecked(vtree.clone(), f.levels.clone().into_vec(), f.output);
    assert!(!rebuilt.levels.is_canonical(rebuilt.output));
    assert_eq!(rebuilt.support().unwrap(), f.support().unwrap());
    rebuilt.marginalize_levels(&[vtree.root()]).unwrap();
    rebuilt.minimize().unwrap();
    assert!(!rebuilt.levels.is_canonical(rebuilt.output));
}

#[test]
fn certified_leaf_queries_borrow_and_still_honor_cancellation() {
    use crate::OperationError;
    use crate::limits::LimitConfig;
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&vtree);
    f.minimize().unwrap();
    let unproven = Tdd::from_levels_unchecked(vtree, f.levels.clone().into_vec(), f.output);
    let eng = Engine::new();
    {
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        // A free-variable scan needs no result allocation; an unproven input needs a copy.
        assert!(eng.support(&f).unwrap().is_empty());
        assert!(eng.implied_literals(&f).unwrap().is_empty());
        assert_eq!(eng.support(&unproven), Err(OperationError::OverBudget));
    }
    let stopped = crate::test_helpers::stopping_engine();
    assert_eq!(stopped.support(&f), Err(OperationError::Stopped));
    assert_eq!(stopped.implied_literals(&f), Err(OperationError::Stopped));
    assert_eq!(stopped.equivalent(&f, &f.clone()), Err(OperationError::Stopped));
    assert!(f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn refused_minimization_does_not_certify_partial_work() {
    let vtree = Arc::new(Vtree::balanced(4));
    let source = Tdd::clause(&vtree, [1, 2, 3]).unwrap();
    let mut f = Tdd::from_levels_unchecked(vtree, source.levels.clone().into_vec(), source.output);
    let eng = Engine::new();
    eng.limits().refuse_nth_reserve(0);
    assert_eq!(eng.minimize(&mut f), Err(crate::OperationError::OverBudget));
    assert!(!f.levels.is_canonical(f.output));
    eng.limits().grant_every_reserve();
    eng.minimize(&mut f).unwrap();
    assert!(f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}
