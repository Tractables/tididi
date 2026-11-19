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
    eng.refuse_nth_reserve(0);
    assert_eq!(eng.minimize(&mut f), Err(crate::OperationError::OverBudget));
    assert!(!f.levels.is_canonical(f.output));
    eng.limits().grant_every_reserve();
    eng.minimize(&mut f).unwrap();
    assert!(f.levels.is_canonical(f.output));
    crate::test_helpers::assert_canonical(&f);
}

/// A diagram on a projection reseated on a second allocation of that
/// projection stays on the leaves it uses and names none true, and conjoins
/// with a diagram on the first allocation.
#[test]
fn a_diagram_reseated_on_another_allocation_of_its_projection_stays_closed() {
    use crate::test_helpers::check::check_trimmed;
    use crate::vtree::{Home, VarId};
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(5));
    let mut f = eng.xor(eng.literal(home.vtree(), 1).unwrap(), eng.clause(home.vtree(), [3, -4]).unwrap()).unwrap();
    let first = Arc::clone(f.vtree());
    assert_eq!(first.num_leaves(), 3);
    let second = Vtree::projection(&home, [1, 3, 4].map(VarId)).unwrap();
    assert!(!Arc::ptr_eq(&first, &second), "the test needs two allocations");
    check_trimmed(&f, true).unwrap();
    let before = f.clone();
    // Safety: `second` is `first` built again: same nodes, indices,
    // variables and home.
    unsafe { f.reseat_vtree_unchecked(&second) };
    assert!(Arc::ptr_eq(f.vtree(), &second));
    check_trimmed(&f, true).unwrap();
    let same = eng.and(f.clone(), before.clone()).unwrap();
    check_trimmed(&same, true).unwrap();
    assert!(eng.equivalent(&same, &before).unwrap());
    let g = eng.negate(before).unwrap();
    assert!(Arc::ptr_eq(g.vtree(), &first));
    assert!(eng.and(f, g).unwrap().is_zero());
}

/// In a debug build, reseating a diagram on a home onto a vtree off every
/// home is refused.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "off every home")]
fn reseating_a_diagram_on_a_home_off_the_home_panics_in_debug() {
    use crate::vtree::Home;
    let home = Home::new(Vtree::balanced(3));
    let mut f = Tdd::clause(home.vtree(), [1, 2, 3]).unwrap();
    let bare = Arc::new(Vtree::balanced(3));
    // Safety: never reached past the debug check.
    unsafe { f.reseat_vtree_unchecked(&bare) };
}
