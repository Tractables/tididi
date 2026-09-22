use std::sync::Arc;
use super::*;
use crate::limits::LimitConfig;
use crate::test_helpers::assert_canonical;
use crate::{Engine, Vtree};

/// Seed 2 draws the root of balanced(4), where either rotation applies.
#[test]
fn restart_rotations_use_the_callers_memory_budget() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2, 3]).unwrap();
    assert_canonical(&f);
    let before = format!("{f:?}");
    let count = f.model_count().unwrap();
    let eng = Engine::new();
    {
        let _budget = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(1)));
        assert_eq!(kick(&eng, &mut f, 1, usize::MAX, &mut Lcg::new(2)), Err(OperationError::OverBudget));
    }
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_eq!(f.model_count().unwrap(), count);
    assert_canonical(&f);

    kick(&eng, &mut f, 1, usize::MAX, &mut Lcg::new(2)).unwrap();
    assert!(!f.vtree().same_tree(&vtree));
    assert_eq!(f.model_count().unwrap(), count);
    assert_canonical(&f);
}

/// An explicit pair bound must apply to the perturbation, not just its later descent.
#[test]
fn restart_rotations_respect_the_pair_bound() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2, 3]).unwrap();
    assert_canonical(&f);
    let before = format!("{f:?}");
    let eng = Engine::new();
    kick(&eng, &mut f, 1, 0, &mut Lcg::new(2)).unwrap();
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_canonical(&f);
}
