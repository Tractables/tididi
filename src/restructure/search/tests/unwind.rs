use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::diagram::{Tdd, TddLevel};
use crate::Engine;
use crate::restructure::relevel::RestructureScratch;
use crate::restructure::search::probe::{probe, ProbeRule};
use crate::restructure::search::RotationObjective;
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};

struct Panicking;

impl RotationObjective for Panicking {
    fn delta(&mut self, _: (&TddLevel, &TddLevel), _: (&TddLevel, &TddLevel)) -> i64 {
        panic!("objective failed")
    }
}

impl ProbeRule for Panicking {}

#[test]
fn an_objective_panic_restores_the_rotation_trial() {
    for kind in [RotationKind::Left, RotationKind::Right] {
        let eng = Engine::new();
        let vtree = Arc::new(Vtree::balanced(4));
        let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
        assert_canonical(&f);
        let before = format!("{f:?}");
        let count = f.model_count().unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            probe(&eng, &mut f, vtree.root(), kind, &mut Panicking, &mut RestructureScratch::default(), usize::MAX)
        }));
        assert!(result.is_err());
        assert_eq!(format!("{f:?}"), before);
        assert!(Arc::ptr_eq(f.vtree(), &vtree));
        assert_eq!(f.model_count().unwrap(), count);
        assert_canonical(&f);
    }
}

struct Reject;

impl RotationObjective for Reject {
    fn delta(&mut self, _: (&TddLevel, &TddLevel), _: (&TddLevel, &TddLevel)) -> i64 { 0 }
}

impl ProbeRule for Reject {}

#[test]
fn a_rejected_rotation_preserves_the_shared_vtree_and_worklists() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    assert_canonical(&f);
    let before = format!("{f:?}");
    assert!(!probe(&eng, &mut f, vtree.root(), RotationKind::Left, &mut Reject, &mut RestructureScratch::default(), usize::MAX).unwrap());
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_canonical(&f);
}

struct AcceptThenFail(bool);

impl RotationObjective for AcceptThenFail {
    fn delta(&mut self, _: (&TddLevel, &TddLevel), _: (&TddLevel, &TddLevel)) -> i64 { -1 }
}

impl ProbeRule for AcceptThenFail {
    fn on_accept(&mut self, _: &Engine, _: &mut Tdd, _: &crate::vtree::rotate::RotationInfo) -> Result<(), crate::OperationError> {
        if self.0 { panic!("accepted callback failed"); }
        Err(crate::OperationError::Stopped)
    }
}

#[test]
fn acceptance_is_committed_before_a_failing_callback() {
    for panic in [false, true] {
        let eng = Engine::new();
        let tree = Arc::new(Vtree::balanced(4));
        let mut f = Tdd::clause(&tree, [1, 2]).unwrap() & Tdd::clause(&tree, [3, 4]).unwrap();
        assert_canonical(&f);
        let count = f.model_count().unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            probe(&eng, &mut f, tree.root(), RotationKind::Left, &mut AcceptThenFail(panic), &mut RestructureScratch::default(), usize::MAX)
        }));
        if panic { assert!(result.is_err()); }
        else { assert!(matches!(result, Ok(Err(crate::OperationError::Stopped)))); }
        assert_ne!(f.vtree().to_text(), tree.to_text());
        assert_eq!(f.model_count().unwrap(), count);
        assert_canonical(&f);
    }
}
