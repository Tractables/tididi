use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::diagram::Tdd;
use crate::Engine;
use crate::restructure::relevel::RestructureScratch;
use crate::restructure::search::probe::{probe, ProbeRule};
use crate::restructure::search::RotationProbe;
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};

struct Panicking;

impl ProbeRule for Panicking {
    fn keeps(&mut self, _: &RotationProbe<'_>, _: &crate::vtree::rotate::RotationInfo) -> bool {
        panic!("objective failed")
    }
}

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

impl ProbeRule for Reject {
    fn keeps(&mut self, _: &RotationProbe<'_>, _: &crate::vtree::rotate::RotationInfo) -> bool { false }
}

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
