use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::diagram::Tdd;
use crate::Engine;
use crate::restructure::search::RotationMove;
use crate::restructure::search::probe::rotate_if_on;
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};

#[test]
fn an_objective_panic_restores_the_rotation_trial() {
    for kind in [RotationKind::Left, RotationKind::Right] {
        let eng = Engine::new();
        let vtree = Arc::new(Vtree::balanced(4));
        let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
        assert_canonical(&f);
        let before = format!("{f:?}");
        let count = f.model_count().unwrap();
        let turn = RotationMove { pivot: vtree.root(), kind };
        let result = catch_unwind(AssertUnwindSafe(|| {
            rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| panic!("objective failed"))
        }));
        assert!(result.is_err());
        assert_eq!(format!("{f:?}"), before);
        assert!(Arc::ptr_eq(f.vtree(), &vtree));
        assert_eq!(f.model_count().unwrap(), count);
        assert_canonical(&f);
    }
}

#[test]
fn a_rejected_rotation_preserves_the_shared_vtree_and_worklists() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    assert_canonical(&f);
    let before = format!("{f:?}");
    let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    assert!(!rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| false).unwrap());
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_canonical(&f);
}
