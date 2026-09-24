//! What the rotation rebuild does when it cannot have the memory it needs.

use std::sync::Arc;

use crate::limits::{LimitConfig, OperationError};
use crate::restructure::search::RotationMove;
use crate::restructure::search::probe::rotate_if_on;
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};
use crate::diagram::Tdd;
use crate::{Engine, and};

#[test]
fn a_rebuild_that_does_not_fit_the_budget_is_refused_not_taken() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = and(Tdd::clause(&vtree, [1, 2, 3]).unwrap(), Tdd::clause(&vtree, [4, 5, 6]).unwrap())
        .unwrap();
    eng.minimize(&mut f).unwrap();
    assert_canonical(&f);
    let before = format!("{f:?}");
    let count = f.model_count().unwrap();

    // One byte is less than the expansion's first buffer, so the refusal comes
    // from the budget rather than from the host running out of memory.
    let _budget = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(1)));
    let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    // The closure keeps whatever it is shown, so a refusal is the only way
    // the probe can come back without keeping the rotation.
    let refused = rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| true);
    assert_eq!(refused, Err(OperationError::OverBudget));

    // A refused probe is a probe that did not happen: the two levels it would
    // have rebuilt are still the ones it read.
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_eq!(f.model_count().unwrap(), count);
    assert_canonical(&f);
}
