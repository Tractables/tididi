//! What the rotation rebuild does when it cannot have the memory it needs.

use std::sync::Arc;

use crate::limits::{LimitConfig, OperationError};
use crate::restructure::relevel::RestructureScratch;
use crate::restructure::search::RotationProbe;
use crate::restructure::search::probe::{ProbeRule, probe};
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};
use crate::diagram::Tdd;
use crate::{Engine, and};

/// Scores every rotation an improvement, so a refusal is the only way the
/// probe can come back without keeping one.
struct Accept;

impl ProbeRule for Accept {
    fn keeps(&mut self, _: &RotationProbe<'_>, _: &crate::vtree::rotate::RotationInfo) -> bool { true }
}

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
    let refused = probe(
        &eng,
        &mut f,
        vtree.root(),
        RotationKind::Left,
        &mut Accept,
        &mut RestructureScratch::default(),
        usize::MAX,
    );
    assert_eq!(refused, Err(OperationError::OverBudget));

    // A refused probe is a probe that did not happen: the two levels it would
    // have rebuilt are still the ones it read.
    assert_eq!(format!("{f:?}"), before);
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_eq!(f.model_count().unwrap(), count);
    assert_canonical(&f);
}
