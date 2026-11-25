use super::*;
use std::sync::Arc;

mod deadline;

#[test]
fn invalid_root_preserves_diagram_and_attempts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = eng.one(&crate::vtree::Home::unchecked(&vtree));
    let before = tdd.model_count().unwrap();
    let invalid = VtreeIdx(vtree.num_nodes() as u32);
    let mut tried = vec![1];
    assert_eq!(eng.rotate_marginal_cluster(&mut tdd, invalid, 8, &mut tried),
        Err(OperationError::LevelNotInVtree(invalid)));
    assert_eq!(tried, [1]);
    assert!(Arc::ptr_eq(tdd.vtree(), &vtree));
    assert_eq!(tdd.model_count().unwrap(), before);
    crate::test_helpers::assert_canonical(&tdd);
}

#[test]
fn short_attempt_storage_grows_without_losing_prior_entries() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = eng.one(&crate::vtree::Home::unchecked(&vtree));
    let mut tried = vec![1];
    assert_eq!(eng.rotate_marginal_cluster(&mut tdd, vtree.root(), 8, &mut tried), Ok(0));
    assert_eq!(tried, [1, 0, 0]);
    assert!(Arc::ptr_eq(tdd.vtree(), &vtree));
    crate::test_helpers::assert_canonical(&tdd);
}

/// Refusals after a committed rotation must keep its vtree with the rebuilt levels.
#[test]
fn allocation_refusals_preserve_cluster_counts() {
    let (mut original, root) = deadline::worthwhile_candidate_tdd();
    original.minimize().unwrap();
    crate::test_helpers::assert_canonical(&original);
    let expected = original.model_count().unwrap();
    let mut refused = 0;
    let mut completed = false;
    for cut in 0..1024 {
        let eng = Engine::new();
        let mut tdd = original.clone();
        let mut tried = vec![0; tdd.vtree().num_nodes()];
        eng.refuse_nth_reserve(cut);
        let result = eng.rotate_marginal_cluster(&mut tdd, root, 8, &mut tried);
        eng.limits().grant_every_reserve();
        assert_eq!(tdd.model_count().unwrap(), expected, "reservation {cut}");
        // A refused closure may leave valid reduction work pending.
        eng.minimize(&mut tdd).unwrap();
        crate::test_helpers::assert_canonical(&tdd);
        assert_eq!(tdd.model_count().unwrap(), expected, "after cleanup, reservation {cut}");
        match result {
            Err(OperationError::OverBudget) => refused += 1,
            Ok(1) => { completed = true; break; }
            other => panic!("unexpected result at reservation {cut}: {other:?}"),
        }
    }
    assert!(refused > 0 && completed);
}

/// The small fixture's rotation would shrink the diagram by 2 pairs, under
/// `CLUSTER_MIN_GAIN`: the pass tries it, declines it and leaves the diagram
/// as it was.
#[test]
fn a_rotation_that_saves_less_than_the_minimum_gain_is_declined() {
    let (mut tdd, root) = deadline::one_candidate_tdd();
    tdd.minimize().unwrap();
    crate::test_helpers::assert_canonical(&tdd);
    let count = tdd.model_count().unwrap();
    let pairs = tdd.pair_count();
    let eng = Engine::new();
    let mut tried = vec![0u8; tdd.vtree().num_nodes()];
    assert_eq!(eng.rotate_marginal_cluster(&mut tdd, root, 8, &mut tried), Ok(0));
    assert_eq!(tried[root.idx()], 0b01, "the left rotation at the root was tried");
    assert_eq!(tdd.pair_count(), pairs);
    assert_eq!(tdd.model_count().unwrap(), count);
    crate::test_helpers::assert_canonical(&tdd);
}
