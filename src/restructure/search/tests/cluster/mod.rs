use super::*;

mod deadline;

#[test]
fn invalid_root_preserves_diagram_and_attempts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = eng.one(&vtree);
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
    let mut tdd = eng.one(&vtree);
    let mut tried = vec![1];
    assert_eq!(eng.rotate_marginal_cluster(&mut tdd, vtree.root(), 8, &mut tried), Ok(0));
    assert_eq!(tried, [1, 0, 0]);
    assert!(Arc::ptr_eq(tdd.vtree(), &vtree));
    crate::test_helpers::assert_canonical(&tdd);
}
