use std::sync::Arc;
use crate::{Engine, Tdd, Vtree, OperationError};
use crate::restructure::{EmbeddingPlan, GraftError};
use crate::vtree::VarId;
use crate::test_helpers::assert_canonical;

#[test]
fn composed_placements_match_sequential_and_direct_embeddings() {
    let source = Arc::new(Vtree::balanced_over(&[VarId(3), VarId(9)]).unwrap());
    let middle = Arc::new(Vtree::linear(4));
    let destination = Arc::new(Vtree::linear(8));
    let first = EmbeddingPlan::new(&source, &middle, |v| if v == VarId(3) { VarId(1) } else { VarId(3) }).unwrap();
    let second = EmbeddingPlan::new(&middle, &destination, |v| VarId(v.0 * 2)).unwrap();
    let both = first.then(&second).unwrap();
    for literals in [vec![], vec![3], vec![-9], vec![3, -9]] {
        let f = Tdd::clause(&source, literals).unwrap();
        assert_canonical(&f);
        let intermediate = first.apply(&f).unwrap();
        let sequential = second.apply(&intermediate).unwrap();
        let composed = both.apply(&f).unwrap();
        let (direct, levels) = f.embed(&destination, |v| if v == VarId(3) { VarId(2) } else { VarId(6) }).unwrap();
        for result in [&intermediate, &sequential, &composed, &direct] { assert_canonical(result); }
        assert!(composed.equivalent(&sequential).unwrap());
        assert!(composed.equivalent(&direct).unwrap());
        assert_eq!(both.levels().as_slice(), levels.as_slice());
        assert!(Arc::ptr_eq(composed.vtree(), &destination));
    }
}

#[test]
fn plans_validate_once_and_reject_wrong_sources_without_using_level_indices() {
    use std::cell::Cell;
    let source = Arc::new(Vtree::balanced(2));
    let destination = Arc::new(Vtree::balanced(4));
    let calls = Cell::new(0);
    let plan = EmbeddingPlan::new(&source, &destination, |var| { calls.set(calls.get() + 1); var }).unwrap();
    assert_eq!(calls.get(), 2);
    for literal in [1, -1, 2, -2] {
        let f = Tdd::clause(&source, [literal]).unwrap();
        let result = plan.apply(&f).unwrap();
        assert_canonical(&f);
        assert_canonical(&result);
    }
    assert_eq!(calls.get(), 2);
    let wrong = Tdd::one(&Arc::new(Vtree::balanced(2)));
    assert_canonical(&wrong);
    assert!(matches!(plan.apply(&wrong), Err(GraftError::Operation(OperationError::VtreeMismatch))));
    assert!(matches!(plan.then(&plan), Err(GraftError::Operation(OperationError::VtreeMismatch))));
    let mut marginal = Tdd::one(&source);
    marginal.marginalize_levels(&[source.root()]).unwrap();
    assert_canonical(&marginal);
    assert!(matches!(plan.apply(&marginal), Err(GraftError::Operation(OperationError::MarginalLevel(_)))));
}

#[test]
fn prepared_placements_recover_from_allocation_refusal_without_rebuilding_the_plan() {
    let source = Arc::new(Vtree::linear(3));
    let destination = Arc::new(Vtree::linear(7));
    let plan = EmbeddingPlan::new(&source, &destination, |v| VarId(v.0 * 2)).unwrap();
    let f = Tdd::clause(&source, [1, -2, 3]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    let mut completed = false;
    for refusal in 0..128 {
        engine.limits().refuse_nth_reserve(refusal);
        let result = engine.embed_with(&f, &plan);
        engine.limits().grant_every_reserve();
        let retry = engine.embed_with(&f, &plan).unwrap();
        assert_canonical(&retry);
        assert_eq!(retry.model_count().unwrap(), 112u32.into());
        match result {
            Ok(result) => { assert_canonical(&result); completed = true; break; }
            Err(error) => assert_eq!(error, GraftError::Operation(OperationError::OverBudget)),
        }
    }
    assert!(completed);
}
