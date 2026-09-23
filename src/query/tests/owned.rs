use crate::{Engine, OperationError, Tdd, Vtree};
use crate::query::{OwnedEvaluator, OwnedModelCounter, PinSemantics, Retention};
use crate::diagram::RationalWeights;
use crate::test_helpers::assert_canonical;
use crate::vtree::VarId;
use std::sync::Arc;

#[test]
fn moved_queries_match_borrowed_queries_and_return_the_original_storage() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, -2, 3]).unwrap();
    assert_canonical(&f);
    for retention in [Retention::All, Retention::Frontier] {
        for convention in [PinSemantics::Evidence, PinSemantics::Cofactor] {
            let original = f.clone();
            let storage = original.levels.as_ptr();
            let mut owned: OwnedModelCounter = original.into_counter_with(retention, convention).unwrap();
            let mut borrowed = f.counter_with(retention, convention).unwrap();
            for code in 0..81 {
                let mut code = code;
                let pins: Vec<_> = (1..=4).map(|var| {
                    let value = match code % 3 { 0 => None, 1 => Some(false), _ => Some(true) };
                    code /= 3;
                    (VarId(var), value)
                }).collect();
                owned.set_pins(&pins).unwrap();
                borrowed.set_pins(&pins).unwrap();
                // Moving a live cache is safe: no field points into its owner.
                let mut moved = Box::new(owned);
                assert_eq!(moved.model_count().unwrap(), borrowed.model_count().unwrap());
                owned = *moved;
            }
            let recovered = owned.into_inner();
            assert_eq!(recovered.levels.as_ptr(), storage);
            assert_eq!(recovered.model_count().unwrap(), f.model_count().unwrap());
            assert_canonical(&recovered);
        }
    }
}

#[test]
fn owned_evaluation_survives_moves_weight_changes_and_failed_observations() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let weights = RationalWeights::unit(3);
    let mut borrowed = f.evaluator(weights.clone()).unwrap();
    let copy = f.clone();
    let storage = copy.levels.as_ptr();
    let owned: OwnedEvaluator<_> = copy.into_evaluator(weights.clone()).unwrap();
    let mut owned = Box::new(owned);
    for literals in [[1, -2], [-1, 2], [1, 2]] {
        borrowed.observe(literals).unwrap();
        owned.observe(literals).unwrap();
        assert_eq!(owned.value().unwrap(), borrowed.value().unwrap());
    }
    assert_eq!(owned.observe([-1, 9]), Err(OperationError::VariableNotInVtree(VarId(9))));
    assert_eq!(owned.value().unwrap(), borrowed.value().unwrap());
    owned.clear_pins();
    owned.replace_algebra(weights.clone());
    assert_eq!(owned.value().unwrap(), f.evaluate(&weights).unwrap());
    let recovered = (*owned).into_inner();
    assert_eq!(storage, recovered.levels.as_ptr());
    assert_canonical(&recovered);
}

#[test]
fn owned_queries_bind_to_limits_and_recover_after_every_reservation_refusal() {
    let vtree = Arc::new(Vtree::balanced(16));
    let f = Tdd::clause(&vtree, [1, 8, 16]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    let expected = f.model_count().unwrap();
    for evaluator in [false, true] {
        let mut completed = false;
        for refusal in 0..128 {
            let mut count = f.clone().into_counter().unwrap();
            let mut value = f.clone().into_evaluator(RationalWeights::unit(16)).unwrap();
            engine.limits().refuse_nth_reserve(refusal);
            let result = if evaluator { value.bind(&engine).value().map(|_| ()) }
                else { count.bind(&engine).model_count().map(|_| ()) };
            engine.limits().grant_every_reserve();
            assert_eq!(count.model_count().unwrap(), expected);
            assert_eq!(value.value().unwrap().to_integer(), expected.clone().into());
            if result.is_ok() { completed = true; break; }
            assert_eq!(result, Err(OperationError::OverBudget));
        }
        assert!(completed);
    }
}
