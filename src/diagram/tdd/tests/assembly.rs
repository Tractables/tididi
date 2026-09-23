use super::*;
use crate::{Engine, OperationError};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;

#[test]
fn checked_assembly_charges_worklists_even_when_levels_are_already_allocated() {
    let tree = Arc::new(Vtree::balanced(4));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let eng = Engine::new();
    let output = source.output;
    let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(Tdd::try_from_levels_on(&eng, tree, source.levels.into_vec(), output).err(), Some(OperationError::OverBudget));
}

#[test]
fn checked_assembly_polls_while_seeding_worklists() {
    let tree = Arc::new(Vtree::linear(16));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let output = source.output;
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(3)), ..StopRules::default()
    }));
    assert_eq!(Tdd::try_from_levels_on(&eng, tree, source.levels.into_vec(), output).err(), Some(OperationError::Stopped));
    assert_eq!(eng.limits().meters().work_units, 3);
}

#[test]
fn operation_assembly_owns_weights_and_recycles_refused_results() {
    use crate::diagram::{Assembly, Arithmetic, RationalWeights, WeightStore};
    let vtree = Arc::new(Vtree::balanced(4));
    let mut source = Tdd::clause(&vtree, [1, -2]).unwrap();
    source.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    assert_canonical(&source);
    let output = source.output();
    for refuse in [false, true] {
        let eng = Engine::new();
        let copy = source.clone();
        let assembly = Assembly::from_levels(&eng, Arc::clone(&vtree), copy.levels.into_vec(), copy.weights);
        let result = if refuse {
            let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            assembly.finish(output)
        } else {
            assembly.finish(output)
        };
        if refuse {
            assert_eq!(result.err(), Some(OperationError::OverBudget));
            assert_eq!(eng.levels().occupancy(), 1);
            let fresh = Assembly::new(&eng, &vtree).unwrap();
            for t in vtree.bottomup() {
                assert_eq!(fresh.level(t).slot_count(), 0);
                assert!(!fresh.level(t).is_marginal());
            }
        } else {
            let result = result.unwrap();
            assert_canonical(&result);
            assert_eq!(result.weighted_value().unwrap().unwrap().into_rational(),
                source.weighted_value().unwrap().unwrap().into_rational());
            assert!(!result.dirty.is_empty());
            assert_eq!(eng.levels().occupancy(), 0, "finished arenas belong to the diagram");
        }
    }
}

#[test]
fn unwinding_discards_unfinished_output_storage() {
    use crate::diagram::Assembly;
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _output = Assembly::new(&eng, &vtree).unwrap();
        panic!("interrupted output construction");
    }));
    assert!(result.is_err());
    assert_eq!(eng.levels().occupancy(), 0);
}
