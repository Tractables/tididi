use super::*;
use crate::limits::LimitConfig;
use crate::query::{KeepAllColumns, KeepFrontier, ModelCounter, PinSemantics, Retention};
use crate::test_helpers::assert_canonical;
use crate::OperationError;

/// Compare every sparse-domain pin assignment with explicit Boolean enumeration.
fn sparse_counts<R: Retention>() {
    let eng = Engine::new();
    let vars = [VarId(19), VarId(2), VarId(71)];
    let mut rotated = Vtree::balanced_over(&vars);
    let root = rotated.root();
    crate::vtree::rotate::rotate_pointers(&mut rotated, root, crate::vtree::RotationKind::Left)
        .unwrap().commit(&mut rotated);
    for tree in [Vtree::balanced_over(&vars), Vtree::linear_from_order(&vars), rotated] {
        let tree = Arc::new(tree);
        for (f, kind) in [(Tdd::one(&tree), 0), (Tdd::zero(&tree), 1), (Tdd::clause(&tree, [3]), 2)] {
            assert_canonical(&f);
            for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
                let mut counter = ModelCounter::<R>::try_new_on(&eng, &f, semantics).unwrap();
                for code in (0..27).chain(std::iter::once(0)) {
                    let mut digits = code;
                    let pins = vars.map(|var| {
                        let pin = match digits % 3 { 0 => None, 1 => Some(false), _ => Some(true) };
                        digits /= 3;
                        counter.set_pin(var, pin).unwrap();
                        pin
                    });
                    let evidence = (0..8).filter(|&bits| {
                        let model = match kind { 0 => true, 1 => false, _ => bits & 2 != 0 };
                        model && pins.iter().enumerate().all(|(i, pin)| pin.is_none_or(|v| v == (bits & (1 << i) != 0)))
                    }).count();
                    let expected = match semantics {
                        PinSemantics::Evidence => evidence,
                        PinSemantics::Cofactor => evidence << pins.iter().filter(|p| p.is_some()).count(),
                    };
                    assert_eq!(counter.try_model_count_on(&eng).unwrap(), BigUint::from(expected));
                    assert_eq!(counter.try_model_count_on(&eng).unwrap(), BigUint::from(expected));
                }
            }
        }
    }
}

#[test]
fn sparse_and_reordered_variables_count_under_both_pin_conventions() {
    sparse_counts::<KeepAllColumns>();
    sparse_counts::<KeepFrontier>();
}

/// Reject holes and out-of-range IDs without disturbing cached or pending evidence.
fn invalid_pins<R: Retention>() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced_over(&[VarId(9), VarId(2)]));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let mut counter = ModelCounter::<R>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.try_model_count_on(&eng).unwrap(), 4u32.into());
    for pending in [false, true] {
        if pending { counter.set_pin(VarId(9), Some(false)).unwrap(); }
        for var in [VarId(0), VarId(3), VarId(10), VarId(u32::MAX)] {
            for value in [None, Some(false), Some(true)] {
                assert_eq!(counter.set_pin(var, value), Err(OperationError::VariableNotInVtree(var)));
            }
        }
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), if pending { 2u32.into() } else { 4u32.into() });
    }
    counter.set_pin(VarId(9), None).unwrap();
    assert_eq!(counter.try_model_count_on(&eng).unwrap(), 4u32.into());
}

#[test]
fn absent_pin_variables_leave_cached_and_pending_counts_unchanged() {
    invalid_pins::<KeepAllColumns>();
    invalid_pins::<KeepFrontier>();
}

/// Refuse summed-out variables while continuing to count pins on structural leaves.
fn marginal_pins<R: Retention>() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    for summed in [tree.children(tree.root()).0, tree.leaf_of(VarId(0)).unwrap()] {
        let mut f = Tdd::clause(&tree, [1, 3]);
        assert_canonical(&f);
        crate::marginal::marginalize_levels(&eng, &mut f, &[summed]).unwrap();
        minimize(&mut f);
        assert_canonical(&f);
        let mut counter = ModelCounter::<R>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 12u32.into());
        counter.set_pin(VarId(2), Some(false)).unwrap();
        for pin in [None, Some(false), Some(true)] {
            assert_eq!(counter.set_pin(VarId(0), pin), Err(OperationError::MarginalLevel(summed)));
        }
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 4u32.into());
        counter.set_pin(VarId(2), None).unwrap();
        assert_eq!(counter.try_model_count_on(&eng).unwrap(), 12u32.into());
    }
}

#[test]
fn summed_out_pin_variables_are_refused_without_losing_pending_pins() {
    marginal_pins::<KeepAllColumns>();
    marginal_pins::<KeepFrontier>();
}

#[test]
fn sparse_pin_storage_fits_a_budget_independent_of_variable_ids() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::leaf(VarId(100_000)));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(1024)));
    let mut counter = ModelCounter::<KeepAllColumns>::try_new_on(&eng, &f, PinSemantics::Evidence).unwrap();
    counter.set_pin(VarId(100_000), Some(true)).unwrap();
    assert_eq!(counter.try_model_count_on(&eng).unwrap(), 1u32.into());
    counter.set_pin(VarId(100_000), None).unwrap();
    assert_eq!(counter.try_model_count_on(&eng).unwrap(), 2u32.into());
}
