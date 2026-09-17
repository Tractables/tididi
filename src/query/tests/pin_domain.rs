use super::*;
use crate::limits::LimitConfig;
use crate::query::{KeepAllColumns, KeepFrontier, PinSemantics, Retention};
use crate::test_helpers::assert_canonical;
use crate::OperationError;

/// Compare every sparse-domain pin assignment with explicit Boolean enumeration.
fn sparse_counts<R: Retention>() {
    let eng = Engine::new();
    let vars = [VarId(20), VarId(3), VarId(72)];
    let mut rotated = Vtree::balanced_over(&vars).unwrap();
    let root = rotated.root();
    crate::vtree::rotate::rotate_pointers(&mut rotated, root, crate::vtree::RotationKind::Left)
        .unwrap().commit(&mut rotated);
    for tree in [Vtree::balanced_over(&vars).unwrap(), Vtree::linear_from_order(&vars).unwrap(), rotated] {
        let tree = Arc::new(tree);
        for (f, kind) in [(Tdd::one(&tree), 0), (Tdd::zero(&tree), 1), (Tdd::clause(&tree, [3]).unwrap(), 2)] {
            assert_canonical(&f);
            for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
                let mut counter = eng.counter_with::<R>(&f, semantics).unwrap();
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
                    assert_eq!(counter.model_count().unwrap(), BigUint::from(expected));
                    assert_eq!(counter.model_count().unwrap(), BigUint::from(expected));
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
    let tree = Arc::new(Vtree::balanced_over(&[VarId(10), VarId(3)]).unwrap());
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let mut counter = eng.counter_with::<R>(&f, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.model_count().unwrap(), 4u32.into());
    for pending in [false, true] {
        if pending { counter.set_pin(VarId(10), Some(false)).unwrap(); }
        for var in [VarId(1), VarId(4), VarId(11), VarId(u32::MAX)] {
            for value in [None, Some(false), Some(true)] {
                assert_eq!(counter.set_pin(var, value), Err(OperationError::VariableNotInVtree(var)));
            }
        }
        assert_eq!(counter.model_count().unwrap(), if pending { 2u32.into() } else { 4u32.into() });
    }
    counter.set_pin(VarId(10), None).unwrap();
    assert_eq!(counter.model_count().unwrap(), 4u32.into());
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
    for summed in [tree.children(tree.root()).0, tree.leaf_of(VarId(1)).unwrap()] {
        let mut f = Tdd::clause(&tree, [1, 3]).unwrap();
        assert_canonical(&f);
        eng.marginalize_levels(&mut f, &[summed]).unwrap();
        f.minimize().unwrap();
        assert_canonical(&f);
        let mut counter = eng.counter_with::<R>(&f, PinSemantics::Evidence).unwrap();
        assert_eq!(counter.model_count().unwrap(), 12u32.into());
        counter.set_pin(VarId(3), Some(false)).unwrap();
        for pin in [None, Some(false), Some(true)] {
            assert_eq!(counter.set_pin(VarId(1), pin), Err(OperationError::MarginalLevel(summed)));
        }
        assert_eq!(counter.observe([3, 1]), Err(OperationError::MarginalLevel(summed)));
        assert_eq!(counter.model_count().unwrap(), 4u32.into());
        counter.set_pin(VarId(3), None).unwrap();
        assert_eq!(counter.model_count().unwrap(), 12u32.into());
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
    let mut counter = eng.counter_with::<KeepAllColumns>(&f, PinSemantics::Evidence).unwrap();
    counter.set_pin(VarId(100_000), Some(true)).unwrap();
    assert_eq!(counter.model_count().unwrap(), 1u32.into());
    counter.set_pin(VarId(100_000), None).unwrap();
    assert_eq!(counter.model_count().unwrap(), 2u32.into());
}

/// Check signed observations against enumeration, including atomic rejection.
fn observed_counts<R: Retention>() {
    let vtree = Arc::new(Vtree::balanced_over(&[VarId(10), VarId(3), VarId(72)]).unwrap());
    let f = Tdd::clause(&vtree, [3, 10]).unwrap();
    assert_canonical(&f);
    for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
        let mut counter = f.counter_with::<R>(semantics).unwrap();
        for code in 0..8 {
            let pins = [10, 3, 72].map(|literal| {
                let bit = match literal { 10 => 0, 3 => 1, _ => 2 };
                if code & (1 << bit) != 0 { literal } else { -literal }
            });
            counter.observe(pins).unwrap();
            let satisfying = code & 3 != 0;
            let expected = if satisfying {
                if semantics == PinSemantics::Evidence { 1u32 } else { 8 }
            } else { 0 };
            assert_eq!(counter.model_count().unwrap(), expected.into());
        }
        counter.clear_pins();
        counter.observe([3, -3, 10]).unwrap();
        let expected: BigUint = if semantics == PinSemantics::Evidence { 2u32 } else { 8 }.into();
        assert_eq!(counter.model_count().unwrap(), expected);
        counter.observe([]).unwrap();
        assert_eq!(counter.model_count().unwrap(), expected);
        for pending in [false, true] {
            if pending { counter.observe([-72]).unwrap(); }
            for (input, error) in [
                (0, OperationError::InvalidLiteral(0)),
                (1, OperationError::VariableNotInVtree(VarId(1))),
                (i32::MIN, OperationError::VariableNotInVtree(VarId(1 << 31))),
            ] {
                assert_eq!(counter.observe([-10, input]), Err(error));
            }
            let expected: BigUint = if semantics == PinSemantics::Cofactor { 8u32 } else if pending { 1 } else { 2 }.into();
            assert_eq!(counter.model_count().unwrap(), expected);
        }
        let eng = Engine::new();
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        counter.bind(&eng).observe([10]).unwrap();
    }
}

#[test]
fn signed_observations_count_and_reject_invalid_batches_atomically() {
    observed_counts::<KeepAllColumns>();
    observed_counts::<KeepFrontier>();
}
