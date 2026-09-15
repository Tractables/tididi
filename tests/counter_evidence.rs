use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::vtree::VarId;
use tididi::limits::{LimitConfig, StopCallback, StopDecision};
use tididi::query::{KeepAllColumns, KeepFrontier, PinSemantics, Retention};
use tididi::test_helpers::assert_canonical;

/// Count a three-variable clause independently for each observation convention.
fn expected_count(pins: [Option<bool>; 3], semantics: PinSemantics) -> usize {
    let evidence = (0..8).filter(|bits| {
        (bits & 1 != 0 || bits & 2 == 0)
            && pins.iter().enumerate().all(|(i, pin)| pin.is_none_or(|v| v == (bits & (1 << i) != 0)))
    }).count();
    match semantics {
        PinSemantics::Evidence => evidence,
        PinSemantics::Cofactor => evidence << pins.iter().filter(|p| p.is_some()).count(),
        _ => unreachable!("the fixture selects a known pin convention"),
    }
}

/// Exercise sparse variables, partial updates and duplicates against enumeration.
fn bulk_counts<R: Retention>() {
    let vars = [VarId(9), VarId(2), VarId(71)];
    let tree = Arc::new(Vtree::balanced_over(&vars));
    let f = Tdd::clause(&tree, [10, -3]).unwrap();
    assert_canonical(&f);
    for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
        let mut counter = f.counter_with::<R>(semantics).unwrap();
        assert_eq!(counter.model_count().unwrap(), 6u32.into());
        for code in 0..27 {
            let mut digits = code;
            let pins = vars.map(|_| {
                let pin = match digits % 3 { 0 => None, 1 => Some(false), _ => Some(true) };
                digits /= 3;
                pin
            });
            let updates = std::array::from_fn::<_, 3, _>(|i| (vars[i], pins[i]));
            counter.set_pins(&updates).unwrap();
            let expected = expected_count(pins, semantics);
            assert_eq!(counter.model_count().unwrap(), expected.into());
            counter.set_pins(&[]).unwrap();
            assert_eq!(counter.model_count().unwrap(), expected.into());

            counter.set_pins(&[(vars[0], Some(false)), (vars[0], None), (vars[0], Some(true))]).unwrap();
            assert_eq!(counter.model_count().unwrap(), expected_count([Some(true), pins[1], pins[2]], semantics).into());
            counter.clear_pins();
            assert_eq!(counter.model_count().unwrap(), 6u32.into());
            counter.clear_pins();
            assert_eq!(counter.model_count().unwrap(), 6u32.into());
        }
        counter.set_pins(&[(vars[0], Some(false)), (vars[2], Some(true))]).unwrap();
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), 6u32.into());
    }
}

#[test]
fn bulk_evidence_matches_enumeration_with_both_policies_and_semantics() {
    bulk_counts::<KeepAllColumns>();
    bulk_counts::<KeepFrontier>();
}

/// Reject a bad middle entry without applying neighboring updates or losing pending ones.
fn invalid_batches<R: Retention>() {
    let tree = Arc::new(Vtree::balanced_over(&[VarId(9), VarId(2), VarId(71)]));
    let f = Tdd::clause(&tree, [10, -3]).unwrap();
    assert_canonical(&f);
    for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
        for pending in [false, true] {
            for invalid in [VarId(0), VarId(3), VarId(72), VarId(u32::MAX)] {
                for invalid_pin in [None, Some(false), Some(true)] {
                    let mut counter = f.counter_with::<R>(semantics).unwrap();
                    counter.set_pin(VarId(9), Some(false)).unwrap();
                    assert_eq!(counter.model_count().unwrap(), expected_count([Some(false), None, None], semantics).into());
                    let pin = if pending { Some(true) } else { Some(false) };
                    if pending { counter.set_pin(VarId(9), pin).unwrap(); }
                    assert_eq!(counter.set_pins(&[
                        (VarId(9), pin.map(|v| !v)), (invalid, invalid_pin), (VarId(71), Some(false)),
                    ]), Err(OperationError::VariableNotInVtree(invalid)));
                    assert_eq!(counter.model_count().unwrap(), expected_count([pin, None, None], semantics).into());
                    counter.clear_pins();
                    assert_eq!(counter.model_count().unwrap(), 6u32.into());
                }
            }
        }
    }
}

#[test]
fn invalid_bulk_updates_preserve_cached_and_pending_evidence() {
    invalid_batches::<KeepAllColumns>();
    invalid_batches::<KeepFrontier>();
}

/// Reject summed-out variables in a batch while retaining live structural evidence.
fn marginal_batches<R: Retention>() {
    let tree = Arc::new(Vtree::balanced(4));
    for summed in [tree.children(tree.root()).0, tree.leaf_of(VarId(0)).unwrap()] {
        let mut f = Tdd::clause(&tree, [1, 3]).unwrap();
        assert_canonical(&f);
        f.marginalize_levels(&[summed]).unwrap();
        f.minimize().unwrap();
        assert_canonical(&f);
        for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
            let mut counter = f.counter_with::<R>(semantics).unwrap();
            assert_eq!(counter.model_count().unwrap(), 12u32.into());
            counter.set_pin(VarId(2), Some(false)).unwrap();
            for pin in [None, Some(false), Some(true)] {
                assert_eq!(counter.set_pins(&[(VarId(2), Some(true)), (VarId(0), pin)]),
                    Err(OperationError::MarginalLevel(summed)));
            }
            let expected = if semantics == PinSemantics::Evidence { 4u32 } else { 8u32 };
            assert_eq!(counter.model_count().unwrap(), expected.into());
            counter.clear_pins();
            assert_eq!(counter.model_count().unwrap(), 12u32.into());
        }
    }
}

#[test]
fn summed_out_bulk_updates_are_atomic_and_remaining_pins_can_be_cleared() {
    marginal_batches::<KeepAllColumns>();
    marginal_batches::<KeepFrontier>();
}

/// Check both owned and borrowed bindings while their engine refuses reads.
fn bound_batches<R: Retention>() {
    let tree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    let stop = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&stop);
    let config = LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
        if trigger.load(Ordering::Relaxed) { StopDecision::Stop } else { StopDecision::Continue }
    })));
    let _limits = engine.limits().scope(config);
    for semantics in [PinSemantics::Evidence, PinSemantics::Cofactor] {
        for owned in [false, true] {
            let mut persistent = f.counter_with::<R>(semantics).unwrap();
            let mut counter = if owned {
                engine.counter_with::<R>(&f, semantics).unwrap()
            } else {
                persistent.bind(&engine)
            };
            counter.set_pins(&[(VarId(0), Some(false)), (VarId(2), Some(true))]).unwrap();
            assert_eq!(counter.model_count().unwrap(), expected_count([Some(false), None, Some(true)], semantics).into());
            stop.store(true, Ordering::Relaxed);
            counter.set_pins(&[(VarId(1), Some(true))]).unwrap();
            assert_eq!(counter.set_pins(&[(VarId(0), Some(true)), (VarId(3), None)]),
                Err(OperationError::VariableNotInVtree(VarId(3))));
            assert_eq!(counter.model_count(), Err(OperationError::Stopped));
            stop.store(false, Ordering::Relaxed);
            assert_eq!(counter.model_count().unwrap(), 0u32.into());
            stop.store(true, Ordering::Relaxed);
            counter.clear_pins();
            assert_eq!(counter.model_count(), Err(OperationError::Stopped));
            stop.store(false, Ordering::Relaxed);
            assert_eq!(counter.model_count().unwrap(), 6u32.into());
            counter.set_pins(&[(VarId(0), Some(true))]).unwrap();
            drop(counter);
            if !owned {
                assert_eq!(persistent.model_count().unwrap(), expected_count([Some(true), None, None], semantics).into());
            }
        }
    }
}

#[test]
fn bulk_updates_preserve_bound_limits_and_survive_temporary_bindings() {
    bound_batches::<KeepAllColumns>();
    bound_batches::<KeepFrontier>();
}
