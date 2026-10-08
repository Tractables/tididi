use super::*;
use std::sync::Arc;
use crate::Vtree;
use crate::test_helpers::assert_canonical;

#[test]
fn interrupted_ancestor_growth_leaves_room_for_further_pin_updates() {
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    let mut completed = false;
    for reserve in 0..16 {
        let mut counter = diagram.counter().unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        counter.set_pin(VarId(1), Some(false)).unwrap();
        engine.limits().refuse_nth_reserve(reserve);
        let result = counter.bind(&engine).model_count();
        engine.limits().grant_every_reserve();
        let capacity = counter.cache.observations.changed.capacity();
        counter.clear_pins();
        let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(v % 2 == 0))).collect();
        counter.set_pins(&pins).unwrap();
        assert_eq!(counter.cache.observations.changed.capacity(), capacity, "pin updates must reuse reserved storage");
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        match result {
            Ok(_) => { completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed);
}

#[test]
fn panicking_refresh_keeps_dirty_membership_reusable() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    engine.limits().pin_reduce_poll_stride(Some(1));
    let mut completed = false;
    for cut in 0..128 {
        let mut counter = diagram.counter().unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        counter.set_pin(VarId(1), Some(false)).unwrap();
        let calls = AtomicUsize::new(0);
        let result = {
            let _limits = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(move |_, _| {
                    assert_ne!(calls.fetch_add(1, Ordering::Relaxed), cut, "interrupted counter refresh");
                    StopDecision::Continue
                }))));
            catch_unwind(AssertUnwindSafe(|| counter.bind(&engine).model_count().unwrap()))
        };
        let capacity = counter.cache.observations.changed.capacity();
        counter.clear_pins();
        let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(false))).collect();
        counter.set_pins(&pins).unwrap();
        assert_eq!(counter.cache.observations.changed.capacity(), capacity);
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
        counter.set_pin(VarId(1), None).unwrap();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(2u32));
        counter.clear_pins();
        assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
        if result.is_ok() {completed = true; break;}
    }
    assert!(completed);
}

#[test]
fn frontier_evidence_invalidates_without_retaining_a_dirty_worklist() {
    let tree = Arc::new(Vtree::balanced(16));
    let diagram = Tdd::one(&tree);
    assert_canonical(&diagram);
    let engine = Engine::new();
    let mut counter = diagram.counter_with(Retention::Frontier, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.cache.observations.changed.capacity(), 0);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
    let pins: Vec<_> = (1..=16).map(|v| (VarId(v), Some(false))).collect();
    counter.set_pins(&pins).unwrap();
    assert!(!counter.cache.observations.evaluated);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32));
    // Repeating observations leaves the root cached, even if allocations would be refused.
    engine.limits().refuse_nth_reserve(0);
    counter.set_pins(&pins).unwrap();
    assert!(counter.cache.observations.evaluated);
    assert_eq!(counter.bind(&engine).model_count().unwrap(), BigUint::from(1u32));
    engine.limits().grant_every_reserve();
    counter.clear_pins();
    assert!(!counter.cache.observations.evaluated);
    assert_eq!(counter.model_count().unwrap(), BigUint::from(1u32) << 16);
    counter.clear_pins();
    assert!(counter.cache.observations.evaluated);
    assert_eq!(counter.cache.observations.changed.capacity(), 0);
    assert!(counter.cache.observations.pins.iter().all(|pin| !pin.dirty));
}

/// The table a fresh counter gives one assignment at a time.
fn one_at_a_time(f: &Tdd, vars: &[VarId], retention: Retention, convention: PinSemantics) -> Vec<BigUint> {
    (0..1usize << vars.len())
        .map(|assignment| {
            let mut counter = f.counter_with(retention, convention).unwrap();
            let pins: Vec<_> = vars.iter().enumerate().map(|(bit, &v)| (v, Some((assignment >> bit) & 1 == 1))).collect();
            counter.set_pins(&pins).unwrap();
            counter.model_count().unwrap()
        })
        .collect()
}

#[test]
fn a_count_table_matches_counting_each_assignment_alone() {
    use crate::test_helpers::random_diagrams;
    for (i, f) in random_diagrams(17, 24, 2..7).iter().enumerate() {
        let leaves: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, var)| var).collect();
        // Up to three variables, in an order other than the vtree's.
        let vars: Vec<VarId> = leaves.iter().rev().step_by(2).take(3).copied().collect();
        for retention in [Retention::All, Retention::Frontier] {
            for convention in [PinSemantics::Evidence, PinSemantics::Cofactor] {
                let mut counter = f.counter_with(retention, convention).unwrap();
                let table = counter.count_table(&vars).unwrap();
                assert_eq!(table, one_at_a_time(f, &vars, retention, convention), "diagram {i}, {retention:?}, {convention:?}");
            }
        }
    }
}

#[test]
fn a_count_table_keeps_the_other_pins_and_restores_its_own() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, -2, 3, 4]).unwrap();
    assert_canonical(&f);
    let mut counter = f.counter().unwrap();
    counter.set_pins(&[(VarId(4), Some(false)), (VarId(2), Some(true))]).unwrap();
    let before = counter.model_count().unwrap();
    // Variable 4 stays false throughout; variable 2 is pinned by the table.
    let table = counter.count_table(&[VarId(2), VarId(3)]).unwrap();
    let expected: Vec<BigUint> = [(false, false), (true, false), (false, true), (true, true)]
        .iter()
        .map(|&(x2, x3)| {
            let mut alone = f.counter().unwrap();
            alone.set_pins(&[(VarId(4), Some(false)), (VarId(2), Some(x2)), (VarId(3), Some(x3))]).unwrap();
            alone.model_count().unwrap()
        })
        .collect();
    assert_eq!(table, expected);
    assert_eq!(counter.model_count().unwrap(), before, "variable 2 is pinned true again and variable 3 is unpinned");
    counter.set_pin(VarId(3), Some(false)).unwrap();
    counter.set_pin(VarId(2), None).unwrap();
    let mut alone = f.counter().unwrap();
    alone.set_pins(&[(VarId(4), Some(false)), (VarId(3), Some(false))]).unwrap();
    assert_eq!(counter.model_count().unwrap(), alone.model_count().unwrap());
}

#[test]
fn an_empty_list_is_the_model_count_and_false_counts_zero_everywhere() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let mut counter = f.counter().unwrap();
    assert_eq!(counter.count_table(&[]).unwrap(), [BigUint::from(6u32)]);
    let zero = Tdd::zero(&vtree);
    assert_canonical(&zero);
    let mut counter = zero.counter().unwrap();
    assert_eq!(counter.count_table(&[VarId(3), VarId(1)]).unwrap(), vec![BigUint::ZERO; 4]);
}

#[test]
fn a_count_table_refuses_bad_lists_before_counting() {
    use crate::test_helpers::{marginal_diagrams, under};
    let vtree = Arc::new(Vtree::balanced(40));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let mut counter = f.counter().unwrap();
    counter.set_pin(VarId(5), Some(true)).unwrap();
    let wide: Vec<VarId> = (1..=31).map(VarId).collect();
    assert_eq!(counter.count_table(&wide), Err(OperationError::TableTooWide { vars: 31 }));
    assert_eq!(counter.count_table(&[VarId(1), VarId(2), VarId(1)]), Err(OperationError::DuplicateVariable(VarId(1))));
    assert_eq!(counter.count_table(&[VarId(1), VarId(41)]), Err(OperationError::VariableNotInVtree(VarId(41))));
    assert_eq!(counter.model_count().unwrap(), BigUint::from(3u32) << 37, "the pins are as they were");

    let (g, summed) = marginal_diagrams(5, 1, 4..6).swap_remove(0);
    let gone = g.vtree().leaf_bottomup().find(|&(leaf, _)| under(g.vtree(), leaf, summed)).expect("a summed leaf").1;
    let mut counter = g.counter().unwrap();
    assert!(matches!(counter.count_table(&[gone]), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn a_bound_count_table_stops_and_then_counts_with_its_pins_restored() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    let vtree = Arc::new(Vtree::balanced(5));
    let f = Tdd::clause(&vtree, [1, -2, 5]).unwrap();
    assert_canonical(&f);
    use std::sync::atomic::{AtomicUsize, Ordering};
    let engine = Engine::new();
    let mut counter = f.counter().unwrap();
    counter.set_pin(VarId(2), Some(false)).unwrap();
    {
        let _stop = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Stop))));
        assert_eq!(counter.bind(&engine).count_table(&[VarId(1), VarId(2)]), Err(OperationError::Stopped));
    }
    assert_eq!(counter.model_count().unwrap(), BigUint::from(16u32), "variable 2 is still false");
    // Pass the checks, then stop at the first count, after the table has
    // pinned its variables.
    let polls = Arc::new(AtomicUsize::new(0));
    {
        let seen = Arc::clone(&polls);
        let _stop = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(move |_, _| {
            if seen.fetch_add(1, Ordering::Relaxed) == 0 { StopDecision::Continue } else { StopDecision::Stop }
        }))));
        assert_eq!(counter.bind(&engine).count_table(&[VarId(1), VarId(2)]), Err(OperationError::Stopped));
    }
    assert!(polls.load(Ordering::Relaxed) >= 2, "the table got past its checks");
    assert_eq!(counter.model_count().unwrap(), BigUint::from(16u32), "variable 2 is false again");
    let table = counter.bind(&engine).count_table(&[VarId(1), VarId(2)]).unwrap();
    assert_eq!(table, one_at_a_time(&f, &[VarId(1), VarId(2)], Retention::All, PinSemantics::Evidence));
}
