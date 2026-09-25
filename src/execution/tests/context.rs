use std::sync::{Arc, Barrier};

use super::*;
use crate::{OperationError, Tdd};
use crate::test_helpers::assert_canonical;
use crate::vtree::VarId;

#[test]
fn shared_context_keeps_diagrams_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Context>();
    send_sync::<Vtree>();
    send_sync::<Tdd>();
}

#[test]
fn nested_batches_have_independent_limits_and_return_one_idle_engine() {
    let context = Arc::new(Context::new());
    let tree = context.bind(Vtree::balanced(3));
    let limited = LimitConfig::none().with_memory_budget_bytes(Some(0));
    context.with_limits(limited, |outer| {
        assert_eq!(outer.clause(&tree, [1, 2]).unwrap_err(), OperationError::OverBudget);
        context.run(|inner| {
            assert!(!std::ptr::eq(outer, inner));
            let f = inner.clause(&tree, [1, 2]).unwrap();
            assert_canonical(&f);
            assert_eq!(inner.model_count(&f).unwrap(), 6u32.into());
        });
        assert_eq!(outer.clause(&tree, [1, 2]).unwrap_err(), OperationError::OverBudget);
    });
    assert!(context.idle.lock().unwrap().is_some());
    context.run(|engine| {
        let f = engine.clause(&tree, [1, 2]).unwrap();
        assert_canonical(&f);
    });
}

#[test]
fn concurrent_batches_can_run_on_the_same_context() {
    let context = Arc::new(Context::new());
    let tree = context.bind(Vtree::balanced(3));
    let ready = Barrier::new(2);
    std::thread::scope(|threads| {
        for literal in [1, 2] {
            let (context, tree, ready) = (&context, &tree, &ready);
            threads.spawn(move || context.run(|engine| {
                ready.wait();
                let f = engine.literal(tree, literal).unwrap();
                assert_canonical(&f);
                assert_eq!(engine.model_count(&f).unwrap(), 4u32.into());
            }));
        }
    });
    assert!(context.idle.lock().unwrap().is_some());
}

#[test]
fn returned_and_unwound_batches_release_callback_captures() {
    use crate::limits::{StopCallback, StopDecision};
    for unwind in [false, true] {
        let context = Arc::new(Context::new());
        let tree = context.bind(Vtree::balanced(2));
        let captured = Arc::clone(&tree);
        let weak = Arc::downgrade(&tree);
        let callback = StopCallback::new(move |_, _| {
            assert_eq!(captured.num_leaves(), 2);
            StopDecision::Continue
        });
        let config = LimitConfig::none().with_stop_callback(Some(callback));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.run(|engine| {
                let _prior = engine.limits().install(config);
                if unwind { panic!("abort batch"); }
            });
        }));
        assert_eq!(outcome.is_err(), unwind);
        assert_eq!(context.idle.lock().unwrap().is_some(), !unwind);
        drop(tree);
        assert!(weak.upgrade().is_none());
        context.run(|engine| assert!(engine.limits().armed().stop_callback().is_none()));
    }
}

#[test]
fn clearing_scratch_releases_only_the_idle_checkout() {
    let context = Arc::new(Context::new());
    context.run(|engine| engine.scratch.apply.subvars.put(engine, vec![17]));
    context.run(|engine| assert_eq!(engine.scratch.apply.subvars.take(engine), vec![17]));
    context.run(|_| {
        context.clear_scratch();
        assert!(context.idle.lock().unwrap().is_none());
    });
    assert!(context.idle.lock().unwrap().is_some());
    context.clear_scratch();
    assert!(context.idle.lock().unwrap().is_none());
}

#[test]
fn derived_shapes_retain_context_without_retaining_tree_identity() {
    let context = Arc::new(Context::new());
    let tree = context.bind(Vtree::balanced(3));
    let clone = Arc::new((*tree).clone());
    assert!(!Arc::ptr_eq(&tree, &clone));
    assert!(Arc::ptr_eq(tree.context(), clone.context()));
    let projected = tree.project_to_vars(|var| (var.0 <= 2).then_some(var), 2).unwrap();
    assert!(Arc::ptr_eq(projected.context(), &context));
    let grafted = Vtree::graft(&[projected], &[VarId(3)]).unwrap();
    assert!(Arc::ptr_eq(grafted.context(), &context));
    assert_eq!(grafted.validate(), Ok(()));
    assert_eq!(grafted.num_leaves(), tree.num_leaves());
    let loaded = Vtree::from_text(&tree.to_text()).unwrap();
    assert!(tree.same_tree(&loaded));
    assert!(!Arc::ptr_eq(loaded.context(), &context));
    context.run(|engine| {
        let bound = engine.bind_vtree(loaded);
        assert!(Arc::ptr_eq(bound.context(), &context));
    });
    let weak = Arc::downgrade(&context);
    drop(grafted);
    drop(clone);
    drop(tree);
    drop(context);
    assert!(weak.upgrade().is_none());
}

#[test]
fn stop_callback_can_reenter_the_same_context() {
    use crate::limits::{StopCallback, StopDecision};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let context = Arc::new(Context::new());
    let tree = context.bind(Vtree::balanced(3));
    let calls = Arc::new(AtomicUsize::new(0));
    let callback = {
        let context = Arc::clone(&context);
        let tree = Arc::clone(&tree);
        let calls = Arc::clone(&calls);
        StopCallback::new(move |_, _| {
            context.run(|engine| {
                let f = engine.literal(&tree, 1).unwrap();
                assert_canonical(&f);
                assert_eq!(engine.model_count(&f).unwrap(), 4u32.into());
            });
            calls.fetch_add(1, Ordering::Relaxed);
            StopDecision::Continue
        })
    };
    let f = context.with_limits(LimitConfig::none().with_stop_callback(Some(callback)), |engine| {
        engine.clause(&tree, [1, 2]).unwrap()
    });
    assert_canonical(&f);
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn grafts_preserve_only_a_context_agreed_by_every_source() {
    let context = Arc::new(Context::new());
    let left = Vtree::leaf(VarId(1)).with_context(Arc::clone(&context));
    let right = Vtree::leaf(VarId(2)).with_context(Arc::clone(&context));
    let shared = Vtree::graft(&[left.clone(), right], &[]).unwrap();
    assert_eq!(shared.validate(), Ok(()));
    assert!(Arc::ptr_eq(shared.context(), &context));
    let mixed = Vtree::graft(&[left, Vtree::leaf(VarId(2))], &[]).unwrap();
    assert_eq!(mixed.validate(), Ok(()));
    assert!(!Arc::ptr_eq(mixed.context(), &context));
    let standalone = Engine::new().bind_vtree(shared);
    assert!(!Arc::ptr_eq(standalone.context(), &context));
}


#[test]
fn ordinary_diagram_methods_can_reenter_from_a_stop_callback() {
    use crate::limits::{StopCallback, StopDecision};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tree = Arc::new(Vtree::balanced(3));
    let calls = Arc::new(AtomicUsize::new(0));
    let callback = {
        let tree = Arc::clone(&tree);
        let calls = Arc::clone(&calls);
        StopCallback::new(move |_, _| {
            let x = crate::literal(&tree, 1).unwrap();
            let y = crate::literal(&tree, 2).unwrap();
            let z = crate::literal(&tree, 3).unwrap();
            assert_canonical(&x);
            assert_canonical(&y);
            assert_canonical(&z);
            let mut f = (x | y) & !z;
            f.minimize().unwrap();
            assert_canonical(&f);
            assert_eq!(f.model_count().unwrap(), 3u32.into());
            assert!(f.satisfying_assignment().unwrap().is_some());
            calls.fetch_add(1, Ordering::Relaxed);
            StopDecision::Continue
        })
    };
    let f = tree.context().with_limits(
        LimitConfig::none().with_stop_callback(Some(callback)),
        |engine| engine.clause(&tree, [1, 2]).unwrap(),
    );
    assert_canonical(&f);
    assert!(calls.load(Ordering::Relaxed) > 0);
    assert_eq!(f.model_count().unwrap(), 6u32.into());
}

#[test]
fn sharing_context_does_not_make_distinct_trees_compatible() {
    let context = Arc::new(Context::new());
    let first = context.bind(Vtree::balanced(3));
    let second = context.bind(Vtree::balanced(3));
    let f = Tdd::clause(&first, [1, 2]).unwrap();
    let g = Tdd::clause(&second, [1, 2]).unwrap();
    assert_canonical(&f);
    assert_canonical(&g);
    assert!(Arc::ptr_eq(f.context(), g.context()));
    assert!(!Arc::ptr_eq(f.vtree(), g.vtree()));
    assert_eq!(crate::and(f.clone(), g.clone()).unwrap_err(), OperationError::VtreeMismatch);
    assert_eq!(crate::or(f.clone(), g.clone()).unwrap_err(), OperationError::VtreeMismatch);
    assert_eq!(f.equivalent(&g), Err(OperationError::VtreeMismatch));
    assert_eq!(f.implies(&g), Err(OperationError::VtreeMismatch));
    assert_eq!(f.model_count().unwrap(), 6u32.into());
    assert_eq!(g.model_count().unwrap(), 6u32.into());
}

#[test]
fn accepted_rotation_keeps_context_and_detaches_only_the_changed_tree() {
    use crate::restructure::search::{RotationObjective, RotationProbe, RotationSearchConfig};
    use crate::test_helpers::eval;
    struct AcceptOnce(bool);
    impl RotationObjective for AcceptOnce {
        fn delta(&mut self, _: &RotationProbe<'_>) -> i64 {
            if std::mem::replace(&mut self.0, false) { -1 } else { 0 }
        }
    }
    let tree = Arc::new(Vtree::balanced(4));
    let original = Tdd::clause(&tree, [1, 3]).unwrap();
    assert_canonical(&original);
    let mut rotated = original.clone();
    let stats = rotated.rotation_search(&mut AcceptOnce(true), &RotationSearchConfig {
        max_sweeps: Some(1),
        ..RotationSearchConfig::default()
    }).unwrap();
    assert_eq!(stats.accepts, 1);
    assert_canonical(&rotated);
    assert!(Arc::ptr_eq(original.vtree(), &tree));
    assert!(!Arc::ptr_eq(rotated.vtree(), &tree));
    assert!(Arc::ptr_eq(rotated.context(), tree.context()));
    for bits in 0u32..16 {
        let assignment = (0..4).map(|var| bits & (1 << var) != 0).collect::<Vec<_>>();
        assert_eq!(eval(&rotated, &assignment), eval(&original, &assignment));
    }
    assert_eq!(rotated.model_count().unwrap(), 12u32.into());
    let companion = crate::literal(rotated.vtree(), 1).unwrap();
    assert_canonical(&companion);
    let mut combined = crate::and(rotated, companion).unwrap();
    combined.minimize().unwrap();
    assert_canonical(&combined);
    assert_eq!(combined.model_count().unwrap(), 8u32.into());
}

#[test]
fn context_reset_preserves_capacity_accounting_but_not_operation_meters() {
    let context = Arc::new(Context::new());
    let bytes = context.run(|engine| {
        engine.scratch.apply.subvars.put(engine, vec![1, 2, 3]);
        engine.limits().charge_in_flight(999);
        engine.scratch.ledger.bytes()
    });
    assert!(bytes > 0);
    context.run(|engine| {
        assert_eq!(engine.scratch.ledger.bytes(), bytes);
        assert_eq!(engine.limits().meters().in_flight_bytes, 0);
        engine.clear_scratch();
        assert_eq!(engine.scratch.ledger.bytes(), 0);
    });
}
