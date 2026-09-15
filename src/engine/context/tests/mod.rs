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
                let f = engine.literal(&tree, literal).unwrap();
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
    context.run(|engine| engine.apply().node_idx.put(vec![17]));
    context.run(|engine| assert_eq!(engine.apply().node_idx.take(), vec![17]));
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
    let projected = tree.project_to_vars(|var| (var.0 < 2).then_some(var), 2).unwrap();
    assert!(Arc::ptr_eq(projected.context(), &context));
    let grafted = Vtree::graft(&[projected], &[VarId(2)]).unwrap();
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
    let left = Vtree::leaf(VarId(0)).with_context(Arc::clone(&context));
    let right = Vtree::leaf(VarId(1)).with_context(Arc::clone(&context));
    let shared = Vtree::graft(&[left.clone(), right], &[]).unwrap();
    assert_eq!(shared.validate(), Ok(()));
    assert!(Arc::ptr_eq(shared.context(), &context));
    let mixed = Vtree::graft(&[left, Vtree::leaf(VarId(1))], &[]).unwrap();
    assert_eq!(mixed.validate(), Ok(()));
    assert!(!Arc::ptr_eq(mixed.context(), &context));
    let standalone = Engine::new().bind_vtree(shared);
    assert!(!Arc::ptr_eq(standalone.context(), &context));
}
