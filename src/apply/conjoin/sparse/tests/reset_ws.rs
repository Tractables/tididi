use crate::Engine;

#[test]
fn clearing_scratch_leaves_active_workspace_independent() {
    let eng = Engine::new();
    {
        let mut outer = eng.sparse().checkout(eng.limits());
        outer.emit_pairs.reserve(256);
        let allocation = outer.emit_pairs.as_ptr();
        eng.clear_scratch();
        let mut inner = eng.sparse().checkout(eng.limits());
        inner.emit_pairs.reserve(128);
        assert_ne!(inner.emit_pairs.as_ptr(), allocation);
        drop(inner);
        assert_eq!(outer.emit_pairs.as_ptr(), allocation);
    }
    assert!(eng.sparse().checkout(eng.limits()).emit_pairs.capacity() >= 256);
    eng.clear_scratch();
    let ws = eng.sparse().checkout(eng.limits());
    assert_eq!(ws.par_buckets.capacity(), 0);
    assert_eq!(ws.emit_pairs.capacity(), 0);
    assert_eq!(ws.rev_entries_c1.capacity(), 0);
}

#[test]
fn unwinding_discards_partially_filled_sparse_workspace() {
    let eng = Engine::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ws = eng.sparse().checkout(eng.limits());
        ws.p2_map.push(42);
        panic!("interrupt scatter");
    }));
    assert!(eng.sparse().checkout(eng.limits()).p2_map.is_empty());
}
