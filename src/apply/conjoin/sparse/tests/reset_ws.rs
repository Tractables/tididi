//! `reset_sparse_ws` must FULLY drop the engine's workspace (all retained
//! capacity), not the conditional per-array trim `release_if_large` applies on
//! the normal apply exit. The engine-owned workspace survives an unwinding
//! panic at full size, so recovery relies on this reset to release it.
use super::reset_sparse_ws;
use crate::Engine;

#[test]
fn reset_drops_all_capacity() {
    // Grow a few representative buffers directly — the private fields are
    // visible from this in-module submodule. `reserve` allocates capacity
    // without faulting pages in (len stays 0), so RSS stays KB-scale.
    let eng = Engine::new();
    {
        let mut ws = eng.sparse().borrow_mut();
        ws.par_buckets.push(Vec::with_capacity(256));
        ws.emit_pairs.reserve(256);
        ws.rev_entries_c1.reserve(256);
    }
    {
        let ws = eng.sparse().borrow();
        assert!(ws.par_buckets.capacity() > 0, "precondition: workspace grown");
        assert!(ws.emit_pairs.capacity() > 0);
        assert!(ws.rev_entries_c1.capacity() > 0);
    }

    reset_sparse_ws(&eng);

    {
        let ws = eng.sparse().borrow();
        assert_eq!(ws.par_buckets.capacity(), 0, "par_buckets released");
        assert_eq!(ws.emit_pairs.capacity(), 0, "emit_pairs released");
        assert_eq!(ws.rev_entries_c1.capacity(), 0, "rev_entries_c1 released");
    }
}
