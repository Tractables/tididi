use crate::limits::pool::{Pool, PooledScratch};

#[derive(Default)]
struct WorkingSet {
    values: Vec<u32>,
    valid: bool,
}

impl PooledScratch for WorkingSet {
    fn prepare(&mut self) {
        self.values.clear();
        self.valid = false;
    }

    fn retain(&mut self) {
        if self.values.capacity() > 16 {
            self.values = Vec::new();
        }
    }
}

#[test]
fn nested_checkouts_own_independent_buffers() {
    let pool = Pool::<WorkingSet>::default();
    let mut outer = pool.checkout();
    outer.values.push(7);
    let outer_allocation = outer.values.as_ptr();
    {
        let mut inner = pool.checkout();
        assert!(inner.values.is_empty());
        inner.values.push(11);
        assert_ne!(inner.values.as_ptr(), outer_allocation);
        assert_eq!(outer.values, [7]);
    }
    assert_eq!(outer.values, [7]);
    drop(outer);
    let reused = pool.checkout();
    assert_eq!(reused.values.as_ptr(), outer_allocation);
    assert!(reused.values.is_empty());
}

#[test]
fn error_exit_retains_capacity_and_invalidates_results() {
    let pool = Pool::<WorkingSet>::default();
    let mut allocation = std::ptr::null();
    let result: Result<(), ()> = {
        let mut scratch = pool.checkout();
        scratch.values.push(7);
        scratch.valid = true;
        allocation = scratch.values.as_ptr();
        Err(())
    };
    assert!(result.is_err());
    let reused = pool.checkout();
    assert_eq!(reused.values.as_ptr(), allocation);
    assert!(!reused.valid);
    assert!(reused.values.is_empty());
}

#[test]
fn unwind_discards_partial_work_and_keeps_pool_usable() {
    let pool = Pool::<WorkingSet>::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scratch = pool.checkout();
        scratch.values.push(7);
        scratch.valid = true;
        panic!("interrupt scratch update");
    }));
    assert!(result.is_err());
    let mut fresh = pool.checkout();
    assert_eq!(fresh.values.capacity(), 0);
    assert!(!fresh.valid);
    fresh.values.push(11);
    let allocation = fresh.values.as_ptr();
    drop(fresh);
    assert_eq!(pool.checkout().values.as_ptr(), allocation);
}

#[test]
fn return_applies_retention_to_each_checkout() {
    let pool = Pool::<WorkingSet>::default();
    {
        let mut scratch = pool.checkout();
        scratch.values.reserve(17);
    }
    assert_eq!(pool.checkout().values.capacity(), 0);
}
