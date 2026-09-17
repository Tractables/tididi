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

    fn retain(&mut self, _lim: &crate::limits::Limits) {
        if self.values.capacity() > 16 {
            self.values = Vec::new();
        }
    }
}

#[test]
fn nested_checkouts_own_independent_buffers() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::<WorkingSet>::default();
    let mut outer = pool.checkout(&lim);
    outer.values.push(7);
    let outer_allocation = outer.values.as_ptr();
    {
        let mut inner = pool.checkout(&lim);
        assert!(inner.values.is_empty());
        inner.values.push(11);
        assert_ne!(inner.values.as_ptr(), outer_allocation);
        assert_eq!(outer.values, [7]);
    }
    assert_eq!(outer.values, [7]);
    drop(outer);
    let reused = pool.checkout(&lim);
    assert_eq!(reused.values.as_ptr(), outer_allocation);
    assert!(reused.values.is_empty());
}

#[test]
fn error_exit_retains_capacity_and_invalidates_results() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::<WorkingSet>::default();
    let allocation;
    let result: Result<(), ()> = {
        let mut scratch = pool.checkout(&lim);
        scratch.values.push(7);
        scratch.valid = true;
        allocation = scratch.values.as_ptr();
        Err(())
    };
    assert!(result.is_err());
    let reused = pool.checkout(&lim);
    assert_eq!(reused.values.as_ptr(), allocation);
    assert!(!reused.valid);
    assert!(reused.values.is_empty());
}

#[test]
fn unwind_discards_partial_work_and_keeps_pool_usable() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::<WorkingSet>::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scratch = pool.checkout(&lim);
        scratch.values.push(7);
        scratch.valid = true;
        panic!("interrupt scratch update");
    }));
    assert!(result.is_err());
    let mut fresh = pool.checkout(&lim);
    assert_eq!(fresh.values.capacity(), 0);
    assert!(!fresh.valid);
    fresh.values.push(11);
    let allocation = fresh.values.as_ptr();
    drop(fresh);
    assert_eq!(pool.checkout(&lim).values.as_ptr(), allocation);
}

#[test]
fn return_applies_retention_to_each_checkout() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::<WorkingSet>::default();
    {
        let mut scratch = pool.checkout(&lim);
        scratch.values.reserve(17);
    }
    assert_eq!(pool.checkout(&lim).values.capacity(), 0);
}

#[test]
fn releasing_oversized_scratch_returns_its_bytes_to_the_meter() {
    use crate::limits::Limits;
    use crate::limits::pool::{SCRATCH_RETAIN_BYTES, release_if_oversized};

    let lim = Limits::new();
    let mut buf: Vec<u64> = Vec::new();
    let entries = SCRATCH_RETAIN_BYTES / std::mem::size_of::<u64>() + 1;
    lim.reserve(&mut buf, entries).unwrap();

    let charged = lim.meters().in_flight_bytes;
    assert!(charged > SCRATCH_RETAIN_BYTES as u64, "the reserve should have charged the meter");

    // The buffer is over the retain cap, so this frees the allocation. The
    // bytes have to come back: the meter only resets at the outermost
    // operation entry, so within one long operation a release that stayed
    // charged would permanently consume headroom the run no longer uses.
    release_if_oversized(&lim, &mut buf);
    assert_eq!(buf.capacity(), 0, "an oversized buffer is released, not kept");
    assert_eq!(lim.meters().in_flight_bytes, 0, "the freed bytes were not returned");
}

#[test]
fn releasing_an_undersized_buffer_keeps_both_the_allocation_and_the_charge() {
    use crate::limits::Limits;
    use crate::limits::pool::release_if_oversized;

    let lim = Limits::new();
    let mut buf: Vec<u64> = Vec::new();
    lim.reserve(&mut buf, 8).unwrap();
    let charged = lim.meters().in_flight_bytes;

    release_if_oversized(&lim, &mut buf);
    assert!(buf.capacity() >= 8, "an under-cap buffer stays warm");
    assert_eq!(lim.meters().in_flight_bytes, charged, "nothing was freed, so nothing is returned");
}
