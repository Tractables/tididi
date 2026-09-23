use crate::limits::pool::{Pool, PooledScratch};

#[derive(Default)]
struct WorkingSet {
    values: Vec<u32>,
    valid: bool,
}

impl PooledScratch for WorkingSet {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        [capacity_bytes(&self.values)].into_iter().sum()
    }

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

#[test]
fn preserving_checkout_keeps_initialized_entries() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::default();
    pool.put(&lim, vec![7u32, 11]);
    let mut scratch = pool.checkout_preserving(&lim);
    assert_eq!(&**scratch, &[7, 11]);
    scratch[1] = 13;
    drop(scratch);
    assert_eq!(&**pool.checkout_preserving(&lim), &[7, 13]);
    assert!(pool.checkout(&lim).is_empty());
}

/// Report large capacities without making a large allocation in a unit test.
#[derive(Default)]
struct Capacity(usize);
impl PooledScratch for Capacity {
    fn prepare(&mut self) {}
    fn retain(&mut self, _: &crate::limits::Limits) {}
    fn retained_bytes(&self) -> usize { self.0 }
}

#[test]
fn independent_pools_share_one_ceiling_and_release_their_claims() {
    use crate::limits::pool::ENGINE_RETAIN_BYTES;
    let lim = crate::limits::Limits::new();
    let first = Pool::<Capacity>::default();
    let second = Pool::<Capacity>::default();
    let half = ENGINE_RETAIN_BYTES / 2;
    first.put(&lim, Capacity(half));
    second.put(&lim, Capacity(half + 1));
    assert_eq!(lim.retained_scratch.get(), half);
    assert!(!second.occupied());
    second.put(&lim, Capacity(half));
    assert_eq!(lim.retained_scratch.get(), ENGINE_RETAIN_BYTES);
    let checked_out = first.take(&lim);
    assert_eq!(lim.retained_scratch.get(), half);
    second.drain(&lim);
    first.put(&lim, checked_out);
    first.drain(&lim);
    assert_eq!(lim.retained_scratch.get(), 0);
}

#[test]
fn nested_return_replaces_its_claim_and_unwind_leaves_other_pools_usable() {
    let lim = crate::limits::Limits::new();
    let pool = Pool::<Capacity>::default();
    let mut outer = pool.checkout(&lim);
    outer.0 = 17;
    { let mut inner = pool.checkout(&lim); inner.0 = 23; }
    assert_eq!(lim.retained_scratch.get(), 23);
    drop(outer);
    assert_eq!(lim.retained_scratch.get(), 17);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _active = pool.checkout(&lim);
        { let mut inner = pool.checkout(&lim); inner.0 = 31; }
        panic!("interrupted outer operation");
    }));
    assert!(result.is_err());
    assert_eq!(lim.retained_scratch.get(), 31);
    pool.drain(&lim);
    assert_eq!(lim.retained_scratch.get(), 0);
}

#[test]
fn recycled_levels_and_scratch_use_the_same_allowance() {
    use crate::limits::pool::ENGINE_RETAIN_BYTES;
    use crate::diagram::{take_levels, return_levels, PoolSlot};
    let eng = crate::Engine::new();
    let blocker = Pool::<Capacity>::default();
    blocker.put(eng.limits(), Capacity(ENGINE_RETAIN_BYTES));
    return_levels(&eng, PoolSlot::First, take_levels(&eng, 3));
    assert_eq!(eng.levels().occupancy(), 0);
    blocker.drain(eng.limits());
    return_levels(&eng, PoolSlot::First, take_levels(&eng, 3));
    assert_eq!(eng.levels().occupancy(), 1);
    assert!(eng.limits().retained_scratch.get() > 0);
    eng.clear_scratch();
    assert_eq!(eng.limits().retained_scratch.get(), 0);
}
