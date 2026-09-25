use crate::execution::pool::{Buffers, Drain, Nested, Pool, PooledScratch, Scratch};

#[derive(Default)]
struct WorkingSet {
    values: Vec<u32>,
    valid: bool,
}

impl Buffers for WorkingSet {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) { visit(&mut self.values); }
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
    let eng = crate::Engine::new();
    let pool = Pool::<WorkingSet>::default();
    let mut outer = pool.checkout(&eng);
    outer.values.push(7);
    let outer_allocation = outer.values.as_ptr();
    {
        let mut inner = pool.checkout(&eng);
        assert!(inner.values.is_empty());
        inner.values.push(11);
        assert_ne!(inner.values.as_ptr(), outer_allocation);
        assert_eq!(outer.values, [7]);
    }
    assert_eq!(outer.values, [7]);
    drop(outer);
    let reused = pool.checkout(&eng);
    assert_eq!(reused.values.as_ptr(), outer_allocation);
    assert!(reused.values.is_empty());
}

#[test]
fn error_exit_retains_capacity_and_invalidates_results() {
    let eng = crate::Engine::new();
    let pool = Pool::<WorkingSet>::default();
    let allocation;
    let result: Result<(), ()> = {
        let mut scratch = pool.checkout(&eng);
        scratch.values.push(7);
        scratch.valid = true;
        allocation = scratch.values.as_ptr();
        Err(())
    };
    assert!(result.is_err());
    let reused = pool.checkout(&eng);
    assert_eq!(reused.values.as_ptr(), allocation);
    assert!(!reused.valid);
    assert!(reused.values.is_empty());
}

#[test]
fn unwind_discards_partial_work_and_keeps_pool_usable() {
    let eng = crate::Engine::new();
    let pool = Pool::<WorkingSet>::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scratch = pool.checkout(&eng);
        scratch.values.push(7);
        scratch.valid = true;
        panic!("interrupt scratch update");
    }));
    assert!(result.is_err());
    let mut fresh = pool.checkout(&eng);
    assert_eq!(fresh.values.capacity(), 0);
    assert!(!fresh.valid);
    fresh.values.push(11);
    let allocation = fresh.values.as_ptr();
    drop(fresh);
    assert_eq!(pool.checkout(&eng).values.as_ptr(), allocation);
}

#[test]
fn return_applies_retention_to_each_checkout() {
    let eng = crate::Engine::new();
    let pool = Pool::<WorkingSet>::default();
    {
        let mut scratch = pool.checkout(&eng);
        scratch.values.reserve(17);
    }
    assert_eq!(pool.checkout(&eng).values.capacity(), 0);
}

#[test]
fn releasing_oversized_scratch_returns_its_bytes_to_the_meter() {
    use crate::limits::Limits;
    use crate::execution::pool::{SCRATCH_RETAIN_BYTES, release_if_oversized};

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
    use crate::execution::pool::release_if_oversized;

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
    let eng = crate::Engine::new();
    let pool = Pool::default();
    pool.put(&eng, vec![7u32, 11]);
    let mut scratch = pool.checkout_preserving(&eng);
    assert_eq!(&**scratch, &[7, 11]);
    scratch[1] = 13;
    drop(scratch);
    assert_eq!(&**pool.checkout_preserving(&eng), &[7, 13]);
    assert!(pool.checkout(&eng).is_empty());
}

/// Report large capacities without making a large allocation in a unit test.
#[derive(Default)]
struct Capacity(usize);
impl crate::limits::Charged for Capacity {
    fn charged_bytes(&self) -> u64 { self.0 as u64 }
}
impl Scratch for Capacity {
    fn release(&mut self) { self.0 = 0; }
}
impl Buffers for Capacity {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) { visit(self); }
}
impl PooledScratch for Capacity {
    fn prepare(&mut self) {}
    fn retain(&mut self, _: &crate::limits::Limits) {}
}

#[test]
fn independent_pools_share_one_ceiling_and_release_their_claims() {
    use crate::execution::pool::ENGINE_RETAIN_BYTES;
    let eng = crate::Engine::new();
    let first = Pool::<Capacity>::default();
    let second = Pool::<Capacity>::default();
    let half = ENGINE_RETAIN_BYTES / 2;
    first.put(&eng, Capacity(half));
    second.put(&eng, Capacity(half + 1));
    assert_eq!(eng.scratch.ledger.bytes(), half);
    assert!(!second.occupied());
    second.put(&eng, Capacity(half));
    assert_eq!(eng.scratch.ledger.bytes(), ENGINE_RETAIN_BYTES);
    let checked_out = first.take(&eng);
    assert_eq!(eng.scratch.ledger.bytes(), half);
    second.drain(&eng);
    first.put(&eng, checked_out);
    first.drain(&eng);
    assert_eq!(eng.scratch.ledger.bytes(), 0);
}

#[test]
fn nested_return_replaces_its_claim_and_unwind_leaves_other_pools_usable() {
    let eng = crate::Engine::new();
    let pool = Pool::<Capacity>::default();
    let mut outer = pool.checkout(&eng);
    outer.0 = 17;
    { let mut inner = pool.checkout(&eng); inner.0 = 23; }
    assert_eq!(eng.scratch.ledger.bytes(), 23);
    drop(outer);
    assert_eq!(eng.scratch.ledger.bytes(), 17);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _active = pool.checkout(&eng);
        { let mut inner = pool.checkout(&eng); inner.0 = 31; }
        panic!("interrupted outer operation");
    }));
    assert!(result.is_err());
    assert_eq!(eng.scratch.ledger.bytes(), 31);
    pool.drain(&eng);
    assert_eq!(eng.scratch.ledger.bytes(), 0);
}

#[test]
fn recycled_levels_and_scratch_use_the_same_allowance() {
    use crate::execution::pool::ENGINE_RETAIN_BYTES;
    use crate::diagram::{take_levels, return_levels, PoolSlot};
    let eng = crate::Engine::new();
    let blocker = Pool::<Capacity>::default();
    blocker.put(&eng, Capacity(ENGINE_RETAIN_BYTES));
    return_levels(&eng, PoolSlot::First, take_levels(&eng, 3));
    assert_eq!(eng.scratch.levels.occupancy(), 0);
    blocker.drain(&eng);
    return_levels(&eng, PoolSlot::First, take_levels(&eng, 3));
    assert_eq!(eng.scratch.levels.occupancy(), 1);
    assert!(eng.scratch.ledger.bytes() > 0);
    eng.clear_scratch();
    assert_eq!(eng.scratch.ledger.bytes(), 0);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a pool is missing")]
fn clearing_scratch_notices_a_parked_pool_it_does_not_list() {
    let eng = crate::Engine::new();
    let unlisted = Pool::<Capacity>::default();
    unlisted.put(&eng, Capacity(1));
    eng.clear_scratch();
}

/// Two scratch buffers with the default retention rule: each is judged on its
/// own bytes.
#[derive(Default)]
struct Pair {
    small: Vec<u64>,
    large: Vec<u64>,
}

impl Buffers for Pair {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.small);
        visit(&mut self.large);
    }
}

impl PooledScratch for Pair {
    fn prepare(&mut self) {}
}

#[test]
fn default_retention_releases_each_listed_buffer_on_its_own_bytes() {
    use crate::execution::pool::SCRATCH_RETAIN_BYTES;
    let eng = crate::Engine::new();
    let pool = Pool::<Pair>::default();
    {
        let mut scratch = pool.checkout(&eng);
        scratch.small.reserve(8);
        // `reserve` does not touch the pages, so the test stays small in memory.
        scratch.large.reserve(SCRATCH_RETAIN_BYTES / std::mem::size_of::<u64>() + 1);
    }
    let mut scratch = pool.checkout(&eng);
    assert!(scratch.small.capacity() >= 8, "an under-cap buffer stays warm");
    assert_eq!(scratch.large.capacity(), 0, "an over-cap buffer is released");
    assert_eq!(scratch.retained_bytes(), scratch.small.capacity() * std::mem::size_of::<u64>());
}

#[test]
fn nested_buffers_are_released_on_their_total_bytes() {
    use crate::execution::pool::{SCRATCH_RETAIN_BYTES, release_if_oversized};
    let lim = crate::limits::Limits::new();
    // Two outer rows whose inner capacities together pass the cap: a count of
    // outer rows would keep them, the byte count must not.
    let per_row = SCRATCH_RETAIN_BYTES / (2 * std::mem::size_of::<(u32, u32)>()) + 1;
    let mut fat: Vec<Vec<(u32, u32)>> = vec![Vec::with_capacity(per_row), Vec::with_capacity(per_row)];
    release_if_oversized(&lim, &mut Nested(&mut fat));
    assert_eq!(fat.capacity(), 0, "few but fat rows are released on bytes");

    let mut thin: Vec<Vec<(u32, u32)>> = (0..1000).map(|_| Vec::with_capacity(4)).collect();
    let before = thin.capacity();
    release_if_oversized(&lim, &mut Nested(&mut thin));
    assert_eq!(thin.capacity(), before, "many small rows are kept");
    assert_eq!(thin.len(), 1000);
}

#[test]
fn inline_small_vectors_count_no_heap_bytes() {
    use crate::limits::Charged;
    use smallvec::SmallVec;
    let mut groups: Vec<SmallVec<[u32; 4]>> = Vec::with_capacity(2);
    groups.push(SmallVec::from_slice(&[1, 2]));
    let spine = (groups.capacity() * std::mem::size_of::<SmallVec<[u32; 4]>>()) as u64;
    assert_eq!(Nested(&mut groups).charged_bytes(), spine);
    groups.push((0..9).collect());
    let spilled = (groups[1].capacity() * std::mem::size_of::<u32>()) as u64;
    assert_eq!(Nested(&mut groups).charged_bytes(), spine + spilled);
}
