use super::*;

impl RangeTable {
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.grow(&Untracked, additional).unwrap();
    }
}

#[test]
fn empty_ranges_stay_allocation_free_through_pool_operations() {
    let eng = crate::Engine::new();
    let mut ranges = RangeTable::default();
    ranges.clear();
    ranges.reserve_exact(eng.limits(), 0).unwrap();
    ranges.shrink_to_fit();
    assert_eq!(ranges.retain(1024), 0);
    assert_eq!(ranges.try_clone_on(eng.limits()).unwrap().charged_bytes(), 0);
    assert!(ranges.0.is_none());
}

#[test]
fn wide_ranges_retry_reservation_and_release_the_cold_header() {
    let eng = crate::Engine::new();
    let mut ranges = RangeTable::default();
    eng.limits().refuse_nth_reserve(0);
    assert_eq!(ranges.grow(eng.limits(), 1), Err(OperationError::OverBudget));
    assert!(ranges.0.is_none());
    eng.limits().grant_every_reserve();
    ranges.grow(eng.limits(), 1).unwrap();
    ranges.push(PairRange { start: 1 << 31, len: 1 << 32 });
    let clone = ranges.try_clone_on(eng.limits()).unwrap();
    assert_eq!(clone, ranges);
    assert_eq!(ranges[0].start, 1 << 31);
    ranges[0].len = 3;
    assert_eq!(clone[0].len, 1 << 32);
    assert_eq!(ranges.charged_bytes(), (ranges.capacity() * std::mem::size_of::<PairRange>() + std::mem::size_of::<Vec<PairRange>>()) as u64);
    ranges.clear();
    ranges.shrink_to_fit();
    assert_eq!(ranges.charged_bytes(), 0);
    assert!(ranges.0.is_none());
}
