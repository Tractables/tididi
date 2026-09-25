use super::*;
use crate::diagram::WeightValue;
use crate::execution::pool::{Pool, PooledScratch, SCRATCH_RETAIN_BYTES};
use crate::test_helpers::{CountVecExt, rat};

fn populated(weighted: bool, n: usize) -> StreamCache {
    let mut cache = StreamCache::default();
    cache.reset(n, Some(weighted));
    if weighted {
        cache.weighted_mut()[n - 1] = Some(vec![WeightValue::Exact(rat(7, 3))]);
    } else {
        let eng = Engine::new();
        let mut counts = CountVec::with_width(&eng, 0);
        counts.push_i(&eng, Count::Big(BigUint::from(1u32) << 192usize));
        cache.int_mut()[n - 1] = Some(counts);
    }
    cache
}

fn table(cache: &StreamCache) -> (usize, usize, usize) {
    match cache {
        StreamCache::Int(cols) => (cols.len(), cols.capacity(), cols.as_ptr() as usize),
        StreamCache::Weighted(cols) => (cols.len(), cols.capacity(), cols.as_ptr() as usize),
        StreamCache::None => panic!("expected a column table"),
    }
}

fn assert_empty(cache: &StreamCache, n: usize, weighted: bool) {
    if weighted {
        assert_eq!(cache.weighted().len(), n);
        assert!(cache.weighted().iter().all(Option::is_none));
    } else {
        assert_eq!(cache.int().len(), n);
        assert!(cache.int().iter().all(Option::is_none));
    }
}

#[test]
fn returning_cache_drops_columns_and_retains_table_allocation() {
    let lim = crate::limits::Limits::new();
    for weighted in [false, true] {
        let mut cache = populated(weighted, 7);
        let (_, capacity, allocation) = table(&cache);
        cache.retain(&lim);
        assert_eq!(table(&cache), (0, capacity, allocation));
        for n in [3, 15, 1, 0, 7] {
            cache.reset(n, Some(weighted));
            assert_empty(&cache, n, weighted);
            cache.retain(&lim);
        }
    }
}

#[test]
fn arithmetic_switch_replaces_the_column_kind() {
    let lim = crate::limits::Limits::new();
    let mut cache = StreamCache::default();
    for weighted in [false, true, false] {
        cache.reset(3, Some(weighted));
        assert_empty(&cache, 3, weighted);
        cache.retain(&lim);
    }
}

#[test]
fn nonstreaming_apply_preserves_parked_table() {
    let lim = crate::limits::Limits::new();
    for weighted in [false, true] {
        let mut cache = populated(weighted, 7);
        let (_, capacity, allocation) = table(&cache);
        cache.retain(&lim);
        cache.reset(3, None);
        assert_eq!(table(&cache), (0, capacity, allocation));
        cache.reset(3, Some(weighted));
        assert_eq!(table(&cache), (3, capacity, allocation));
        assert_empty(&cache, 3, weighted);
    }
}

#[test]
fn nested_streaming_caches_keep_independent_tables() {
    let eng = crate::Engine::new();
    for weighted in [false, true] {
        let pool = Pool::<StreamCache>::default();
        let mut outer = pool.checkout(&eng);
        outer.reset(3, Some(weighted));
        let mut inner = pool.checkout(&eng);
        inner.reset(7, Some(weighted));
        assert_ne!(table(&outer).2, table(&inner).2);
        assert_empty(&outer, 3, weighted);
        drop(inner);
        let allocation = table(&outer).2;
        drop(outer);
        let mut reused = pool.checkout(&eng);
        reused.reset(3, Some(weighted));
        assert_eq!(table(&reused).2, allocation);
        assert_empty(&reused, 3, weighted);
    }
}

#[test]
fn returning_cache_releases_oversized_table_capacity() {
    let lim = crate::limits::Limits::new();
    let int_slots = SCRATCH_RETAIN_BYTES / std::mem::size_of::<Option<CountVec>>() + 1;
    let weighted_slots = SCRATCH_RETAIN_BYTES / std::mem::size_of::<Option<Vec<WeightValue>>>() + 1;
    for mut cache in [
        StreamCache::Int(Vec::with_capacity(int_slots)),
        StreamCache::Weighted(Vec::with_capacity(weighted_slots)),
    ] {
        cache.retain(&lim);
        assert_eq!(table(&cache).1, 0);
    }
}
