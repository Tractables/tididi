use super::*;
use crate::diagram::WeightValue;
use crate::limits::pool::{Pool, SCRATCH_RETAIN_BYTES};
use crate::test_helpers::{CountVecExt, rat};

fn populated(weighted: bool, n: usize) -> StreamCache {
    let mut cache = StreamCache::take(&Pool::default(), n, Some(weighted));
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
    for weighted in [false, true] {
        let pool = Pool::default();
        let cache = populated(weighted, 7);
        let (_, capacity, allocation) = table(&cache);
        cache.put(&pool);
        let parked = pool.take();
        assert_eq!(table(&parked), (0, capacity, allocation));
    }
}

#[test]
fn reused_cache_matches_current_tree_size() {
    for weighted in [false, true] {
        let pool = Pool::default();
        populated(weighted, 7).put(&pool);
        for n in [3, 15, 1, 0, 7] {
            let cache = StreamCache::take(&pool, n, Some(weighted));
            assert_empty(&cache, n, weighted);
            cache.put(&pool);
        }
    }
}

#[test]
fn arithmetic_switch_replaces_the_column_kind() {
    let pool = Pool::default();
    for weighted in [false, true, false] {
        let cache = StreamCache::take(&pool, 3, Some(weighted));
        assert_empty(&cache, 3, weighted);
        cache.put(&pool);
    }
}

#[test]
fn nonstreaming_apply_preserves_parked_table() {
    for weighted in [false, true] {
        let pool = Pool::default();
        let cache = populated(weighted, 7);
        let (_, capacity, allocation) = table(&cache);
        cache.put(&pool);
        let disabled = StreamCache::take(&pool, 3, None);
        assert!(matches!(disabled, StreamCache::None));
        disabled.put(&pool);
        let reused = StreamCache::take(&pool, 3, Some(weighted));
        assert_eq!(table(&reused), (3, capacity, allocation));
        assert_empty(&reused, 3, weighted);
    }
}

#[test]
fn nested_streaming_caches_keep_independent_tables() {
    for weighted in [false, true] {
        let pool = Pool::default();
        populated(weighted, 7).put(&pool);
        let outer = StreamCache::take(&pool, 3, Some(weighted));
        let inner = StreamCache::take(&pool, 7, Some(weighted));
        assert_ne!(table(&outer).2, table(&inner).2);
        assert_empty(&outer, 3, weighted);
        inner.put(&pool);
        let outer_allocation = table(&outer).2;
        outer.put(&pool);
        let reused = StreamCache::take(&pool, 3, Some(weighted));
        assert_eq!(table(&reused).2, outer_allocation);
        assert_empty(&reused, 3, weighted);
    }
}

#[test]
fn returning_cache_releases_oversized_table_capacity() {
    let pool = Pool::default();
    let int_slots = SCRATCH_RETAIN_BYTES / std::mem::size_of::<Option<CountVec>>() + 1;
    let weighted_slots = SCRATCH_RETAIN_BYTES / std::mem::size_of::<Option<Vec<WeightValue>>>() + 1;
    for cache in [
        StreamCache::Int(Vec::with_capacity(int_slots)),
        StreamCache::Weighted(Vec::with_capacity(weighted_slots)),
    ] {
        cache.put(&pool);
        assert_eq!(table(&pool.take()).1, 0);
    }
}
