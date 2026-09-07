    use super::*;

    #[test]
    fn level_pairs_iter_of_unpacked_matches_slice() {
        // PairsIter::Slice and PairsIter::Inline branches: yield the same
        // pairs as pairs_of_idx on an unpacked level.
        let mut lvl = TddLevel::new();
        // Push some multi-pair nodes and an inline pair.
        let multi_pairs = vec![
            InputPair { left: LocalNodeIdx(1), right: LocalNodeIdx(2) },
            InputPair { left: LocalNodeIdx(3), right: LocalNodeIdx(4) },
            InputPair { left: LocalNodeIdx(5), right: LocalNodeIdx(6) },
        ];
        lvl.try_push_internal_node(&multi_pairs).unwrap();
        lvl.try_push_internal_node(&[InputPair {
            left: LocalNodeIdx(99),
            right: LocalNodeIdx(100),
        }]).unwrap();

        let from_slice: Vec<InputPair> = lvl.pairs_of_idx(0).to_vec();
        let from_iter: Vec<InputPair> = lvl.pairs_iter_of_idx(0).collect();
        assert_eq!(from_slice, from_iter);

        let inline_slice: Vec<InputPair> = lvl.pairs_of_idx(1).to_vec();
        let inline_iter: Vec<InputPair> = lvl.pairs_iter_of_idx(1).collect();
        assert_eq!(inline_slice, inline_iter);
        assert_eq!(inline_slice.len(), 1);
    }

    #[test]
    fn level_pairs_iter_of_idx_size_hint() {
        // ExactSizeIterator: size_hint correctly reports the count for all
        // four variants.
        let mut lvl = TddLevel::new();
        lvl.try_push_internal_node(&[
            InputPair { left: LocalNodeIdx(1), right: LocalNodeIdx(2) },
            InputPair { left: LocalNodeIdx(3), right: LocalNodeIdx(4) },
            InputPair { left: LocalNodeIdx(5), right: LocalNodeIdx(6) },
        ]).unwrap();
        let it = lvl.pairs_iter_of_idx(0);
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn test_take_levels_fresh_allocation() {
        let levels = take_levels(5);
        assert_eq!(levels.len(), 5);
        for level in &levels {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
            assert!(!level.has_multi_pair());
        }
    }

    #[test]
    fn test_pool_roundtrip() {
        // Take fresh, return, take again — should reuse
        let levels = take_levels(3);
        assert_eq!(levels.len(), 3);
        return_levels(levels);
        let levels2 = take_levels(3);
        assert_eq!(levels2.len(), 3);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }
    }

    #[test]
    fn test_pool_size_mismatch_resizes() {
        // A parked entry of the wrong length is RESIZED to the request, not
        // discarded — that is what keeps the arenas warm when components of
        // different variable counts alternate. Both directions, and every level
        // handed out is still empty (the reset barrier is not skipped).
        let _ = LEVELS_POOL.with(|cell| cell.take());
        let _ = LEVELS_POOL2.with(|cell| cell.take());

        // Shrink: return a Vec of size 5, then request size 3.
        let levels = take_levels(5);
        return_levels(levels);
        let levels2 = take_levels(3);
        assert_eq!(levels2.len(), 3);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }

        // Grow: return that size-3 Vec, then request size 6.
        return_levels(levels2);
        let levels3 = take_levels(6);
        assert_eq!(levels3.len(), 6);
        for level in &levels3 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }
    }

    #[test]
    fn test_pool2_roundtrip() {
        // Test secondary pool
        let levels = take_levels(4);
        return_levels2(levels);
        let levels2 = take_levels(4);
        assert_eq!(levels2.len(), 4);
    }

    #[test]
    fn test_reset_levels_clears_state() {
        let mut levels = take_levels(2);
        // Dirty the levels with internal nodes
        let dummy = InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) };
        levels[0].push_internal_node(&[dummy]);
        levels[1].push_internal_node(&[dummy, dummy]);
        assert_eq!(levels[0].nodes.len(), 1);
        assert!(levels[1].has_multi_pair());
        // Reset and verify clean
        for level in &mut levels {
            reset_level(level);
        }
        for level in &levels {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
            assert!(!level.has_multi_pair());
        }
    }

    /// Regression test: a level returned to the pool with bloated `pairs`
    /// capacity (left behind by a giant apply intermediate that minimized
    /// down to few nodes but never shrank its pair arena) must not survive
    /// pool recycle. Bug observed `mc2020_track1_185` under
    /// a 4 GiB heap cap: `clause_to_tdd`'s pooled levels contained a
    /// level with `pairs.capacity()` = 2 GiB (real allocator bytes, retained
    /// because the pool retention check only inspected `nodes.capacity()`).
    /// The retained-capacity accounting reported 2 GiB on a one-clause TDD, `BatchBudget`
    /// fired immediately, recovery had no operand to v-split, and the
    /// compile bailed as "Compilation failed". See pool-pairs-bloat bug.
    #[test]
    fn test_pool_shrinks_oversized_pair_capacity() {
        // Pre-flush the pool to make this test deterministic regardless of
        // prior thread-local state.
        let _ = LEVELS_POOL.with(|cell| cell.take());
        let _ = LEVELS_POOL2.with(|cell| cell.take());
        let mut levels = take_levels(3);
        // Simulate the pathological-apply legacy: a level with tiny content
        // but huge `pairs` capacity.
        levels[1].pairs.reserve(8_000_000);
        let bloated_cap = levels[1].pairs.capacity();
        assert!(bloated_cap >= 8_000_000, "reserve didn't grow capacity");
        // The pool retention gate (nodes ≤ 4M) passes — total node capacity is
        // 0 — so this Vec is retained. The oversized pair arena must be gone
        // by the time it is parked, not merely by the time it is handed out:
        // otherwise the bytes sit in the thread-local for the whole gap until
        // some later consumer asks for levels of this length.
        return_levels(levels);
        let parked = LEVELS_POOL
            .with(|cell| cell.take())
            .expect("levels passing the retention gate must be parked");
        let parked_cap = parked[1].pairs.capacity();
        assert!(
            parked_cap < bloated_cap,
            "pool parked a level with bloated pairs.capacity() = {} \
             (bloated {}), expected shrink at return",
            parked_cap, bloated_cap,
        );
        LEVELS_POOL.with(|cell| cell.set(Some(parked)));
        // Take the recycled Vec back — the levels it hands out are the ones
        // that were trimmed above.
        let levels2 = take_levels(3);
        let kept_cap = levels2[1].pairs.capacity();
        assert!(
            kept_cap < bloated_cap,
            "pool recycled a level with bloated pairs.capacity() = {} \
             (pre-bloat {}), expected shrink",
            kept_cap, bloated_cap,
        );
        // Belt-and-suspenders: cap is bounded by the per-arena threshold
        // (no hardcoded number — assert against the threshold the
        // implementation uses).
        use std::mem::size_of;
        assert!(
            kept_cap.saturating_mul(size_of::<InputPair>()) <= MAX_LEVEL_ARENA_BYTES,
            "pairs.capacity() = {} ({} bytes) exceeds MAX_LEVEL_ARENA_BYTES = {}",
            kept_cap, kept_cap.saturating_mul(size_of::<InputPair>()),
            MAX_LEVEL_ARENA_BYTES,
        );
    }

    #[test]
    fn test_pool_returns_clean_levels() {
        // Dirty some levels, return to pool, take back — should be clean
        let mut levels = take_levels(2);
        let dummy = InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) };
        levels[0].push_internal_node(&[dummy]);
        levels[1].push_internal_node(&[dummy, dummy]);
        return_levels(levels);
        let levels2 = take_levels(2);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
            assert!(!level.has_multi_pair());
        }
    }

    #[test]
    fn test_tdd_level_width() {
        let mut level = TddLevel::new();
        assert_eq!(level.width(), 0);
        let dummy = InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) };
        level.push_internal_node(&[dummy]);
        assert_eq!(level.width(), 1);
        level.push_internal_node(&[dummy]);
        assert_eq!(level.width(), 2);
    }

    #[test]
    fn test_push_internal_node() {
        let mut level = TddLevel::new();
        let pairs = [
            InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(1) },
            InputPair { left: LocalNodeIdx(1), right: LocalNodeIdx(0) },
        ];
        let idx = level.push_internal_node(&pairs);
        assert_eq!(idx, LocalNodeIdx(0));
        assert_eq!(level.width(), 1);
        assert_eq!(level.pairs.len(), 2);
        assert!(level.has_multi_pair());
        let node_pairs = level.pairs_of_idx(0);
        assert_eq!(node_pairs.len(), 2);
        assert_eq!(node_pairs[0], pairs[0]);
        assert_eq!(node_pairs[1], pairs[1]);
    }

    #[test]
    fn test_tdd_is_zero() {
        use crate::vtree::Vtree;
        use std::sync::Arc;
        let vtree = Arc::new(Vtree::balanced(2));
        let levels = take_levels(vtree.num_nodes());
        let tdd = Tdd::with_levels(
            vtree.clone(),
            levels,
            TddNodeId { vtree: vtree.root(), local: ZERO },
        );
        assert!(tdd.is_zero());
    }

    #[test]
    fn test_node_size() {
        assert_eq!(std::mem::size_of::<TddNodeData>(), 8, "TddNodeData should be 8 bytes (packed struct)");
    }

    #[test]
    fn test_encode_multi_promotes_to_extended_on_huge_start() {
        // A pair_start at 2^31 triggers the extended encoding even though
        // pair_len is small. Verifies the side-table round-trips the stored values.
        let mut level = TddLevel::new();
        let huge_start = 1usize << 31;
        let data = level.encode_multi(huge_start, 3);
        level.nodes.push(data);
        assert!(data.is_multi(), "huge-start node should be multi");
        assert!(data.is_multi_extended(), "huge-start node should promote to extended");
        assert!(!data.is_leaf(), "extended multi must not be mis-flagged as leaf");
        assert!(!data.is_inline(), "extended multi must not be mis-flagged as inline");
        assert_eq!(level.multi_start_at(0), huge_start);
        assert_eq!(level.multi_len_at(0), 3);
        assert_eq!(level.pair_count_at(0), 3);
        assert_eq!(level.ext.len(), 1);
    }

    #[test]
    fn test_encode_multi_promotes_to_extended_on_huge_len() {
        // A pair_len at 2^31 triggers the extended encoding. We don't actually
        // allocate that much arena memory — encode_multi only stores the count
        // and ext_idx; the arena is the caller's concern.
        let mut level = TddLevel::new();
        let huge_len = 1usize << 31;
        let data = level.encode_multi(0, huge_len);
        level.nodes.push(data);
        assert!(data.is_multi_extended(), "huge-len node should be extended");
        assert!(!data.is_leaf(), "extended multi must not be mis-flagged as leaf — this is the bug that caused qmr-100 UNSAT");
        assert_eq!(level.multi_len_at(0), huge_len);
    }

    #[test]
    fn test_encode_multi_stays_normal_for_small_values() {
        // Normal-sized multi nodes don't allocate an ext slot — the packed
        // 8-byte encoding handles them.
        let mut level = TddLevel::new();
        let data = level.encode_multi(100, 5);
        level.nodes.push(data);
        assert!(data.is_multi_normal(), "small multi should stay in packed form");
        assert!(!data.is_multi_extended());
        assert_eq!(level.multi_start_at(0), 100);
        assert_eq!(level.multi_len_at(0), 5);
        assert_eq!(level.ext.len(), 0, "no ext slot allocated for packed form");
    }

    #[test]
    fn test_extended_set_pair_len_updates_side_table() {
        // Shrinking an extended node's pair_len must update the side table,
        // not the node's b field (which is the extended-form sentinel).
        let mut level = TddLevel::new();
        let data = level.encode_multi(0, 1 << 31);
        level.nodes.push(data);
        level.set_pair_len(0, 100);
        assert!(level.nodes[0].is_multi_extended(), "still extended after shrink");
        assert_eq!(level.multi_len_at(0), 100);
    }

    #[test]
    fn test_reset_levels_clears_ext() {
        let mut level = TddLevel::new();
        let data = level.encode_multi(1 << 31, 3);
        level.nodes.push(data);
        assert_eq!(level.ext.len(), 1);
        let mut levels = vec![level];
        for level in &mut levels {
            reset_level(level);
        }
        assert_eq!(levels[0].ext.len(), 0);
        assert_eq!(levels[0].nodes.len(), 0);
        assert!(!levels[0].has_multi_pair());
    }

    #[test]
    fn pairs_view_into_unpacked_returns_direct_borrow() {
        // On unpacked levels, pairs_view_into returns a slice equivalent to
        // pairs_of_idx — no decoding, no copy (semantically; the test asserts
        // value equivalence, not identity).
        let mut lvl = TddLevel::new();
        let multi: Vec<InputPair> = (0..7u32)
            .map(|i| InputPair { left: LocalNodeIdx(i), right: LocalNodeIdx(i + 20) })
            .collect();
        lvl.try_push_internal_node(&multi).unwrap();
        lvl.try_push_internal_node(&[InputPair {
            left: LocalNodeIdx(50),
            right: LocalNodeIdx(60),
        }]).unwrap();

        let mut scratch = Vec::new();
        let view = lvl.pairs_view_into(0, &mut scratch);
        assert_eq!(view, &multi[..]);
        // Inline node: scratch is used to materialize.
        let view = lvl.pairs_view_into(1, &mut scratch);
        assert_eq!(view.len(), 1);
        assert_eq!(view[0], InputPair { left: LocalNodeIdx(50), right: LocalNodeIdx(60) });
    }

