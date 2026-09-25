    use super::*;
    use crate::Engine;

    #[test]
    fn level_pairs_iter_of_unpacked_matches_slice() {
        // PairsIter::Slice and PairsIter::Inline branches: yield the same
        // pairs as `pairs_of_idx` on an unpacked level.
        let mut lvl = TddLevel::new();
        // Push some multi-pair nodes and an inline pair.
        let three = vec![
            ChildPair::new(NodeIdx(1), NodeIdx(2)),
            ChildPair::new(NodeIdx(3), NodeIdx(4)),
            ChildPair::new(NodeIdx(5), NodeIdx(6)),
        ];
        lvl.push_internal_node(&three);
        lvl.push_internal_node(&[ChildPair::new(NodeIdx(99), NodeIdx(100))]);

        let from_slice: Vec<ChildPair> = lvl.pairs_of_idx(0).to_vec();
        let from_iter: Vec<ChildPair> = lvl.pairs_iter_of_idx(0).collect();
        assert_eq!(from_slice, from_iter);

        let inline_slice: Vec<ChildPair> = lvl.pairs_of_idx(1).to_vec();
        let inline_iter: Vec<ChildPair> = lvl.pairs_iter_of_idx(1).collect();
        assert_eq!(inline_slice, inline_iter);
        assert_eq!(inline_slice.len(), 1);
    }

    #[test]
    fn level_pairs_iter_of_idx_size_hint() {
        // ExactSizeIterator: `size_hint` correctly reports the count for all
        // four variants.
        let mut lvl = TddLevel::new();
        lvl.push_internal_node(&[
            ChildPair::new(NodeIdx(1), NodeIdx(2)),
            ChildPair::new(NodeIdx(3), NodeIdx(4)),
            ChildPair::new(NodeIdx(5), NodeIdx(6)),
        ]);
        let it = lvl.pairs_iter_of_idx(0);
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn test_take_levels_fresh_allocation() {
        let eng = &Engine::new();
        let levels = take_levels(eng, 5);
        assert_eq!(levels.len(), 5);
        for level in &levels {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
            assert!(!level.has_multi_pair());
        }
    }

    #[test]
    fn a_refused_level_array_growth_answers_over_budget() {
        let eng = &Engine::new();
        eng.limits().refuse_nth_reserve(0);
        assert_eq!(try_take_levels(eng, 3).unwrap_err(), crate::OperationError::OverBudget);
        // The allocator-only take does not consult the budget.
        let levels = take_levels(eng, 3);
        assert_eq!(levels.len(), 3);
        return_levels(eng, PoolSlot::First, levels);
        eng.limits().grant_every_reserve();
        // A parked array long enough for the request grows nothing.
        eng.limits().refuse_nth_reserve(0);
        assert_eq!(try_take_levels(eng, 2).unwrap().len(), 2);
        eng.limits().grant_every_reserve();
        assert_eq!(try_take_levels(eng, 5).unwrap().len(), 5);
    }

    #[test]
    fn test_pool_roundtrip() {
        let eng = &Engine::new();
        // Take fresh, return, take again — should reuse
        let levels = take_levels(eng, 3);
        assert_eq!(levels.len(), 3);
        return_levels(eng, PoolSlot::First, levels);
        let levels2 = take_levels(eng, 3);
        assert_eq!(levels2.len(), 3);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }
    }

    #[test]
    fn test_pool_size_mismatch_resizes() {
        let eng = &Engine::new();
        // A parked entry of the wrong length is RESIZED to the request, not
        // discarded — that is what keeps the arenas warm when components of
        // different variable counts alternate. Both directions, and every level
        // handed out is still empty (the reset barrier is not skipped).
        eng.clear_scratch();

        // Shrink: return a Vec of size 5, then request size 3.
        let levels = take_levels(eng, 5);
        return_levels(eng, PoolSlot::First, levels);
        let levels2 = take_levels(eng, 3);
        assert_eq!(levels2.len(), 3);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }

        // Grow: return that size-3 Vec, then request size 6.
        return_levels(eng, PoolSlot::First, levels2);
        let levels3 = take_levels(eng, 6);
        assert_eq!(levels3.len(), 6);
        for level in &levels3 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
        }
    }

    #[test]
    fn test_pool2_roundtrip() {
        let eng = &Engine::new();
        // Test secondary pool
        let levels = take_levels(eng, 4);
        return_levels(eng, PoolSlot::Second, levels);
        let levels2 = take_levels(eng, 4);
        assert_eq!(levels2.len(), 4);
    }

    #[test]
    fn test_reset_levels_clears_state() {
        let eng = &Engine::new();
        let mut levels = take_levels(eng, 2);
        // Dirty the levels with internal nodes
        let dummy = ChildPair::new(NodeIdx(0), NodeIdx(0));
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
    /// pool recycle: a retention check that inspects only `nodes.capacity()`
    /// keeps a level whose `pairs` arena is arbitrarily large.
    /// The retained-capacity accounting then reports that arena against a
    /// one-clause diagram and the apply budget fires immediately.
    #[test]
    fn test_pool_shrinks_oversized_pair_capacity() {
        let eng = &Engine::new();
        // Pre-flush the pool to make this test deterministic regardless of what
        // the engine's pool already holds.
        eng.clear_scratch();
        let mut levels = take_levels(eng, 3);
        // A level with tiny content but huge `pairs` capacity.
        levels[1].pairs.reserve(8_000_000);
        let bloated_cap = levels[1].pairs.capacity();
        assert!(bloated_cap >= 8_000_000, "reserve didn't grow capacity");
        // The pool retention gate (nodes ≤ 4M) passes — total node capacity is
        // 0 — so this Vec is retained. The oversized pair arena must be gone
        // by the time it is parked, not merely by the time it is handed out:
        // otherwise the bytes sit in the pool for the whole gap until some
        // later consumer asks for levels of this length.
        return_levels(eng, PoolSlot::First, levels);
        // Take the recycled Vec back — same length, so the pool hands back the
        // very entry it parked, trimmed.
        let levels2 = take_levels(eng, 3);
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
            kept_cap.saturating_mul(size_of::<ChildPair>()) <= super::pool::MAX_LEVEL_ARENA_BYTES,
            "pairs.capacity() = {} ({} bytes) exceeds MAX_LEVEL_ARENA_BYTES = {}",
            kept_cap, kept_cap.saturating_mul(size_of::<ChildPair>()),
            super::pool::MAX_LEVEL_ARENA_BYTES,
        );
    }

    #[test]
    fn test_pool_returns_clean_levels() {
        let eng = &Engine::new();
        // Dirty some levels, return to pool, take back — should be clean
        let mut levels = take_levels(eng, 2);
        let dummy = ChildPair::new(NodeIdx(0), NodeIdx(0));
        levels[0].push_internal_node(&[dummy]);
        levels[1].push_internal_node(&[dummy, dummy]);
        return_levels(eng, PoolSlot::First, levels);
        let levels2 = take_levels(eng, 2);
        for level in &levels2 {
            assert_eq!(level.nodes.len(), 0);
            assert!(level.pairs.is_empty());
            assert!(!level.has_multi_pair());
        }
    }

    #[test]
    fn test_tdd_level_width() {
        let mut level = TddLevel::new();
        assert_eq!(level.slot_count(), 0);
        let dummy = ChildPair::new(NodeIdx(0), NodeIdx(0));
        level.push_internal_node(&[dummy]);
        assert_eq!(level.slot_count(), 1);
        level.push_internal_node(&[dummy]);
        assert_eq!(level.slot_count(), 2);
    }

    #[test]
    fn test_push_internal_node() {
        let mut level = TddLevel::new();
        let pairs = [
            ChildPair::new(NodeIdx(0), NodeIdx(1)),
            ChildPair::new(NodeIdx(1), NodeIdx(0)),
        ];
        let idx = level.push_internal_node(&pairs);
        assert_eq!(idx, NodeIdx(0));
        assert_eq!(level.slot_count(), 1);
        assert_eq!(level.pairs.len(), 2);
        assert!(level.has_multi_pair());
        let node_pairs = level.pairs_of_idx(0);
        assert_eq!(node_pairs.len(), 2);
        assert_eq!(node_pairs[0], pairs[0]);
        assert_eq!(node_pairs[1], pairs[1]);
    }

    #[test]
    fn test_tdd_is_zero() {
        let eng = &Engine::new();
        use crate::vtree::Vtree;
        use std::sync::Arc;
        let vtree = Arc::new(Vtree::balanced(2));
        let levels = take_levels(eng, vtree.num_nodes());
        let tdd = Tdd::from_levels_unchecked(
            vtree.clone(),
            levels,
            TddNodeId { vtree: vtree.root(), local: ZERO },
        );
        assert!(tdd.is_zero());
    }

    #[test]
    fn test_node_size() {
        assert_eq!(std::mem::size_of::<EncodedNode>(), 8, "EncodedNode should be 8 bytes (packed struct)");
    }

    #[test]
    fn test_encode_multi_promotes_to_ranged_on_huge_start() {
        // A pair_start at 2^31 triggers the ranged encoding even though
        // pair_len is small. Verifies the side-table round-trips the stored values.
        let mut level = TddLevel::new();
        let huge_start = 1usize << 31;
        let data = level.encode_multi(huge_start, 3);
        level.nodes.push(data);
        assert!(
            matches!(data.kind(), NodeKind::MultiRanged(_)),
            "huge-start node should promote to ranged, got {:?}", data.kind()
        );
        assert_eq!(level.pair_range_at(0), huge_start..huge_start + 3);
        assert_eq!(level.pair_count_at(0), 3);
        assert_eq!(level.ranges.len(), 1);
    }

    #[test]
    fn test_encode_multi_promotes_to_ranged_on_huge_len() {
        // A pair_len at 2^31 triggers the ranged encoding. We don't actually
        // allocate that much arena memory — `encode_multi` only stores the count
        // and `range_idx`; the arena is the caller's concern.
        let mut level = TddLevel::new();
        let huge_len = 1usize << 31;
        let data = level.encode_multi(0, huge_len);
        level.nodes.push(data);
        assert!(
            matches!(data.kind(), NodeKind::MultiRanged(_)),
            "huge-len node should be ranged, got {:?} — mis-reading it as a leaf is \
             what made qmr-100 come back UNSAT", data.kind()
        );
        assert_eq!(level.pair_range_at(0).len(), huge_len);
    }

    #[test]
    fn test_encode_multi_stays_normal_for_small_values() {
        // Normal-sized multi nodes don't allocate an ranges slot — the packed
        // 8-byte encoding handles them.
        let mut level = TddLevel::new();
        let data = level.encode_multi(100, 5);
        level.nodes.push(data);
        assert!(
            matches!(data.kind(), NodeKind::Multi { .. }),
            "small multi should stay in packed form, got {:?}", data.kind()
        );
        assert_eq!(level.pair_range_at(0), 100..105);
        assert_eq!(level.ranges.len(), 0, "no ranges slot allocated for packed form");
    }

    #[test]
    fn test_ranged_set_pair_len_updates_side_table() {
        // Shrinking an ranged node's pair_len must update the side table,
        // not the node's b field (which is the ranged-form sentinel).
        let mut level = TddLevel::new();
        let data = level.encode_multi(0, 1 << 31);
        level.nodes.push(data);
        level.set_pair_len(0, 100);
        assert!(
            matches!(level.nodes[0].kind(), NodeKind::MultiRanged(_)),
            "still ranged after shrink"
        );
        assert_eq!(level.pair_range_at(0).len(), 100);
    }

    #[test]
    fn test_reset_levels_clears_ext() {
        let mut level = TddLevel::new();
        let data = level.encode_multi(1 << 31, 3);
        level.nodes.push(data);
        assert_eq!(level.ranges.len(), 1);
        let mut levels = vec![level];
        for level in &mut levels {
            reset_level(level);
        }
        assert_eq!(levels[0].ranges.len(), 0);
        assert_eq!(levels[0].nodes.len(), 0);
        assert!(!levels[0].has_multi_pair());
    }


mod try_from_levels {
    use std::sync::Arc;

    use num_bigint::BigUint;


    use crate::diagram::{
        CountOverflow, ChildPair, NodeIdx, ValueRef, Tdd, TddBuildError, TddLevel, EncodedNode,
        TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO,
    };
    use crate::vtree::{Vtree, VtreeIdx};

    // (x1 ∧ x2) ∨ x3 over balanced(4); returns the levels and the output id.
    /// The invariant list on levels assembled outside a builder, which is what
    /// these tests check: `check_levels` is the body a `finish` runs.
    fn try_from_levels(
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
    ) -> Result<Tdd, TddBuildError> {
        crate::diagram::builder::check_levels(&vtree, &levels, output, None)?;
        Ok(Tdd::from_levels_unchecked(vtree, levels, output))
    }

    fn build(vtree: &Vtree) -> (Vec<TddLevel>, TddNodeId) {
        let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
        let root = vtree.root();
        let (l, r) = vtree.children(root);
        let and = levels[l.idx()]
            .push_internal_node(&[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]);
        let x3 = levels[r.idx()]
            .push_internal_node(&[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]);
        let all = levels[r.idx()]
            .push_internal_node(&[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)]);
        // Pairs of one node are mutually exclusive: the second pair covers
        // ¬(x1 ∧ x2) explicitly.
        let nand = levels[l.idx()].push_internal_node(&[
            ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX),
            ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX),
        ]);
        let out = levels[root.idx()].push_internal_node(&[
            ChildPair::new(and, all),
            ChildPair::new(nand, x3),
        ]);
        (levels, TddNodeId { vtree: root, local: out })
    }

    #[test]
    fn well_formed_diagram_counts_and_minimizes() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (levels, out) = build(&vtree);
        let mut f = try_from_levels(vtree.clone(), levels, out).unwrap();
        // (x1∧x2)∨x3 has 10 models over 4 variables.
        assert_eq!(f.model_count().unwrap(), BigUint::from(10u32));
        // `all` and `x3` overlap (x3 ⊂ all): not canonical, but minimize accepts it.
        f.minimize().unwrap();
        assert_eq!(f.model_count().unwrap(), BigUint::from(10u32));
    }

    #[test]
    fn zero_output_is_accepted() {
        let vtree = Arc::new(Vtree::balanced(2));
        let levels = vec![TddLevel::new(); vtree.num_nodes()];
        let out = TddNodeId { vtree: vtree.root(), local: ZERO };
        let f = try_from_levels(vtree, levels, out).unwrap();
        assert!(f.is_zero());
    }

    #[test]
    fn level_count_mismatch() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (mut levels, out) = build(&vtree);
        levels.pop();
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::LevelCountMismatch { expected: 7, found: 6 })
        );
    }

    #[test]
    fn non_empty_leaf_level() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (mut levels, out) = build(&vtree);
        let (l, _) = vtree.children(vtree.root());
        let (leaf, _) = vtree.children(l);
        levels[leaf.idx()]
            .push_internal_node(&[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)]);
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::NonEmptyLeafLevel(leaf))
        );
    }

    #[test]
    fn marginal_leaf_level_with_a_foreign_column() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (mut levels, out) = build(&vtree);
        let (l, _) = vtree.children(vtree.root());
        let (leaf, _) = vtree.children(l);
        levels[leaf.idx()].become_marginal(vec![2, 1, 1], None);
        assert!(try_from_levels(vtree.clone(), levels.clone(), out).is_ok());
        levels[leaf.idx()].become_marginal(vec![2, 1, 1, 5], None);
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::NonEmptyLeafLevel(leaf))
        );
    }

    #[test]
    fn child_index_out_of_range() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (mut levels, out) = build(&vtree);
        let root = vtree.root();
        let (_, r) = vtree.children(root);
        let bad = ChildPair::new(ONE_LEAF_IDX, NodeIdx(7));
        let node = levels[root.idx()].push_internal_node(&[bad]);
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::ChildIndexOutOfRange { level: root, node, pair: bad, child: r })
        );
    }

    #[test]
    fn leaf_index_past_the_three_implicit_nodes() {
        let vtree = Arc::new(Vtree::balanced(2));
        let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
        let root = vtree.root();
        let (l, _) = vtree.children(root);
        let bad = ChildPair::new(NodeIdx(3), NEG_LEAF_IDX);
        let node = levels[root.idx()].push_internal_node(&[bad]);
        let out = TddNodeId { vtree: root, local: node };
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::ChildIndexOutOfRange { level: root, node, pair: bad, child: l })
        );
    }

    #[test]
    fn reserved_bit_rejected() {
        let vtree = Arc::new(Vtree::balanced(2));
        let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
        let root = vtree.root();
        // Staged by hand: the encoder debug-asserts against a reserved side.
        let bad = ChildPair::new(POS_LEAF_IDX, ZERO);
        levels[root.idx()].nodes.push(EncodedNode { a: bad.left.raw(), b: bad.right.raw() });
        let node = NodeIdx(0);
        let out = TddNodeId { vtree: root, local: node };
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::ReservedBitSet { level: root, node, pair: bad })
        );
    }

    #[test]
    fn bad_output() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (levels, out) = build(&vtree);
        let off = TddNodeId { vtree: out.vtree, local: NodeIdx(out.local.0 + 1) };
        assert_eq!(
            try_from_levels(vtree.clone(), levels.clone(), off).err(),
            Some(TddBuildError::BadOutput(off))
        );
        let (l, _) = vtree.children(vtree.root());
        let wrong_level = TddNodeId { vtree: l, local: NodeIdx(0) };
        assert_eq!(
            try_from_levels(vtree, levels, wrong_level).err(),
            Some(TddBuildError::BadOutput(wrong_level))
        );
    }

    // Marginal left child with an overflowed slot; `backed` says whether the
    // side table carries its exact value.
    fn marginal_case(backed: bool) -> (Arc<Vtree>, Vec<TddLevel>, TddNodeId, VtreeIdx) {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
        let root = vtree.root();
        let (l, r) = vtree.children(root);
        for (leaf, _) in vtree.leaf_bottomup() {
            levels[leaf.idx()].become_marginal(Vec::new(), None);
        }
        let big = if backed {
            Some(CountOverflow::from_iter([(0u32, BigUint::from(1u32) << 130)]))
        } else {
            None
        };
        levels[l.idx()].become_marginal(vec![u128::MAX, 5], big);
        levels[r.idx()].become_marginal(vec![3], None);
        let out = levels[root.idx()].push_internal_node(&[
            ChildPair::new(ValueRef::Slot(0).side().unwrap(), ValueRef::Slot(0).side().unwrap()),
            ChildPair::new(ValueRef::Slot(1).side().unwrap(), ValueRef::Inline(2).side().unwrap()),
        ]);
        (vtree, levels, TddNodeId { vtree: root, local: out }, l)
    }

    #[test]
    fn marginal_children_resolve_and_count() {
        let (vtree, levels, out, _) = marginal_case(true);
        let f = try_from_levels(vtree, levels, out).unwrap();
        let expected = (BigUint::from(1u32) << 130) * 3u32 + BigUint::from(10u32);
        assert_eq!(f.model_count().unwrap(), expected);
    }

    #[test]
    fn overflow_without_value() {
        let (vtree, levels, out, l) = marginal_case(false);
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::OverflowWithoutValue { level: l, slot: 0 })
        );
    }

    #[test]
    fn marginal_slot_out_of_range() {
        let (vtree, mut levels, out, l) = marginal_case(true);
        let root = vtree.root();
        let bad = ChildPair::new(ValueRef::Slot(2).side().unwrap(), ValueRef::Inline(1).side().unwrap());
        let node = levels[root.idx()].push_internal_node(&[bad]);
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::ChildIndexOutOfRange { level: root, node, pair: bad, child: l })
        );
    }

    #[test]
    fn marginal_level_over_structural_child() {
        let vtree = Arc::new(Vtree::balanced(4));
        let (mut levels, _) = build(&vtree);
        let root = vtree.root();
        let (l, _) = vtree.children(root);
        levels[root.idx()] = TddLevel::new();
        levels[root.idx()].become_marginal(vec![1], None);
        let out = TddNodeId { vtree: root, local: NodeIdx(0) };
        assert_eq!(
            try_from_levels(vtree, levels, out).err(),
            Some(TddBuildError::MarginalNotDownwardClosed { level: root, child: l })
        );
    }
}

    #[test]
    fn checked_inline_node_reuse_does_not_request_allocation() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use crate::limits::{LimitConfig, MemoryHooks};
        let requests = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&requests);
        let hooks = MemoryHooks::new(move |_| { observed.fetch_add(1, Ordering::Relaxed); }, || 0, || None, || {});
        let eng = Engine::new();
        let mut level = TddLevel::new();
        level.nodes.reserve(1);
        let _scope = eng.limits().scope(LimitConfig::none().with_memory_hooks(hooks));
        let pair = ChildPair::new(NodeIdx(0), NodeIdx(0));
        let index = level.push_node(eng.limits(), &[pair]).unwrap();
        assert_eq!(index, NodeIdx(0));
        assert_eq!(level.pairs_of_idx(0), &[pair]);
        assert_eq!(requests.load(Ordering::Relaxed), 0);
    }
