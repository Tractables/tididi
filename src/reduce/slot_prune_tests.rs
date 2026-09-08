use crate::check::marg::test_fixtures::{toy, toy_weighted, BIG};
use crate::check::marg::check_slot_count_uniqueness;
use super::*;

/// Extract the exact `BigRational`s from a weighted store slice (these tests
/// run in the default exact mode, so every value is `WeightVal::Exact`).
/// `pub(super)` so the sibling `compact_store_in_place_tests` module shares this
/// one extractor instead of keeping a second copy in sync.
pub(super) fn exact_vals(vals: &[crate::query::WeightVal]) -> Vec<num_rational::BigRational> {
    vals.iter()
        .map(|v| match v {
            crate::query::WeightVal::Log(_) => panic!("test expected exact-mode value"),
            v => v.as_rational().into_owned(),
        })
        .collect()
}

/// Weighted boundary value-dedup (the weighted analogue of
/// `prune_merges_equal_value_referenced_slots`): two referenced slots holding
/// equal `BigRational` values must merge to one output slot, parent refs to
/// both rewritten onto the survivor, and `retired_marg_width` (the weighted
/// width carrier) SET to the new length. Fails on `main`, where the weighted
/// branch early-returns default stats and leaves the store full-width.
#[test]
fn weighted_prune_merges_equal_value_slots() {
    use crate::weight_store::Precision;
    use crate::query::RationalWeights;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let r = |a: i64, b: i64| BigRational::new(BigInt::from(a), BigInt::from(b));

    // toy_weighted → balanced(3), marginal side INTERNAL. Two parent nodes:
    // node0 right-refs slot0,
    // node1 right-refs slot1; both slots hold 3/7.
    let ws = crate::weight_store::WeightStore::new(
        RationalWeights::from_weights(&[(r(1, 2), r(1, 2))]),
        Precision::Exact,
    );
    let mut tdd = toy_weighted(ws, vec![r(3, 7), r(3, 7)], &[&[(0, 0)], &[(0, 1)]]);
    let stats = prune_marg_slots(&mut tdd);

    let (v, parent, side) = boundary_marginal_levels(&tdd)[0];
    let new_vals = exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());
    let width = tdd.levels[v.idx()].retired_marg_width;
    let mut buf = crate::marg_slots::RefSlotScratch::default();
    let refs = referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut buf);

    assert_eq!(new_vals.len(), 1, "equal-valued slots must merge to one");
    assert_eq!(new_vals[0], r(3, 7), "survivor keeps the value");
    assert_eq!(width, 1, "retired_marg_width must be SET to the new width");
    assert_eq!(stats.slots_freed, 1, "one duplicate slot freed");
    assert_eq!(stats.values_merged, 1, "one value-dedup merge");
    assert_eq!(stats.value_merged_levels, vec![v.0], "merged level reported for twin-scan");
    assert_eq!(refs, vec![0], "both parent refs remap to the merged slot 0");
}

/// Weighted orphan compaction (no dedup): three distinct-valued slots, only
/// slot 1 referenced → store compacts to that one value at index 0, the parent
/// ref remaps, and two slots are freed. Fails on `main` (full-width store).
#[test]
fn weighted_prune_compacts_orphans() {
    use crate::weight_store::Precision;
    use crate::query::RationalWeights;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let r = |a: i64, b: i64| BigRational::new(BigInt::from(a), BigInt::from(b));

    let ws = crate::weight_store::WeightStore::new(
        RationalWeights::from_weights(&[(r(1, 2), r(1, 2))]),
        Precision::Exact,
    );
    let mut tdd = toy_weighted(ws, vec![r(1, 1), r(2, 1), r(3, 1)], &[&[(0, 1)]]);
    let stats = prune_marg_slots(&mut tdd);

    let (v, parent, side) = boundary_marginal_levels(&tdd)[0];
    let new_vals = exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());
    let width = tdd.levels[v.idx()].retired_marg_width;
    let mut buf = crate::marg_slots::RefSlotScratch::default();
    let refs = referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut buf);

    assert_eq!(new_vals, vec![r(2, 1)], "only the referenced slot's value survives");
    assert_eq!(width, 1, "retired_marg_width SET to compacted width");
    assert_eq!(stats.slots_freed, 2, "two orphan slots freed");
    assert_eq!(stats.values_merged, 0, "no value-dedup (all distinct)");
    assert_eq!(refs, vec![0], "parent ref remapped to compacted slot 0");
}

/// Boundary compaction: unreferenced slots drop, the surviving ref is
/// remapped onto the compacted index.
#[test]
fn prune_compacts_boundary_store_and_remaps() {
    // Slot 1 referenced; slots 0 and 2 orphaned.
    let mut tdd = toy(vec![BIG + 7, BIG + 1, BIG + 7], &[&[(0, 1)]]);
    let stats = prune_marg_slots(&mut tdd);
    assert_eq!(stats.slots_freed, 2);
    let v = {
        let mut it = boundary_marginal_levels(&tdd).into_iter();
        it.next().unwrap().0
    };
    let counts = tdd.levels[v.idx()].marginal_counts.as_ref().unwrap();
    assert_eq!(counts.as_slice(), &[BIG + 1]);
    // The parent ref now points at compacted slot 0.
    let mut buf = crate::marg_slots::RefSlotScratch::default();
    let (_, parent, side) = boundary_marginal_levels(&tdd)[0];
    let refs = referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut buf);
    assert_eq!(refs, vec![0]);
}

/// `prune_marg_slots` decreases `node_count()` honestly (surviving
/// circuit only) while tallying freed slots into `retired_marg_width` /
/// `retired_marg_total()` for the minimize-gate threshold-offset logic.
#[test]
fn prune_shrinks_total_nodes_and_tallies_retired() {
    let mut tdd = toy(vec![BIG + 7, BIG + 1, BIG + 7], &[&[(0, 1)]]);
    let nodes_before = tdd.node_count();
    prune_marg_slots(&mut tdd);
    // Boundary level: 3 slots, 1 referenced → 2 freed.
    let v = {
        let mut it = boundary_marginal_levels(&tdd).into_iter();
        it.next().unwrap().0
    };
    assert_eq!(
        tdd.levels[v.idx()].retired_marg_width,
        2,
        "boundary sweep must retire 2 freed slots"
    );
    assert_eq!(
        tdd.retired_marg_total(),
        2,
        "retired_marg_total() must equal the freed count"
    );
    assert_eq!(
        tdd.node_count(),
        nodes_before - 2,
        "node_count() must decrease by the freed count (honest metric)"
    );
}

/// A fully-referenced store is untouched.
#[test]
fn prune_keeps_dense_store() {
    let mut tdd = toy(vec![BIG + 1, BIG + 2], &[&[(0, 0), (1, 1)]]);
    let stats = prune_marg_slots(&mut tdd);
    assert_eq!(stats.slots_freed, 0);
}

/// A marginalized level whose vtree PARENT level is also marginal (a "dead
/// deep store") must cost nothing after slot-prune: its counts were consumed
/// by the parent at cascade-marginalize time and the store is unreachable.
/// `prune_marg_slots` must clear and shrink both `marginal_counts` and
/// `marginal_counts_big` to zero capacity while leaving the level in
/// marginal mode. The root (output) marginal store is exempt.
#[test]
fn deep_marginal_store_cleared_to_zero_footprint() {
    use crate::diagram::{LocalNodeIdx, TddLevel, TddNodeId};
    use crate::vtree::Vtree;
    use num_bigint::BigUint;
    use std::sync::Arc;

    // balanced(4): 4 leaves + 3 internal nodes = 7 nodes total.
    // After reindex_bottomup the root is at index num_nodes-1.
    let vtree = Arc::new(Vtree::balanced(4));
    let root = vtree.root();
    let (_v_left, v_right) = vtree.children(root);

    // v_right must be an internal vtree node (has a parent = root).
    assert!(
        matches!(*vtree.node(v_right), crate::vtree::VtreeNode::Internal { .. }),
        "v_right must be internal so it has the root as parent"
    );

    // 3 slots: a plain count, the u128::MAX overflow sentinel, and another plain count.
    // The big side-table must be present and carry Some(BigUint) at the sentinel slot.
    let deep_counts: Vec<u128> = vec![42u128, u128::MAX, 99u128];
    let deep_big: Option<crate::diagram::BigSide> =
        Some([(1u32, BigUint::from(1_000_000_u64))].into_iter().collect());
    let deep_slot_count = deep_counts.len(); // 3

    // Root (output level): marginal with 2 slots — must remain untouched.
    let root_counts: Vec<u128> = vec![100u128, 200u128];
    let root_slot_count = root_counts.len(); // 2

    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();

    // Make v_right a dead deep store: marginal with 3 slots (one big-sentinel).
    // Its parent (root) is also marginal → no pairs reference it → unreachable.
    levels[v_right.idx()].make_marginal(deep_counts, deep_big);

    // Make root the output (marginal) level. Root has no parent, so it is always
    // exempt from the deep-store loop regardless; the out_v check is the belt.
    levels[root.idx()].make_marginal(root_counts, None);

    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = crate::diagram::Tdd::with_levels(vtree, levels, output);

    // Precondition: both levels are marginal, deep store has 3 slots.
    assert_eq!(tdd.levels[v_right.idx()].width(), deep_slot_count);
    assert_eq!(tdd.levels[root.idx()].width(), root_slot_count);

    let stats = prune_marg_slots(&mut tdd);

    // ── Stats ────────────────────────────────────────────────────────────
    assert_eq!(
        stats.stores_cleared, 1,
        "exactly one dead deep store must be cleared"
    );
    assert_eq!(
        stats.slots_freed, deep_slot_count,
        "all {} slots of the deep store must be freed", deep_slot_count
    );

    // ── Deep level: zero footprint, still in marginal mode ───────────────
    let deep = &tdd.levels[v_right.idx()];

    assert!(
        deep.is_marginal(),
        "deep level must remain in marginal mode after clearing"
    );

    let counts_vec = deep.marginal_counts.as_ref()
        .expect("marginal_counts must stay Some (keeps marginal mode)");
    assert_eq!(counts_vec.len(), 0, "deep store len must be 0 after clear");
    assert_eq!(
        counts_vec.capacity(), 0,
        "deep store capacity must be 0 after shrink_to_fit (zero allocation)"
    );

    // marginal_counts_big: present and emptied (stays Some, not collapsed to None).
    let big_side = deep.marginal_counts_big.as_ref()
        .expect("marginal_counts_big must stay Some after clear (clear path keeps it)");
    assert_eq!(big_side.len(), 0, "big side-table must hold no entries after clear");
    assert_eq!(
        big_side.bytes(), 0,
        "big side-table must own no heap after shrink_to_fit"
    );

    assert_eq!(
        deep.width(), 0,
        "width() must be 0 (delegates to marginal_counts.len())"
    );

    // nodes/pairs/ext are already empty after make_marginal; confirm no resurrection.
    assert_eq!(deep.nodes.len(), 0, "no node slots on cleared deep level");
    assert_eq!(deep.pairs.len(), 0, "no pair arena entries on cleared deep level");

    // retired_marg_width captures freed slots for minimize-gate bookkeeping.
    assert_eq!(
        deep.retired_marg_width, deep_slot_count as u32,
        "retired_marg_width must equal the freed slot count"
    );

    // ── Root (output) store: untouched ────────────────────────────────────
    let root_level = &tdd.levels[root.idx()];
    let root_counts_vec = root_level.marginal_counts.as_ref()
        .expect("root marginal_counts must still be Some");
    assert_eq!(
        root_counts_vec.len(), root_slot_count,
        "root (output) store must be untouched: still {} slots", root_slot_count
    );
}

/// Value-dedup in the boundary compaction pass:
/// two referenced slots holding equal values must merge to one output slot,
/// and parent refs to both must be rewritten to the surviving slot.
///
/// This is where slot-count uniqueness is established for apply-emit-born stores
/// (the emit site is forbidden from deduping — see conjoin/mod.rs comment).
#[test]
fn prune_merges_equal_value_referenced_slots() {
    // Two nodes, each referencing one slot; both slots have equal value BIG+42.
    // Node 0: right-ref = 0 (slot 0 → BIG+42)
    // Node 1: right-ref = 1 (slot 1 → BIG+42)
    // After prune: store collapses to 1 slot; both refs become 0.
    let mut tdd = toy(vec![BIG + 42, BIG + 42], &[&[(0, 0)], &[(0, 1)]]);

    // Pre-condition: slot-count uniqueness is violated (duplicate slot values).
    assert!(
        check_slot_count_uniqueness(&tdd).is_err(),
        "pre-prune: slot values must start out duplicated"
    );

    let stats = prune_marg_slots(&mut tdd);

    // (a) Unique values: slot-count uniqueness holds after prune.
    check_slot_count_uniqueness(&tdd)
        .expect("post-prune: no duplicate slot values");

    // (b) Store collapsed to 1 slot; 1 slot freed.
    let v = boundary_marginal_levels(&tdd).into_iter().next().unwrap().0;
    let counts = tdd.levels[v.idx()].marginal_counts.as_ref().unwrap();
    assert_eq!(counts.len(), 1, "equal slots must merge to one output slot");
    assert_eq!(counts[0], BIG + 42, "surviving slot must hold the original value");
    assert_eq!(stats.slots_freed, 1, "one duplicate slot must be freed");

    // (c) Parent refs both decode to slot 0 after remap.
    let mut buf = crate::marg_slots::RefSlotScratch::default();
    let (_, parent, side) = boundary_marginal_levels(&tdd)[0];
    let refs = referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut buf);
    assert_eq!(refs, vec![0], "both parent refs must decode to the merged slot 0");
}

/// The sweep scratch pool carries CAPACITY across sweeps, never state: a
/// leftover `referenced` list would compact the next boundary store to the
/// PREVIOUS level's slot set, and a leftover `remap` would be re-resized over
/// rather than rebuilt. Pins the take-side clear.
#[test]
fn sweep_scratch_is_cleared_on_take() {
    let mut dirty = RefSlotScratch::default();
    dirty.referenced.push(7);
    return_sweep_scratch(dirty, vec![1, 2, 3]);

    let (slots, remap) = take_sweep_scratch();
    assert!(slots.referenced.is_empty(), "referenced must be cleared on take");
    assert!(remap.is_empty(), "remap must be cleared on take");
}

#[cfg(test)]
mod compact_store_in_place_tests {
    use super::*;
    use crate::check::marg::test_fixtures::{toy, BIG};
    use num_bigint::BigUint;

    /// In-place boundary compaction on a sparse referenced set: the compacted
    /// store holds ONE entry per surviving slot, in ascending referenced order,
    /// with equal values deduped onto the first survivor that carries them.
    /// Expectations are hand-derived from the fixture below:
    ///
    /// store: slot0 = Big(b9), slot1 = BIG+5, slot2 = Big(b1), slot3 = Big(b1),
    ///        slot4 = BIG+5, slot5 = BIG+7;  referenced = [1, 2, 3, 5].
    ///
    /// - slot1 → new 0 (first survivor); slot0's `b9` must NOT linger at new 0
    /// - slot2 → new 1, carrying `b1` down with it
    /// - slot3 → new 1 as a value-dedup merge (equal `BigUint`)
    /// - slot5 → new 2
    /// - slot0 and slot4 are orphans and must not seed the dedup map (slot4
    ///   holds the same value as the surviving slot1, so a leaked orphan would
    ///   show up as a bogus extra merge)
    #[test]
    fn compact_store_in_place_dedups_and_moves_survivors() {
        let b1 = BigUint::from(u128::MAX) + BigUint::from(1u32);
        let b9 = BigUint::from(u128::MAX) + BigUint::from(9u32);
        let mut tdd = toy(
            vec![u128::MAX, BIG + 5, u128::MAX, u128::MAX, BIG + 5, BIG + 7],
            &[&[(0, 1)]],
        );
        let v = boundary_marginal_levels(&tdd)[0].0;
        tdd.levels[v.idx()].marginal_counts_big = Some(
            [(0u32, b9), (2u32, b1.clone()), (3u32, b1.clone())].into_iter().collect(),
        );

        let mut remap = vec![u32::MAX; 6];
        let (new_len, values_merged) =
            IntFold::compact_store(&mut tdd, v, &[1, 2, 3, 5], &mut remap);

        assert_eq!(new_len, 3, "three distinct values survive");
        assert_eq!(values_merged, 1, "slot3 merges onto slot2's compacted slot");
        let level = &tdd.levels[v.idx()];
        assert_eq!(
            level.marginal_counts.as_deref().unwrap(),
            &[BIG + 5, u128::MAX, BIG + 7],
            "survivors move down in referenced order",
        );
        let big = level.marginal_counts_big.as_ref().unwrap();
        assert_eq!(
            (big.len(), big.get(1)),
            (1, Some(&b1)),
            "big side table tracks the moves; no stale BigUint at a reused slot, \
             and the orphaned `b9` is dropped rather than carried along",
        );
        assert_eq!(
            remap,
            vec![u32::MAX, 0, 1, 1, u32::MAX, 2],
            "orphans keep the u32::MAX sentinel; merged refs share a slot",
        );
    }

    /// Weighted twin of the test above, on the same sparse referenced set. The
    /// weight store moves survivors by SWAP (a `WeightVal` is not `Copy`), so
    /// each step writes two slots instead of one — this pins that the extra
    /// write still lands only on already-consumed positions, and that nothing
    /// swapped up out of the prefix survives the truncate. Values are
    /// past-`u128` rationals at scattered slots, so a clobbered or stale slot
    /// cannot coincidentally match the expected one.
    ///
    /// store: slot0 = 3H, slot1 = H/7, slot2 = 2H/5, slot3 = 2H/5,
    ///        slot4 = H/7, slot5 = 3H/11;  referenced = [1, 2, 3, 5].
    ///
    /// - slot1 → new 0 (first survivor); slot0's `3H` must NOT linger at new 0
    /// - slot2 → new 1, carrying `2H/5` down with it
    /// - slot3 → new 1 as a value-dedup merge (equal rational)
    /// - slot5 → new 2
    /// - slot0 and slot4 are orphans and must not seed the dedup map (slot4
    ///   repeats the surviving slot1's value, so a leaked orphan would show up
    ///   as a bogus extra merge)
    #[test]
    fn weighted_compact_store_in_place_dedups_and_moves_survivors() {
        use crate::query::RationalWeights;
        use crate::weight_store::Precision;
        use crate::check::marg::test_fixtures::toy_weighted;
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // H is past `u128`, the weighted analogue of the integer test's
        // `BigUint` entries.
        let h = BigInt::from(u128::MAX) * BigInt::from(u128::MAX);
        let wr = |n: u32, d: u32| BigRational::new(&h * BigInt::from(n), BigInt::from(d));
        let (v_a, v_b, v_c) = (wr(1, 7), wr(2, 5), wr(3, 11));

        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        let ws = crate::weight_store::WeightStore::new(
            RationalWeights::from_weights(&[(half.clone(), half)]),
            Precision::Exact,
        );
        let mut tdd = toy_weighted(
            ws,
            vec![wr(9, 3), v_a.clone(), v_b.clone(), v_b.clone(), v_a.clone(), v_c.clone()],
            &[&[(0, 1)]],
        );
        let v = boundary_marginal_levels(&tdd)[0].0;

        let mut remap = vec![u32::MAX; 6];
        let (new_len, values_merged) =
            WeightFold::compact_store(&mut tdd, v, &[1, 2, 3, 5], &mut remap);

        let vals = super::exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());

        assert_eq!(new_len, 3, "three distinct values survive");
        assert_eq!(values_merged, 1, "slot3 merges onto slot2's compacted slot");
        assert_eq!(vals, vec![v_a, v_b, v_c], "survivors move down in referenced order");
        assert_eq!(
            remap,
            vec![u32::MAX, 0, 1, 1, u32::MAX, 2],
            "orphans keep the u32::MAX sentinel; merged refs share a slot",
        );
    }
}
