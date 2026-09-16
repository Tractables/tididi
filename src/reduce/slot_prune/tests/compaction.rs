use crate::test_helpers::{toy, toy_weighted, BIG};
use crate::test_helpers::check::marginal::check_slot_count_uniqueness;
use super::*;

/// Extract the exact `BigRational`s from a weighted store slice (these tests
/// run in the default exact mode, so every value is `WeightValue::Exact`).
/// `pub(super)` so the sibling `compact_store_in_place_tests` module shares this
/// one extractor instead of keeping a second copy in sync.
pub(super) fn exact_vals(values: &[crate::diagram::WeightValue]) -> Vec<num_rational::BigRational> {
    values.iter()
        .map(|v| match v {
            crate::diagram::WeightValue::Log(_) => panic!("test expected exact-mode value"),
            v => v.as_rational().into_owned(),
        })
        .collect()
}

/// Weighted boundary value-dedup (the weighted analogue of
/// `prune_merges_equal_value_referenced_slots`): two referenced slots holding
/// equal `BigRational` values must merge to one output slot, parent refs to
/// both rewritten onto the survivor, and `weight_width` (the weighted
/// width carrier) SET to the new length. Fails on `main`, where the weighted
/// branch early-returns default stats and leaves the store full-width.
#[test]
fn weighted_prune_merges_equal_value_slots() {
    let eng = &crate::Engine::new();
    use crate::diagram::Arithmetic;
    use crate::diagram::{LiteralWeights, RationalWeights};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let r = |a: i64, b: i64| BigRational::new(BigInt::from(a), BigInt::from(b));

    // toy_weighted → balanced(3), marginal side INTERNAL. Two parent nodes:
    // node0 right-refs slot0,
    // node1 right-refs slot1; both slots hold 3/7.
    let ws = crate::diagram::WeightStore::new(
        RationalWeights::from_literals(&[LiteralWeights { negative: r(1, 2), positive: r(1, 2) }]),
        Arithmetic::ExactRational,
    );
    let mut tdd = toy_weighted(ws, vec![r(3, 7), r(3, 7)], &[&[(0, 0)], &[(0, 1)]]);
    let (v, parent, side) = boundary_marginal_levels(&tdd)[0];
    let slots_before = tdd.weights().unwrap().level(v.idx()).unwrap().len();
    let stats = prune_value_slots(eng, &mut tdd);

    let new_vals = exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());
    let width = tdd.levels[v.idx()].weight_width();
    let mut buf = crate::value::slots::RefSlotScratch::default();
    let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut buf);

    assert_eq!(new_vals.len(), 1, "equal-valued slots must merge to one");
    assert_eq!(new_vals[0], r(3, 7), "survivor keeps the value");
    assert_eq!(width, 1, "weight_width must be SET to the new width");
    assert_eq!(slots_before - new_vals.len(), 1, "one duplicate slot freed");
    assert_eq!(stats.values_merged, 1, "one value-dedup merge");
    assert_eq!(stats.value_merged_levels, vec![v.0], "merged level reported for twin-scan");
    assert_eq!(refs, vec![0], "both parent refs remap to the merged slot 0");
}

/// Weighted orphan compaction (no dedup): three distinct-valued slots, only
/// slot 1 referenced → store compacts to that one value at index 0, the parent
/// ref remaps, and two slots are freed. Fails on `main` (full-width store).
#[test]
fn weighted_prune_compacts_orphans() {
    let eng = &crate::Engine::new();
    use crate::diagram::Arithmetic;
    use crate::diagram::{LiteralWeights, RationalWeights};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let r = |a: i64, b: i64| BigRational::new(BigInt::from(a), BigInt::from(b));

    let ws = crate::diagram::WeightStore::new(
        RationalWeights::from_literals(&[LiteralWeights { negative: r(1, 2), positive: r(1, 2) }]),
        Arithmetic::ExactRational,
    );
    let mut tdd = toy_weighted(ws, vec![r(1, 1), r(2, 1), r(3, 1)], &[&[(0, 1)]]);
    let (v, parent, side) = boundary_marginal_levels(&tdd)[0];
    let slots_before = tdd.weights().unwrap().level(v.idx()).unwrap().len();
    let stats = prune_value_slots(eng, &mut tdd);

    let new_vals = exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());
    let width = tdd.levels[v.idx()].weight_width();
    let mut buf = crate::value::slots::RefSlotScratch::default();
    let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut buf);

    assert_eq!(new_vals, vec![r(2, 1)], "only the referenced slot's value survives");
    assert_eq!(width, 1, "weight_width SET to compacted width");
    assert_eq!(slots_before - new_vals.len(), 2, "two orphan slots freed");
    assert_eq!(stats.values_merged, 0, "no value-dedup (all distinct)");
    assert_eq!(refs, vec![0], "parent ref remapped to compacted slot 0");
}

/// Boundary compaction: unreferenced slots drop, the surviving ref is
/// remapped onto the compacted index.
#[test]
fn prune_compacts_boundary_store_and_remaps() {
    let eng = &crate::Engine::new();
    // Slot 1 referenced; slots 0 and 2 orphaned.
    let mut tdd = toy(vec![BIG + 7, BIG + 1, BIG + 7], &[&[(0, 1)]]);
    let v = boundary_marginal_levels(&tdd)[0].0;
    let slots_before = tdd.levels[v.idx()].marginal_counts().unwrap().len();
    prune_value_slots(eng, &mut tdd);
    let counts = tdd.levels[v.idx()].marginal_counts().unwrap();
    assert_eq!(counts, &[BIG + 1]);
    assert_eq!(slots_before - counts.len(), 2, "two orphan slots freed");
    // The parent ref now points at compacted slot 0.
    let mut buf = crate::value::slots::RefSlotScratch::default();
    let (_, parent, side) = boundary_marginal_levels(&tdd)[0];
    let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut buf);
    assert_eq!(refs, vec![0]);
}

/// `prune_value_slots` decreases `node_count()` honestly (surviving
/// circuit only) while tallying freed slots into `retired_marginal_slots` /
/// `retired_marginal_slots()` for the minimize-gate threshold-offset logic.
#[test]
fn prune_shrinks_total_nodes_and_tallies_retired() {
    let eng = &crate::Engine::new();
    let mut tdd = toy(vec![BIG + 7, BIG + 1, BIG + 7], &[&[(0, 1)]]);
    let nodes_before = tdd.node_count();
    prune_value_slots(eng, &mut tdd);
    // Boundary level: 3 slots, 1 referenced → 2 freed.
    let v = {
        let mut it = boundary_marginal_levels(&tdd).into_iter();
        it.next().unwrap().0
    };
    assert_eq!(
        tdd.levels[v.idx()].retired_marginal_slots(),
        2,
        "boundary sweep must retire 2 freed slots"
    );
    assert_eq!(
        tdd.retired_marginal_slots(),
        2,
        "retired_marginal_slots must equal the freed count"
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
    let eng = &crate::Engine::new();
    let mut tdd = toy(vec![BIG + 1, BIG + 2], &[&[(0, 0), (1, 1)]]);
    let v = boundary_marginal_levels(&tdd)[0].0;
    let before = tdd.levels[v.idx()].marginal_counts().unwrap().to_vec();
    prune_value_slots(eng, &mut tdd);
    assert_eq!(tdd.levels[v.idx()].marginal_counts().unwrap(), &before[..]);
}

/// Value-dedup in the boundary compaction pass:
/// Two referenced slots holding equal values must merge to one output slot,
/// and parent refs to both must be rewritten to the surviving slot.
///
/// This is where slot-count uniqueness is established for apply-emit-born stores
/// (the emit site is forbidden from deduping — see conjoin/mod.rs comment).
#[test]
fn prune_merges_equal_value_referenced_slots() {
    let eng = &crate::Engine::new();
    // Two nodes, each referencing one slot; both slots have equal value BIG+42.
    // Node 0: right-ref = 0 (slot 0 → BIG+42)
    // Node 1: right-ref = 1 (slot 1 → BIG+42)
    // after prune: store collapses to 1 slot; both refs become 0.
    let mut tdd = toy(vec![BIG + 42, BIG + 42], &[&[(0, 0)], &[(0, 1)]]);

    // Pre-condition: slot-count uniqueness is violated (duplicate slot values).
    assert!(
        check_slot_count_uniqueness(&tdd).is_err(),
        "pre-prune: slot values must start out duplicated"
    );

    let v = boundary_marginal_levels(&tdd)[0].0;
    let slots_before = tdd.levels[v.idx()].marginal_counts().unwrap().len();
    prune_value_slots(eng, &mut tdd);

    // (a) Unique values: slot-count uniqueness holds after prune.
    check_slot_count_uniqueness(&tdd)
        .expect("post-prune: no duplicate slot values");

    // (b) Store collapsed to 1 slot; 1 slot freed.
    let counts = tdd.levels[v.idx()].marginal_counts().unwrap();
    assert_eq!(counts.len(), 1, "equal slots must merge to one output slot");
    assert_eq!(counts[0], BIG + 42, "surviving slot must hold the original value");
    assert_eq!(slots_before - counts.len(), 1, "one duplicate slot must be freed");

    // (c) Parent refs both decode to slot 0 after remap.
    let mut buf = crate::value::slots::RefSlotScratch::default();
    let (_, parent, side) = boundary_marginal_levels(&tdd)[0];
    let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut buf);
    assert_eq!(refs, vec![0], "both parent refs must decode to the merged slot 0");
}

/// The sweep scratch pool carries CAPACITY across sweeps, never state: a
/// leftover `referenced` list would compact the next boundary store to the
/// PREVIOUS level's slot set, and a leftover `remap` would be re-resized over
/// rather than rebuilt. Pins the take-side clear.
#[test]
fn sweep_scratch_is_cleared_on_take() {
    let eng = &crate::Engine::new();
    let mut dirty = RefSlotScratch::default();
    dirty.referenced.push(7);
    eng.reduce_scratch().slot_prune_slots.put(dirty);
    eng.reduce_scratch().slot_prune_remap.put(vec![1, 2, 3]);

    let slots = eng.reduce_scratch().slot_prune_slots.checkout();
    let remap = eng.reduce_scratch().slot_prune_remap.checkout();
    assert!(slots.referenced.is_empty(), "referenced must be cleared on take");
    assert!(remap.is_empty(), "remap must be cleared on take");
}

#[cfg(test)]
mod compact_store_in_place_tests {
    use super::*;
    use crate::test_helpers::{toy, BIG};
    use num_bigint::BigUint;

    /// In-place boundary compaction on a sparse referenced set: the compacted
    /// store holds one entry per surviving slot, in ascending referenced order,
    /// with equal values deduped onto the first survivor that carries them.
    /// Expectations are hand-derived from the fixture below:
    ///
    /// store: slot0 = Big(b9), slot1 = BIG+5, slot2 = Big(b1), slot3 = Big(b1),
    ///        slot4 = BIG+5, slot5 = BIG+7;  referenced = [1, 2, 3, 5].
    ///
    /// - slot1 → new 0 (first survivor); slot0's `b9` must not linger at new 0
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
        let counts = tdd.levels[v.idx()].marginal_counts().unwrap().to_vec();
        tdd.levels[v.idx()].set_counts_state(
            counts,
            Some([(0u32, b9), (2u32, b1.clone()), (3u32, b1.clone())].into_iter().collect()),
        );

        let mut remap = vec![u32::MAX; 6];
        let (new_len, values_merged) =
            IntFold::compact_store(&mut tdd, v, &[1, 2, 3, 5], &mut remap);

        assert_eq!(new_len, 3, "three distinct values survive");
        assert_eq!(values_merged, 1, "slot3 merges onto slot2's compacted slot");
        let level = &tdd.levels[v.idx()];
        assert_eq!(
            level.marginal_counts().unwrap(),
            &[BIG + 5, u128::MAX, BIG + 7],
            "survivors move down in referenced order",
        );
        let big = level.marginal_counts_big().unwrap();
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
    /// weight store moves survivors by SWAP (a `WeightValue` is not `Copy`), so
    /// each step writes two slots instead of one — this pins that the extra
    /// write still lands only on already-consumed positions, and that nothing
    /// swapped up out of the prefix survives the truncate. Values are
    /// past-`u128` rationals at scattered slots, so a clobbered or stale slot
    /// cannot coincidentally match the expected one.
    ///
    /// store: slot0 = 3H, slot1 = H/7, slot2 = 2H/5, slot3 = 2H/5,
    ///        slot4 = H/7, slot5 = 3H/11;  referenced = [1, 2, 3, 5].
    ///
    /// - slot1 → new 0 (first survivor); slot0's `3H` must not linger at new 0
    /// - slot2 → new 1, carrying `2H/5` down with it
    /// - slot3 → new 1 as a value-dedup merge (equal rational)
    /// - slot5 → new 2
    /// - slot0 and slot4 are orphans and must not seed the dedup map (slot4
    ///   repeats the surviving slot1's value, so a leaked orphan would show up
    ///   as a bogus extra merge)
    #[test]
    fn weighted_compact_store_in_place_dedups_and_moves_survivors() {
        use crate::diagram::{LiteralWeights, RationalWeights};
        use crate::diagram::Arithmetic;
        use crate::test_helpers::toy_weighted;
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // H is past `u128`, the weighted analogue of the integer test's
        // `BigUint` entries.
        let h = BigInt::from(u128::MAX) * BigInt::from(u128::MAX);
        let wr = |n: u32, d: u32| BigRational::new(&h * BigInt::from(n), BigInt::from(d));
        let (v_a, v_b, v_c) = (wr(1, 7), wr(2, 5), wr(3, 11));

        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        let ws = crate::diagram::WeightStore::new(
            RationalWeights::from_literals(&[LiteralWeights { negative: half.clone(), positive: half }]),
            Arithmetic::ExactRational,
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

        let values = super::exact_vals(tdd.weights().unwrap().level(v.idx()).unwrap());

        assert_eq!(new_len, 3, "three distinct values survive");
        assert_eq!(values_merged, 1, "slot3 merges onto slot2's compacted slot");
        assert_eq!(values, vec![v_a, v_b, v_c], "survivors move down in referenced order");
        assert_eq!(
            remap,
            vec![u32::MAX, 0, 1, 1, u32::MAX, 2],
            "orphans keep the u32::MAX sentinel; merged refs share a slot",
        );
    }
}
