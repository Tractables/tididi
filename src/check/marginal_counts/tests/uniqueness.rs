use num_bigint::BigUint;
use crate::check::marginal::check_store_counts;
use crate::marginal::dedup_fresh_store;
use crate::diagram::{BigSide, MarginalSide, ValueRef};

// ── dedup_fresh_store for marginalize-time stores ────────────────

/// Two nodes with equal small counts → dedup merges them to one slot.
/// Parent refs 0 and 1 both remap to the single surviving slot 0.
#[test]
fn dedup_fresh_store_merges_equal_small_counts() {
    // counts[0] = 42, counts[1] = 42 — duplicate
    let counts = vec![42u128, 42u128];
    let big: Option<BigSide> = None;
    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);
    assert_eq!(new_counts.len(), 1, "two equal slots must collapse to one");
    assert_eq!(new_counts[0], 42u128);
    assert!(new_big.is_none() || new_big.as_ref().unwrap().is_empty());
    assert_eq!(remap[0], 0, "canonical slot stays 0");
    assert_eq!(remap[1], 0, "duplicate remaps to canonical slot 0");
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("born store must satisfy invariant 10 immediately");
}

/// Two nodes with distinct small counts → no dedup; remap is identity.
#[test]
fn dedup_fresh_store_distinct_small_counts_unchanged() {
    let counts = vec![10u128, 20u128];
    let big: Option<BigSide> = None;
    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);
    assert_eq!(new_counts, vec![10u128, 20u128]);
    assert_eq!(remap, vec![0, 1]);
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("born store must satisfy invariant 10 immediately");
}

/// Two nodes with equal BigUint counts (behind the OVERFLOW sentinel) →
/// dedup merges them to one slot, with one BigUint entry in new_big.
#[test]
fn dedup_fresh_store_merges_equal_big_counts() {
    let big_val = BigUint::from(u128::MAX as u64) * BigUint::from(3u32);
    let counts = vec![u128::MAX, u128::MAX]; // both OVERFLOW sentinels
    let big: Option<BigSide> = Some(
        [(0u32, big_val.clone()), (1u32, big_val.clone())].into_iter().collect(),
    );
    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);
    assert_eq!(new_counts.len(), 1, "two equal Big slots must collapse to one");
    assert_eq!(new_counts[0], u128::MAX);
    let nb = new_big.as_ref().expect("new_big must exist for Big slots");
    assert_eq!(nb.len(), 1, "the merged-away slot's entry is dropped, not kept");
    assert_eq!(nb.get(0), Some(&big_val));
    assert_eq!(remap[0], 0);
    assert_eq!(remap[1], 0);
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("born store must satisfy invariant 10 immediately");
}

/// Parent refs (as bare slot indices) are correctly remapped through the
/// dedup remap table. Two slots {0: count=7, 1: count=7} collapse to slot 0;
/// a parent ref holding bare index 1 must become index 0 after remap.
#[test]
fn dedup_fresh_store_ref_remap_is_correct() {
    let counts = vec![7u128, 7u128];
    let (_, _, remap) = dedup_fresh_store(counts, None);
    // Simulate a parent pair ref that was minted as bare slot 1.
    let old_ref: u32 = 1; // bare slot index (pre-tagger)
    let new_ref: u32 = remap[old_ref as usize];
    assert_eq!(new_ref, 0, "remapped ref must point to the canonical slot");
    // After tagging (ValueRef::slot_raw), the consumer would decode correctly.
    let tagged = ValueRef::slot_raw(new_ref);
    assert_eq!(ValueRef::from_raw(MarginalSide(tagged)), ValueRef::Slot(0));
}

// ── dedup_fresh_store duplicate-merge, independent of any call site. The apply
// streaming emit does not call it: invariant 10 for those stores is established
// at the post-tagger slot prune, by prune_value_slots.

/// Two nodes with equal small counts → dedup_fresh_store merges to one slot.
/// Both node_idx grid entries remap to the single surviving slot 0.
#[test]
fn streaming_emit_dedup_equal_counts() {
    // Two cells with count=99 each (duplicate).
    let counts = vec![99u128, 99u128];
    let big: Option<BigSide> = None;
    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);
    assert_eq!(new_counts.len(), 1);
    // Simulate node_idx grid for two cells: [0, 1] (bare slot indices).
    let mut node_idx: Vec<u32> = vec![0, 1];
    const NO_PRODUCT: u32 = u32::MAX;
    let all_identity = remap.iter().enumerate().all(|(i, &r)| r == i as u32);
    assert!(!all_identity, "equal counts must not produce identity remap");
    for entry in node_idx.iter_mut() {
        if *entry != NO_PRODUCT {
            *entry = remap[*entry as usize];
        }
    }
    assert_eq!(node_idx[0], 0, "first cell maps to slot 0");
    assert_eq!(node_idx[1], 0, "duplicate cell also maps to slot 0");
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("deduped store must satisfy invariant 10");
}

/// Two nodes with equal BigUint counts behind OVERFLOW sentinel.
#[test]
fn streaming_emit_dedup_equal_big_counts() {
    let big_val = BigUint::from(u128::MAX as u64) + BigUint::from(1u32);
    let counts = vec![u128::MAX, u128::MAX];
    let big: Option<BigSide> = Some(
        [(0u32, big_val.clone()), (1u32, big_val.clone())].into_iter().collect(),
    );
    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);
    assert_eq!(new_counts.len(), 1, "equal Big counts must collapse");
    let mut node_idx = [0u32, 1u32];
    const NO_PRODUCT: u32 = u32::MAX;
    for entry in node_idx.iter_mut() {
        if *entry != NO_PRODUCT { *entry = remap[*entry as usize]; }
    }
    assert_eq!(node_idx[0], 0);
    assert_eq!(node_idx[1], 0);
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("deduped store must satisfy invariant 10");
}

// ── In-place compaction (mirrors `slot_prune`'s `compact_store_in_place_*`) ──

/// Dedup compacts the fast count column the caller handed over IN PLACE —
/// survivors move down inside the input buffer, no second column beside it —
/// and re-files the sparse overflow entries under their compacted slot.
/// Expectations are hand-derived from the fixture below:
///
/// store: slot0 = 5, slot1 = 5, slot2 = 9, slot3 = Big(b1), slot4 = 9.
///
/// - slot0 → new 0 (first survivor, a self-write)
/// - slot1 → new 0 as a value-dedup merge
/// - slot2 → new 1, moving DOWN over the already-consumed slot1
/// - slot3 → new 2, its overflow entry moved (not cloned) to key 2
/// - slot4 → new 1 as a value-dedup merge onto the moved slot
///
/// The allocation check is what pins "in place": `truncate` never reallocates,
/// and the guarded `shrink_to_fit` cannot fire at this capacity, so a returned
/// buffer at a fresh address would mean a second column was built after all.
#[test]
fn dedup_fresh_store_compacts_in_place() {
    let b1 = BigUint::from(u128::MAX) + BigUint::from(1u32);
    let counts = vec![5u128, 5, 9, u128::MAX, 9];
    let big: Option<BigSide> = Some([(3u32, b1.clone())].into_iter().collect());
    let counts_addr = counts.as_ptr();

    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);

    assert_eq!(new_counts, vec![5u128, 9, u128::MAX], "survivors move down in slot order");
    let nb = new_big.as_ref().expect("the overflow table survives compaction");
    assert_eq!(nb.get(2), Some(&b1), "the overflow entry is re-filed under its compacted slot");
    assert_eq!(nb.len(), 1, "and nothing is left behind at the vacated slot");
    assert_eq!(remap, vec![0, 0, 1, 2, 1], "merged slots share their canonical's compacted index");
    assert_eq!(new_counts.as_ptr(), counts_addr, "counts compacted in the input allocation");
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("compacted store must satisfy invariant 10");
}

/// Several DISTINCT overflow entries at scattered slots all survive dedup, and
/// each keeps its exact value under its new key. This is the sparse table's
/// load-bearing case: keys are slot indices, so every survivor whose index
/// moved must be re-filed, and a merged-away neighbour must not shift the
/// wrong entry down.
///
/// store: 0 = Big(b0), 1 = 7, 2 = 7, 3 = Big(b1), 4 = Big(b0), 5 = 8, 6 = Big(b2)
///
/// - slot0 → new 0, Big(b0) survives
/// - slot1 → new 1
/// - slot2 → new 1 (value-dedup merge onto slot1)
/// - slot3 → new 2, Big(b1) re-filed 3 → 2
/// - slot4 → new 0 (value-dedup merge: Big(b0) already has a slot)
/// - slot5 → new 3
/// - slot6 → new 4, Big(b2) re-filed 6 → 4
#[test]
fn dedup_fresh_store_rekeys_scattered_big_entries() {
    let b0 = BigUint::from(u128::MAX) + BigUint::from(1u32);
    let b1 = BigUint::from(u128::MAX) * BigUint::from(3u32);
    let b2 = BigUint::from(u128::MAX) * BigUint::from(u128::MAX);
    let counts = vec![u128::MAX, 7, 7, u128::MAX, u128::MAX, 8, u128::MAX];
    let big: Option<BigSide> = Some(
        [
            (0u32, b0.clone()),
            (3u32, b1.clone()),
            (4u32, b0.clone()),
            (6u32, b2.clone()),
        ]
        .into_iter()
        .collect(),
    );

    let (new_counts, new_big, remap) = dedup_fresh_store(counts, big);

    assert_eq!(new_counts, vec![u128::MAX, 7, u128::MAX, 8, u128::MAX]);
    assert_eq!(remap, vec![0, 1, 1, 2, 0, 3, 4]);
    let nb = new_big.as_ref().expect("distinct Big slots keep an overflow table");
    assert_eq!(nb.len(), 3, "one entry per surviving Big slot, duplicate dropped");
    assert_eq!(nb.get(0), Some(&b0), "b0 keeps its value at its unchanged key");
    assert_eq!(nb.get(2), Some(&b1), "b1 re-filed 3 -> 2");
    assert_eq!(nb.get(4), Some(&b2), "b2 re-filed 6 -> 4");
    // Every ref lands on a slot carrying the value it was minted with.
    for (old, &new) in remap.iter().enumerate() {
        let new = new as usize;
        match old {
            0 | 4 => assert_eq!(nb.get(new), Some(&b0)),
            3 => assert_eq!(nb.get(new), Some(&b1)),
            6 => assert_eq!(nb.get(new), Some(&b2)),
            1 | 2 => assert_eq!(new_counts[new], 7),
            _ => assert_eq!(new_counts[new], 8),
        }
    }
    check_store_counts(&new_counts, new_big.as_ref())
        .expect("compacted store must satisfy invariant 10");
}
