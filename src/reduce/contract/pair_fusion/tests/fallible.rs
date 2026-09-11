use super::*;
use crate::diagram::{ValueRef, NodeIdx};
use crate::engine::Engine;
use crate::diagram::*;
use crate::diagram::{MarginalSide, TddLevel, TddNodeId};
use crate::vtree::{Vtree, VtreeNode};
use std::sync::Arc;

/// Slots the marginal level (the root's right child) holds.
fn marginal_slots(tdd: &Tdd) -> usize {
    let VtreeNode::Internal { right, .. } = *tdd.vtree.node(tdd.vtree.root()) else {
        panic!("root must be internal")
    };
    tdd.levels[right.idx()].marginal_counts().unwrap().len()
}

/// Build a minimal diagram with one boundary-marginal level carrying a single
/// fusable group: the root holds one internal node with two pairs sharing
/// the same x-side index and distinct marginal-side indices `{0, 1}`.
/// `fuse_pairs` must fuse those into one fresh slot.
fn fusable_tdd() -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    // Right child: marginal level with two distinct slots.
    levels[right.idx()].set_counts_state(vec![5u128, 7u128], None);
    // Root: one internal node, two pairs same x (left=0), distinct marginal (right=0,1).
    levels[root.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(0), right: NodeIdx(0) },
        InputPair { left: NodeIdx(0), right: NodeIdx(1) },
    ]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}

/// Happy path: with no apply budget set, the fusion succeeds. The fused
/// count (5 + 7 = 12) fits the inline width, so it is emitted as an inline
/// ref at the parent pair — no fresh slot is allocated.
#[test]
fn p_fusion_succeeds_without_budget() {
    let eng = Engine::new();
    let mut tdd = fusable_tdd();
    let (slots_before, size_before) = (marginal_slots(&tdd), tdd.size());
    let stats = fuse_pairs(&eng, &mut tdd).expect("no budget → must not over-budget");
    assert_eq!(marginal_slots(&tdd), slots_before, "small fused count must inline, not allocate a slot");
    assert_eq!(stats.fusion_groups, 1);
    assert_eq!(size_before - tdd.size(), 1);
}

/// A 1-byte apply budget makes the very first guarded growth trip
/// `OverBudget`: every growth in `fuse_pairs` is charged, so a pair-list
/// explosion is catchable instead of aborting the process in the allocator.
#[test]
fn p_fusion_over_budget_is_catchable() {
    let mut tdd = fusable_tdd();
    // Guard scope ends before asserting, so the tripped budget can't leak
    // into the assert (or any later test on a reused thread).
    let r = {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(1));
        fuse_pairs(&eng, &mut tdd)
    };
    assert!(matches!(r, Err(ApplyError::OverBudget)),
        "tiny budget must surface OverBudget, not abort or silently succeed; got {r:?}");
}

// ── Tests: fusion of inline-ref parent pairs ─────────────────────────────
//
// `fusable_tdd()` uses bare slot indices (bit-30 clear) in parent pairs.
// The two tests below use INLINE marginal refs (bit-30 set = MARGINAL_OVERFLOW_TAG)
// directly in the parent pair fields, exercising the
// `ValueRef::Inline` branch of `sum_marginal_counts`.

/// Build a diagram whose parent node has two pairs that share the same x-side
/// index and carry inline marginal refs with counts `c0` and `f`.
///
/// The marginal level has no slots at all — the inline counts are
/// self-contained in the pair fields.
fn inline_fusable_tdd(c0: u32, f: u32) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    // Right child: marginal level with ZERO slots (inline refs are self-contained).
    levels[right.idx()].set_counts_state(vec![], None);
    // Root: one internal node, two pairs sharing x=0, with INLINE marginal refs.
    // Bit-30 (MARGINAL_OVERFLOW_TAG) set marks these as inline count refs.
    let r0_raw = ValueRef::inline_raw(c0 as u128).expect("test inline count must fit inline encoding");
    let r1_raw = ValueRef::inline_raw(f as u128).expect("test inline count must fit inline encoding");
    levels[root.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(0), right: NodeIdx(r0_raw) },
        InputPair { left: NodeIdx(0), right: NodeIdx(r1_raw) },
    ]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}

/// One parent node with pairs `(x, Inline(5))` and `(x, Inline(7))`;
/// `fuse_pairs` must fuse them. Sum = 12.
///
/// 12 fits a ref, so the fused pair carries an inline ref and no new slot is
/// pushed.
/// One pair leaves the pair list.
#[test]
fn fusion_sums_inline_inline_pairs() {
    let eng = Engine::new();
    let mut tdd = inline_fusable_tdd(5, 7);
    let (slots_before, size_before) = (marginal_slots(&tdd), tdd.size());
    let stats = fuse_pairs(&eng, &mut tdd).expect("inline+inline fusion must not over-budget");
    // One fusion group eliminated one pair.
    assert_eq!(stats.fusion_groups, 1);
    assert_eq!(size_before - tdd.size(), 1);
    // Sum 12 fits inline (12 ≤ MARGINAL_INLINE_MAX in default env) → no new slot.
    assert_eq!(marginal_slots(&tdd), slots_before, "fused count 12 must be inlined, not slotted");
    // The surviving pair's marginal-side ref must decode to count 12.
    let root = tdd.vtree.root();
    let right = match tdd.vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("root must be internal"),
    };
    let counts = tdd.levels[right.idx()].marginal_counts().unwrap();
    let pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(pairs.len(), 1, "fusion must collapse two pairs to one");
    let fused_raw = pairs[0].right.0;
    let fused_count = match ValueRef::from_raw(MarginalSide(fused_raw)) {
        ValueRef::Inline(v) => v as u128,
        ValueRef::Slot(s) => counts[s as usize],
    };
    assert_eq!(fused_count, 12u128, "fused inline+inline count must be 5+7=12; got {fused_count}");
}

/// One parent node with pairs `(x, Inline(5))` and `(x, slot s)` where
/// `counts[s]` = 1<<40, too wide to fit a ref.
///
/// Because the sum (1<<40)+5 is also too wide, the fused result
/// must be a SLOT ref (bit-30 clear). A new slot is pushed (since no
/// existing slot carries that exact count), so the store grows by one and the
/// fused pair's marginal ref is a slot whose count decodes to `(1<<40)+5`.
#[test]
fn fusion_sums_inline_plus_slot_into_slot() {
    let eng = Engine::new();

    const BIG: u128 = 1u128 << 40; // too wide to fit a ref

    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    // Right child: marginal level with one slot carrying count BIG.
    levels[right.idx()].set_counts_state(vec![BIG], None);
    // Root: one internal node; left pair has inline count 5, right pair is slot 0.
    let inline_5_raw = ValueRef::inline_raw(5u128).expect("test inline count must fit inline encoding");
    let slot_0_raw = ValueRef::slot_raw(0); // bare slot index 0 (bit-30 clear)
    levels[root.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(0), right: NodeIdx(inline_5_raw) },
        InputPair { left: NodeIdx(0), right: NodeIdx(slot_0_raw) },
    ]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    let (slots_before, size_before) = (marginal_slots(&tdd), tdd.size());
    let stats = fuse_pairs(&eng, &mut tdd).expect("inline+slot fusion must not over-budget");
    assert_eq!(stats.fusion_groups, 1);
    assert_eq!(size_before - tdd.size(), 1);
    // Sum BIG+5 doesn't fit inline → a new slot must be allocated.
    assert_eq!(marginal_slots(&tdd) - slots_before, 1, "sum (1<<40)+5 is too wide for a ref; must allocate a slot");

    // The fused pair's marginal ref must be a SLOT (bit-30 clear).
    let counts = tdd.levels[right.idx()].marginal_counts().unwrap();
    let pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(pairs.len(), 1, "fusion must collapse two pairs to one");
    let fused_raw = pairs[0].right.0;
    let fused_ref = ValueRef::from_raw(MarginalSide(fused_raw));
    assert!(
        matches!(fused_ref, ValueRef::Slot(_)),
        "fused count (1<<40)+5 must be a slot ref (bit-30 clear); got {fused_ref:?}",
    );
    // The slot's count must be the exact sum.
    let ValueRef::Slot(s) = fused_ref else { unreachable!() };
    let fused_count = counts[s as usize];
    assert_eq!(
        fused_count, BIG + 5,
        "slot must hold (1<<40)+5; got {fused_count}",
    );
}

/// Identical-ref group: a parent node holds two pairs that are LITERALLY
/// identical — same x-side AND same marginal slot `(x=0, slot 0)`. Fusion
/// groups by the full occurrence MULTISET (no dedup), so the two occurrences
/// must sum to `2 × count`, not collapse to a single `count`.
///
/// This pins multiplicity preservation in the `by_x` grouping: a set-based
/// grouping would
/// silently dedup the two identical refs and halve this node's contribution.
/// Slot 0 carries count 6 ⇒ the fused count must be 12.
#[test]
fn fusion_sums_identical_ref_occurrences() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    // One marginal slot, count 6.
    levels[right.idx()].set_counts_state(vec![6u128], None);
    // Root node 0: two IDENTICAL pairs (x=0, slot 0).
    levels[root.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(0), right: NodeIdx(0) },
        InputPair { left: NodeIdx(0), right: NodeIdx(0) },
    ]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    let (slots_before, size_before) = (marginal_slots(&tdd), tdd.size());
    let stats = fuse_pairs(&eng, &mut tdd).expect("identical-ref fusion must not over-budget");
    assert_eq!(stats.fusion_groups, 1, "the two identical pairs form one fusion group");
    assert_eq!(size_before - tdd.size(), 1, "a group of size 2 removes one pair");
    assert_eq!(marginal_slots(&tdd), slots_before, "fused count 12 inlines under the default threshold");

    let counts = tdd.levels[right.idx()].marginal_counts().unwrap();
    let pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(pairs.len(), 1, "fusion must collapse the two identical pairs to one");
    let fused_count = match ValueRef::from_raw(MarginalSide(pairs[0].right.0)) {
        ValueRef::Inline(v) => v as u128,
        ValueRef::Slot(s) => counts[s as usize],
    };
    assert_eq!(
        fused_count, 12u128,
        "two occurrences of slot-0 (count 6) must sum to 12, not dedup to 6; got {fused_count}",
    );
}

/// Two independent identical-ref groups in one parent node must fuse
/// separately: pairs at x=0 (slots {0,1,2}) and x=1 (slots {3,4}) partition
/// by the non-marginal-side index, each summing only its own group's counts.
/// The pairs are interleaved to prove grouping is keyed by x (FxHashMap),
/// not by position. Exercises the multi-group partition and the >2-element
/// summation that the single-group tests above do not.
#[test]
fn fusion_partitions_two_independent_x_groups() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    // Five slots: x=0 → {0:3, 1:5, 2:7} (sum 15); x=1 → {3:11, 4:13} (sum 24).
    levels[right.idx()].set_counts_state(vec![3u128, 5, 7, 11, 13], None);
    // Interleave the two groups: grouping must be by x, not by pair order.
    levels[root.idx()].push_internal_node(&[
        InputPair { left: NodeIdx(0), right: NodeIdx(0) },
        InputPair { left: NodeIdx(1), right: NodeIdx(3) },
        InputPair { left: NodeIdx(0), right: NodeIdx(1) },
        InputPair { left: NodeIdx(1), right: NodeIdx(4) },
        InputPair { left: NodeIdx(0), right: NodeIdx(2) },
    ]);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    let (slots_before, size_before) = (marginal_slots(&tdd), tdd.size());
    let stats = fuse_pairs(&eng, &mut tdd).expect("two-group fusion must not over-budget");
    assert_eq!(stats.fusion_groups, 2, "x=0 and x=1 are two independent fusion groups");
    // x=0 (3 pairs) removes 2; x=1 (2 pairs) removes 1.
    assert_eq!(size_before - tdd.size(), 3);
    assert_eq!(marginal_slots(&tdd), slots_before, "sums 15 and 24 both inline under the default threshold");

    let counts = tdd.levels[right.idx()].marginal_counts().unwrap();
    let pairs = tdd.levels[root.idx()].pairs_of_idx(0);
    assert_eq!(pairs.len(), 2, "five pairs in two groups collapse to two fused pairs");
    // Map each surviving fused pair by its preserved x-side (left) index.
    let mut by_left: std::collections::HashMap<u32, u128> = std::collections::HashMap::new();
    for p in pairs {
        let c = match ValueRef::from_raw(MarginalSide(p.right.0)) {
            ValueRef::Inline(v) => v as u128,
            ValueRef::Slot(s) => counts[s as usize],
        };
        by_left.insert(p.left.0, c);
    }
    assert_eq!(by_left.get(&0).copied(), Some(15u128), "x=0 group must sum 3+5+7=15");
    assert_eq!(by_left.get(&1).copied(), Some(24u128), "x=1 group must sum 11+13=24");
}
