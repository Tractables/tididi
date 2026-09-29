//! A node with many plans tests "fused away" on a bitmap of its planned
//! x-refs instead of in the plan map: the level it leaves is the one the map
//! leaves, pair for pair and in the same order.

use super::*;
use super::super::plan::{allocate_fusion_slots, collect_fusion_plans, SORT_MIN};
use super::super::rewrite::{rebuild_parent_level_with, BITMAP_BITS_PER_PLAN, BITMAP_MIN_PLANS};
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, ValueRef};
use crate::test_helpers::toy;
use crate::value::IntFold;
use std::collections::BTreeMap;

/// A seeded xorshift draw below `m`.
fn draws(seed: u64) -> impl FnMut(u32) -> u32 {
    let mut state = seed | 1;
    move |m: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(m)) as u32
    }
}

/// Three root nodes over 64 marginal slots:
///
/// - a wide node, over the sorting threshold and not ascending, whose x's
///   repeat over a dense range — thousands of plans, a bitmap well under the
///   width cap — with, past its planned range, x's that occur once (no plan
///   and no bit) and, inside it, x's that occur once too;
/// - a narrow node, fused through the map at any threshold;
/// - a node with as many plans as the wide one but x's spread so far apart
///   that the bitmap would pass the width cap, so it keeps the map.
fn three_nodes(seed: u64) -> (Tdd, [usize; 3]) {
    let slots = 64u32;
    let counts: Vec<u128> = (0..slots).map(|s| u128::from(s) * 5 + 2).collect();
    let mut next = draws(seed);
    let dense = 9_000u32;
    let mut wide: Vec<(u32, u32)> = (0..(SORT_MIN as u32 + 30_000)).map(|_| (next(dense), next(slots))).collect();
    wide.extend((0..500).map(|i| (dense + 1_000_000 + 7 * i, next(slots))));
    wide.extend((0..200).map(|i| (dense + 10 + i, next(slots))));
    // Shuffled, so the node does not ascend by x and is grouped by sorting.
    for i in (1..wide.len()).rev() {
        wide.swap(i, next(i as u32 + 1) as usize);
    }
    let narrow: Vec<(u32, u32)> = vec![(3, 1), (4, 2), (3, 5), (3, 1), (9, 0)];
    let spread_xs = 6_000u32;
    let spread: Vec<(u32, u32)> =
        (0..3 * spread_xs).map(|_| (next(spread_xs) * 100_003, next(slots))).collect();
    let tdd = toy(counts, &[&wide, &narrow, &spread]);
    (tdd, [wide.len(), narrow.len(), spread.len()])
}

/// Per root node, each explicit ref's summed marginal count.
fn sums(tdd: &Tdd) -> Vec<BTreeMap<u32, u128>> {
    let root = tdd.vtree.root();
    let (_, right) = tdd.vtree.children(root);
    let counts = tdd.levels[right.idx()].marginal_counts().unwrap();
    let level = &tdd.levels[root.idx()];
    (0..level.nodes.len())
        .map(|n| {
            let mut by_x = BTreeMap::new();
            for p in level.pairs_of_idx(n) {
                let c = match ChildDecoder::marginal().value(EncodedChildRef::from_raw(p.right.0)) {
                    ValueRef::Inline(v) => v as u128,
                    ValueRef::Slot(s) => counts[s as usize],
                };
                *by_x.entry(p.left.0).or_insert(0) += c;
            }
            by_x
        })
        .collect()
}

/// The root's pair lists, node by node, and the marginal level's counts.
fn layout(tdd: &Tdd) -> (Vec<Vec<ChildPair>>, Vec<u128>) {
    let root = tdd.vtree.root();
    let (_, right) = tdd.vtree.children(root);
    let level = &tdd.levels[root.idx()];
    let pairs = (0..level.nodes.len()).map(|n| level.pairs_of_idx(n).to_vec()).collect();
    (pairs, tdd.levels[right.idx()].marginal_counts().unwrap().to_vec())
}

/// Plan, allocate and rewrite the root level as the sweep does, with the
/// bitmap taken from `bitmap_min_plans` plans on; returns the plans per node.
fn fuse_with(tdd: &mut Tdd, bitmap_min_plans: usize) -> [usize; 3] {
    let eng = Engine::new();
    let root = tdd.vtree.root();
    let (_, right) = tdd.vtree.children(root);
    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    let mut plans = collect_fusion_plans::<IntFold>(&eng, tdd, root, right, ChildSide::Right, &mut scratch.pair_fusion)
        .expect("no limits armed");
    let any_inline = allocate_fusion_slots::<IntFold>(&eng, tdd, right, &mut plans).expect("no limits armed");
    rebuild_parent_level_with(&eng, tdd, root, ChildSide::Right, any_inline, &plans, bitmap_min_plans)
        .expect("no limits armed");
    let mut per_node = [0usize; 3];
    for plan in &plans {
        per_node[plan.node_idx] += 1;
    }
    per_node
}

#[test]
fn the_bitmap_leaves_the_level_the_map_leaves() {
    for seed in [0x9E37_79B9_7F4A_7C15u64, 0x2545_F491_4F6C_DD1D, 42] {
        let (base, _) = three_nodes(seed);
        let before = sums(&base);
        let mut by_map = base.clone();
        let mut by_bits = base.clone();
        let mut by_default = base.clone();
        let plans = fuse_with(&mut by_map, usize::MAX);
        assert_eq!(fuse_with(&mut by_bits, 1), plans);
        assert_eq!(fuse_with(&mut by_default, BITMAP_MIN_PLANS), plans);
        // The wide node takes the bitmap under the default threshold, the
        // spread one is past the width cap, the narrow one under the count.
        assert!(plans[0] >= BITMAP_MIN_PLANS, "wide node plans: {}", plans[0]);
        assert!(plans[2] >= BITMAP_MIN_PLANS, "spread node plans: {}", plans[2]);
        assert!(6_000usize * 100_003 / BITMAP_BITS_PER_PLAN >= plans[2]);
        assert!(plans[1] < BITMAP_MIN_PLANS);
        let want = layout(&by_map);
        assert_eq!(layout(&by_bits), want, "seed {seed:#x}: bitmap at every node");
        assert_eq!(layout(&by_default), want, "seed {seed:#x}: bitmap at the default threshold");
        assert_eq!(sums(&by_bits), before, "fusion keeps every explicit ref's summed count");
        let level = &by_bits.levels[by_bits.vtree.root().idx()];
        for n in 0..3 {
            let mut xs: Vec<u32> = level.pairs_of_idx(n).iter().map(|p| p.left.0).collect();
            xs.sort_unstable();
            xs.dedup();
            assert_eq!(xs.len(), level.pair_count_at(n), "node {n} holds one pair per explicit ref");
        }
    }
}

/// The bitmap is zero between nodes: two wide nodes over the same x's, one
/// after the other, each fuse all of their own groups and only those.
#[test]
fn a_second_wide_node_sees_no_bit_of_the_first() {
    let slots = 16u32;
    let counts: Vec<u128> = (0..slots).map(|s| u128::from(s) + 1).collect();
    let mut next = draws(7);
    let first: Vec<(u32, u32)> = (0..3 * BITMAP_MIN_PLANS as u32).map(|i| (i / 3, next(slots))).collect();
    // The second node repeats only the odd x's; each even x occurs once.
    let second: Vec<(u32, u32)> = (0..BITMAP_MIN_PLANS as u32 * 2)
        .flat_map(|x| if x % 2 == 1 { vec![(x, next(slots)), (x, next(slots))] } else { vec![(x, next(slots))] })
        .collect();
    let base = toy(counts, &[&first, &second]);
    let before = sums(&base);
    let mut by_map = base.clone();
    let mut by_bits = base.clone();
    fuse_with_two(&mut by_map, usize::MAX);
    fuse_with_two(&mut by_bits, 1);
    assert_eq!(layout(&by_bits), layout(&by_map));
    assert_eq!(sums(&by_bits), before);
}

/// [`fuse_with`] for a root of two nodes.
fn fuse_with_two(tdd: &mut Tdd, bitmap_min_plans: usize) {
    let eng = Engine::new();
    let root = tdd.vtree.root();
    let (_, right) = tdd.vtree.children(root);
    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    let mut plans = collect_fusion_plans::<IntFold>(&eng, tdd, root, right, ChildSide::Right, &mut scratch.pair_fusion)
        .expect("no limits armed");
    let any_inline = allocate_fusion_slots::<IntFold>(&eng, tdd, right, &mut plans).expect("no limits armed");
    rebuild_parent_level_with(&eng, tdd, root, ChildSide::Right, any_inline, &plans, bitmap_min_plans)
        .expect("no limits armed");
}
