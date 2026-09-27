//! A narrow node probes a window of the grouping table, not the whole of it,
//! and a stop fires between nodes.

use super::*;
use crate::diagram::{ChildDecoder, EncodedChildRef, ValueRef};
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::test_helpers::toy;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Per root node, each explicit-side ref's summed marginal count: what fusion
/// must preserve however it groups.
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

/// Whether every root node holds one pair per explicit-side ref.
fn fused(tdd: &Tdd, n: usize) -> bool {
    let level = &tdd.levels[tdd.vtree.root().idx()];
    let pairs = level.pairs_of_idx(n);
    let mut xs: Vec<u32> = pairs.iter().map(|p| p.left.0).collect();
    xs.sort_unstable();
    xs.dedup();
    xs.len() == pairs.len()
}

/// One wide node, then many narrow ones: the table grows to the wide node's
/// width, and each narrow node groups within its own window of it. Seeded
/// pseudo-random x's collide inside a window as they would in a table of
/// the node's own size.
fn wide_then_narrow() -> Tdd {
    let slots = 64u32;
    let counts: Vec<u128> = (0..slots).map(|s| u128::from(s) * 3 + 1).collect();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move |m: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(m)) as u32
    };
    let mut lists: Vec<Vec<(u32, u32)>> = Vec::new();
    // 4096 pairs over 700 x's: the table grows to 8192 cells.
    lists.push((0..4096).map(|_| (next(700), next(slots))).collect());
    for i in 0..300 {
        let width = 2 + (i % 7) as usize;
        // Wide x's, so a whole-table hash and a windowed one place them
        // differently; few distinct, so most narrow nodes have a group.
        let base = next(1 << 20);
        lists.push((0..width).map(|_| (base + next(3) * 4099, next(slots))).collect());
    }
    let lists: Vec<&[(u32, u32)]> = lists.iter().map(Vec::as_slice).collect();
    toy(counts, &lists)
}

#[test]
fn narrow_nodes_after_a_wide_one_fuse_every_group() {
    let eng = Engine::new();
    let mut tdd = wide_then_narrow();
    let before = sums(&tdd);
    let stats = fuse_pairs(&eng, &mut tdd).expect("no limits armed");
    assert!(stats.fusion_groups > 300, "most narrow nodes carry a group: {}", stats.fusion_groups);
    assert_eq!(sums(&tdd), before, "fusion keeps every explicit ref's summed count");
    for n in 0..before.len() {
        assert!(fused(&tdd, n), "node {n} still holds two pairs at one explicit ref");
    }
}

/// An engine whose stop callback lets `calls` checks pass and stops the next.
fn stopping_after(calls: usize) -> Engine {
    let seen = Arc::new(AtomicUsize::new(0));
    let callback = StopCallback::new(move |_, _| {
        if seen.fetch_add(1, Ordering::Relaxed) < calls { StopDecision::Continue } else { StopDecision::Stop }
    });
    let eng = Engine::new();
    let _prior = eng.limits().install(LimitConfig::none().with_stop_callback(Some(callback)));
    eng.limits().pin_reduce_poll_stride(Some(1));
    eng
}

#[test]
fn a_stop_while_grouping_leaves_the_level_as_it_was() {
    let mut tdd = wide_then_narrow();
    let before = sums(&tdd);
    let pairs = tdd.pair_count();
    let r = fuse_pairs(&stopping_after(5), &mut tdd);
    assert!(matches!(r, Err(OperationError::Stopped)), "got {r:?}");
    assert_eq!(tdd.pair_count(), pairs, "grouping only reads");
    assert_eq!(sums(&tdd), before);
}

#[test]
fn a_stop_while_rewriting_leaves_whole_nodes_fused_and_the_rest_intact() {
    let mut tdd = wide_then_narrow();
    let before = sums(&tdd);
    let nodes = before.len();
    // With a stride of one the grouping tests once per node, then the
    // rewrite once per node it fuses: let the grouping and three rewrites
    // pass.
    let r = fuse_pairs(&stopping_after(nodes + 3), &mut tdd);
    assert!(matches!(r, Err(OperationError::Stopped)), "got {r:?}");
    assert_eq!(sums(&tdd), before, "a node is rewritten whole or not at all");
    assert!(fused(&tdd, 0), "the first node was rewritten before the stop");
    assert!((0..nodes).any(|n| !fused(&tdd, n)), "the stop left later groups unfused");
    // What was left is a redex, and a later sweep fuses it.
    fuse_pairs(&Engine::new(), &mut tdd).expect("no limits armed");
    assert_eq!(sums(&tdd), before);
    for n in 0..nodes {
        assert!(fused(&tdd, n), "node {n} still holds two pairs at one explicit ref");
    }
}

/// A node whose pairs ascend by explicit ref is grouped by its runs, and its
/// plans come in the order the table would emit them: first occurrence, which
/// is ascending x. Sums too wide to inline are minted as slots in plan order,
/// so the slots ascend with x.
#[test]
fn an_ascending_node_is_grouped_by_runs_in_first_occurrence_order() {
    let big = 1u128 << 40;
    let counts: Vec<u128> = (0..4).map(|s| big + s).collect();
    let node: &[(u32, u32)] = &[(0, 0), (0, 1), (2, 2), (5, 0), (5, 2), (5, 3), (9, 1), (9, 1)];
    let mut tdd = toy(counts.clone(), &[node]);
    let before = sums(&tdd);
    let stats = fuse_pairs(&Engine::new(), &mut tdd).expect("no limits armed");
    assert_eq!(stats.fusion_groups, 3);
    assert_eq!(sums(&tdd), before);
    assert!(fused(&tdd, 0));
    let root = tdd.vtree.root();
    let mut slot_of: Vec<(u32, u32)> = tdd.levels[root.idx()]
        .pairs_of_idx(0)
        .iter()
        .filter(|p| p.left.0 != 2)
        .map(|p| match ChildDecoder::marginal().value(p.right) {
            ValueRef::Slot(s) => (p.left.0, s),
            ValueRef::Inline(_) => panic!("a sum of 2^40-wide counts is not inline"),
        })
        .collect();
    slot_of.sort_unstable();
    assert_eq!(slot_of, vec![(0, 4), (5, 5), (9, 6)], "slots minted in ascending x");
}

/// Wide ascending nodes, runs of every length, next to shuffled ones that go
/// through the table: both keep every sum and leave one pair per ref.
#[test]
fn ascending_and_shuffled_nodes_fuse_alike() {
    let counts: Vec<u128> = (0..32).map(|s| s * 5 + 2).collect();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |m: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(m)) as u32
    };
    let mut lists: Vec<Vec<(u32, u32)>> = Vec::new();
    for i in 0..40 {
        let width = 2 + next(if i % 4 == 0 { 3000 } else { 30 }) as usize;
        let mut list: Vec<(u32, u32)> = (0..width).map(|_| (next(width as u32 / 2 + 1), next(32))).collect();
        if i % 2 == 0 {
            list.sort_unstable();
        }
        lists.push(list);
    }
    let lists: Vec<&[(u32, u32)]> = lists.iter().map(Vec::as_slice).collect();
    let mut tdd = toy(counts, &lists);
    let before = sums(&tdd);
    fuse_pairs(&Engine::new(), &mut tdd).expect("no limits armed");
    assert_eq!(sums(&tdd), before);
    for n in 0..before.len() {
        assert!(fused(&tdd, n), "node {n} still holds two pairs at one explicit ref");
    }
}

/// A diagram whose root holds one node of `pairs`, both children marginal:
/// counts `3s + 1` in slot `s` on each side, so either side can be fused.
fn both_marginal(pairs: &[(u32, u32)], slots: u32) -> Tdd {
    use crate::diagram::{ChildPair, NodeIdx, TddLevel, TddNodeId};
    use crate::vtree::Vtree;
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let (left, right) = vtree.children(root);
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let counts: Vec<u128> = (0..slots).map(|s| u128::from(s) * 3 + 1).collect();
    levels[left.idx()].set_counts_state(counts.clone(), None);
    levels[right.idx()].set_counts_state(counts, None);
    let pairs: Vec<ChildPair> = pairs.iter().map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r))).collect();
    levels[root.idx()].push_internal_node(&pairs);
    Tdd::from_levels_unchecked(vtree, levels, TddNodeId { vtree: root, local: NodeIdx(0) })
}

/// A wide node out of order is sorted into runs, and its plans are the
/// table's: the same groups with the same sums, in first-occurrence order,
/// on either side and over explicit refs of any width.
#[test]
fn a_wide_node_is_grouped_alike_by_sorting_and_by_the_table() {
    use super::super::plan::{SORT_MIN, group_by_scatter, group_by_sorting};
    use crate::diagram::ChildSide;
    use crate::reduce::contract::scratch::PFusionScratch;
    use crate::value::IntFold;
    let eng = Engine::new();
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut next = move |m: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(m)) as u32
    };
    let slots = 1 << 12;
    // (pairs, distinct explicit refs, where they start): few and many
    // groups, and refs past 2^30 as an inline-tagged side carries.
    for (width, refs, base) in [(SORT_MIN, 40u32, 0u32), (3 * SORT_MIN + 5, 20_000, 7), (SORT_MIN + 1, 900, 1 << 30)] {
        let pairs: Vec<(u32, u32)> = (0..width).map(|_| (base + next(refs), next(slots))).collect();
        for side in [ChildSide::Right, ChildSide::Left] {
            let flipped: Vec<(u32, u32)> = match side {
                ChildSide::Right => pairs.clone(),
                ChildSide::Left => pairs.iter().map(|&(x, m)| (m, x)).collect(),
            };
            let tdd = both_marginal(&flipped, slots);
            let root = tdd.vtree.root();
            let (left, right) = tdd.vtree.children(root);
            let v = match side { ChildSide::Right => right, ChildSide::Left => left };
            let level = &tdd.levels[root.idx()];
            let mut sc = PFusionScratch::default();
            let mut by_table = Vec::new();
            group_by_scatter::<IntFold>(&eng, &tdd, level, v, side, 0, &mut by_table, &mut sc).expect("no limits armed");
            let mut by_sort = Vec::new();
            group_by_sorting::<IntFold>(eng.limits(), &tdd, v, side, 0, level.pairs_of_idx(0), &mut by_sort, &mut sc)
                .expect("no limits armed");
            let view = |plans: &[PlanEntry<crate::value::Count>]| -> Vec<(usize, u32, crate::value::Count)> {
                plans.iter().map(|p| (p.node_idx, p.x_idx, p.value.clone())).collect()
            };
            assert!(by_table.len() > 1, "the node has groups to fuse");
            assert_eq!(view(&by_sort), view(&by_table), "width {width}, side {side:?}");
        }
    }
}
