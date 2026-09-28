use super::*;
use crate::Engine;
use crate::diagram::{ChildPair, LeafLabel, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::rng::Lcg;
use crate::vtree::Vtree;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The twin groups of a child level, found by listing every node's sorted
/// contexts: members ascending, groups by their lowest member.
fn listed_groups(parent: &TddLevel, child: &TddLevel, side: ChildSide) -> Vec<Vec<u32>> {
    let mut contexts: Vec<Vec<(u32, u32)>> = vec![Vec::new(); child.slot_count()];
    for (p, node) in parent.nodes.iter().enumerate() {
        for pair in parent.pairs_of(node) {
            let (t, s) = split_pair(pair, side);
            let t = resolve_target(child.child_decoder(), t).expect("a node ref");
            contexts[t as usize].push((p as u32, s));
        }
    }
    let mut by_contexts: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
    for (t, mut c) in contexts.into_iter().enumerate() {
        c.sort_unstable();
        by_contexts.entry(c).or_default().push(t as u32);
    }
    let mut groups: Vec<Vec<u32>> = by_contexts.into_values().filter(|g| g.len() > 1).collect();
    groups.sort();
    groups
}

/// The twin search finds the groups a listing of every node's contexts
/// gives, on either side. Parent nodes that draw from few siblings repeat
/// contexts, and twins occur; parent nodes that draw from many seldom
/// repeat a sibling, and the search stops before fingerprinting unless two
/// nodes no pair names are twins of one another.
#[test]
fn the_twin_groups_are_the_nodes_with_equal_contexts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(11);
    let mut scratch = ContractScratch::default();
    let mut seen_groups = 0;
    let mut stopped_early = 0;
    for round in 0..300 {
        // Every third round draws siblings from a wide level, so most
        // parent nodes repeat none and the levels often have no twin.
        let widths = [1 + rng.below(40) as u32, 1 + rng.below(if round % 3 == 0 { 400 } else { 6 }) as u32];
        let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
        for (v, w) in [(v_left, widths[0]), (v_right, widths[1])] {
            for _ in 0..w {
                levels[v.idx()].push_internal_node(&[filler]);
            }
        }
        let mut used: HashSet<(u32, u32)> = HashSet::new();
        let mut pairs: Vec<ChildPair> = Vec::new();
        for _ in 0..1 + rng.below(30) {
            pairs.clear();
            let most = if rng.below(2) == 0 { 3 } else { 40 };
            for _ in 0..1 + rng.below(most) {
                let (l, r) = (rng.below(widths[0] as u64) as u32, rng.below(widths[1] as u64) as u32);
                if used.insert((l, r)) {
                    pairs.push(ChildPair::new(NodeIdx(l), NodeIdx(r)));
                }
            }
            if !pairs.is_empty() {
                levels[root.idx()].push_internal_node(&pairs[..]);
            }
        }
        let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
        for (child, side) in [(v_left, ChildSide::Left), (v_right, ChildSide::Right)] {
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[child.idx()], side);
            let entries = ContextEntries {
                parent_level: &tdd.levels[root.idx()],
                t1_side: side,
                t1_view: tdd.levels[child.idx()].child_decoder(),
                early_stop: true,
            };
            let width = tdd.levels[child.idx()].slot_count();
            if entries.no_twin(eng.limits(), &mut scratch, width).expect("no_twin") {
                assert!(want.is_empty(), "round {round}, {side:?}: a level with twins passed the early stop");
                stopped_early += 1;
            }
            let found = find_twin_groups(&eng, &tdd, child, root, side, &mut scratch).expect("find_twin_groups");
            let mut got: Vec<Vec<u32>> = Vec::new();
            if found {
                for (g, &start) in scratch.group_starts.iter().enumerate() {
                    let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                    got.push(scratch.flat_groups[start as usize..end].to_vec());
                }
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(seen_groups > 100, "the fixtures hold twins: {seen_groups}");
    assert!(stopped_early > 20, "the search stops early on some fixtures: {stopped_early}");
}
