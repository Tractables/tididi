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
    for (p, node) in parent.nodes().iter().enumerate() {
        for pair in parent.pairs_iter_of(&node) {
            let (t, s) = split_pair(&pair, side);
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

/// The sibling table's stamp wraps to 1 only after every cell is zeroed, so
/// a cell stamped 1 before the wrap is not read as a sibling the node filed.
#[test]
fn a_wrapped_stamp_reads_no_older_cell() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    for v in [v_left, v_right] {
        for _ in 0..4 {
            levels[v.idx()].push_internal_node(&[filler]);
        }
    }
    // One parent node pairing left node i with right node i: no sibling
    // repeats, and every node is named.
    let pairs: Vec<ChildPair> = (0..4).map(|i| ChildPair::new(NodeIdx(i), NodeIdx(i))).collect();
    levels[root.idx()].push_internal_node(&pairs);
    let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
    let entries = ContextEntries {
        parent_level: &tdd.levels[root.idx()],
        t1_side: ChildSide::Left,
        early_stop: true,
    };
    // Every cell of the node's window stamped 1 and filing one of its
    // siblings: read under the wrapped stamp, each would be a repeat.
    let mut scratch = ContractScratch {
        twin_local: (0..8u64).map(|i| (1 << 32) | (i % 4)).collect(),
        twin_generation: u32::MAX,
        ..ContractScratch::default()
    };
    assert!(entries.no_twin(eng.limits(), &mut scratch, 4).expect("no_twin"));
    assert_eq!(scratch.twin_generation, 1);
}

/// A parent node wider than the sibling table's first size doubles the
/// table as it files: distinct siblings pass the test, and a sibling filed
/// before the table doubled is still found when a later pair repeats it.
#[test]
fn a_doubled_table_keeps_the_siblings_filed_before() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let width = 3 * TWIN_TABLE_START_CELLS as u32;
    for repeat in [false, true] {
        let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
        for v in [v_left, v_right] {
            for _ in 0..width {
                levels[v.idx()].push_internal_node(&[filler]);
            }
        }
        // Left node i paired with right node i: every left node is named
        // once, beside a sibling no other pair names. The repeating fixture's
        // last pair names right node 7 again, after the table has doubled.
        let mut pairs: Vec<ChildPair> = (0..width).map(|i| ChildPair::new(NodeIdx(i), NodeIdx(i))).collect();
        if repeat {
            pairs.last_mut().expect("a pair").right = ChildPair::new(NodeIdx(0), NodeIdx(7)).right;
        }
        levels[root.idx()].push_internal_node(&pairs);
        let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
        let entries = ContextEntries {
            parent_level: &tdd.levels[root.idx()],
            t1_side: ChildSide::Left,
            early_stop: true,
        };
        let mut scratch = ContractScratch::default();
        let passed = entries.no_twin(eng.limits(), &mut scratch, width as usize).expect("no_twin");
        assert_eq!(passed, !repeat, "repeat {repeat}");
        assert!(scratch.twin_local.len() >= 4 * TWIN_TABLE_START_CELLS, "the table doubled: {}", scratch.twin_local.len());
    }
}

/// Levels whose left nodes one parent pair names each, some given a second
/// pair or none: the radix grouping finds the groups a listing of every
/// node's contexts gives wherever it runs, on either side, and declines a
/// level where a node has no entry or several. A level past the width the
/// search tries it at is grouped through `find_twin_groups`.
#[test]
fn single_entry_levels_are_grouped_by_sorting() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(23);
    let mut scratch = ContractScratch::default();
    let (mut grouped, mut with_groups, mut declined) = (0, 0, 0);
    for round in 0..80u32 {
        let wide = round == 78;
        let width = if wide { groups::SINGLE_ENTRY_MIN_WIDTH as u32 + 4321 } else { 1 + rng.below(3000) as u32 };
        let siblings = 1 + rng.below(if round % 2 == 0 { 8 } else { 3000 }) as u32;
        let parents = 1 + rng.below(if round % 3 == 0 { 1 } else { 20 }) as usize;
        let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
        for (v, w) in [(v_left, width), (v_right, siblings)] {
            for _ in 0..w {
                levels[v.idx()].push_internal_node(&[filler]);
            }
        }
        // Each left node beside one sibling in one parent node; every fifth
        // round node 0 gets a second pair, every seventh round one node gets
        // none.
        let mut by_parent: Vec<Vec<ChildPair>> = vec![Vec::new(); parents];
        let twice = round % 5 == 4 && (parents > 1 || siblings > 1);
        let never = round % 7 == 6 && width > 1;
        for t in 0..width {
            if never && t == width / 2 {
                continue;
            }
            let s = rng.below(siblings as u64) as u32;
            let p = rng.below(parents as u64) as usize;
            by_parent[p].push(ChildPair::new(NodeIdx(t), NodeIdx(s)));
            if twice && t == 0 {
                // Another sibling in the same parent node, or the same
                // sibling in another.
                let (p, s) = if siblings > 1 { (p, (s + 1) % siblings) } else { ((p + 1) % parents, s) };
                by_parent[p].push(ChildPair::new(NodeIdx(t), NodeIdx(s)));
            }
        }
        for pairs in &mut by_parent {
            pairs.sort_unstable();
            if !pairs.is_empty() {
                levels[root.idx()].push_internal_node(&pairs[..]);
            }
        }
        let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
        for (child, side) in [(v_left, ChildSide::Left), (v_right, ChildSide::Right)] {
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[child.idx()], side);
            let width = tdd.levels[child.idx()].slot_count();
            let entries = ContextEntries {
                parent_level: &tdd.levels[root.idx()],
                t1_side: side,
                early_stop: true,
            };
            scratch.flat_groups.clear();
            scratch.group_starts.clear();
            let found = if wide {
                Some(find_twin_groups(&eng, &tdd, child, root, side, &mut scratch).expect("find_twin_groups"))
            } else {
                groups::group_single_entries(&eng, &entries, width, &mut scratch).expect("group_single_entries")
            };
            let Some(found) = found else {
                if side == ChildSide::Left {
                    assert!(twice || never, "round {round}: a level of one entry a node was declined");
                }
                declined += 1;
                continue;
            };
            if side == ChildSide::Left && !wide {
                assert!(!twice && !never, "round {round}: a node with no entry or two was grouped");
            }
            let mut got: Vec<Vec<u32>> = Vec::new();
            if found {
                for (g, &start) in scratch.group_starts.iter().enumerate() {
                    let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                    got.push(scratch.flat_groups[start as usize..end].to_vec());
                }
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            grouped += 1;
            with_groups += usize::from(found);
        }
    }
    assert!(grouped > 50 && with_groups > 20 && declined > 20,
        "grouped {grouped}, with groups {with_groups}, declined {declined}");
}
