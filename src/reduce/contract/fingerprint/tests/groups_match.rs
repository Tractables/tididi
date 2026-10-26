use super::*;
use crate::Engine;
use crate::execution::pool::PooledScratch;
use crate::diagram::{ChildPair, LeafLabel, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::rng::Lcg;
use crate::vtree::Vtree;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::reduce::contract::scratch::Marks;

/// The twin groups of a child level, found by listing every node's sorted
/// contexts: members ascending, groups by their lowest member. A node no
/// pair names has no twin.
fn listed_groups(parent: &TddLevel, child: &TddLevel, side: ChildSide) -> Vec<Vec<u32>> {
    let mut contexts: Vec<Vec<(u32, u32)>> = vec![Vec::new(); child.slot_count()];
    for (p, node) in parent.nodes().iter().enumerate() {
        for pair in parent.pairs_iter_of(&node) {
            let (t, s) = split_pair(&pair, side);
            contexts[t as usize].push((p as u32, s));
        }
    }
    let mut by_contexts: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
    for (t, mut c) in contexts.into_iter().enumerate().filter(|(_, c)| !c.is_empty()) {
        c.sort_unstable();
        by_contexts.entry(c).or_default().push(t as u32);
    }
    let mut groups: Vec<Vec<u32>> = by_contexts.into_values().filter(|g| g.len() > 1).collect();
    groups.sort();
    groups
}

/// Put a parent node's pairs, sorted by node, in one of three orders by
/// `order`: as they are, by sibling, or shuffled.
fn reorder(pairs: &mut [(u32, u32)], order: usize, rng: &mut Lcg) {
    match order % 3 {
        0 => {}
        1 => pairs.sort_unstable_by_key(|&(t, s)| (s, t)),
        _ => {
            for i in (1..pairs.len()).rev() {
                pairs.swap(i, rng.below(i as u64 + 1) as usize);
            }
        }
    }
}

/// The rounds a randomized test runs: `full` natively and `few` under Miri,
/// which interprets each round far slower. The few rounds still take every
/// path a test reaches through the pair views; the counts of what the
/// fixtures hold are asserted on full runs only.
fn rounds(full: u32, few: u32) -> u32 {
    if cfg!(miri) { few } else { full }
}

/// The twin search finds the groups a listing of every node's contexts
/// gives, on either side. Parent nodes that draw from few siblings repeat
/// contexts, and twins occur; parent nodes that draw from many seldom
/// repeat a sibling, and the search stops before fingerprinting unless two
/// nodes no pair names share their empty context.
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
    let mut marked = 0;
    for round in 0..rounds(300, 24) {
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
            let found = twin_groups(&eng, &tdd, child, root, side, &mut scratch, round % 2 == 1);
            let mut got: Vec<Vec<u32>> = Vec::new();
            if found {
                for (g, &start) in scratch.group_starts.iter().enumerate() {
                    let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                    got.push(scratch.flat_groups[start as usize..end].to_vec());
                }
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            if let (true, Some(marks)) = (found, scratch.marked) {
                check_marks(&tdd.levels[child.idx()], &scratch, marks);
                marked += 1;
            }
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 100, "the fixtures hold twins: {seen_groups}");
    assert!(cfg!(miri) || stopped_early > 20, "the search stops early on some fixtures: {stopped_early}");
    assert!(cfg!(miri) || marked > 20, "the refinement marks the members it lists: {marked}");
}

/// [`find_twin_groups`] with every group listed whole: a search that marks
/// the members lists each group's lowest member alone ([`list_members`]).
fn twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    child: VtreeIdx,
    parent: VtreeIdx,
    side: ChildSide,
    scratch: &mut ContractScratch,
    screen: bool,
) -> bool {
    let found = find_twin_groups(eng, tdd, child, parent, side, scratch, screen).expect("find_twin_groups");
    list_members(eng.limits(), scratch).expect("list_members");
    found
}

/// The marks a search left with its groups are the merge's: every member
/// but a group's lowest absorbed into the lowest, no other node absorbed,
/// and each group's pairs counted at its class.
fn check_marks(child: &TddLevel, scratch: &ContractScratch, marks: Marks) {
    let dropped = crate::reduce::contract::merge::DROPPED;
    let (groups, starts) = (&scratch.flat_groups, &scratch.group_starts);
    let absorbed = |t: u32| scratch.remap.absorbed[(t / 64) as usize] >> (t % 64) & 1 == 1;
    for (g, &start) in starts.iter().enumerate() {
        let end = starts.get(g + 1).map_or(groups.len(), |&e| e as usize);
        let keep = groups[start as usize];
        assert!(!absorbed(keep), "a group's lowest member survives");
        for &t in &groups[start as usize + 1..end] {
            assert!(absorbed(t), "member {t} is absorbed");
            assert_eq!(scratch.remap.final_remap[t as usize].0, keep | dropped, "member {t} merges into {keep}");
        }
    }
    let mut inline = 0;
    for (g, &start) in starts.iter().enumerate() {
        let end = starts.get(g + 1).map_or(groups.len(), |&e| e as usize);
        let members = &groups[start as usize..end];
        let pairs: usize = members.iter().map(|&t| child.pair_count_at(t as usize)).sum();
        let keep = members[0] as usize;
        assert_eq!(scratch.classes[scratch.twin_class[keep] as usize].hits as usize, pairs, "group {g}'s pairs");
        inline += members.iter().filter(|&&t| !child.node(t as usize).kind().pairs_in_arena()).count();
    }
    assert_eq!(marks.inline, inline, "the members of one pair");
    let mut members: Vec<(u32, u32)> = Vec::new();
    for (g, &start) in starts.iter().enumerate() {
        let end = starts.get(g + 1).map_or(groups.len(), |&e| e as usize);
        members.extend(groups[start as usize + 1..end].iter().map(|&t| (t, g as u32)));
    }
    members.sort_unstable();
    let by_node: Vec<u32> = members.iter().map(|&(_, g)| g).collect();
    assert_eq!(scratch.member_groups, by_node, "each member's group, in node order");
    let words = child.slot_count().div_ceil(64);
    let count: usize = scratch.remap.absorbed[..words].iter().map(|w| w.count_ones() as usize).sum();
    assert_eq!(count, groups.len() - starts.len(), "only the members are absorbed");
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
    // Wider than the nodes the test sorts, so that it files them.
    let width = refine::SMALL_PAIRS as u32 + 4;
    for v in [v_left, v_right] {
        for _ in 0..width {
            levels[v.idx()].push_internal_node(&[filler]);
        }
    }
    // One parent node pairing left node i with right node i: no sibling
    // repeats, and every node is named.
    let pairs: Vec<ChildPair> = (0..width).map(|i| ChildPair::new(NodeIdx(i), NodeIdx(i))).collect();
    levels[root.idx()].push_internal_node(&pairs);
    let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
    let entries = ContextEntries {
        parent_level: &tdd.levels[root.idx()],
        t1_side: ChildSide::Left,
        early_stop: true,
    };
    // Every cell of the node's window stamped 1 and filing one of its
    // siblings: read under the wrapped stamp, each would be a repeat.
    let cells = (2 * width as usize).next_power_of_two().min(TWIN_TABLE_START_CELLS) as u64;
    let mut scratch = ContractScratch {
        twin_local: (0..cells).map(|i| (1 << 32) | (i % u64::from(width))).collect(),
        twin_generation: u32::MAX,
        ..ContractScratch::default()
    };
    assert!(entries.no_twin(eng.limits(), &mut scratch, width as usize).expect("no_twin"));
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
    for round in 0..rounds(80, 7) {
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
                Some(twin_groups(&eng, &tdd, child, root, side, &mut scratch, round % 2 == 1))
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
    assert!(cfg!(miri) || (grouped > 50 && with_groups > 20 && declined > 20),
        "grouped {grouped}, with groups {with_groups}, declined {declined}");
}

/// Wide levels with planted twin classes, under parent nodes of a few pairs
/// and every fiftieth of many: the search finds the groups a listing of
/// every node's contexts gives, on either side.
#[test]
fn wide_levels_with_planted_twins_find_every_twin() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(31);
    let mut scratch = ContractScratch::default();
    let mut seen_groups = 0;
    for round in 0..rounds(12, 1) {
        let width = 4096 + rng.below(2000) as u32;
        let siblings = if round % 3 == 0 { 8 } else { 1 + rng.below(4000) as u32 };
        // Twin classes of one to four nodes, in a shuffled order.
        let mut order: Vec<u32> = (0..width).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut classes: Vec<&[u32]> = Vec::new();
        let mut rest = &order[..];
        while !rest.is_empty() {
            let k = (1 + rng.below(4) as usize).min(rest.len());
            classes.push(&rest[..k]);
            rest = &rest[k..];
        }
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let mut used: HashSet<(usize, u32)> = HashSet::new();
            let mut pairs: Vec<ChildPair> = Vec::new();
            for p in 0..width as usize / 2 {
                pairs.clear();
                // Every fiftieth parent node is wide and draws from a few
                // siblings, so the table files many shared pairs.
                let (k, range) = if p % 50 == 0 { (20 + rng.below(100), siblings.min(6)) } else { (1 + rng.below(4), siblings) };
                for _ in 0..k {
                    let c = rng.below(classes.len() as u64) as usize;
                    let s = rng.below(u64::from(range)) as u32;
                    if used.insert((c, s)) {
                        for &t in classes[c] {
                            let (l, r) = if side == ChildSide::Left { (t, s) } else { (s, t) };
                            pairs.push(ChildPair::new(NodeIdx(l), NodeIdx(r)));
                        }
                    }
                }
                if !pairs.is_empty() {
                    levels[root.idx()].push_internal_node(&pairs[..]);
                }
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            // A new diagram: the scratch is checked out afresh, as a
            // contraction checks it out.
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, round % 2 == 1);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 1000, "the fixtures hold twins: {seen_groups}");
}

/// Nodes first named together with one sibling set and told apart only by
/// later parent nodes, some of them wide: a parent node names one member of
/// a planted twin class alone, or every member but one, as well as whole
/// classes, and some parent nodes hold a pair twice. The wide parent nodes
/// list their pairs in node order, in sibling order or shuffled. The search
/// finds the groups a listing of every node's context multiset gives, on
/// either side.
#[test]
fn classes_are_split_by_later_contexts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(47);
    let mut shuffle = Lcg::new(48);
    let mut scratch = ContractScratch::default();
    let mut seen_groups = 0;
    for round in 0..rounds(40, 4) {
        let width = 256 + rng.below(600) as u32;
        let siblings = 2 + rng.below(if round % 2 == 0 { 3 } else { 40 }) as u32;
        let mut order: Vec<u32> = (0..width).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut classes: Vec<&[u32]> = Vec::new();
        let mut rest = &order[..];
        while !rest.is_empty() {
            let k = (1 + rng.below(5) as usize).min(rest.len());
            classes.push(&rest[..k]);
            rest = &rest[k..];
        }
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let pair = |t: u32, s: u32| if side == ChildSide::Left { ChildPair::new(NodeIdx(t), NodeIdx(s)) } else { ChildPair::new(NodeIdx(s), NodeIdx(t)) };
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for p in 0..width as usize / 3 {
                pairs.clear();
                // Every fifteenth parent node is wide.
                let draws = if p % 15 == 0 { 3 + rng.below(12) } else { 1 + rng.below(3) };
                for _ in 0..draws {
                    let class = classes[rng.below(classes.len() as u64) as usize];
                    let s = rng.below(u64::from(siblings)) as u32;
                    match rng.below(6) {
                        // One member alone: the class splits here.
                        0 => pairs.push((class[rng.below(class.len() as u64) as usize], s)),
                        // Every member but one.
                        1 if class.len() > 2 => {
                            let skip = rng.below(class.len() as u64) as usize;
                            pairs.extend(class.iter().enumerate().filter(|&(j, _)| j != skip).map(|(_, &t)| (t, s)));
                        }
                        _ => pairs.extend(class.iter().map(|&t| (t, s))),
                    }
                }
                pairs.sort_unstable();
                // Every fourth round keeps a pair drawn twice, as a parent
                // node may only in a diagram with a marginal level: there
                // the sibling level is marginal.
                if round % 4 != 3 {
                    pairs.dedup();
                }
                reorder(&mut pairs, p / 15, &mut shuffle);
                if !pairs.is_empty() {
                    let node: Vec<ChildPair> = pairs.iter().map(|&(t, s)| pair(t, s)).collect();
                    levels[root.idx()].push_internal_node(&node[..]);
                }
            }
            if round % 4 == 3 {
                levels[other.idx()].become_marginal(vec![1; siblings as usize], None);
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            // A new diagram: the scratch is checked out afresh, as a
            // contraction checks it out.
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, round % 2 == 1);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 200, "the fixtures hold twins: {seen_groups}");
}

/// Wide parent nodes whose nodes' sibling sets are nested, each a prefix of
/// one order of the siblings, so that a node's degree there names its
/// siblings, and wide parent nodes whose sets are not, over planted twin
/// classes some parent nodes split; nodes of different classes share
/// degrees, and some wide parent nodes name every member but one of a
/// class. The search finds the groups a listing of every node's contexts
/// gives, on either side. Their pairs come in node order, where the
/// degree split reads those of more than four siblings, in sibling order or
/// shuffled.
#[test]
fn nested_wide_nodes_are_split_by_degree() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(59);
    let mut shuffle = Lcg::new(60);
    let mut scratch = ContractScratch::default();
    let mut seen_groups = 0;
    for round in 0..rounds(40, 2) {
        let width = 64 + rng.below(400) as u32;
        let siblings = 4 + rng.below(if round % 2 == 0 { 8 } else { 60 }) as u32;
        let mut order: Vec<u32> = (0..width).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut classes: Vec<&[u32]> = Vec::new();
        let mut rest = &order[..];
        while !rest.is_empty() {
            let k = (1 + rng.below(4) as usize).min(rest.len());
            classes.push(&rest[..k]);
            rest = &rest[k..];
        }
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let pair = |t: u32, s: u32| if side == ChildSide::Left { ChildPair::new(NodeIdx(t), NodeIdx(s)) } else { ChildPair::new(NodeIdx(s), NodeIdx(t)) };
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            let mut by: Vec<u32> = (0..siblings).collect();
            for p in 0..width as usize / 4 {
                pairs.clear();
                if p % 3 == 0 {
                    // Nested: each class drawn takes a prefix of one order
                    // of the siblings, its degree drawn from few.
                    for i in (1..by.len()).rev() {
                        by.swap(i, rng.below(i as u64 + 1) as usize);
                    }
                    for _ in 0..2 + rng.below(8) {
                        let class = classes[rng.below(classes.len() as u64) as usize];
                        let d = 1 + rng.below(u64::from(siblings.min(8))) as usize;
                        let skip = if class.len() > 2 && rng.below(4) == 0 { rng.below(class.len() as u64) as usize } else { usize::MAX };
                        for (j, &t) in class.iter().enumerate() {
                            if j != skip {
                                pairs.extend(by[..d].iter().map(|&s| (t, s)));
                            }
                        }
                    }
                } else {
                    for _ in 0..1 + rng.below(if p % 3 == 1 { 3 } else { 12 }) {
                        let class = classes[rng.below(classes.len() as u64) as usize];
                        let s = rng.below(u64::from(siblings)) as u32;
                        if rng.below(5) == 0 {
                            pairs.push((class[rng.below(class.len() as u64) as usize], s));
                        } else {
                            pairs.extend(class.iter().map(|&t| (t, s)));
                        }
                    }
                }
                pairs.sort_unstable();
                pairs.dedup();
                reorder(&mut pairs, p / 3, &mut shuffle);
                if !pairs.is_empty() {
                    let node: Vec<ChildPair> = pairs.iter().map(|&(t, s)| pair(t, s)).collect();
                    levels[root.idx()].push_internal_node(&node[..]);
                }
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            // A new diagram: the scratch is checked out afresh, as a
            // contraction checks it out.
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, round % 2 == 1);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 100, "the fixtures hold twins: {seen_groups}");
}

/// Wide parent nodes most of whose pairs name nodes a parent node of one
/// pair, refined first, has left without a twin, beside none, one, two, a
/// few or many pairs of nodes still in a class: whole planted classes with
/// one sibling, or single members, which split their class. Their pairs
/// come in node order, in sibling order or shuffled. The search finds the
/// groups a listing of every node's contexts gives, on either side.
#[test]
fn wide_nodes_of_mostly_lone_nodes_are_split_by_the_rest() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(71);
    let mut shuffle = Lcg::new(72);
    let mut scratch = ContractScratch::default();
    let mut seen_groups = 0;
    for round in 0..rounds(40, 2) {
        let width = 200 + rng.below(300) as u32;
        let siblings = 2 + rng.below(if round % 2 == 0 { 4 } else { 50 }) as u32;
        let mut order: Vec<u32> = (0..width).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        // Classes of one to four nodes; the nodes of the first three
        // quarters of them each have a parent node of their own.
        let mut classes: Vec<&[u32]> = Vec::new();
        let mut rest = &order[..];
        while !rest.is_empty() {
            let k = (1 + rng.below(4) as usize).min(rest.len());
            classes.push(&rest[..k]);
            rest = &rest[k..];
        }
        let lone: Vec<u32> = classes[..classes.len() * 3 / 4].iter().flat_map(|c| c.iter().copied()).collect();
        let kept = &classes[classes.len() * 3 / 4..];
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let pair = |t: u32, s: u32| if side == ChildSide::Left { ChildPair::new(NodeIdx(t), NodeIdx(s)) } else { ChildPair::new(NodeIdx(s), NodeIdx(t)) };
            for &t in &lone {
                levels[root.idx()].push_internal_node(&[pair(t, rng.below(u64::from(siblings)) as u32)]);
            }
            for class in kept {
                let s = rng.below(u64::from(siblings)) as u32;
                let node: Vec<ChildPair> = class.iter().map(|&t| pair(t, s)).collect();
                levels[root.idx()].push_internal_node(&node[..]);
            }
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for p in 0..width as usize / 8 {
                pairs.clear();
                for _ in 0..refine::SMALL_PAIRS + 1 + rng.below(60) as usize {
                    pairs.push((lone[rng.below(lone.len() as u64) as usize], rng.below(u64::from(siblings)) as u32));
                }
                let want = [0, 1, 2, 5, 20][rng.below(5) as usize];
                let mut live = 0;
                while live < want {
                    let class = kept[rng.below(kept.len() as u64) as usize];
                    let s = rng.below(u64::from(siblings)) as u32;
                    if want - live < class.len() || rng.below(3) == 0 {
                        pairs.push((class[rng.below(class.len() as u64) as usize], s));
                        live += 1;
                    } else {
                        pairs.extend(class.iter().map(|&t| (t, s)));
                        live += class.len();
                    }
                }
                pairs.sort_unstable();
                pairs.dedup();
                reorder(&mut pairs, p, &mut shuffle);
                let node: Vec<ChildPair> = pairs.iter().map(|&(t, s)| pair(t, s)).collect();
                levels[root.idx()].push_internal_node(&node[..]);
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            // A new diagram: the scratch is checked out afresh, as a
            // contraction checks it out.
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, round % 2 == 1);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 100, "the fixtures hold twins: {seen_groups}");
}

/// Wide parent nodes over one to six siblings, on planted twin classes:
/// each class a parent node names gets a set of siblings, given to all its
/// members, to all but one, to one, or split between two codes, so that
/// the classes split into their members of each code and the rest. Their
/// pairs come in node order, in sibling order, or shuffled. Parent nodes of
/// up to four siblings are split by codes, by their runs' where their pairs
/// come in node order, and wider ones by degree or by the sibling buckets.
/// The search finds the groups a listing of every node's contexts gives,
/// on either side.
#[test]
fn scattered_wide_nodes_of_few_siblings_are_split_by_codes() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(71);
    let mut scratch = ContractScratch::default();
    let (mut seen_groups, mut few, mut many) = (0, 0, 0);
    for round in 0..rounds(60, 2) {
        let width = 64 + rng.below(700) as u32;
        let siblings = 1 + rng.below(6) as u32;
        let mut order: Vec<u32> = (0..width).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut classes: Vec<&[u32]> = Vec::new();
        let mut rest = &order[..];
        while !rest.is_empty() {
            let k = (1 + rng.below(6) as usize).min(rest.len());
            classes.push(&rest[..k]);
            rest = &rest[k..];
        }
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let pair = |t: u32, s: u32| if side == ChildSide::Left { ChildPair::new(NodeIdx(t), NodeIdx(s)) } else { ChildPair::new(NodeIdx(s), NodeIdx(t)) };
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for p in 0..width as usize / 4 {
                pairs.clear();
                // Most parent nodes are wide.
                let draws = if p % 4 == 0 { 1 + rng.below(2) } else { 3 + rng.below(10) };
                for _ in 0..draws {
                    let class = classes[rng.below(classes.len() as u64) as usize];
                    let code = 1 + rng.below((1u64 << siblings) - 1);
                    let set: Vec<u32> = (0..siblings).filter(|&s| code >> s & 1 == 1).collect();
                    let give = |pairs: &mut Vec<(u32, u32)>, t: u32, set: &[u32]| pairs.extend(set.iter().map(|&s| (t, s)));
                    match rng.below(5) {
                        0 => give(&mut pairs, class[rng.below(class.len() as u64) as usize], &set),
                        1 if class.len() > 2 => {
                            let skip = rng.below(class.len() as u64) as usize;
                            for (j, &t) in class.iter().enumerate() {
                                if j != skip {
                                    give(&mut pairs, t, &set);
                                }
                            }
                        }
                        2 if class.len() > 1 => {
                            // Two codes: the first half of the class gets
                            // `set`, the rest one other sibling more or less.
                            let other_set: Vec<u32> = (0..siblings).filter(|&s| (code ^ 1) >> s & 1 == 1).collect();
                            let half = class.len() / 2;
                            for &t in &class[..half] {
                                give(&mut pairs, t, &set);
                            }
                            for &t in &class[half..] {
                                give(&mut pairs, t, if other_set.is_empty() { &set } else { &other_set });
                            }
                        }
                        _ => {
                            for &t in class {
                                give(&mut pairs, t, &set);
                            }
                        }
                    }
                }
                pairs.sort_unstable();
                pairs.dedup();
                reorder(&mut pairs, p / 4, &mut rng);
                if pairs.len() > refine::SMALL_PAIRS {
                    let named: HashSet<u32> = pairs.iter().map(|&(_, s)| s).collect();
                    if named.len() <= 4 {
                        few += 1;
                    } else {
                        many += 1;
                    }
                }
                if !pairs.is_empty() {
                    let node: Vec<ChildPair> = pairs.iter().map(|&(t, s)| pair(t, s)).collect();
                    levels[root.idx()].push_internal_node(&node[..]);
                }
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, false);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 200, "the fixtures hold twins: {seen_groups}");
    assert!(cfg!(miri) || (few > 1000 && many > 200), "wide parent nodes of few and of more siblings: {few}, {many}");
}

/// Parent nodes of three and four pairs naming one or two nodes, the shapes
/// read without a sort, mixed with ones naming three nodes or more: the
/// twin groups are the nodes with equal contexts, on either side and in
/// any pair order.
#[test]
fn few_pair_parent_nodes_of_two_nodes_split_as_their_contexts_say() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let mut rng = Lcg::new(83);
    let mut scratch = ContractScratch::default();
    let (mut seen_groups, mut two_two) = (0, 0);
    for round in 0..rounds(200, 6) {
        let width = 2 + rng.below(12) as u32;
        let siblings = 1 + rng.below(3) as u32;
        for side in [ChildSide::Left, ChildSide::Right] {
            let (target, other) = if side == ChildSide::Left { (v_left, v_right) } else { (v_right, v_left) };
            let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
            for (v, w) in [(target, width), (other, siblings)] {
                for _ in 0..w {
                    levels[v.idx()].push_internal_node(&[filler]);
                }
            }
            let pair = |t: u32, s: u32| if side == ChildSide::Left { ChildPair::new(NodeIdx(t), NodeIdx(s)) } else { ChildPair::new(NodeIdx(s), NodeIdx(t)) };
            // Nodes 2k and 2k + 1 are twins unless a parent node of the
            // last kind below names one without the other.
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for _ in 0..2 + rng.below(3 * width as u64) {
                pairs.clear();
                let any = |rng: &mut Lcg| rng.below(width as u64) as u32;
                let sibling = |rng: &mut Lcg| rng.below(siblings as u64) as u32;
                match rng.below(10) {
                    // Both nodes of a would-be twin pair, with the same one
                    // or two siblings each.
                    0..=7 => {
                        let a = any(&mut rng) & !1;
                        let (s0, s1) = (sibling(&mut rng), sibling(&mut rng));
                        for t in [a, a + 1] {
                            if t < width {
                                pairs.extend([(t, s0), (t, s1)]);
                            }
                        }
                    }
                    // Two nodes, or the same node twice, of up to two
                    // siblings each, or three nodes of one sibling each.
                    _ => {
                        let a = any(&mut rng);
                        let b = if rng.below(4) == 0 { a } else { any(&mut rng) };
                        for t in [a, b] {
                            for _ in 0..1 + rng.below(2) {
                                pairs.push((t, sibling(&mut rng)));
                            }
                        }
                        if rng.below(5) == 0 {
                            pairs.push((any(&mut rng), sibling(&mut rng)));
                        }
                    }
                }
                pairs.sort_unstable();
                pairs.dedup();
                for i in (1..pairs.len()).rev() {
                    pairs.swap(i, rng.below(i as u64 + 1) as usize);
                }
                let nodes: HashSet<u32> = pairs.iter().map(|&(t, _)| t).collect();
                if pairs.len() == 4 && nodes.len() == 2 {
                    two_two += 1;
                }
                if !pairs.is_empty() {
                    let node: Vec<ChildPair> = pairs.iter().map(|&(t, s)| pair(t, s)).collect();
                    levels[root.idx()].push_internal_node(&node[..]);
                }
            }
            let tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: NodeIdx(0) });
            let want = listed_groups(&tdd.levels[root.idx()], &tdd.levels[target.idx()], side);
            scratch.prepare();
            let found = twin_groups(&eng, &tdd, target, root, side, &mut scratch, false);
            let mut got: Vec<Vec<u32>> = Vec::new();
            for (g, &start) in scratch.group_starts.iter().enumerate() {
                let end = scratch.group_starts.get(g + 1).map_or(scratch.flat_groups.len(), |&e| e as usize);
                got.push(scratch.flat_groups[start as usize..end].to_vec());
            }
            assert_eq!(found, !want.is_empty(), "round {round}, {side:?}");
            assert_eq!(got, want, "round {round}, {side:?}");
            seen_groups += want.len();
        }
    }
    assert!(cfg!(miri) || seen_groups > 100, "the fixtures hold twins: {seen_groups}");
    assert!(cfg!(miri) || two_two > 300, "parent nodes of two nodes of two siblings each: {two_two}");
}
