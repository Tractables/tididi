use super::*;
use crate::diagram::*;
use crate::test_helpers::pair;
use crate::limits::Limits;

/// Every node's pair slice, in arena order — the whole of a level's
/// observable content (`pairs_vec` resolves inline, normal-multi and
/// extended nodes alike).
fn content(level: &TddLevel) -> Vec<Vec<ChildPair>> {
    (0..level.nodes().len())
        .map(|i| level.pairs_vec(i).to_vec())
        .collect()
}

/// Arena slots actually owned by live nodes.
fn live_pairs(level: &TddLevel) -> usize {
    (0..level.nodes().len()).map(|i| level.arena_pairs_at(i)).sum()
}

/// One round of tail merges over the nodes from `first` on: merge each
/// adjacent pair (survivor first) by appending their union at the arena tail,
/// then drop the absorbed ones, so each union leaves both source ranges dead
/// behind it, as passes that shrink or move pair lists in place do.
fn merge_adjacent_round(level: &mut TddLevel, first: usize) {
    let n = level.nodes().len();
    let mut dead = 0;
    let mut keep = first;
    while keep + 1 < n {
        let union: Vec<ChildPair> = [keep, keep + 1].iter().flat_map(|&i| level.pairs_vec(i).to_vec()).collect();
        dead += level.arena_pairs_at(keep) + level.arena_pairs_at(keep + 1);
        let start = level.pairs.len();
        level.pairs.stored_mut().extend_from_slice(&union);
        let node = level.encode_multi(start, union.len());
        level.nodes.stored_mut()[keep] = node;
        keep += 2;
    }
    let mut i = 0;
    level.nodes.stored_mut().retain(|_| {
        let absorbed = i > first && (i - first) % 2 == 1;
        i += 1;
        !absorbed
    });
    level.note_dead_pairs(dead);
}

/// Two merge rounds mint more dead arena than live, so the sweep fires; it
/// must reclaim the garbage without changing a single node's pair slice.
#[test]
fn contraction_garbage_is_swept_leaving_content_identical() {
    let n = TddLevel::PAIRS_COMPACT_MIN_DEAD;
    let mut level = TddLevel::new();
    // Two nodes that survive both rounds untouched, so the sweep has to
    // handle them alongside the tail survivors — and in ascending-start, not
    // node, order (these keep the LOWEST starts while the merged survivors
    // sit at the tail). First an extended (side-table) node, staged by hand
    // since the encoder only goes extended past 2^31 slots: its start lives
    // in `ranges` rather than the node word. Then an inline node, which
    // owns no arena slot at all.
    level.pairs.stored_mut().extend([pair(1, 7), pair(2, 7)]);
    level.ranges.push(PairRange { start: 0, len: 2 });
    level.nodes.stored_mut().push(EncodedNode::multi_ranged(0));
    level.push_internal_node(&[pair(3, 4)]);
    let first_multi = level.nodes().len();
    for i in 0..n {
        level.push_internal_node(&[pair(i as u32, 0), pair(i as u32, 1)]);
    }

    merge_adjacent_round(&mut level, first_multi);
    merge_adjacent_round(&mut level, first_multi);

    let before = content(&level);
    let before_len = level.pairs.len();
    let live = live_pairs(&level);
    assert!(live < before_len, "the merge rounds must leave dead arena behind");
    assert!(
        level.compact_pairs_if_stale(),
        "{} dead slots in a {before_len}-slot arena must trip the compaction trigger",
        level.dead_pairs,
    );
    assert_eq!(
        content(&level), before,
        "the sweep must preserve every node's pair slice and its order",
    );
    assert!(
        level.pairs.len() < before_len,
        "the arena must shrink: {} slots, was {before_len}", level.pairs.len(),
    );
    assert_eq!(
        level.pairs.len(), live,
        "the swept arena must hold exactly the live pairs",
    );
    assert_eq!(level.dead_pairs, 0, "the sweep must reset the garbage count");
    // Nothing left to reclaim, so the trigger must not fire again.
    assert!(!level.compact_pairs_if_stale());
}

/// A parent node already in the extended (side-table) encoding that shrinks to
/// a single non-inlinable pair must rewrite its own `ranges` entry, not
/// mint a second one and abandon the first.
///
/// Hand-encoded because the arm needs a length-≥2 extended node, which only a
/// pair arena past the 31-bit start/length encoding mints in a real run.
#[test]
fn a_shrunk_extended_parent_node_inlines_its_survivor() {
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(2));
    let root = vtree.root();
    let mut levels: Vec<TddLevel> =
        (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    // An extended node staged by hand (the encoder only goes extended past
    // 2^31 slots): its sole survivor goes inline, and its `ranges` entry
    // is left in place rather than replaced.
    let sibling = NodeIdx(3);
    let parent = &mut levels[root.idx()];
    parent.pairs = vec![
        ChildPair::new(NodeIdx(0), sibling),
        ChildPair::new(NodeIdx(1), sibling),
    ]
    .into();
    parent.ranges = vec![PairRange { start: 0, len: 2 }].into();
    parent.nodes = vec![EncodedNode::multi_ranged(0)].into();
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    // T1 twins {0, 1} merged into 0: the parent's second pair is dropped.
    let remap = remap_of(&[0, 0], &[]);

    rewrite_parent(&Limits::new(), &mut tdd, root, ChildSide::Left, &remap, None);

    let parent = &tdd.levels[root.idx()];
    assert_eq!(parent.ranges.len(), 1, "no range entry is added for an inlined survivor");
    assert!(matches!(parent.node(0).kind(), NodeKind::Inline(_)), "the sole survivor is stored inline");
    assert_eq!(
        parent.pairs_vec(0),
        &[ChildPair::new(NodeIdx(0), sibling)],
    );
    assert_eq!(parent.pairs.len(), 0, "an inline survivor keeps no arena slot");
    assert_eq!(parent.dead_pairs, 0, "the rewrite leaves no dead pairs");
}

/// The parent of the merged level `[0, 0, 2, 3, 3]` (twins {0, 1} and {3, 4}),
/// its nodes' pairs placed by `place` (`(start, pairs)` per node, `None` for
/// an inline node) in an arena of `len` slots, rewritten.
fn rewritten_parent(len: usize, place: &[(Option<usize>, Vec<ChildPair>)]) -> TddLevel {
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(2));
    let root = vtree.root();
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let parent = &mut levels[root.idx()];
    let mut arena = vec![ChildPair::new(NodeIdx(9), NodeIdx(9)); len];
    let mut nodes = Vec::new();
    for (start, pairs) in place {
        match start {
            Some(start) => {
                arena[*start..*start + pairs.len()].copy_from_slice(pairs);
                nodes.push(EncodedNode::multi_pair(*start as u32, pairs.len() as u32));
            }
            None => nodes.push(EncodedNode::inline(pairs[0])),
        }
    }
    parent.pairs = arena.into();
    parent.nodes = nodes.into();
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    let remap = remap_of(&[0, 0, 2, 3, 3], &[]);
    rewrite_parent(&Limits::new(), &mut tdd, root, ChildSide::Left, &remap, None);
    tdd.levels[root.idx()].clone()
}

/// A parent whose ranges are in node order has its kept pairs moved down as
/// they are rewritten: the arena is left with the live pairs only, in node
/// order, gaps and dropped pairs gone. Out of node order, each node shrinks
/// in place and its tail is counted dead.
#[test]
fn a_parent_in_node_order_is_rewritten_without_dead_pairs() {
    let p = |l: u32, r: u32| ChildPair::new(NodeIdx(l), NodeIdx(r));
    let place = [
        (Some(0), vec![p(0, 10), p(1, 10), p(2, 11)]),
        (None, vec![p(2, 10)]),
        (Some(4), vec![p(0, 12), p(1, 12), p(3, 13), p(4, 13)]),
    ];
    let parent = rewritten_parent(8, &place);
    assert_eq!(content(&parent), vec![vec![p(0, 10), p(1, 11)], vec![p(1, 10)], vec![p(0, 12), p(2, 13)]]);
    assert_eq!(parent.pairs.len(), 4, "only the live pairs are left");
    assert_eq!(parent.pair_range_at(2).start, 2, "the arena is in node order");
    assert_eq!(parent.dead_pairs, 0);

    // Node 0's range after node 1's: shrunk in place.
    let place = [(Some(2), vec![p(0, 10), p(1, 10), p(2, 11)]), (Some(0), vec![p(3, 12), p(4, 12)])];
    let parent = rewritten_parent(5, &place);
    assert_eq!(content(&parent), vec![vec![p(0, 10), p(1, 11)], vec![p(2, 12)]]);
    assert_eq!(parent.pairs.len(), 5, "the arena is not moved");
    assert_eq!(parent.dead_pairs, 3, "node 0's tail and node 1's whole range are dead");
}

/// A refused arena reservation retains merge capacity, and retry clears its plans.
#[test]
fn refused_merge_reuses_buffers_without_replaying_stale_plans() {
    let eng = Engine::new();
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(4));
    let parent = vtree.root();
    let (child, sibling) = vtree.children(parent);
    let mut levels = take_levels(&eng, vtree.num_nodes());
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let a = levels[child.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let b = levels[child.idx()].push_internal_node(&[ChildPair::new(neg, pos)]);
    let s = levels[sibling.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let output = levels[parent.idx()].push_internal_node(&[
        ChildPair::new(a, s), ChildPair::new(b, s),
    ]);
    let mut tdd = Tdd::from_levels_unchecked(
        vtree, levels, TddNodeId { vtree: parent, local: output },
    );
    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    scratch.flat_groups = vec![0, 1];
    scratch.group_starts = vec![0];
    scratch.remap.final_remap = vec![NodeIdx(0); 2];
    scratch.merge.sel.reserve(8);
    let allocation = scratch.merge.sel.as_ptr();

    // Reserves before the arena's: the absorbed bits, then the planner's
    // three buffers (`sel`, `group_plans` and `resolve_keeps`; a diagram
    // without a marginal level has no overlap filter to reserve for). The
    // fifth is the survivor's pairs.
    eng.limits().refuse_nth_reserve(4);
    let result = contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch);
    eng.limits().grant_every_reserve();
    assert_eq!(result, Err(OperationError::OverBudget));
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(scratch.merge.group_plans.len(), 1);
    assert_eq!(tdd.levels[child.idx()].slot_count(), 2);
    assert_eq!(tdd.model_count().unwrap(), 4u32.into());

    // Return through the engine pool before retrying the same contraction.
    drop(scratch);
    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch), Ok(1));
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(scratch.merge.group_plans.len(), 1);
    assert_eq!(tdd.model_count().unwrap(), 4u32.into());
    drop(scratch);
    tdd.minimize().unwrap();
    crate::test_helpers::assert_canonical(&tdd);
}

/// Twins of a diagram with no marginal level have disjoint supports, so
/// their merge plan concatenates every member and never fills the marginal
/// map, a pass over every level of the diagram.
#[test]
fn a_plain_merge_leaves_the_marginal_map_unfilled() {
    let eng = Engine::new();
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(4));
    let parent = vtree.root();
    let (child, sibling) = vtree.children(parent);
    let mut levels = take_levels(&eng, vtree.num_nodes());
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let a = levels[child.idx()].push_internal_node(&[ChildPair::new(pos, pos)]);
    let b = levels[child.idx()].push_internal_node(&[ChildPair::new(neg, pos)]);
    let s = levels[sibling.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let output = levels[parent.idx()].push_internal_node(&[
        ChildPair::new(a, s), ChildPair::new(b, s),
    ]);
    let mut tdd = Tdd::from_levels_unchecked(
        vtree, levels, TddNodeId { vtree: parent, local: output },
    );
    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    scratch.flat_groups = vec![0, 1];
    scratch.group_starts = vec![0];
    assert_eq!(contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch), Ok(1));
    assert!(!scratch.has_marginal_below_valid);
    assert_eq!(tdd.model_count().unwrap(), 4u32.into());
}

/// Twins of an implicit level merge as those of its stored copy do
/// ([`same_as_stored`](crate::test_helpers::same_as_stored)). The diagram
/// has a marginal level, so the plan runs the overlap filter, which reads
/// every member's pairs; the merged level is stored.
#[test]
fn twins_of_an_implicit_level_merge_as_its_stored_copys_do() {
    crate::test_helpers::same_as_stored(twins_of_a_level_of_cubes);
}

/// The diagrams of [`twins_of_an_implicit_level_merge_as_its_stored_copys_do`]:
/// its diagram after the twins of its level of cubes merge, closed, and
/// after a reduction.
fn twins_of_a_level_of_cubes() -> Vec<Tdd> {
    let eng = Engine::new();
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(4));
    let parent = vtree.root();
    let (child, sibling) = vtree.children(parent);
    let literal = [NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::Neg as u32)];
    let one = NodeIdx(LeafLabel::One as u32);
    let mut levels = take_levels(&eng, vtree.num_nodes());
    // The four cubes over the child's two variables, a pair a node: an
    // affine level, which a low floor holds as its description.
    for m in 0..4 {
        levels[child.idx()].push_internal_node(&[ChildPair::new(literal[m / 2], literal[m % 2])]);
    }
    let s0 = levels[sibling.idx()].push_internal_node(&[ChildPair::new(literal[0], one)]);
    let s1 = levels[sibling.idx()].push_internal_node(&[ChildPair::new(literal[1], one)]);
    // Cubes 0 and 3 are paired with `s0` only, 1 and 2 with `s1` only: two
    // twin groups, in a parent no description fits.
    let output = levels[parent.idx()].push_internal_node(&[
        ChildPair::new(NodeIdx(0), s0), ChildPair::new(NodeIdx(1), s1),
        ChildPair::new(NodeIdx(2), s1), ChildPair::new(NodeIdx(3), s0),
    ]);
    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: parent, local: output });
    let stored = crate::test_helpers::stored_levels_forced();
    assert_eq!(tdd.levels[child.idx()].implicit().is_none(), stored, "the level of cubes is implicit unless forced stored");
    // Too wide to fit a reference, so the sibling's nodes stay slots.
    assert_can_make_marginal(&tdd.levels, &vtree, sibling);
    tdd.levels[sibling.idx()].become_marginal(vec![(1u128 << 40) + 2, (1u128 << 40) + 3], None);
    crate::diagram::inline_small_marginal_refs(&mut tdd, None);
    let before = tdd.model_count().unwrap();

    let mut scratch = eng.scratch.reduce.contract.checkout(&eng);
    scratch.flat_groups = vec![0, 3, 1, 2];
    scratch.group_starts = vec![0, 2];
    assert_eq!(contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch), Ok(2));
    assert!(scratch.has_marginal_below_valid, "the plan read the marginal map, then filtered");
    drop(scratch);
    assert!(tdd.levels[child.idx()].implicit().is_none(), "the merged level is stored");
    assert_eq!(tdd.model_count().unwrap(), before);
    // Closed as the end of a reduction closes the levels it changed.
    let mut merged = tdd.clone();
    merged.close_levels();
    tdd.minimize().unwrap();
    crate::test_helpers::assert_canonical(&tdd);
    assert_eq!(tdd.model_count().unwrap(), before);
    vec![merged, tdd]
}

/// The rebuild unions each group into its survivor (inline, packed and
/// ranged members alike), drops the absorbed members and a content-equal
/// redirected one, keeps every other node's pairs, and leaves the arena
/// holding exactly the live pairs in node order.
#[test]
fn the_rebuild_unions_groups_and_leaves_no_dead_arena() {
    let p = |l: u32, r: u32| ChildPair::new(NodeIdx(l), NodeIdx(r));
    let mut level = TddLevel::new();
    // Node 0 ranged (staged by hand), 1 inline, 2 and 3 packed, 4 inline,
    // 5 packed, 6 inline; garbage between the ranges.
    level.pairs = vec![
        p(9, 9), p(1, 0), p(1, 1), p(9, 9), p(3, 0), p(3, 1), p(3, 2),
        p(2, 0), p(2, 1), p(9, 9), p(5, 0), p(5, 1),
    ]
    .into();
    level.ranges = vec![PairRange { start: 1, len: 2 }].into();
    level.nodes = vec![
        EncodedNode::multi_ranged(0),
        EncodedNode::inline(p(4, 0)),
        EncodedNode::multi_pair(7, 2),
        EncodedNode::multi_pair(4, 3),
        EncodedNode::inline(p(6, 0)),
        EncodedNode::multi_pair(10, 2),
        EncodedNode::inline(p(7, 0)),
    ]
    .into();
    level.note_dead_pairs(3);
    let before: Vec<Vec<ChildPair>> = content(&level);
    // Groups {0, 3, 6} concat, {1, 4} concat, {2, 5} a content-equal
    // redirect of 5 onto 2.
    let mut bufs = MergeBuffers {
        sel: vec![0, 3, 6, 1, 4, 2, 5],
        group_plans: vec![
            GroupPlan { action: GroupAction::Concat, start: 0, end: 3 },
            GroupPlan { action: GroupAction::Concat, start: 3, end: 5 },
            GroupPlan { action: GroupAction::DupRedirect, start: 5, end: 7 },
        ],
        ..Default::default()
    };
    let mut remap = MergeRemap { final_remap: vec![NodeIdx(0); 7], absorbed: vec![0], ..Default::default() };
    let policy = MergePolicy { plain_level: true, parent_marginal: true, t1_scalable: false, disjoint_supports: false };
    assert_eq!(mark_group_members(&policy, &mut remap, &mut bufs), 4);
    let lim = Limits::new();
    let arenas = remap_and_reserve(&lim, &level, &mut remap, &bufs).unwrap();
    assert_eq!(
        remap.final_remap,
        remap_of(&[0, 1, 2, 0, 1, 2, 0], &[5]).final_remap,
        "the survivors numbered in node order, each member at its survivor",
    );
    rebuild_level(&mut level, arenas, &remap, &bufs, true);

    let cat = |nodes: &[usize]| -> Vec<ChildPair> { nodes.iter().flat_map(|&i| before[i].clone()).collect() };
    assert_eq!(content(&level), vec![cat(&[0, 3, 6]), cat(&[1, 4]), cat(&[2])]);
    assert_eq!(level.pairs.len(), live_pairs(&level), "no dead arena is left");
    assert_eq!(level.pairs.len(), 10);
    assert_eq!(level.pairs.capacity(), 10, "the arena is reserved at its exact size");
    assert_eq!(level.dead_pairs, 0);
    assert_eq!(level.ranges.len(), 0, "every rebuilt range fits the packed encoding");
    let starts: Vec<usize> = (0..3).map(|i| level.pair_range_at(i).start).collect();
    assert_eq!(starts, vec![0, 6, 8], "the arena is in node order");
}

/// The survivors outside every group keep their pairs, copied a run of
/// adjacent ranges at a time: each node lands at its offset in its run, a
/// group's union between two runs, and a range that does not continue the
/// run (node 3's, after the absorbed node 5's) starts a new one.
#[test]
fn the_rebuild_copies_untouched_ranges_in_runs() {
    let p = |l: u32, r: u32| ChildPair::new(NodeIdx(l), NodeIdx(r));
    let mut level = TddLevel::new();
    level.pairs = (0..14).map(|i| p(i, 100 + i)).collect::<Vec<_>>().into();
    level.nodes = vec![
        EncodedNode::multi_pair(0, 2),
        EncodedNode::multi_pair(2, 3),
        EncodedNode::inline(p(50, 50)),
        EncodedNode::multi_pair(8, 2),
        EncodedNode::multi_pair(10, 2),
        EncodedNode::multi_pair(5, 3),
        EncodedNode::multi_pair(12, 2),
    ]
    .into();
    let before: Vec<Vec<ChildPair>> = content(&level);
    let mut bufs = MergeBuffers {
        sel: vec![1, 5],
        group_plans: vec![GroupPlan { action: GroupAction::Concat, start: 0, end: 2 }],
        ..Default::default()
    };
    let mut remap = MergeRemap { final_remap: vec![NodeIdx(0); 7], absorbed: vec![0], ..Default::default() };
    let policy = MergePolicy { plain_level: true, parent_marginal: false, t1_scalable: false, disjoint_supports: true };
    assert_eq!(mark_group_members(&policy, &mut remap, &mut bufs), 1);
    let lim = Limits::new();
    let arenas = remap_and_reserve(&lim, &level, &mut remap, &bufs).unwrap();
    rebuild_level(&mut level, arenas, &remap, &bufs, true);

    let union: Vec<ChildPair> = [1, 5].iter().flat_map(|&i| before[i].clone()).collect();
    let expected = vec![before[0].clone(), union, before[2].clone(), before[3].clone(), before[4].clone(), before[6].clone()];
    assert_eq!(content(&level), expected);
    assert_eq!(level.pairs.len(), 14);
    let starts: Vec<usize> = [0, 1, 3, 4, 5].iter().map(|&i| level.pair_range_at(i).start).collect();
    assert_eq!(starts, vec![0, 2, 8, 10, 12], "the arena is in node order");
}
