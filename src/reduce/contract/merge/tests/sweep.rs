use super::*;
use crate::diagram::*;
use crate::test_helpers::pair;

/// Every node's pair slice, in arena order — the whole of a level's
/// observable content (`pairs_of_idx` resolves inline, normal-multi and
/// extended nodes alike).
fn content(level: &TddLevel) -> Vec<Vec<ChildPair>> {
    (0..level.nodes.len())
        .map(|i| level.pairs_of_idx(i).to_vec())
        .collect()
}

/// Arena slots actually owned by live nodes.
fn live_pairs(level: &TddLevel) -> usize {
    (0..level.nodes.len()).map(|i| level.arena_pairs_at(i)).sum()
}

/// One contraction round over the nodes from `first` on: merge each adjacent
/// pair (survivor first), then drop the absorbed ones — the same two steps
/// `contract_twins` runs, so each union lands at the arena tail with both
/// source ranges left dead behind it.
fn merge_adjacent_round(level: &mut TddLevel, first: usize) {
    let n = level.nodes.len();
    // Stand-in for `contract_twins`' hoisted grand reserve: the concat path
    // asserts the capacity is already there.
    let extra: usize = (first..n).map(|i| level.pair_count_at(i)).sum();
    level.pairs.reserve(extra);
    let mut merge_target: Vec<u32> = (0..n as u32).collect();
    let mut keep = first;
    while keep + 1 < n {
        let total = level.pair_count_at(keep) + level.pair_count_at(keep + 1);
        concat_twin_pairs(level, keep, &[keep as u32, keep as u32 + 1], total, /*allow_dups=*/ false, /*diagram_marginal=*/ false);
        merge_target[keep + 1] = keep as u32;
        keep += 2;
    }
    compact_explicit_level(level, &merge_target);
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
    level.pairs.extend([pair(1, 7), pair(2, 7)]);
    level.ranges.push(PairRange { start: 0, len: 2 });
    level.nodes.push(EncodedNode::multi_ranged(0));
    level.push_internal_node(&[pair(3, 4)]);
    let first_multi = level.nodes.len();
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
    ];
    parent.ranges = vec![PairRange { start: 0, len: 2 }];
    parent.nodes = vec![EncodedNode::multi_ranged(0)];
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    // T1 twins {0, 1} merged into 0: the parent's second pair is dropped.
    let remap = MergeRemap {
        merge_target: vec![0, 0],
        duplicate_redirect: vec![false, false],
        final_remap: vec![NodeIdx(0), NodeIdx(0)],
    };

    rewrite_parent(&mut tdd, root, ChildSide::Left, &remap);

    let parent = &tdd.levels[root.idx()];
    assert_eq!(parent.ranges.len(), 1, "no range entry is added for an inlined survivor");
    assert!(matches!(parent.nodes[0].kind(), NodeKind::Inline(_)), "the sole survivor is stored inline");
    assert_eq!(
        parent.pairs_of_idx(0),
        &[ChildPair::new(NodeIdx(0), sibling)],
    );
    assert_eq!(parent.dead_pairs, 2, "the whole old range is abandoned");
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
    let mut scratch = eng.scratch.reduce.contract.checkout(eng.limits());
    scratch.flat_groups = vec![0, 1];
    scratch.group_starts = vec![0];
    scratch.remap.merge_target = vec![0, 1];
    scratch.remap.final_remap = vec![NodeIdx(0); 2];
    scratch.remap.duplicate_redirect = vec![false; 2];
    scratch.merge.sel.reserve(8);
    let allocation = scratch.merge.sel.as_ptr();

    // Reserves before the arena's: the redirect flags, then the planner's
    // eight buffers (`sel`, `group_plans`, the five overlap-filter buffers and
    // `resolve_keeps`). The tenth is the survivor's pairs.
    eng.limits().refuse_nth_reserve(9);
    let result = contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch);
    eng.limits().grant_every_reserve();
    assert_eq!(result, Err(OperationError::OverBudget));
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(scratch.merge.group_plans.len(), 1);
    assert_eq!(tdd.levels[child.idx()].slot_count(), 2);
    assert_eq!(tdd.model_count().unwrap(), 4u32.into());

    // Return through the engine pool before retrying the same contraction.
    drop(scratch);
    let mut scratch = eng.scratch.reduce.contract.checkout(eng.limits());
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(contract_twins(&eng, &mut tdd, child, parent, ChildSide::Left, &mut scratch), Ok(1));
    assert_eq!(scratch.merge.sel.as_ptr(), allocation);
    assert_eq!(scratch.merge.group_plans.len(), 1);
    assert_eq!(tdd.model_count().unwrap(), 4u32.into());
    drop(scratch);
    tdd.minimize().unwrap();
    crate::test_helpers::assert_canonical(&tdd);
}
