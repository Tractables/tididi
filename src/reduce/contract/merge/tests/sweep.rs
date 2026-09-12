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
        merge_two_internal_twins(level, keep, keep + 1, /*allow_dups=*/ false);
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
    // sit at the tail). First an extended (side-table) len-1 node, whose
    // start lives in `multi_pairs` rather than the node word: a left ref with
    // MULTI_BIT set cannot be inlined, which is what routes
    // `push_internal_node` to the extended encoding. Then an inline node,
    // which owns no arena slot at all.
    level.push_internal_node(&[pair(1 << 31, 7)]);
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
/// a single non-inlinable pair must rewrite its own `multi_pairs` entry, not
/// mint a second one and abandon the first.
///
/// Hand-encoded because the arm needs a length-≥2 extended node, which only a
/// pair arena past the 31-bit start/length encoding mints in a real run.
#[test]
fn a_shrunk_extended_parent_node_rewrites_its_own_range_entry() {
    let vtree = std::sync::Arc::new(crate::vtree::Vtree::balanced(2));
    let root = vtree.root();
    let mut levels: Vec<TddLevel> =
        (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    // A right-side ref with bit 31 set: the survivor cannot go back inline, so
    // the re-encoding has to keep a `multi_pairs` range for it.
    let sibling = NodeIdx(1 << 31 | 3);
    let parent = &mut levels[root.idx()];
    parent.pairs = vec![
        ChildPair::new(NodeIdx(0), sibling),
        ChildPair::new(NodeIdx(1), sibling),
    ];
    parent.multi_pairs = vec![MultiPairRange { start: 0, len: 2 }];
    parent.nodes = vec![EncodedNode::multi_ranged(0)];
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);

    // T1 twins {0, 1} merged into 0: the parent's second pair is dropped.
    let mut scratch = ContractScratch {
        merge_target: vec![0, 0],
        duplicate_redirect: vec![false, false],
        final_remap: vec![NodeIdx(0), NodeIdx(0)],
        ..Default::default()
    };

    rewrite_parent(&mut tdd, root, ChildSide::Left, &mut scratch);

    let parent = &tdd.levels[root.idx()];
    assert_eq!(
        parent.multi_pairs.len(), 1,
        "the shrunk node must keep its own range entry, not abandon it",
    );
    assert_eq!(parent.multi_pairs[0], MultiPairRange { start: 0, len: 1 });
    assert_eq!(
        parent.pairs_of_idx(0),
        &[ChildPair::new(NodeIdx(0), sibling)],
    );
}
