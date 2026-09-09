use super::*;
use crate::vtree::Vtree;

/// Restriction compacts each node's pair list inside the arena range that
/// node already owns — the level is never rebuilt into a second arena.
/// Fixture, conditioning the LEFT leaf to ⊤ (Pos pairs survive and lose
/// their label, Neg pairs drop, One passes through):
///
/// - node0, multi `[(Pos,One), (Neg,Pos), (One,Neg)]` → `[(One,One), (One,Neg)]`
///   — the third pair moves down over an already-read slot, one slot is abandoned
/// - node1, inline `(Pos,One)` → inline `(One,One)`
/// - node2, inline `(Neg,One)` → emptied to the zero-pair placeholder
///
/// The arena length is what pins "in place": the former rebuild replaced the
/// arena with a freshly pushed one holding only the two live pairs, whereas
/// the shrink leaves the abandoned slot behind and reports it dead.
#[test]
fn rewrite_for_restrict_shrinks_pair_lists_in_place() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut tdd = constant_one(eng, &vtree);
    let level = &mut tdd.levels[root.idx()];
    level.clear();
    level.push_internal_node(&[
        InputPair { left: POS, right: ONE },
        InputPair { left: NEG, right: POS },
        InputPair { left: ONE, right: NEG },
    ]);
    level.push_internal_node(&[InputPair { left: POS, right: ONE }]);
    level.push_internal_node(&[InputPair { left: NEG, right: ONE }]);
    let arena_len_before = level.pairs.len();

    rewrite_for_restrict(&mut tdd, root, ChildSide::Left, Polarity::Positive);

    let level = &tdd.levels[root.idx()];
    assert_eq!(level.nodes.len(), 3, "node indices are preserved");
    assert_eq!(
        level.pairs_of_idx(0),
        &[InputPair { left: ONE, right: ONE }, InputPair { left: ONE, right: NEG }],
        "survivors compacted into the node's own range, sorted",
    );
    assert_eq!(
        level.pairs_of_idx(1),
        &[InputPair { left: ONE, right: ONE }],
        "the inline node's restricted pair stays inline",
    );
    assert!(level.pairs_of_idx(2).is_empty(), "an all-dropped node keeps no pairs");
    assert_eq!(level.pairs.len(), arena_len_before, "no second arena, and no growth");
    assert_eq!(level.dead_pairs, 1, "the one abandoned slot is reported as dead");
}
