//! The pair arena after a prune: a dropped node's range is counted dead, and
//! the arena is swept once enough of it is dead.

use std::sync::Arc;

use crate::diagram::{ChildPair, Tdd, TddLevel, ZERO};
use crate::reduce::prune::{PruneScope, prune_unreachable};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use crate::Engine;

/// A clause diagram whose root level holds `copies` further nodes, each a
/// copy of the root node's pair list, all unreachable from the output.
fn with_unreachable_copies(copies: usize) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let mut tdd = Tdd::clause(&vtree, [1, 2]).unwrap();
    let root = vtree.root().idx();
    let pairs: Vec<ChildPair> = tdd.levels[root].pairs_of_idx(tdd.output.local.idx()).to_vec();
    assert!(pairs.len() >= 2, "the fixture needs a node whose pairs live in the arena");
    for _ in 0..copies {
        tdd.levels[root].push_internal_node(&pairs);
    }
    tdd
}

/// The arena slots the level's live nodes own.
fn live_pairs(level: &TddLevel) -> usize {
    (0..level.nodes.len()).map(|i| level.arena_pairs_at(i)).sum()
}

#[test]
fn prune_counts_the_dropped_nodes_pair_ranges_as_dead() {
    let eng = Engine::new();
    let mut tdd = with_unreachable_copies(10);
    let root = tdd.vtree.root().idx();
    let per_node = tdd.levels[root].arena_pairs_at(0);
    prune_unreachable(&eng, &mut tdd, PruneScope::Whole).unwrap();
    assert_canonical(&tdd);
    assert_eq!(tdd.levels[root].nodes.len(), 1);
    // Below the sweep threshold the slack stays, and is on record.
    assert_eq!(tdd.levels[root].dead_pairs as usize, 10 * per_node);
    assert_eq!(tdd.levels[root].pairs.len(), 11 * per_node);
}

#[test]
fn prune_sweeps_the_arena_once_most_of_it_is_dead() {
    let eng = Engine::new();
    let copies = TddLevel::PAIRS_COMPACT_MIN_DEAD;
    let mut tdd = with_unreachable_copies(copies);
    let root = tdd.vtree.root().idx();
    let before = tdd.levels[root].pairs.len();
    assert!(before > 2 * TddLevel::PAIRS_COMPACT_MIN_DEAD);
    prune_unreachable(&eng, &mut tdd, PruneScope::Whole).unwrap();
    assert_canonical(&tdd);
    let level = &tdd.levels[root];
    assert_eq!(level.nodes.len(), 1);
    assert_eq!(level.pairs.len(), live_pairs(level), "the sweep leaves no dead slot in the arena");
    assert_eq!(level.dead_pairs, 0);
    assert_eq!(tdd.model_count().unwrap(), 3u32.into());
}

#[test]
fn prune_of_the_constant_false_diagram_frees_every_arena() {
    let eng = Engine::new();
    let mut tdd = with_unreachable_copies(10);
    tdd.output.local = ZERO;
    prune_unreachable(&eng, &mut tdd, PruneScope::Whole).unwrap();
    for level in &tdd.levels {
        assert!(level.nodes.is_empty());
        assert!(level.pairs.is_empty());
        assert!(level.ranges.is_empty());
        assert_eq!(level.dead_pairs, 0);
    }
}
