//! The semantic rewrites of an implicit level against the same rewrites of
//! its stored copy: the same pairs, the level built stored.

use crate::apply::falsity::propagate_false_nodes;
use crate::diagram::{ChildDecoder, ValueRef};
use crate::marginal::marginalize_leaf_inline;
use crate::test_helpers::{sorted_pairs, with_stored_copy, x_decision_diagram};

/// Emptying the nodes of the level below that the `¬x2` pairs name drops
/// those pairs: what is left of each node, its two `(x2, ·)` pairs, is built
/// stored, as the sweep leaves the level's stored copy.
#[test]
fn the_falsity_sweep_builds_what_is_left_of_an_implicit_level() {
    let (mut implicit, v, w) = x_decision_diagram(32);
    assert!(implicit.levels[v.idx()].implicit().is_some(), "the fixture's level is implicit");
    let below = &mut implicit.levels[w.idx()];
    for j in (1..below.nodes().len()).step_by(2) {
        below.nodes.stored_mut()[j] = below.encode_multi(0, 0);
    }
    let mut stored = with_stored_copy(&implicit, v);
    propagate_false_nodes(&mut implicit);
    propagate_false_nodes(&mut stored);
    assert_eq!(sorted_pairs(&implicit), sorted_pairs(&stored));
    let (level, oracle) = (&implicit.levels[v.idx()], &stored.levels[v.idx()]);
    assert_eq!(level.pair_count_at(0), 2);
    assert!(level.implicit().is_none(), "the rest is built stored");
    assert_eq!((level.pairs.len(), level.pairs.capacity(), level.dead_pairs), (oracle.pairs.len(), oracle.pairs.capacity(), oracle.dead_pairs));
}

/// Summing out x2 makes its two labels one count: the `(x2, ·)` and
/// `(¬x2, ·)` pairs of a node then name the nodes below in the same
/// context, twins the reduction that follows merges. The move of the
/// level's slots is not one to one, so the level is stored for it.
#[test]
fn summing_out_a_leaf_under_an_implicit_level_stores_it_for_the_reduction() {
    let (mut implicit, v, _) = x_decision_diagram(32);
    assert!(implicit.levels[v.idx()].implicit().is_some(), "the fixture's level is implicit");
    let mut stored = with_stored_copy(&implicit, v);
    let vtree = std::sync::Arc::clone(&implicit.vtree);
    let x2 = vtree.children(v).0;
    marginalize_leaf_inline(&mut implicit, x2, &vtree);
    marginalize_leaf_inline(&mut stored, x2, &vtree);
    assert_eq!(sorted_pairs(&implicit), sorted_pairs(&stored));
    let level = &implicit.levels[v.idx()];
    assert!(level.pairs_iter_of_idx(0).all(|p| matches!(ChildDecoder::marginal().value(p.left), ValueRef::Inline(1))));
    assert!(level.implicit().is_none(), "a move that is not one to one stores the level");
}
