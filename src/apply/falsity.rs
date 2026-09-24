//! Dropping pairs from a level and propagating the falsity that leaves behind.
//!
//! Conditioning, care restriction and negation's fill test share these: a
//! rewrite that filters one level's pair lists in place, and the bottom-up
//! sweep that empties every node left computing ⊥.

use std::sync::Arc;

use crate::diagram::{sort_pairs, ChildDecoder, ChildPair, EncodedChildRef, EncodedNode, NodeKind, Tdd, TddLevel, ZERO};

/// Propagate falsity upward after pairs were dropped in place, so that no
/// node left in the diagram computes ⊥.
///
/// A node whose every pair was dropped computes ⊥, which invariant 2
/// (`docs/architecture.md`) forbids, and every reduction rule that follows
/// assumes such a node is already gone. One bottom-up pass drops every pair
/// whose structural child is empty, which empties further nodes above and
/// cascades. What is left unreferenced is removed by `prune_unreachable` in
/// the reduction that follows; an emptied output is collapsed to the
/// sentinel. Conditioning and care restriction both end in this pass.
///
/// Marginal levels are passed over: their structure is summed out, so they hold
/// no node that a dropped pair could have emptied.
pub(super) fn propagate_false_nodes(tdd: &mut Tdd) {
    let vtree = Arc::clone(&tdd.vtree);
    for (vi, left, right) in vtree.internal_bottomup() {
        if tdd.levels[vi.idx()].is_marginal() { continue; }
        // Leaf labels and marginal values do not name structural nodes.
        let left_structural = tdd.is_structural_internal(left);
        let right_structural = tdd.is_structural_internal(right);
        let [parent, left_level, right_level] = tdd.levels
            .get_disjoint_mut([vi.idx(), left.idx(), right.idx()])
            .expect("a parent and its children are distinct levels");
        let has_empty = |structural: bool, level: &TddLevel| {
            structural && (0..level.nodes.len()).any(|i| empty_node(level, i))
        };
        if parent.nodes.is_empty()
            || !(has_empty(left_structural, left_level) || has_empty(right_structural, right_level))
        { continue; }
        let dead = |structural: bool, level: &TddLevel, child: EncodedChildRef| {
            child == ZERO.into()
                || (structural && empty_node(level, ChildDecoder::structural().node(child).idx()))
        };
        rewrite_level_pairs(parent, |_, _, _, pair| {
            if dead(left_structural, left_level, pair.left) || dead(right_structural, right_level, pair.right) {
                None
            } else {
                Some(pair)
            }
        });
        tdd.invalidate(vi);
    }
    let output = tdd.output;
    if empty_node(&tdd.levels[output.vtree.idx()], output.local.idx()) {
        tdd.output.local = ZERO;
    }
}

/// Whether node `i` owns no pairs.
pub(super) fn empty_node(level: &TddLevel, i: usize) -> bool {
    level.pair_count_at(i) == 0
}

/// Rewrite a level's pair lists in place through `rewrite_pair`, which sees
/// each pair with its node's index, its position in that node and the node's
/// pair count, and drop every pair it answers `None` for. Answers whether any node was left with
/// no pairs at all.
///
/// Both of conditioning's rewrites and the care restriction's are this pass
/// under a different predicate.
pub(super) fn rewrite_level_pairs(
    level: &mut TddLevel,
    mut rewrite_pair: impl FnMut(usize, usize, usize, ChildPair) -> Option<ChildPair>,
) -> bool {
    let n_nodes = level.nodes.len();
    if n_nodes == 0 {
        return false;
    }

    let mut emptied = false;
    let mut dead = 0usize;
    for i in 0..n_nodes {
        if let NodeKind::Inline(p) = level.nodes[i].kind() {
            // The single pair lives in the node's own two words, not the arena.
            match rewrite_pair(i, 0, 1, p) {
                Some(np) => {
                    level.nodes[i] = EncodedNode::inline(np);
                }
                None => {
                    // Emptied: the slot holds no pair, which `propagate_false_nodes`
                    // reads as the node computing false and drops every reference to.
                    let empty = level.encode_multi(0, 0);
                    level.nodes[i] = empty;
                    emptied = true;
                }
            }
            continue;
        }

        let range = level.pair_range_at(i);
        let (start, old_len) = (range.start, range.len());
        let pairs = level.pairs_mut(i);
        let mut w = 0usize;
        for r in 0..old_len {
            if let Some(np) = rewrite_pair(i, r, old_len, pairs[r]) {
                // `w <= r`, so this write is at or below a slot already read.
                pairs[w] = np;
                w += 1;
            }
        }
        sort_pairs(&mut pairs[..w]);
        if w < old_len {
            // As in the inline arm, an emptied node computes false, and the
            // falsity sweep drops what still names it.
            dead += level.reencode_shrunk(i, start, old_len, w);
            emptied |= w == 0;
        }
    }

    level.note_dead_pairs(dead);
    // The sweep's precondition holds: every node kept a prefix of its own
    // range, so live ranges stay pairwise disjoint.
    level.compact_pairs_if_stale();
    emptied
}
