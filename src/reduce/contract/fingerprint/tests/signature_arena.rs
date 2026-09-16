use super::*;
use crate::Engine;
use crate::diagram::{ChildPair, LeafLabel, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::Vtree;
use std::sync::Arc;

/// Number of unique-fingerprint (provably twin-free) nodes at the child level.
const N_UNIQUE: u32 = 4;
/// Parent pairs referencing each unique node — its whole signature.
const UNIQUE_FAN_OUT: u32 = 10;
/// 2-member twin groups at the child level.
const N_TWIN_GROUPS: u32 = 3;
/// Every parent pair at the level, i.e. what a fan-out-sized signature arena
/// would have to hold.
const TOTAL_FAN_OUT: usize = (N_UNIQUE * UNIQUE_FAN_OUT + 2 * N_TWIN_GROUPS) as usize;
/// The twin candidates' share of that fan-out — the only rows any signature
/// comparison can reach.
const CANDIDATE_MASS: usize = (2 * N_TWIN_GROUPS) as usize;

/// The signature arena must be sized by candidate mass, not by the level's
/// total parent-pair fan-out.
///
/// Fixture — child level = the root's left child, `N_UNIQUE + 2 *
/// N_TWIN_GROUPS` nodes:
///   * nodes `0..N_UNIQUE` get one parent each, holding `UNIQUE_FAN_OUT` pairs
///     with distinct siblings ⇒ each fingerprint is unique to its node, so
///     none of them is a twin candidate, and between them they own all but
///     `CANDIDATE_MASS` of the level's parent pairs;
///   * the remaining nodes are paired off, two per parent, both members
///     sharing the *same* sibling ⇒ identical contexts ⇒ one twin group each.
///
/// Candidates are therefore a MAJORITY of the nodes (6 of 10) while owning a
/// small minority of the fan-out — the twin-dense shape in which the arena
/// must still hold candidate rows only.
#[test]
fn signature_arena_holds_candidate_rows_only() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);

    let mut levels: Vec<TddLevel> =
        (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();

    // Neither child level's node CONTENT participates in context twin
    // detection (only the parent pairs do), so one shape serves for all of
    // them; what matters is the node COUNT at each level.
    let filler = ChildPair::new(NodeIdx(LeafLabel::Pos as u32), NodeIdx(LeafLabel::One as u32));
    let child_width = (N_UNIQUE + 2 * N_TWIN_GROUPS) as usize;
    for _ in 0..child_width {
        levels[v_left.idx()].push_internal_node(&[filler]);
    }
    for _ in 0..UNIQUE_FAN_OUT {
        levels[v_right.idx()].push_internal_node(&[filler]);
    }

    // One parent per unique node: UNIQUE_FAN_OUT distinct sibling contexts.
    let mut pairs: Vec<ChildPair> = Vec::new();
    for target in 0..N_UNIQUE {
        pairs.clear();
        for sibling in 0..UNIQUE_FAN_OUT {
            pairs.push(ChildPair::new(NodeIdx(target), NodeIdx(sibling)));
        }
        levels[root.idx()].push_internal_node(&pairs[..]);
    }
    // One parent per twin group: both members share the identical context.
    for group in 0..N_TWIN_GROUPS {
        let first = N_UNIQUE + 2 * group;
        let sibling = NodeIdx(group);
        levels[root.idx()].push_internal_node(&[
            ChildPair::new(NodeIdx(first), sibling),
            ChildPair::new(NodeIdx(first + 1), sibling),
        ]);
    }

    let tdd = Tdd::from_levels_unchecked(
        vtree,
        levels,
        TddNodeId { vtree: root, local: NodeIdx(0) },
    );

    let mut scratch = ContractScratch::default();
    let found = find_twin_groups(&eng, &tdd, root, ChildSide::Left, child_width, &mut scratch)
        .expect("find_twin_groups");

    assert!(found, "the {} same-context pairs are twin groups", N_TWIN_GROUPS);
    assert_eq!(scratch.group_starts, vec![0, 2, 4]);
    assert_eq!(scratch.flat_groups, vec![4, 5, 6, 7, 8, 9]);
    // The scratch is fresh (no pooled high-water mark), so the arena's length
    // after this one call is exactly what the call sized it to.
    assert_eq!(
        scratch.entries.len(),
        CANDIDATE_MASS,
        "signature arena must hold the {} candidate rows, not the level's {} parent pairs",
        CANDIDATE_MASS,
        TOTAL_FAN_OUT,
    );
}
