//! Searches that list the nodes a contraction changed.
//!
//! A parent the sweep reaches only through a contraction above it has its
//! children searched at the nodes that contraction changed. In a debug build
//! every such search is checked against a search of the whole level, and
//! each child is searched whole again after its joint fixpoint for a twin
//! left behind, so these tests also run those checks.

use super::*;
use crate::diagram::{ChildPair, LeafLabel, NodeIdx, TddLevel, TddNodeId};
use crate::reduce::contract::tests::listed_searches;

use crate::Engine;
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use std::sync::Arc;

/// Over eight variables, the root's left level holds two pairs of twins,
/// `{A, B}` and `{C, D}`, and a fifth node `E` of its own. Merging each pair
/// unions its pair lists, which makes twins of the nodes the survivors name
/// one level down, `{a1, a2}` and `{a3, a4}`, while `a5`, which only `E`
/// names, is not listed. The root is the only level on the worklist, so the
/// left level is reached through its contraction and its left child is
/// searched by list. (Its right child is not searched at all: the merge
/// leaves every node of the left level one pair.)
#[test]
fn a_contraction_below_the_worklist_is_found_by_listing() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (l, r) = vtree.children(root);
    let (ll, lr) = vtree.children(l);
    let (rl, rr) = vtree.children(r);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let pair = ChildPair::new;

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let a1 = levels[ll.idx()].push_internal_node(&[pair(pos, pos)]);
    let a2 = levels[ll.idx()].push_internal_node(&[pair(neg, pos)]);
    let a3 = levels[ll.idx()].push_internal_node(&[pair(pos, neg)]);
    let a4 = levels[ll.idx()].push_internal_node(&[pair(neg, neg)]);
    let a5 = levels[ll.idx()].push_internal_node(&[pair(pos, one)]);
    let b1 = levels[lr.idx()].push_internal_node(&[pair(pos, pos)]);
    let b2 = levels[lr.idx()].push_internal_node(&[pair(neg, one)]);
    let b3 = levels[lr.idx()].push_internal_node(&[pair(pos, neg)]);
    let node_a = levels[l.idx()].push_internal_node(&[pair(a1, b1)]);
    let node_b = levels[l.idx()].push_internal_node(&[pair(a2, b1)]);
    let node_c = levels[l.idx()].push_internal_node(&[pair(a3, b2)]);
    let node_d = levels[l.idx()].push_internal_node(&[pair(a4, b2)]);
    let node_e = levels[l.idx()].push_internal_node(&[pair(a5, b3)]);
    let c0 = levels[rl.idx()].push_internal_node(&[pair(pos, pos)]);
    let c1 = levels[rl.idx()].push_internal_node(&[pair(neg, neg)]);
    let c2 = levels[rl.idx()].push_internal_node(&[pair(pos, neg)]);
    let d0 = levels[rr.idx()].push_internal_node(&[pair(pos, pos)]);
    let r0 = levels[r.idx()].push_internal_node(&[pair(c0, d0)]);
    let r1 = levels[r.idx()].push_internal_node(&[pair(c1, d0)]);
    let r2 = levels[r.idx()].push_internal_node(&[pair(c2, d0)]);
    // A and B are x2 x3 x4 beside x1 and x1', C and D are x2' x3' beside x1
    // and x1', and E is x1 x3 x4': the root's left sides are disjoint.
    let top = levels[root.idx()].push_internal_node(&[
        pair(node_a, r0),
        pair(node_b, r0),
        pair(node_c, r1),
        pair(node_d, r1),
        pair(node_e, r2),
    ]);

    let mut tdd = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: top });
    let count = tdd.model_count().unwrap();
    tdd.seed_contract_worklist([root.0]);
    let before = listed_searches();
    contract_all_twins(&eng, &mut tdd).expect("no budget is set");
    let listed = listed_searches() - before;

    assert_eq!(tdd.levels[l.idx()].slot_count(), 3, "{{A, B}} and {{C, D}} merge under the root");
    assert_eq!(tdd.levels[ll.idx()].slot_count(), 3, "{{a1, a2}} and {{a3, a4}} merge under the survivors");
    assert_eq!(tdd.levels[lr.idx()].slot_count(), 3, "b1, b2, b3 keep their contexts");
    assert!(listed >= 1, "the left level's left child is searched by list: {listed}");
    assert_eq!(tdd.model_count().unwrap(), count, "contraction keeps the function");
    tdd.minimize().unwrap();
    assert_canonical(&tdd);
    assert_eq!(tdd.model_count().unwrap(), count);
}
