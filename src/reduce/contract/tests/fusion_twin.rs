//! Where pair fusion and twin contraction meet: a fusion redex that creates
//! twins, and the fork-down content twins above a boundary parent.

use std::sync::Arc;

use super::sweep::contract_all_twins;
use crate::Engine;
use crate::diagram::*;
use crate::test_helpers::check::marginal::{check_pair_fusion_saturation, check_twin_canonicality};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Constructs a level where closing pair fusion redexes creates structural twins:
/// One root node holds four pairs referencing two explicit nodes A and B.
/// A's group uses slots 0 and 1 (`COUNT_A` + `COUNT_B` = `COUNT_SUM`); B's group
/// uses slots 2 and 3 (`COUNT_C` + `COUNT_D` = `COUNT_SUM`). Before fusion, A and B
/// have different parent contexts, so they are not twins. After fusion, both
/// groups yield the same summed count `COUNT_SUM` and map to the same surviving
/// slot — giving A and B identical contexts. The duplicate redirect merge keeps
/// both pairs remapped onto the survivor and pair fusion folds them to `2 * COUNT_SUM`.
#[test]
fn fusion_creates_twin_both_closed_in_one_call() {
    const COUNT_A: u128 = 1_000_000_000_000;
    const COUNT_B: u128 = 2_000_000_000_000;
    const COUNT_C: u128 = 500_000_000_000;
    const COUNT_D: u128 = 2_500_000_000_000;
    const COUNT_SUM: u128 = COUNT_A + COUNT_B;

    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), VtreeNode::Internal { .. }));

    let (pos, one) = (
        NodeIdx(LeafLabel::Pos as u32),
        NodeIdx(LeafLabel::One as u32),
    );
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();

    let a = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, one)]);
    let b = levels[v_left.idx()].push_internal_node(&[ChildPair::new(pos, one)]);

    levels[v_right.idx()].become_marginal(vec![COUNT_A, COUNT_B, COUNT_C, COUNT_D], None);
    let slot = |s: u32| NodeIdx(ValueRef::slot_raw(s));
    levels[root.idx()].push_internal_node(&[
        ChildPair::new(a, slot(0)),
        ChildPair::new(a, slot(1)),
        ChildPair::new(b, slot(2)),
        ChildPair::new(b, slot(3)),
    ]);

    let output = TddNodeId {
        vtree: root,
        local: NodeIdx(0),
    };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    inline_small_marginal_refs(&mut tdd, None);

    assert!(check_pair_fusion_saturation(&tdd, None).is_err());

    tdd.seed_contract_worklist([root.0]);
    contract_all_twins(&eng, &mut tdd).expect("contract_all_twins");

    check_pair_fusion_saturation(&tdd, None).unwrap();
    check_twin_canonicality(&tdd).unwrap();
    assert_eq!(tdd.levels[v_left.idx()].slot_count(), 1);

    let root_pairs = tdd.levels[root.idx()].pairs_vec(0);
    assert_eq!(root_pairs.len(), 1);
    let marginal_counts = tdd.levels[v_right.idx()].marginal_counts().unwrap();
    let count =
        match ChildDecoder::marginal().value(EncodedChildRef::from_raw(root_pairs[0].right.0)) {
            ValueRef::Slot(s) => marginal_counts[s as usize],
            ValueRef::Inline(c) => c as u128,
        };
    assert_eq!(count, 2 * COUNT_SUM);
}
