//! A pair fusion that runs to the end is recorded on its parent, and the
//! next fusion there is skipped until an edit pushes the level.

use super::*;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, NodeIdx, TddLevel, TddNodeId, ValueRef};
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::test_helpers::{assert_same_shape, compile_clauses_on, rand_cnf, toy, vtree_shapes, CnfShape};
use crate::vtree::rng::Lcg;
use crate::vtree::Vtree;
use std::sync::Arc;

/// An engine that stops at its first poll, with a poll at every node: a
/// fusion that reads a node is stopped, and one that is skipped is not.
fn stopping() -> Engine {
    let eng = Engine::new();
    let callback = StopCallback::new(|_, _| StopDecision::Stop);
    let _prior = eng.limits().install(LimitConfig::none().with_stop_callback(Some(callback)));
    eng.limits().pin_reduce_poll_stride(Some(1));
    eng
}

/// Two root nodes over a marginal right child, the first with a group at
/// explicit ref 0.
fn one_group() -> Tdd {
    toy((0..4).map(|s| s * 3 + 1).collect(), &[&[(0, 0), (0, 1), (1, 2)], &[(1, 3), (2, 0)]])
}

#[test]
fn a_fusion_run_to_the_end_is_recorded_and_the_next_one_skipped() {
    let mut tdd = one_group();
    let root = tdd.vtree.root();
    assert!(!tdd.dirty.is_fused(root.0), "a diagram starts with no record");
    let stats = fuse_pairs(&Engine::new(), &mut tdd).expect("no limits armed");
    assert_eq!(stats.fusion_groups, 1);
    assert!(tdd.dirty.is_fused(root.0));
    let stats = fuse_pairs(&stopping(), &mut tdd).expect("a recorded parent is not read");
    assert_eq!(stats.fusion_groups, 0);
    tdd.invalidate(root);
    assert!(!tdd.dirty.is_fused(root.0), "an edit of the parent drops its record");
    let r = fuse_pairs(&stopping(), &mut tdd);
    assert!(matches!(r, Err(OperationError::Stopped)), "the parent is read again: {r:?}");
}

#[test]
fn a_stopped_fusion_records_nothing() {
    let mut tdd = one_group();
    let root = tdd.vtree.root();
    let r = fuse_pairs(&stopping(), &mut tdd);
    assert!(matches!(r, Err(OperationError::Stopped)), "got {r:?}");
    assert!(!tdd.dirty.is_fused(root.0));
}

/// Under a parent with both children marginal, a fusion on one side can
/// mint a group on the other, so it is not recorded.
#[test]
fn a_parent_with_both_children_marginal_is_not_recorded() {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let (left, right) = vtree.children(root);
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let counts: Vec<u128> = (0..4).map(|s| s * 3 + 1).collect();
    levels[left.idx()].set_counts_state(counts.clone(), None);
    levels[right.idx()].set_counts_state(counts, None);
    let pairs: Vec<ChildPair> = [(0, 0), (0, 1), (1, 0), (1, 1)]
        .iter()
        .map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r)))
        .collect();
    levels[root.idx()].push_internal_node(&pairs);
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, TddNodeId { vtree: root, local: NodeIdx(0) });
    fuse_pairs(&Engine::new(), &mut tdd).expect("no limits armed");
    assert!(!tdd.dirty.is_fused(root.0));
}

/// Random formulas on every vtree shape, a level marginalized at a time
/// and minimized after each: the diagrams are the same whether the fusions
/// a record skips or lists run whole or not, and some minimization starts
/// at a recorded boundary parent.
#[test]
fn a_skipped_fusion_is_one_that_finds_no_group() {
    let mut rng = Lcg::new(29);
    let mut recorded = 0;
    for num_vars in 4..9u32 {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let eng = Engine::new();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 2 * num_vars as usize, width: 3 });
            let mut kept = compile_clauses_on(&eng, &vtree, &clauses);
            if kept.is_zero() {
                continue;
            }
            let mut forced = kept.clone();
            let below_root: Vec<VtreeIdx> =
                (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
            for step in 0..4 {
                let target = below_root[rng.below(below_root.len() as u64) as usize];
                eng.marginalize_levels(&mut kept, &[target]).unwrap();
                recorded += (0..vtree.num_nodes() as u32).filter(|&t| kept.dirty.is_fused(t)).count();
                eng.minimize(&mut kept).unwrap();
                eng.marginalize_levels(&mut forced, &[target]).unwrap();
                forced.dirty.forget_all_fused();
                eng.minimize(&mut forced).unwrap();
                forced.dirty.forget_all_fused();
                assert_same_shape(&kept, &forced, &format!("{shape} over {num_vars} variables, step {step}"));
            }
        }
    }
    assert!(recorded > 0, "no minimization started at a recorded parent");
}

/// Two nodes of a boundary parent recorded fused, neither holding a group,
/// are twins under the root. Their merge unions their pairs into a node that
/// holds two groups, and the fusion after it reads that node only.
#[test]
fn a_merge_at_a_recorded_parent_is_fused_at_its_survivors() {
    // The root over variable 1 and the boundary parent, which is over
    // variable 2 and the marginal variable 3.
    let vtree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nL 2 3\nI 3 1 2\nI 4 0 3\n").unwrap());
    let root = vtree.root();
    let (_, boundary) = vtree.children(root);
    let (_, marginal) = vtree.children(boundary);
    let label = |l: LeafLabel| NodeIdx(l as u32);
    let slot = |s: u32| NodeIdx(ValueRef::slot_raw(s));
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    const BIG: u128 = 1 << 40;
    levels[marginal.idx()].become_marginal(vec![BIG, BIG + 1, BIG + 2, BIG + 3], None);
    let (pos, neg) = (label(LeafLabel::Pos), label(LeafLabel::Neg));
    let a = levels[boundary.idx()].push_internal_node(&[ChildPair::new(pos, slot(0)), ChildPair::new(neg, slot(2))]);
    let b = levels[boundary.idx()].push_internal_node(&[ChildPair::new(pos, slot(1)), ChildPair::new(neg, slot(3))]);
    let one = label(LeafLabel::One);
    levels[root.idx()].push_internal_node(&[ChildPair::new(one, a), ChildPair::new(one, b)]);
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, TddNodeId { vtree: root, local: NodeIdx(0) });
    let eng = Engine::new();
    assert_eq!(fuse_pairs(&eng, &mut tdd).expect("no limits armed").fusion_groups, 0);
    assert!(tdd.dirty.is_fused(boundary.0), "a fusion that found nothing records its parent");

    let listed_before = crate::reduce::contract::tests::listed_fusions();
    tdd.seed_contract_worklist([root.0]);
    crate::reduce::contract::sweep::contract_all_twins(&eng, &mut tdd).expect("no limits armed");
    assert!(crate::reduce::contract::tests::listed_fusions() > listed_before, "the fusion read the whole parent");
    let level = &tdd.levels[boundary.idx()];
    assert_eq!(level.nodes().len(), 1, "the twins merged");
    let counts = tdd.levels[marginal.idx()].marginal_counts().unwrap();
    let mut fused: Vec<(u32, u128)> = level
        .pairs_vec(0)
        .iter()
        .map(|p| match ChildDecoder::marginal().value(EncodedChildRef::from_raw(p.right.0)) {
            ValueRef::Slot(s) => (p.left.0, counts[s as usize]),
            ValueRef::Inline(c) => (p.left.0, c as u128),
        })
        .collect();
    fused.sort_unstable();
    let mut want = vec![(pos.0, 2 * BIG + 1), (neg.0, 2 * BIG + 5)];
    want.sort_unstable();
    assert_eq!(fused, want, "each label's two pairs fused into one");
}
