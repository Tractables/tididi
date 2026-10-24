//! A pair fusion that runs to the end is recorded on its parent, and the
//! next fusion there is skipped until an edit pushes the level.

use super::*;
use crate::diagram::{ChildPair, NodeIdx, TddLevel, TddNodeId};
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
/// a record skips run or not, and some minimization starts at a recorded
/// boundary parent.
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
