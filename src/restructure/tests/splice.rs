use std::sync::Arc;

use crate::diagram::{ChildPair, LeafLabel, NodeIdx, TddLevel, TddNodeId};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use crate::{Engine, Tdd, literal};

/// Over eight variables, the right operand's level over x5 x6 holds two
/// twins, x5 x6 and x5' x6', which its parent pairs with the same x7 x8; the
/// parent is on the operand's worklist, and no node is unreachable, so only
/// that worklist entry leads a reduction to them. The splice keeps it.
#[test]
fn a_splice_keeps_what_the_moved_levels_owe() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let root = vtree.root();
    let (l, r) = vtree.children(root);
    let (ll, lr) = vtree.children(l);
    let (rl, rr) = vtree.children(r);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let pair = ChildPair::new;

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let free_ll = levels[ll.idx()].push_internal_node(&[pair(one, one)]);
    let free_lr = levels[lr.idx()].push_internal_node(&[pair(one, one)]);
    let free = levels[l.idx()].push_internal_node(&[pair(free_ll, free_lr)]);
    let equal_true = levels[rl.idx()].push_internal_node(&[pair(pos, pos)]);
    let equal_false = levels[rl.idx()].push_internal_node(&[pair(neg, neg)]);
    let both = levels[rr.idx()].push_internal_node(&[pair(pos, pos)]);
    let twins = levels[r.idx()].push_internal_node(&[pair(equal_true, both), pair(equal_false, both)]);
    let top = levels[root.idx()].push_internal_node(&[pair(free, twins)]);
    let mut right = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: top });
    right.seed_contract_worklist([r.0]);

    let mut f = literal(&vtree, 1).unwrap();
    f.minimize().unwrap();
    // Safety: both operands are on `vtree` and unweighted, and each holds
    // one pair at the root whose other side is true.
    unsafe { f.splice_subtree_unchecked(&eng, right, root) };
    f.minimize().unwrap();
    assert_eq!(f.levels[rl.idx()].slot_count(), 1, "the twins merge");
    assert_canonical(&f);
    // x1, x5 = x6, x7 and x8, with x2, x3 and x4 free.
    assert_eq!(f.model_count().unwrap(), 16u32.into());
}
