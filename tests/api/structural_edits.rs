//! Storage edits for callers that establish diagram invariants themselves.

use std::sync::Arc;
use tididi::{and, literal, Engine, Tdd, Vtree};
use tididi::diagram::{ChildPair, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
use tididi::test_helpers::assert_canonical;

#[test]
fn unchecked_builder_matches_checked_construction() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let build = || {
        let mut builder = Tdd::builder(&eng, &vtree).unwrap();
        let local = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
        (builder, TddNodeId { vtree: root, local })
    };
    let (builder, output) = build();
    let checked = builder.finish(output).unwrap();
    let (builder, output) = build();
    // Safety: one root pair names valid implicit literal nodes at the two leaves.
    let unchecked = unsafe { builder.finish_unchecked(output) };
    assert_eq!(unchecked.model_count().unwrap(), checked.model_count().unwrap());
    assert_canonical(&checked);
    assert_canonical(&unchecked);
}

#[test]
fn reseating_identical_topology_preserves_the_diagram() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut diagram = Tdd::clause(&vtree, [1, -3]).unwrap();
    let count = diagram.model_count().unwrap();
    let replacement = Arc::new((*vtree).clone());
    assert!(!Arc::ptr_eq(&vtree, &replacement));
    // Safety: cloning preserves every index, child relation, and variable label.
    unsafe { diagram.reseat_vtree_unchecked(&replacement); }
    assert!(Arc::ptr_eq(diagram.vtree(), &replacement));
    assert_eq!(diagram.model_count().unwrap(), count);
    assert_canonical(&diagram);
}

#[test]
fn independent_subtree_splice_matches_conjunction() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut left = literal(&vtree, 1).unwrap();
    let right = literal(&vtree, -4).unwrap();
    let mut expected = and(left.clone(), right.clone()).unwrap();
    expected.minimize().unwrap();
    // Safety: the operands constrain opposite root children on the same vtree,
    // with one wrapper pair apiece, true free sides, and no weights.
    unsafe { left.splice_subtree_unchecked(&eng, right, vtree.root()); }
    assert_eq!(left.model_count().unwrap(), expected.model_count().unwrap());
    assert_eq!(left.model_count().unwrap(), 4u32.into());
    assert_canonical(&left);
    assert_canonical(&expected);
}

#[test]
fn summing_out_a_root_of_several_nodes_follows_the_output() {
    // Three root nodes with 2, 2 and 3 models, the output the second: the
    // first two values merge when the root is summed out.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let second = b.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: second }).unwrap();
    assert_eq!(f.model_count().unwrap(), 2u32.into());
    f.marginalize_levels(&[root]).unwrap();
    assert_eq!(f.model_count().unwrap(), 2u32.into());
}
