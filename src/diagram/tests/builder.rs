use crate::diagram::{ChildPair, TddNodeId, POS_LEAF_IDX, NEG_LEAF_IDX, ONE_LEAF_IDX};
use crate::{Engine, Tdd};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use std::sync::Arc;

#[test]
fn interning_indexes_prior_pushes_and_copy_replaces_the_entire_level() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = tree.root();
    let pair = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let other = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let mut builder = Tdd::builder(&eng, &tree);
    let first = builder.push(root, &pair);
    assert_eq!(builder.intern(root, &pair), first);
    let second = builder.push(root, &other);
    assert_eq!(builder.intern(root, &other), second);
    assert_ne!(builder.push(root, &pair), first);
    assert_eq!(builder.intern(root, &pair), first);
    let source = Tdd::clause(&tree, [1]).unwrap();
    assert_canonical(&source);
    builder.replace_level(root, source.level_view(root)).unwrap();
    builder.replace_level(root, source.level_view(root)).unwrap();
    assert_eq!(builder.level(root).slot_count(), source.level(root).slot_count());
    let index = builder.intern(root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]);
    assert_eq!(index, source.output().local);
    let result = builder.finish(TddNodeId { vtree: root, local: index }).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 2u32.into());
}

#[test]
fn finish_rejects_a_deleted_output() {
    use crate::diagram::{EncodedNode, LevelView, TddBuildError};
    let tree = Arc::new(Vtree::balanced(2));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let root = tree.root();
    let mut level = source.level(root).clone();
    level.nodes[0] = EncodedNode::tombstone();
    level.n_tombstones = 1;
    let mut builder = Tdd::builder(&Engine::new(), &tree);
    builder.replace_level(root, LevelView::unweighted(&level).unwrap()).unwrap();
    assert_eq!(builder.finish(source.output()).unwrap_err(), TddBuildError::BadOutput(source.output()));
}

#[test]
fn finish_rejects_a_reference_to_a_deleted_child() {
    use crate::diagram::{EncodedNode, LevelView, NodeIdx, TddBuildError};
    let tree = Arc::new(Vtree::balanced(4));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let root = tree.root();
    let (left, right) = tree.children(root);
    for child in [left, right] {
        let mut builder = Tdd::builder(&Engine::new(), &tree);
        for t in tree.bottomup() { builder.replace_level(t, source.level_view(t)).unwrap(); }
        let mut level = source.level(child).clone();
        level.nodes[0] = EncodedNode::tombstone();
        level.n_tombstones = 1;
        builder.replace_level(child, LevelView::unweighted(&level).unwrap()).unwrap();
        assert_eq!(builder.finish(source.output()).unwrap_err(), TddBuildError::DeadChild {
            level: root, node: NodeIdx(0), child: TddNodeId { vtree: child, local: NodeIdx(0) },
        });
    }
}

#[test]
fn finish_allows_unused_deleted_slots() {
    use crate::diagram::{EncodedNode, LevelView};
    let tree = Arc::new(Vtree::balanced(2));
    let source = Tdd::one(&tree);
    assert_canonical(&source);
    let root = tree.root();
    let mut level = source.level(root).clone();
    level.nodes.push(EncodedNode::tombstone());
    level.n_tombstones = 1;
    let mut builder = Tdd::builder(&Engine::new(), &tree);
    builder.replace_level(root, LevelView::unweighted(&level).unwrap()).unwrap();
    let mut result = builder.finish(source.output()).unwrap();
    assert_eq!(Engine::new().is_sat(&result), Ok(true));
    result.minimize().unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 4u32.into());
}
