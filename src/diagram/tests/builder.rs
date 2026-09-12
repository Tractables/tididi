use crate::diagram::{InputPair, TddNodeId, POS_LEAF_IDX, NEG_LEAF_IDX, ONE_LEAF_IDX};
use crate::{Engine, Tdd};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use std::sync::Arc;

#[test]
fn interning_indexes_prior_pushes_and_copy_replaces_the_entire_level() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = tree.root();
    let pair = [InputPair { left: POS_LEAF_IDX, right: NEG_LEAF_IDX }];
    let other = [InputPair { left: NEG_LEAF_IDX, right: POS_LEAF_IDX }];
    let mut builder = Tdd::build(&eng, &tree);
    let first = builder.push(root, &pair);
    assert_eq!(builder.intern(root, &pair), first);
    let second = builder.push(root, &other);
    assert_eq!(builder.intern(root, &other), second);
    assert_ne!(builder.push(root, &pair), first);
    assert_eq!(builder.intern(root, &pair), first);
    let source = Tdd::clause(&tree, [1]);
    assert_canonical(&source);
    builder.copy_level(root, source.level_view(root)).unwrap();
    builder.copy_level(root, source.level_view(root)).unwrap();
    assert_eq!(builder.level(root).width(), source.level(root).width());
    let index = builder.intern(root, &[InputPair { left: POS_LEAF_IDX, right: ONE_LEAF_IDX }]);
    assert_eq!(index, source.output().local);
    let result = builder.finish(TddNodeId { vtree: root, local: index }).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count(), 2u32.into());
}
