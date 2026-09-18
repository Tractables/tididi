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

/// Assemble `x1 ∧ ¬x2` over `tree` through the fallible methods, returning the
/// output node for [`TddBuilder::finish`].
fn fill_through_fallible(
    eng: &Engine,
    b: &mut crate::diagram::TddBuilder,
    tree: &Arc<Vtree>,
) -> Result<TddNodeId, crate::OperationError> {
    let root = tree.root();
    let (left, right) = tree.children(root);
    b.try_reserve(eng, left, 4, 4)?;
    let pair = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let l = b.try_push(eng, left, &pair)?;
    let again = b.try_intern(eng, left, &pair)?;
    let r = b.try_push(eng, right, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)])?;
    let out = b.try_intern(eng, root, &[ChildPair::new(l, r)])?;
    assert_eq!(again, l, "interning an existing pair list appends nothing");
    Ok(TddNodeId { vtree: root, local: out })
}

#[test]
fn the_fallible_methods_build_what_the_infallible_ones_do() {
    let tree = Arc::new(Vtree::balanced(4));
    let root = tree.root();
    let (left, right) = tree.children(root);
    let eng = Engine::new();

    let mut b = Tdd::builder(&eng, &tree);
    let l = b.push(left, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]);
    let r = b.intern(right, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)]);
    let o = b.intern(root, &[ChildPair::new(l, r)]);
    let want = b.finish(TddNodeId { vtree: root, local: o }).unwrap();

    let mut b = Tdd::try_builder(&eng, &tree).unwrap();
    let output = fill_through_fallible(&eng, &mut b, &tree).unwrap();
    let got = b.finish(output).unwrap();

    assert_canonical(&want);
    assert_canonical(&got);
    crate::test_helpers::assert_same_shape(&want, &got, "fallible against infallible");
    assert_eq!(got.model_count().unwrap(), 4u32.into());
}

#[test]
fn every_refusal_point_answers_over_budget_and_returns_the_buffers() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut refused = 0;
    for cut in 0..10u32 {
        let eng = Engine::new();
        let mut b = Tdd::try_builder(&eng, &tree).unwrap();
        eng.limits().refuse_nth_reserve(cut);
        match fill_through_fallible(&eng, &mut b, &tree) {
            Ok(output) => {
                eng.limits().grant_every_reserve();
                let f = b.finish(output).unwrap();
                assert_canonical(&f);
                assert_eq!(f.model_count().unwrap(), 4u32.into());
            }
            Err(e) => {
                assert_eq!(e, crate::OperationError::OverBudget, "cut {cut}");
                eng.limits().grant_every_reserve();
                b.abandon(&eng);
                assert_eq!(eng.levels().occupancy(), 1, "cut {cut} left a buffer outside the pool");
                refused += 1;
            }
        }
    }
    assert!(refused > 0, "no reservation was refused across the sweep");
}

#[test]
fn a_refused_level_take_leaves_try_builder_over_budget() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    eng.limits().refuse_nth_reserve(0);
    assert_eq!(Tdd::try_builder(&eng, &tree).unwrap_err(), crate::OperationError::OverBudget);
    eng.limits().grant_every_reserve();
    assert!(Tdd::try_builder(&eng, &tree).is_ok());
}

#[test]
fn a_full_level_answers_index_overflow() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = tree.root();
    let mut b = Tdd::try_builder(&eng, &tree).unwrap();
    eng.limits().pin_level_width_cap(Some(1));
    let first = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let second = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let one = b.try_push(&eng, root, &first).unwrap();
    // Interning a pair list the level already holds reads the index instead of
    // appending, so the cap does not apply to it.
    assert_eq!(b.try_intern(&eng, root, &first).unwrap(), one);
    assert_eq!(b.try_push(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    assert_eq!(b.try_intern(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    let f = b.finish(TddNodeId { vtree: root, local: one }).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 1u32.into());
}
