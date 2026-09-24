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
    let mut builder = Tdd::builder(&eng, &tree).unwrap();
    let first = builder.push(&eng, root, &pair).unwrap();
    assert_eq!(builder.intern(&eng, root, &pair).unwrap(), first);
    let second = builder.push(&eng, root, &other).unwrap();
    assert_eq!(builder.intern(&eng, root, &other).unwrap(), second);
    assert_ne!(builder.push(&eng, root, &pair).unwrap(), first);
    assert_eq!(builder.intern(&eng, root, &pair).unwrap(), first);
    let source = Tdd::clause(&tree, [1]).unwrap();
    assert_canonical(&source);
    builder.replace_level(&eng, root, source.level_view(root)).unwrap();
    builder.replace_level(&eng, root, source.level_view(root)).unwrap();
    assert_eq!(builder.level(root).slot_count(), source.level(root).slot_count());
    let index = builder.intern(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    assert_eq!(index, source.output().local);
    let result = builder.finish(TddNodeId { vtree: root, local: index }).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 2u32.into());
}

/// Assemble `x1 ∧ ¬x2` over `tree`, returning the output node for
/// [`TddBuilder::finish`].
fn fill(
    eng: &Engine,
    b: &mut crate::diagram::TddBuilder,
    tree: &Arc<Vtree>,
) -> Result<TddNodeId, crate::OperationError> {
    let root = tree.root();
    let (left, right) = tree.children(root);
    b.reserve(eng, left, 4, 4)?;
    let pair = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let l = b.push(eng, left, &pair)?;
    let again = b.intern(eng, left, &pair)?;
    let r = b.push(eng, right, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)])?;
    let out = b.intern(eng, root, &[ChildPair::new(l, r)])?;
    assert_eq!(again, l, "interning an existing pair list appends nothing");
    Ok(TddNodeId { vtree: root, local: out })
}

#[test]
fn every_refusal_point_answers_over_budget_and_returns_the_buffers() {
    let tree = Arc::new(Vtree::balanced(4));
    let mut refused = 0;
    for cut in 0..10u32 {
        let eng = Engine::new();
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        eng.limits().refuse_nth_reserve(cut);
        match fill(&eng, &mut b, &tree) {
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
fn a_refused_level_take_leaves_the_builder_over_budget() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    eng.limits().refuse_nth_reserve(0);
    assert_eq!(Tdd::builder(&eng, &tree).unwrap_err(), crate::OperationError::OverBudget);
    eng.limits().grant_every_reserve();
    assert!(Tdd::builder(&eng, &tree).is_ok());
}

#[test]
fn a_full_level_answers_index_overflow() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = tree.root();
    let mut b = Tdd::builder(&eng, &tree).unwrap();
    eng.limits().pin_level_width_cap(Some(1));
    let first = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let second = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let one = b.push(&eng, root, &first).unwrap();
    // Interning a pair list the level already holds reads the index instead of
    // appending, so the cap does not apply to it.
    assert_eq!(b.intern(&eng, root, &first).unwrap(), one);
    assert_eq!(b.push(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    assert_eq!(b.intern(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    let f = b.finish(TddNodeId { vtree: root, local: one }).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 1u32.into());
}

/// A node survives a refused index update; retry must find it rather than append a duplicate.
#[test]
fn interning_recovers_after_an_index_update_is_refused() {
    use crate::diagram::NodeIdx;
    use crate::OperationError;

    for multiple_pairs in [false, true] {
        let vtree = Arc::new(Vtree::balanced(2));
        let root = vtree.root();
        let eng = Engine::new();
        let mut builder = Tdd::builder(&eng, &vtree).unwrap();
        let mut first = vec![ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)];
        let mut second = vec![ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)];
        if multiple_pairs {
            first.push(ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX));
            second.push(ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX));
        }
        let original = builder.intern(&eng, root, &first).unwrap();
        builder.reserve(&eng, root, 1, second.len()).unwrap();
        // The append fits the reservation, so the first reserve the push
        // takes is its hash-table update: refuse that one.
        eng.limits().refuse_nth_reserve(0);
        assert_eq!(builder.push(&eng, root, &second), Err(OperationError::OverBudget));
        eng.limits().grant_every_reserve();
        assert_eq!(builder.level(root).slot_count(), 2);
        // Rebuilding the discarded index may also be refused and retried.
        eng.limits().refuse_nth_reserve(0);
        assert_eq!(builder.intern(&eng, root, &second), Err(OperationError::OverBudget));
        eng.limits().grant_every_reserve();
        let recovered = builder.intern(&eng, root, &second).unwrap();
        assert_eq!(recovered, NodeIdx(1));
        assert_eq!(builder.intern(&eng, root, &first).unwrap(), original);
        assert_eq!(builder.level(root).slot_count(), 2);

        // The unfinished level has an unused node and may have uncontracted twins.
        let mut result = builder.finish(TddNodeId { vtree: root, local: recovered }).unwrap();
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), if multiple_pairs { 2u32.into() } else { 1u32.into() });
    }
}
