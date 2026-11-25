//! Storage edits for callers that establish diagram invariants themselves.

use tididi::vtree::Home;
use tididi::{Engine, Tdd, Vtree};
use tididi::diagram::{ChildPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
use tididi::test_helpers::assert_canonical;

#[test]
fn unchecked_builder_matches_checked_construction() {
    let eng = Engine::new();
    let vtree = Home::new(Vtree::balanced(2));
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
fn summing_out_a_root_of_several_nodes_follows_the_output() {
    // Three root nodes with 1, 1 and 2 models, the output the second: the
    // first two values merge when the root is summed out.
    let eng = Engine::new();
    let vtree = std::sync::Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let second = b.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: second }).unwrap();
    assert_eq!(f.model_count().unwrap(), 1u32.into());
    f.marginalize_levels(&[root]).unwrap();
    assert_eq!(f.model_count().unwrap(), 1u32.into());
}
