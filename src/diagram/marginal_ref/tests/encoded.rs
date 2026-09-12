use super::*;
use crate::diagram::{ChildPair, EncodedChildRef, EncodedNode};

#[test]
fn encoded_references_preserve_pair_layout() {
    assert_eq!(size_of::<EncodedChildRef>(), size_of::<u32>());
    assert_eq!(align_of::<EncodedChildRef>(), align_of::<u32>());
    assert_eq!(size_of::<ChildPair>(), size_of::<EncodedNode>());
    assert_eq!(align_of::<ChildPair>(), align_of::<EncodedNode>());
    assert_eq!(std::mem::offset_of!(ChildPair, left), 0);
    assert_eq!(std::mem::offset_of!(ChildPair, right), 4);
}

#[test]
fn child_level_determines_reference_meaning() {
    for value in [ValueRef::Inline(0), ValueRef::Inline(7), ValueRef::Inline(MARGINAL_INLINE_MAX), ValueRef::Slot(7)] {
        let side = value.side();
        assert_eq!(EncodedChildRef::from_raw(side.raw()), side);
        assert_eq!(ChildDecoder::marginal().child(side), ChildRef::Value(value));
        assert_eq!(ChildDecoder::structural().child(side), ChildRef::Node(NodeIdx(side.raw())));
    }
}

#[test]
fn inline_and_arena_pairs_preserve_tagged_references() {
    let tagged = ChildPair::new(ValueRef::Inline(7).side(), ValueRef::Slot(9).side());
    let other = ChildPair::new(NodeIdx(1), NodeIdx(2));
    let mut level = TddLevel::new();
    let inline = level.try_push_internal_node(&[tagged]).unwrap();
    let arena = level.try_push_internal_node(&[tagged, other]).unwrap();
    for (node, expected) in [(inline, &[tagged][..]), (arena, &[tagged, other][..])] {
        assert_eq!(level.pairs_of_idx(node.idx()), expected);
        assert_eq!(level.pairs_iter_of_idx(node.idx()).collect::<Vec<_>>(), expected);
    }
}
