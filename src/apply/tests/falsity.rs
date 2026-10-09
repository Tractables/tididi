use super::*;

#[test]
fn emptiness_handles_inline_normal_and_wide_nodes() {
    let mut level = TddLevel::new();
    level.nodes.stored_mut().push(EncodedNode::inline(crate::test_helpers::pair(0, 1)));
    assert!(!empty_node(&level, 0));
    // Wide offsets exercise the side table without allocating a pair arena.
    for (start, len) in [(0, 0), (0, 2), (1usize << 31, 0), (1usize << 31, 2)] {
        let node = level.encode_multi(start, len);
        let i = level.nodes.stored().len();
        level.nodes.stored_mut().push(node);
        assert_eq!(empty_node(&level, i), len == 0);
    }
}

#[test]
fn implied_nodes_have_nonempty_pair_lists() {
    let mut level = TddLevel::new();
    for i in 0..32 {
        level.push_internal_node(&[
            crate::test_helpers::pair(0, 2 * i),
            crate::test_helpers::pair(1, 2 * i + 1),
        ]);
    }
    crate::test_helpers::describe(&mut level);
    assert!(level.implicit().is_some());
    for i in 0..level.nodes().len() { assert!(!empty_node(&level, i)); }
}
