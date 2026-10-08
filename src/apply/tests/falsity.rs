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
    let (diagram, v, _) = crate::test_helpers::x_decision_diagram(32);
    crate::test_helpers::assert_canonical(&diagram);
    let level = diagram.level(v);
    assert!(level.implicit().is_some());
    for i in 0..level.nodes().len() { assert!(!empty_node(level, i)); }
}
