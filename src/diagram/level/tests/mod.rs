use super::*;

mod support;

/// Growing a pair list charges each arena's capacity increase exactly once.
#[test]
fn appended_pairs_charge_only_new_capacity() {
    use crate::limits::Charged;
    let eng = crate::Engine::new();
    let mut level = TddLevel::new();
    let first = ChildPair::new(crate::diagram::POS_LEAF_IDX, crate::diagram::NEG_LEAF_IDX);
    let other = ChildPair::new(crate::diagram::NEG_LEAF_IDX, crate::diagram::POS_LEAF_IDX);
    let node = level.push_node(eng.limits(), &[first]).unwrap();
    for i in 0..12 {
        if i % 3 == 0 { level.push_node(eng.limits(), &[first, other]).unwrap(); }
        let before = level.charged_bytes();
        let charged = eng.limits().meters().in_flight_bytes;
        level.push_pair_onto_node(&eng, node.idx(), other).unwrap();
        assert_eq!(eng.limits().meters().in_flight_bytes - charged, level.charged_bytes() - before);
    }
}

/// A node whose pairs an iterator yields is stored as the same pairs in a
/// slice would be: inline for one pair, in the arena for more, with the
/// same charge.
#[test]
fn a_node_pushed_from_an_iterator_is_the_node_pushed_from_a_slice() {
    use crate::limits::Charged;
    let pairs: Vec<ChildPair> = (0..9u32)
        .map(|k| ChildPair::new(crate::diagram::NodeIdx(k / 3), crate::diagram::NodeIdx(k % 3)))
        .collect();
    for len in [1, 2, 9] {
        let (eng, other) = (crate::Engine::new(), crate::Engine::new());
        let (mut from_slice, mut from_iter) = (TddLevel::new(), TddLevel::new());
        for _ in 0..2 {
            let a = from_slice.push_node(eng.limits(), &pairs[..len]).unwrap();
            let b = from_iter.push_node_from(other.limits(), len, pairs[..len].iter().copied()).unwrap();
            assert_eq!(a, b);
            assert_eq!(from_slice.pairs_of_idx(a.idx()), from_iter.pairs_of_idx(b.idx()));
        }
        assert_eq!(from_slice.nodes(), from_iter.nodes());
        assert_eq!(from_slice.charged_bytes(), from_iter.charged_bytes());
        assert_eq!(eng.limits().meters().in_flight_bytes, other.limits().meters().in_flight_bytes);
    }
}
