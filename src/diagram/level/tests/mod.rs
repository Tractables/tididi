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
            assert_eq!(from_slice.pairs_vec(a.idx()), from_iter.pairs_vec(b.idx()));
        }
        assert_eq!(from_slice.nodes.stored(), from_iter.nodes.stored());
        assert_eq!(from_slice.charged_bytes(), from_iter.charged_bytes());
        assert_eq!(eng.limits().meters().in_flight_bytes, other.limits().meters().in_flight_bytes);
    }
}

/// A stored level copied over a level with room keeps that level's
/// allocations and charges nothing, and reads as the clone does.
#[test]
fn a_copy_over_a_level_with_room_reuses_its_arenas() {
    let eng = crate::Engine::new();
    let pair = |l: u32, r: u32| ChildPair::new(crate::diagram::NodeIdx(l), crate::diagram::NodeIdx(r));
    let mut from = TddLevel::new();
    from.push_node(eng.limits(), &[pair(0, 1)]).unwrap();
    from.push_node(eng.limits(), &[pair(0, 2), pair(1, 0), pair(2, 2)]).unwrap();
    from.push_node(eng.limits(), &[pair(1, 1), pair(2, 0)]).unwrap();
    let mut out = TddLevel::new();
    for k in 0..8 {
        out.push_node(eng.limits(), &[pair(k, 0), pair(k, 1), pair(k, 2)]).unwrap();
    }
    out.clear();
    let (nodes, pairs) = (out.nodes.stored().as_ptr(), out.pairs.stored().unwrap().as_ptr());
    let charged = eng.limits().meters().in_flight_bytes;
    from.try_copy_over(eng.limits(), &mut out).unwrap();
    assert_eq!(eng.limits().meters().in_flight_bytes, charged);
    assert_eq!((out.nodes.stored().as_ptr(), out.pairs.stored().unwrap().as_ptr()), (nodes, pairs));
    let mut clone = TddLevel::new();
    from.try_clone_over(eng.limits(), 0, 0, &mut clone).unwrap();
    assert_eq!(out.node_count(), clone.node_count());
    for i in 0..clone.node_count() {
        assert_eq!(out.pairs_vec(i), clone.pairs_vec(i));
    }
    assert_eq!(out.live_pairs(), clone.live_pairs());
    assert_eq!(out.nodes.stored(), clone.nodes.stored());
    assert_eq!(out.pairs.stored(), clone.pairs.stored());
}
