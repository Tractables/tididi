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

/// Every way a node is pushed refuses the node past the level's width cap
/// with `IndexOverflow`, here pinned at three nodes, and leaves the level
/// as it was: a slice, an owned buffer, a range of the arena
/// and a reservation of room for nodes written in place.
#[test]
fn a_node_past_the_width_cap_is_refused() {
    use crate::OperationError::IndexOverflow;
    use crate::diagram::{NEG_LEAF_IDX, POS_LEAF_IDX};
    let eng = crate::Engine::new();
    let lim = eng.limits();
    let (a, b) = (ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX));
    let mut level = TddLevel::new();
    lim.pin_level_width_cap(Some(3));
    level.push_node(lim, &[a]).unwrap();
    level.push_node(lim, &[a, b]).unwrap();
    level.push_node(lim, &[b, a]).unwrap();
    let (nodes, pairs) = (level.node_count(), level.pairs.len());
    assert_eq!(level.push_node(lim, &[b]), Err(IndexOverflow));
    assert_eq!(level.push_node(lim, &[a, b]), Err(IndexOverflow));
    assert_eq!(level.try_push_multi_by_range(lim.level_width_cap(), 0, 2), Err(IndexOverflow));
    assert_eq!(crate::diagram::reserve_node_room(lim, level.nodes.stored_mut(), 1), Err(IndexOverflow));
    assert_eq!((level.node_count(), level.pairs.len()), (nodes, pairs));
    crate::diagram::reserve_node_room(lim, level.nodes.stored_mut(), 0).unwrap();
    lim.pin_level_width_cap(None);
    level.push_node(lim, &[b]).unwrap();
    level.try_push_multi_by_range(lim.level_width_cap(), 0, 2).unwrap();
    assert_eq!(level.node_count(), 5);
}
