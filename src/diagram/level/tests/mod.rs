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
