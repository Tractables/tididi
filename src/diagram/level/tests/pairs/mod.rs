mod sorting_networks;

/// The uneven scan reads node counts a run at a time: it finds the first
/// node of another count wherever it falls against the runs, a hint that
/// still holds first, none on an even level, and reads a ranged node's
/// count off the side table.
#[test]
fn uneven_node_finds_the_first_node_of_another_count() {
    use super::{ChildPair, EncodedNode, TddLevel};
    use crate::diagram::PairRange;
    let inline = EncodedNode::inline(ChildPair::new(crate::diagram::NodeIdx(0), crate::diagram::NodeIdx(1)));
    let node = |k: u32| if k == 1 { inline } else { EncodedNode::multi_pair(0, k) };
    for k in [1u32, 2, 5] {
        for nodes in [1usize, 2, 16, 17, 63, 64, 65, 80, 81, 200] {
            let even = |level: &mut TddLevel| {
                let words = level.nodes.stored_mut();
                words.clear();
                words.extend((0..nodes).map(|_| node(k)));
            };
            let mut level = TddLevel::new();
            even(&mut level);
            assert_eq!(level.uneven_node(k as usize), None, "{nodes} nodes of {k}");
            for at in (1..nodes).filter(|i| [1, 2, 15, 16, 17, 63, 64, 79, 80, 81, 127, 128, 199].contains(i) || *i == nodes - 1) {
                even(&mut level);
                level.nodes.stored_mut()[at] = node(k + 1);
                level.uneven = 0;
                assert_eq!(level.uneven_node(k as usize), Some(at), "{nodes} nodes of {k}, node {at}");
                if at + 1 < nodes {
                    level.nodes.stored_mut()[nodes - 1] = node(k + 1);
                    level.uneven = (nodes - 1) as u16;
                    assert_eq!(level.uneven_node(k as usize), Some(nodes - 1), "the hint first");
                }
            }
        }
    }
    // A ranged node holds the count its side table entry gives.
    let mut level = TddLevel::new();
    level.ranges.push(PairRange { start: 0, len: 3 });
    level.ranges.push(PairRange { start: 0, len: 4 });
    level.nodes.stored_mut().extend((0..100).map(|_| EncodedNode::multi_pair(0, 3)));
    level.nodes.stored_mut()[70] = EncodedNode::multi_ranged(0);
    assert_eq!(level.uneven_node(3), None);
    level.nodes.stored_mut()[90] = EncodedNode::multi_ranged(1);
    assert_eq!(level.uneven_node(3), Some(90));
}

/// `try_pairs_read` reserves the room an implicit node's pairs take through
/// the limits, a refusal coming back as `OverBudget`, and reads a stored
/// node's pairs in place without asking for a reserve.
#[test]
fn try_pairs_read_reserves_only_what_an_implicit_level_generates() {
    use crate::test_helpers::{with_stored_copy, x_decision_diagram};
    let (implicit, v, _) = x_decision_diagram(32);
    let stored = with_stored_copy(&implicit, v);
    let (level, kept) = (&implicit.levels[v.idx()], &stored.levels[v.idx()]);
    let per_node = level.implicit().expect("the fixture's level is implicit").pairs_per_node();
    let eng = crate::Engine::new();
    let lim = eng.limits();
    let mut buf = Vec::new();
    lim.refuse_nth_reserve(0);
    assert_eq!(level.try_pairs_read(lim, 0, &mut buf).err(), Some(crate::OperationError::OverBudget));
    lim.refuse_nth_reserve(0);
    assert_eq!(kept.try_pairs_read(lim, 0, &mut buf).unwrap(), kept.pairs_read(0, &mut Vec::new()));
    assert!(lim.refusal_pending(), "a stored node reserves nothing");
    lim.grant_every_reserve();
    let _op = lim.begin_operation();
    for i in 0..level.node_count() {
        let read = level.try_pairs_read(lim, i, &mut buf).unwrap().to_vec();
        assert_eq!(read, kept.pairs_read(i, &mut Vec::new()), "node {i}");
    }
    assert_eq!(buf.capacity(), per_node);
    let bytes = (per_node * std::mem::size_of::<super::ChildPair>()) as u64;
    assert_eq!(lim.meters().in_flight_bytes, bytes);
}

/// Reading a node past a stored level's end names the node.
#[test]
#[should_panic(expected = "node 999 of a level of")]
fn pairs_read_past_a_stored_level_names_the_node() {
    use crate::test_helpers::{with_stored_copy, x_decision_diagram};
    let (implicit, v, _) = x_decision_diagram(32);
    let stored = with_stored_copy(&implicit, v);
    stored.levels[v.idx()].pairs_read(999, &mut Vec::new());
}

/// As [`pairs_read_past_a_stored_level_names_the_node`], through the limits.
#[test]
#[should_panic(expected = "node 999 of a level of")]
fn try_pairs_read_past_a_stored_level_names_the_node() {
    use crate::test_helpers::{with_stored_copy, x_decision_diagram};
    let (implicit, v, _) = x_decision_diagram(32);
    let stored = with_stored_copy(&implicit, v);
    let eng = crate::Engine::new();
    let _ = stored.levels[v.idx()].try_pairs_read(eng.limits(), 999, &mut Vec::new());
}
