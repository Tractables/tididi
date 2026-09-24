//! Stopping the encoding at each of its polls.

use super::soundness::assert_sound;
use super::*;

/// Whether node or slot `id` comes before `at` in the encoding's order.
fn before(f: &Tdd, id: TddNodeId, at: EncodePoint) -> bool {
    let (level, slot) = match at {
        EncodePoint::Level { level, .. } => (level, 0),
        EncodePoint::Node { level, slot } => (level, slot),
    };
    let rank = |t: VtreeIdx| f.vtree().internal_bottomup().position(|(u, _, _)| u == t).expect("an internal level");
    (rank(id.vtree), id.local.idx()) < (rank(level), slot)
}

/// Every poll a full encoding of `f` makes: before each internal level, and
/// before every 1,024th stored node of a structural one.
fn expected_polls(f: &Tdd) -> Vec<EncodePoint> {
    f.vtree().internal_bottomup().enumerate().flat_map(|(complete, (level, _, _))| {
        std::iter::once(EncodePoint::Level { level, complete })
            .chain((0..f.levels[level.idx()].nodes().len()).step_by(1024).map(move |slot| EncodePoint::Node { level, slot }))
    }).collect()
}

/// Stop an encoding of `f` at each of its polls in turn and compare it with
/// the full encoding: the same calls up to the stop and none after, the
/// literals of the nodes before the stop and no others, and the counts of
/// both. With `sound`, check the kept nodes against evaluation too.
fn assert_cuts(f: &Tdd, sound: bool) {
    let (full, whole) = encode(f, None);
    let polls = whole.polls();
    assert_eq!(polls, expected_polls(f));
    let reachable = f.reachable_nodes();
    for (k, &at) in polls.iter().enumerate() {
        let (encoding, sink) = encode(f, Some(k));
        assert_eq!(sink.calls[..], whole.calls[..sink.calls.len()], "stop at {at:?}");
        assert_eq!(sink.calls.last(), Some(&Call::Poll(at)));
        assert_eq!(encoding.stop(), Some(at));
        let (mut kept, mut skipped) = (0, 0);
        for id in internal_nodes(f) {
            let encoded = reachable[id.vtree.idx()][id.local.idx()] && f.levels[id.vtree.idx()].nodes()[id.local.idx()].is_internal();
            let keep = encoded && before(f, id, at);
            assert_eq!(encoding.literal(id), if keep { full.literal(id) } else { None }, "{id:?} with a stop at {at:?}");
            kept += u64::from(keep);
            skipped += u64::from(encoded && !keep);
        }
        // The positive summed-out values at internal levels from the stop on
        // are cleared and counted too.
        for (t, _, _) in f.vtree().internal_bottomup() {
            let Some(counts) = f.levels[t.idx()].marginal_counts() else { continue };
            for (s, &count) in counts.iter().enumerate() {
                let id = TddNodeId { vtree: t, local: NodeIdx(s as u32) };
                skipped += u64::from(reachable[t.idx()][s] && count != 0 && !before(f, id, at));
            }
        }
        assert_eq!((encoding.encoded_nodes(), encoding.skipped()), (kept, skipped), "stop at {at:?}");
        if kept == 0 {
            // Nothing is encoded, and the sink holds at most the true
            // variable's clause.
            assert!(sink.calls.iter().filter(|call| matches!(call, Call::Clause(_))).count() <= 1);
        }
        if sound { assert_sound(f, &encoding, &sink.store); }
    }
}

#[test]
fn every_poll_cuts_the_hand_built_diagrams() {
    assert_cuts(&chain().0, true);
    assert_cuts(&super::golden::inline_marginal().0, true);
    for second in [crate::diagram::LeafLabel::Neg, crate::diagram::LeafLabel::Pos] {
        assert_cuts(&super::golden::marginal_boundary(second).0, true);
    }
}

#[test]
fn every_poll_cuts_seeded_diagrams() {
    for f in random_diagrams(0x7a11, 12, 3..6) { assert_cuts(&f, true); }
    for (f, _) in marginal_diagrams(0x7a12, 12, 4..6) { assert_cuts(&f, true); }
}

#[test]
fn a_level_wider_than_the_poll_stride_is_polled_inside() {
    // x_i ↔ x_{i+11} on the linear order: the level whose left leaf is x12
    // holds one node per assignment of x1 to x11.
    let vtree = Arc::new(Vtree::linear(22));
    let clauses: Vec<Vec<i32>> = (1..=11).flat_map(|i| [vec![-i, i + 11], vec![i, -(i + 11)]]).collect();
    let f = compile_clauses(&vtree, &clauses);
    assert_canonical(&f);
    let wide = vtree.leaf_of(VarId(12)).and_then(|leaf| vtree.node(leaf).parent()).unwrap();
    assert_eq!(f.levels[wide.idx()].nodes().len(), 2048);
    assert!(expected_polls(&f).contains(&EncodePoint::Node { level: wide, slot: 1024 }));
    assert_cuts(&f, false);
}

#[test]
fn a_stop_before_the_first_node_leaves_only_the_true_variable() {
    // Bonsai's prefix fixture: (x1 ∨ x3) ∧ (x2 ∨ ¬x3) on the reversed linear
    // vtree, with its two internal levels.
    let vtree = Arc::new(Vtree::reverse_linear(3));
    let f = compile_clauses(&vtree, &[vec![1, 3], vec![2, -3]]);
    assert_canonical(&f);
    let [lower, root]: [VtreeIdx; 2] = f.vtree().internal_bottomup().map(|(t, _, _)| t).collect::<Vec<_>>().try_into().unwrap();
    let nodes = |t: VtreeIdx| f.levels[t.idx()].nodes().len() as u64;

    // Stopped before the first level: nothing is encoded, and the caller
    // retires the activation literal and gives up.
    let (encoding, sink) = encode(&f, Some(0));
    assert_eq!(encoding.stop(), Some(EncodePoint::Level { level: lower, complete: 0 }));
    assert_eq!((encoding.encoded_nodes(), encoding.skipped()), (0, nodes(lower) + nodes(root)));
    // Variables 1 to 3 are the diagram's, 4 is the activation literal and 5
    // the true variable.
    assert_eq!(sink.store.clauses(), [vec![5, -4]]);
    let mut store = sink.store;
    store.retire();
    assert_eq!(store.clauses().last(), Some(&vec![-4]));

    // Stopped before the second: the lower level is kept.
    let (encoding, _) = encode(&f, Some(2));
    assert_eq!(encoding.stop(), Some(EncodePoint::Level { level: root, complete: 1 }));
    assert_eq!((encoding.encoded_nodes(), encoding.skipped()), (nodes(lower), nodes(root)));
}
