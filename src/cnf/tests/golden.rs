//! The exact calls the equivalence encoding makes on hand-built diagrams.
//!
//! The order of the variables and clauses is part of the scheme's contract:
//! a caller that numbers variables as the store does gets the same solver
//! input from one release to the next. These transcripts pin it: the true
//! variable at the first leaf that is not summed out, every node's variable
//! before any pair's, the single-pair rule on the stored pair count, and the
//! polls.

use super::*;
use crate::diagram::{ChildPair, LeafLabel, NEG_LEAF_IDX, POS_LEAF_IDX};

/// The calls as text, naming each polled level by `names`.
fn transcript(calls: &[Call], names: &[(VtreeIdx, &str)]) -> Vec<String> {
    let name = |t: VtreeIdx| names.iter().find(|(u, _)| *u == t).map(|(_, n)| *n).expect("every polled level is named");
    calls.iter().map(|call| match call {
        Call::Fresh(var) => format!("fresh {var}"),
        Call::Leaf(var) => format!("leaf x{var}"),
        Call::Clause(body) => format!("clause {}", body.iter().map(i32::to_string).collect::<Vec<_>>().join(" ")),
        Call::Poll(EncodePoint::Level { level, complete }) => format!("level {} after {complete}", name(*level)),
        Call::Poll(EncodePoint::Node { level, slot }) => format!("node {} {slot}", name(*level)),
    }).collect()
}

#[test]
fn the_chain_is_encoded_bottom_up_with_node_variables_first() {
    let (f, [b, a, root]) = chain();
    let (encoding, sink) = encode(&f, None);
    // Variables 1 to 4 are the diagram's and 5 is the activation literal.
    assert_eq!(transcript(&sink.calls, &[(b, "b"), (a, "a"), (root, "root")]), [
        // The true variable at the first leaf in bottom-up order, then each
        // leaf's literal in that order.
        "fresh 6", "clause 6", "leaf x3", "leaf x4", "leaf x2", "leaf x1",
        // The reachable nodes: three of four on b, two of three on a, and the
        // output.
        "fresh 7", "fresh 8", "fresh 9", "fresh 10", "fresh 11", "fresh 12",
        "level b after 0", "node b 0",
        "clause -7 3", "clause -7 4", "clause 7 -3 -4",
        "clause -8 3", "clause -8 -4", "clause 8 -3 4",
        "clause -9 -3", "clause -9 4", "clause 9 3 -4",
        "level a after 1", "node a 0",
        "fresh 13", "clause -13 2", "clause -13 7", "clause 13 -2 -7",
        "fresh 14", "clause -14 -2", "clause -14 8", "clause 14 2 -8",
        "clause -10 13 14", "clause 10 -13", "clause 10 -14",
        "clause -11 -2", "clause -11 9", "clause 11 2 -9",
        "level root after 2", "node root 0",
        "fresh 15", "clause -15 1", "clause -15 10", "clause 15 -1 -10",
        "fresh 16", "clause -16 -1", "clause -16 11", "clause 16 1 -11",
        "clause -12 15 16", "clause 12 -15", "clause 12 -16",
    ]);
    // Every clause reached the store gated by the activation literal.
    assert!(sink.store.clauses().iter().all(|clause| clause.last() == Some(&-5)));
    let node = |t: VtreeIdx, i: u32| encoding.literal(TddNodeId { vtree: t, local: NodeIdx(i) });
    assert_eq!([node(b, 0), node(b, 1), node(b, 2), node(b, 3)], [Some(7), Some(8), Some(9), None]);
    assert_eq!([node(a, 0), node(a, 1), node(a, 2), node(root, 0)], [Some(10), Some(11), None, Some(12)]);
    let leaf = f.vtree().leaf_of(VarId(3)).unwrap();
    let labels = [LeafLabel::One, LeafLabel::Pos, LeafLabel::Neg].map(|label| encoding.literal(TddNodeId { vtree: leaf, local: NodeIdx(label as u32) }));
    assert_eq!(labels, [Some(6), Some(3), Some(-3)]);
    assert_eq!((encoding.encoded_nodes(), encoding.skipped(), encoding.stop()), (6, 0, None));
    assert!(encoding.erasure_certified());
}

#[test]
fn inline_counts_read_as_true_when_positive_and_dead_at_zero() {
    let (f, [v4, v5, root]) = inline_marginal();
    let (encoding, sink) = encode(&f, None);
    assert_eq!(transcript(&sink.calls, &[(v4, "v4"), (v5, "v5"), (root, "root")]), [
        // x2 is summed out: no literal is asked for it.
        "fresh 6", "clause 6", "leaf x1", "leaf x3", "leaf x4",
        "fresh 7", "fresh 8", "fresh 9", "fresh 10", "fresh 11",
        "level v4 after 0", "node v4 0",
        // 7 ↔ x1 ∧ ⊤, and the dead node's variable is false.
        "clause -7 1", "clause 7 -1",
        "clause -8",
        "level v5 after 1", "node v5 0",
        "clause -9 3", "clause -9 6", "clause 9 -3 -6",
        "clause -10 -3", "clause -10 6", "clause 10 3 -6",
        "level root after 2", "node root 0",
        "fresh 12", "clause -12 7", "clause -12 9", "clause 12 -7 -9",
        "fresh 13", "clause -13 8", "clause -13 10", "clause 13 -8 -10",
        "clause -11 12 13", "clause 11 -12", "clause 11 -13",
    ]);
    assert!(encoding.erasure_certified());
}

#[test]
fn the_single_pair_rule_counts_stored_pairs() {
    // One node over (x1, x2) with x2 summed out: `x1 · 5 ∨ ¬x1 · 0`. Only
    // the first pair can be true, but the node stores two, so the pair gets
    // a variable of its own.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    sum_out_leaf(&eng, &vtree, &mut builder, vtree.children(root).1);
    let out = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, inline_side(5)), ChildPair::new(NEG_LEAF_IDX, inline_side(0))]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: out }).unwrap();
    let (encoding, sink) = encode(&f, None);
    assert_eq!(transcript(&sink.calls, &[(root, "root")]), [
        "fresh 4", "clause 4", "leaf x1",
        "fresh 5",
        "level root after 0", "node root 0",
        "fresh 6", "clause -6 1", "clause 6 -1",
        "clause -5 6", "clause 5 -6",
    ]);
    assert_eq!(encoding.literal(f.output()), Some(5));
    assert!(encoding.erasure_certified());
}

#[test]
fn without_a_structural_leaf_there_is_no_true_variable() {
    // x1 ∨ x2 with both leaves summed out: one pair of two positive counts,
    // and the output's is the only variable.
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1, 2]).unwrap();
    let (left, right) = tree.children(tree.root());
    f.marginalize_levels(&[left, right]).unwrap();
    assert_marginal_canonical(&f);
    let (encoding, sink) = encode(&f, None);
    assert_eq!(transcript(&sink.calls, &[(tree.root(), "root")]), [
        "fresh 4",
        "level root after 0", "node root 0",
        "clause 4",
    ]);
    // A pair whose sides are both summed out reads as true, so the
    // certificate cannot hold.
    assert!(!encoding.erasure_certified());
}

#[test]
fn count_slots_read_as_true_and_the_certificate_follows_the_primes() {
    for (second, n2, certified) in [(LeafLabel::Neg, ["clause -9 -1", "clause 9 1"], true), (LeafLabel::Pos, ["clause -9 1", "clause 9 -1"], false)] {
        let (f, [m, v4, v5, root]) = marginal_boundary(second);
        let (encoding, sink) = encode(&f, None);
        let expected: Vec<&str> = [
            // The first leaf in bottom-up order is x2, below the summed-out
            // level but not summed out itself.
            "fresh 7", "clause 7", "leaf x2", "leaf x5", "leaf x1", "leaf x3", "leaf x4",
            "fresh 8", "fresh 9", "fresh 10", "fresh 11",
            // A summed-out level has no node and no node poll, but counts as
            // complete.
            "level m after 0",
            "level v4 after 1", "node v4 0",
            "clause -8 1", "clause 8 -1",
        ].into_iter().chain(n2).chain([
            "level v5 after 2", "node v5 0",
            "clause -10 3", "clause -10 7", "clause 10 -3 -7",
            "level root after 3", "node root 0",
            "fresh 12", "clause -12 8", "clause -12 10", "clause 12 -8 -10",
            "fresh 13", "clause -13 9", "clause -13 10", "clause 13 -9 -10",
            "clause -11 12 13", "clause 11 -12", "clause 11 -13",
        ]).collect();
        assert_eq!(transcript(&sink.calls, &[(m, "m"), (v4, "v4"), (v5, "v5"), (root, "root")]), expected, "{second:?}");
        // With a shared prime the two nodes both read as x1 once the slots
        // read as true, although their counts kept them apart.
        assert_eq!(encoding.erasure_certified(), certified, "{second:?}");
        for i in 0..2 { assert_eq!(encoding.literal(TddNodeId { vtree: m, local: NodeIdx(i) }), None); }
    }
}
