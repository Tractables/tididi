//! The exact calls the probe makes on hand-built diagrams.
//!
//! The order of the solves, reads and clauses is part of the probe's
//! contract: a caller with the same solver gets the same answers from one
//! release to the next. These transcripts pin it: the enumeration's fresh
//! variable, its reads of every encoded node and its blocking clauses, the
//! walk's order by dead sides and counts, the propagation, the witness
//! skips, and the polls.

use super::*;
use crate::diagram::{ChildPair, NEG_LEAF_IDX, POS_LEAF_IDX};

/// The calls as text, naming each node by its level's name in `names` and
/// its index.
fn transcript(calls: &[Call], names: &[(VtreeIdx, &str)]) -> Vec<String> {
    let node = |id: TddNodeId| {
        let name = names.iter().find(|(t, _)| *t == id.vtree).map(|(_, n)| *n).expect("every probed level is named");
        format!("{name}{}", id.local.idx())
    };
    let status = |s: SolveStatus| match s { SolveStatus::Sat => "sat", SolveStatus::Unsat => "unsat", SolveStatus::Unknown => "unknown" };
    let list = |literals: &[i32]| literals.iter().map(i32::to_string).collect::<Vec<_>>().join(" ");
    calls.iter().map(|call| match call {
        Call::Fresh(var) => format!("fresh {var}"),
        Call::Add(clause) => format!("add {}", list(clause)),
        Call::Solve(assumptions, SolveCall::Round { round }, s) => format!("solve {} round {round}: {}", list(assumptions), status(*s)),
        Call::Solve(assumptions, SolveCall::Probe { node: id }, s) => format!("solve {} probe {}: {}", list(assumptions), node(*id), status(*s)),
        Call::Value(literal, value) => format!("value {literal} {value}"),
        Call::Poll(ProbePoint::Begin) => "begin".to_owned(),
        Call::Poll(ProbePoint::Round { round }) => format!("round {round}"),
        Call::Poll(ProbePoint::Solve { node: id }) => format!("next {}", node(*id)),
        Call::Poll(ProbePoint::Probed { node: id, status: s }) => format!("probed {} {}", node(*id), status(*s)),
    }).collect()
}

/// Counts, with the given fields set.
macro_rules! stats {
    ($($field:ident: $value:expr),* $(,)?) => {{
        let mut stats = ProbeStats::default();
        $(stats.$field = $value;)*
        stats
    }};
}

// On the chain, variables 1 to 4 are the diagram's, 5 is the activation
// literal and 6 the true variable. Level b's reachable nodes `x3 x4`,
// `x3 ¬x4` and `¬x3 x4` are 7, 8 and 9; level a's are 10 and 11; the output
// is 12, and the pairs take 13 to 16.

#[test]
fn the_enumeration_reads_every_node_and_kills_what_no_round_saw() {
    // x1 ¬x2 x3 ¬x4 is the only model: its slice is b1, a0 and the output.
    let (f, [b, a, root]) = chain();
    let (outcome, _, host) = probe(&f, &[vec![1], vec![-2], vec![3], vec![-4]], 4, Witness::NodeLiterals);
    assert_eq!(transcript(&host.calls, &[(b, "b"), (a, "a"), (root, "root")]), [
        "begin", "fresh 17",
        "round 0", "solve 5 17 round 0: sat",
        "value 7 false", "value 8 true", "value 9 false", "value 10 true", "value 11 false", "value 12 true",
        "add -17 -8 -10 -12",
        "round 1", "solve 5 17 round 1: unsat",
        "add -17",
        // The walk probes nothing: b0, b2 and a1 are dead, and the rest were
        // seen true.
    ]);
    assert_eq!(dead_nodes(&f, &outcome), [
        TddNodeId { vtree: b, local: NodeIdx(0) }, TddNodeId { vtree: b, local: NodeIdx(2) }, TddNodeId { vtree: a, local: NodeIdx(1) },
    ]);
    assert_eq!(outcome.stats(), stats! {
        rounds: 2, rounds_sat: 1, rounds_unsat: 1, dead: 3, enumeration_dead: 3,
        round_witnesses: 3, witness_skips: 3, value_reads: 6, reordered_levels: 2,
    });
}

#[test]
fn the_walk_reads_the_nodes_not_yet_seen_and_propagates_dead_sides() {
    let (f, [b, a, root]) = chain();
    let (outcome, _, host) = probe(&f, &[vec![1], vec![-2], vec![3], vec![-4]], 0, Witness::NodeLiterals);
    assert_eq!(transcript(&host.calls, &[(b, "b"), (a, "a"), (root, "root")]), [
        "begin",
        "next b0", "solve 5 7 probe b0: unsat", "probed b0 unsat",
        // The model's reads skip the dead b0 and include b1 itself.
        "next b1", "solve 5 8 probe b1: sat",
        "value 8 true", "value 9 false", "value 10 true", "value 11 false", "value 12 true",
        "probed b1 sat",
        "next b2", "solve 5 9 probe b2: unsat", "probed b2 unsat",
        // On a, both nodes have one dead pair; a1 has the smaller count and
        // no other pair, so it dies without a solve. a0 and the output were
        // seen true.
    ]);
    assert_eq!(dead_nodes(&f, &outcome), [
        TddNodeId { vtree: b, local: NodeIdx(0) }, TddNodeId { vtree: b, local: NodeIdx(2) }, TddNodeId { vtree: a, local: NodeIdx(1) },
    ]);
    assert_eq!(outcome.stats(), stats! {
        probes: 3, probes_unsat: 2, dead: 3, witness_skips: 2, value_reads: 5, reordered_levels: 2,
    });
}

#[test]
fn a_dead_side_comes_before_a_smaller_count() {
    // x3 x4 is excluded. On a, a0 has a pair into it and count 2, a1 has
    // count 1 and no dead side: a0 is probed first.
    let (f, [b, a, root]) = chain();
    let (outcome, _, host) = probe(&f, &[vec![-3, -4]], 0, Witness::None);
    assert_eq!(transcript(&host.calls, &[(b, "b"), (a, "a"), (root, "root")]), [
        "begin",
        "next b0", "solve 5 7 probe b0: unsat", "probed b0 unsat",
        "next b1", "solve 5 8 probe b1: sat", "probed b1 sat",
        "next b2", "solve 5 9 probe b2: sat", "probed b2 sat",
        "next a0", "solve 5 10 probe a0: sat", "probed a0 sat",
        "next a1", "solve 5 11 probe a1: sat", "probed a1 sat",
        "next root0", "solve 5 12 probe root0: sat", "probed root0 sat",
    ]);
    assert_eq!(dead_nodes(&f, &outcome), [TddNodeId { vtree: b, local: NodeIdx(0) }]);
    assert_eq!(outcome.stats(), stats! { probes: 6, probes_unsat: 1, dead: 1, reordered_levels: 1 });
}

#[test]
fn without_dead_sides_a_smaller_count_comes_first() {
    let (f, [b, a, root]) = chain();
    let (outcome, _, host) = probe(&f, &[], 0, Witness::None);
    assert_eq!(transcript(&host.calls, &[(b, "b"), (a, "a"), (root, "root")]), [
        "begin",
        "next b0", "solve 5 7 probe b0: sat", "probed b0 sat",
        "next b1", "solve 5 8 probe b1: sat", "probed b1 sat",
        "next b2", "solve 5 9 probe b2: sat", "probed b2 sat",
        "next a1", "solve 5 11 probe a1: sat", "probed a1 sat",
        "next a0", "solve 5 10 probe a0: sat", "probed a0 sat",
        "next root0", "solve 5 12 probe root0: sat", "probed root0 sat",
    ]);
    assert_eq!(outcome.stats(), stats! { probes: 6 });
}

#[test]
fn a_summed_out_side_is_never_dead_and_a_zero_count_dies_before_any_solve() {
    // Variables 1 to 4 are the diagram's and 5 the activation literal; 6 is
    // the true variable, v4's nodes are 7 and 8, v5's 9 and 10, the output
    // 11, and its pairs 12 and 13. Every model of x1 ∧ x3 makes v4 0, v5 0
    // and the output true.
    let (f, [v4, v5, root]) = inline_marginal();
    let (outcome, _, host) = probe(&f, &[vec![1], vec![3]], 4, Witness::NodeLiterals);
    assert_eq!(transcript(&host.calls, &[(v4, "v4"), (v5, "v5"), (root, "root")]), [
        "begin", "fresh 14",
        "round 0", "solve 5 14 round 0: sat",
        // The dead node is read too.
        "value 7 true", "value 8 false", "value 9 true", "value 10 false", "value 11 true",
        "add -14 -7 -9 -11",
        "round 1", "solve 5 14 round 1: unsat",
        "add -14",
    ]);
    assert_eq!(dead_nodes(&f, &outcome), [TddNodeId { vtree: v4, local: NodeIdx(1) }, TddNodeId { vtree: v5, local: NodeIdx(1) }]);
    // Only the output's pair into the dead node counts toward a reordering:
    // v4's side of count 0 is summed out.
    assert_eq!(outcome.stats(), stats! {
        rounds: 2, rounds_sat: 1, rounds_unsat: 1, dead: 2, zero_count_dead: 1, enumeration_dead: 1,
        round_witnesses: 3, witness_skips: 3, value_reads: 5, reordered_levels: 1,
    });
}

#[test]
fn an_output_whose_pairs_all_have_dead_sides_dies_without_a_solve() {
    // On the linear vtree over three variables, level a = (x2, x3) holds
    // `x2 ¬x3` and `¬x2 ¬x3`, and the output joins x1 with the first and
    // ¬x1 with the second.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::linear(3));
    let root = vtree.root();
    let a = vtree.children(root).1;
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let n0 = builder.push(&eng, a, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let n1 = builder.push(&eng, a, &[ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let out = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, n0), ChildPair::new(NEG_LEAF_IDX, n1)]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: out }).unwrap();
    // x3 is forced true, so both nodes are dead, and then the output. The
    // activation literal is 4, the true variable 5, and a's nodes 6 and 7.
    let (outcome, _, host) = probe(&f, &[vec![3]], 0, Witness::None);
    assert_eq!(transcript(&host.calls, &[(a, "a"), (root, "root")]), [
        "begin",
        "next a0", "solve 4 6 probe a0: unsat", "probed a0 unsat",
        "next a1", "solve 4 7 probe a1: unsat", "probed a1 unsat",
    ]);
    assert!(outcome.is_dead(f.output()));
    assert_eq!(outcome.stats(), stats! { probes: 2, probes_unsat: 2, dead: 3, reordered_levels: 1 });
}

#[test]
fn candidates_sort_by_score_then_count_then_index() {
    // Bonsai's table: scores, counts, and the order they give.
    for (scores, counts, order) in [
        (vec![0, 3, 1, 2], vec![0, 0, 0, 0], vec![1, 3, 2, 0]),
        (vec![2, 2, 1], vec![7, 7, 7], vec![0, 1, 2]),
        (vec![3, 3], vec![100, 10], vec![1, 0]),
        (vec![0, 0, 0], vec![5, 1, 3], vec![1, 2, 0]),
    ] {
        let mut sorted: Vec<usize> = (0..scores.len()).collect();
        super::super::walk::sort_candidates(&mut sorted, &scores, &counts);
        assert_eq!(sorted, order, "scores {scores:?}, counts {counts:?}");
    }
}
