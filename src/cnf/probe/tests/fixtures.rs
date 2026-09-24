//! Bonsai's probe tests, on the exhaustive store in place of a solver.
//!
//! The store picks false first wherever it branches, which is the phase
//! bonsai forces on its solver in the boundary tests. Bonsai's inline and
//! shared-prime fixtures pair the diagram with clauses that do not imply it;
//! only the walk's answers hold there whatever the solver's models, so the
//! tests that kill nodes on those fixtures run without enumeration.

use num_bigint::BigUint;

use super::*;
use crate::apply::FilterOutcome;
use crate::diagram::{ChildDecoder, ChildRef, LeafLabel, ValueRef};

/// The diagram `filter_nodes` leaves once the dead nodes are removed.
fn filtered(f: &Tdd, outcome: &ProbeOutcome) -> Tdd {
    match f.filter_nodes(|id| !outcome.is_dead(id)).unwrap() {
        FilterOutcome::Filtered { tdd, .. } => tdd,
        other => panic!("expected a filtered diagram, got {other:?}"),
    }
}

#[test]
fn inline_summed_out_sides_are_probed_and_filtered_natively() {
    // The count-0 node dies by its count; the node with count 5 lives and
    // keeps its inline side through the filter.
    let (f, [v4, _, _]) = inline_marginal();
    let (outcome, _, _) = probe(&f, &[vec![1, 2]], 0, Witness::None);
    assert_eq!(dead_nodes(&f, &outcome), [TddNodeId { vtree: v4, local: NodeIdx(1) }]);
    assert_eq!((outcome.stats().zero_count_dead, outcome.stats().probes_unsat), (1, 0));
    let g = filtered(&f, &outcome);
    let level = &g.levels[v4.idx()];
    assert_eq!(level.nodes().len(), 1, "only the live node stays on v4");
    let pairs = level.pairs_of_idx(0);
    assert_eq!(pairs.len(), 1);
    assert_eq!(ChildDecoder::marginal().child(pairs[0].right), ChildRef::Value(ValueRef::Inline(5)));
    let out = g.output();
    assert_eq!(g.levels[out.vtree.idx()].pairs_of_idx(out.local.idx()).len(), 1, "the output drops its pair into the dead node");
}

/// Probe the boundary fixture under x3 with enumeration and return what the
/// probe did. The output has a model under x3 in both forms.
fn boundary(second: LeafLabel) -> (Tdd, CnfEncoding, ProbeOutcome) {
    let (f, _) = marginal_boundary(second);
    let (outcome, encoding, _) = probe(&f, &[vec![3]], 4, Witness::NodeLiterals);
    assert!(!outcome.is_dead(f.output()), "{second:?}: the output is live");
    assert_eq!(outcome.stats().enumeration_dead, 0, "{second:?}: the enumeration killed a live node");
    (f, encoding, outcome)
}

#[test]
fn the_enumeration_sees_every_slice_on_a_certified_boundary() {
    // With distinct primes the nodes of v4 read as x1 and ¬x1 once the slots
    // read as true: two slices, each seen once, and nothing dies. Reading
    // the slots as free variables would let a model hide both nodes.
    let (f, encoding, outcome) = boundary(LeafLabel::Neg);
    assert!(encoding.erasure_certified());
    assert!(dead_nodes(&f, &outcome).is_empty());
    let stats = outcome.stats();
    assert_eq!((stats.rounds_sat, stats.rounds_unsat, stats.round_witnesses, stats.probes), (2, 1, 4, 0));
}

#[test]
fn the_enumeration_kills_nothing_on_a_shared_prime_boundary() {
    // With a shared prime both nodes of v4 read as x1, so the slices nest:
    // the first model, with x1 false, sees only v5's node, its blocking
    // clause excludes every model, and the enumeration ends with the live
    // nodes unseen. The certificate fails and the walk probes them instead.
    let (f, encoding, outcome) = boundary(LeafLabel::Pos);
    assert!(!encoding.erasure_certified());
    assert!(dead_nodes(&f, &outcome).is_empty());
    let stats = outcome.stats();
    assert_eq!((stats.rounds_sat, stats.rounds_unsat, stats.round_witnesses, stats.probes), (1, 1, 1, 1));
}

/// Bonsai's prefix fixture: `(x1 ∨ x3) ∧ (x2 ∨ ¬x3)` on the reversed linear
/// vtree, whose lower level has two nodes that need x1, and the clauses with
/// `¬x1` added.
fn prefix_fixture() -> (Tdd, Vec<Vec<i32>>) {
    let vtree = Arc::new(Vtree::reverse_linear(3));
    let clauses = vec![vec![1, 3], vec![2, -3]];
    let f = compile_clauses(&vtree, &clauses);
    let phi = [clauses, vec![vec![-1]]].concat();
    (f, phi)
}

#[test]
fn an_encoded_prefix_is_probed() {
    // Cut after the lower level: its two dead nodes are found, the output's
    // level has no literal and is not asked about, and the filtered diagram
    // keeps the one model with x1 false.
    let (f, phi) = prefix_fixture();
    for (rounds, witness) in [(4, Witness::NodeLiterals), (0, Witness::None)] {
        let (encoding, mut host) = Host::over(&f, &phi, Some(1));
        assert!(encoding.skipped() >= 1 && encoding.literal(f.output()).is_none());
        let outcome = host.probe(&f, &encoding, rounds, witness);
        assert!(outcome.stats().dead >= 2, "rounds {rounds}: {:?}", outcome.stats());
        assert!(!outcome.is_dead(f.output()));
        let unit = compile_clauses(f.vtree(), &[vec![-1]]);
        assert_eq!(crate::and(filtered(&f, &outcome), unit).unwrap().model_count().unwrap(), BigUint::from(1u32));
    }
}

#[test]
fn an_empty_prefix_is_neither_probed_nor_killed() {
    // Bonsai gives up on such a pass; a probe that runs anyway finds no
    // candidate, and its enumeration has no node to kill.
    let (f, phi) = prefix_fixture();
    for (rounds, witness) in [(4, Witness::NodeLiterals), (0, Witness::None)] {
        let (encoding, mut host) = Host::over(&f, &phi, Some(0));
        assert_eq!(encoding.encoded_nodes(), 0);
        let outcome = host.probe(&f, &encoding, rounds, witness);
        assert!(dead_nodes(&f, &outcome).is_empty(), "rounds {rounds}");
        assert_eq!(outcome.stats().probes, 0);
    }
}

#[test]
fn a_false_diagram_has_nothing_to_probe() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::zero(&tree);
    let (outcome, _, host) = probe(&f, &[], 0, Witness::NodeLiterals);
    assert_eq!(host.calls, [Call::Poll(ProbePoint::Begin)]);
    assert_eq!(outcome.stats(), ProbeStats::default());
    assert!(!outcome.is_dead(f.output()));
}
