//! Stopping the probe at each of its polls, refusing its allocations,
//! cancelling it, and what it refuses.

use super::*;
use crate::diagram::LeafLabel;
use crate::limits::{LimitConfig, StopAt, StopRules};

/// Diagrams with clauses that imply them, where a probe makes every kind of
/// call.
fn fixtures() -> Vec<(Tdd, Vec<Vec<i32>>)> {
    vec![
        (chain().0, vec![vec![1], vec![-2], vec![3], vec![-4]]),
        (chain().0, vec![vec![1], vec![3], vec![2, -4], vec![-2, 4]]),
        (inline_marginal().0, vec![vec![1], vec![3]]),
        (marginal_boundary(LeafLabel::Neg).0, vec![vec![3]]),
        (marginal_boundary(LeafLabel::Pos).0, vec![vec![1], vec![3]]),
    ]
}

/// The rounds and witness settings each fixture is probed with.
const SETTINGS: [(u32, Witness); 3] = [(0, Witness::None), (0, Witness::NodeLiterals), (4, Witness::NodeLiterals)];

/// The probe's counts and dead nodes.
fn result(outcome: &ProbeOutcome) -> (ProbeStats, Vec<Vec<bool>>) {
    (outcome.stats(), outcome.dead().to_vec())
}

#[test]
fn a_break_at_each_poll_ends_what_the_poll_names() {
    for (f, phi) in fixtures() {
        for (rounds, witness) in SETTINGS {
            let (full, _, whole) = probe(&f, &phi, rounds, witness);
            let polls: Vec<ProbePoint> = whole.calls.iter().filter_map(|call| match call { Call::Poll(at) => Some(*at), _ => None }).collect();
            for (k, &at) in polls.iter().enumerate() {
                let (encoding, mut host) = Host::over(&f, &phi, None);
                host.stop_at = Some(k);
                let outcome = host.probe(&f, &encoding, rounds, witness);
                let what = format!("{rounds} rounds, {witness:?}, break at {at:?}");
                let at_poll = host.calls.iter().position(|call| *call == Call::Poll(at)).unwrap();
                assert_eq!(host.calls[..=at_poll], whole.calls[..=at_poll], "{what}");
                for (id, dead) in internal_nodes(&f).into_iter().map(|id| (id, outcome.is_dead(id))) {
                    assert!(!dead || full.is_dead(id), "{what}: {id:?} died only after a break");
                }
                match at {
                    // The rest of the probe is skipped.
                    ProbePoint::Begin | ProbePoint::Solve { .. } | ProbePoint::Probed { .. } => {
                        assert_eq!(host.calls.len(), at_poll + 1, "{what}");
                    }
                    // The enumeration ends and its variable is retired; the
                    // walk still decides every node.
                    ProbePoint::Round { .. } => {
                        let Call::Fresh(block) = host.calls[1] else { panic!("{what}: the enumeration's variable comes first") };
                        assert_eq!(host.calls[at_poll + 1], Call::Add(vec![-block]), "{what}");
                        assert_eq!(outcome.dead(), full.dead(), "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_refused_allocation_leaves_the_calls_so_far_given() {
    for (f, phi) in fixtures() {
        for (rounds, witness) in SETTINGS {
            let (full, _, whole) = probe(&f, &phi, rounds, witness);
            let counts = f.node_counts_u128().unwrap();
            let mut completed = false;
            for nth in 0..4096 {
                let (encoding, mut host) = Host::over(&f, &phi, None);
                let eng = Engine::new();
                eng.limits().refuse_nth_reserve(nth);
                let probed = eng.probe_nodes(&f, &encoding, ProbeOrder::BottomUp { counts: &counts }, rounds, witness, &mut host);
                eng.limits().grant_every_reserve();
                match probed {
                    Err(error) => {
                        assert_eq!(error, OperationError::OverBudget);
                        assert_eq!(host.calls, whole.calls[..host.calls.len()], "refusal {nth}");
                    }
                    Ok(outcome) => {
                        assert_eq!(host.calls, whole.calls);
                        assert_eq!(result(&outcome), result(&full));
                        completed = true;
                        break;
                    }
                }
            }
            assert!(completed, "every reservation must be covered");
        }
    }
}

#[test]
fn cancellation_stops_between_nodes_with_the_calls_so_far_given() {
    for (f, phi) in fixtures() {
        for (rounds, witness) in SETTINGS {
            let (full, _, whole) = probe(&f, &phi, rounds, witness);
            let counts = f.node_counts_u128().unwrap();
            let mut stopped_inside = false;
            for units in 0.. {
                let (encoding, mut host) = Host::over(&f, &phi, None);
                let eng = Engine::new();
                let _guard = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
                    unconditional: Some(StopAt::WorkUnits(units)), ..StopRules::default()
                }));
                eng.limits().pin_reduce_poll_stride(Some(1));
                match eng.probe_nodes(&f, &encoding, ProbeOrder::BottomUp { counts: &counts }, rounds, witness, &mut host) {
                    Err(error) => {
                        assert_eq!(error, OperationError::Stopped);
                        assert_eq!(host.calls, whole.calls[..host.calls.len()], "a stop after {units} units");
                        stopped_inside |= !host.calls.is_empty();
                    }
                    Ok(outcome) => {
                        assert_eq!(host.calls, whole.calls);
                        assert_eq!(result(&outcome), result(&full));
                        break;
                    }
                }
            }
            assert!(stopped_inside, "some stop must fall after the first call");
        }
    }
}

#[test]
fn a_fresh_variable_the_table_cannot_hold_is_refused() {
    let (f, _) = chain();
    for bad in [0, i32::MIN] {
        let (encoding, mut host) = Host::over(&f, &[], None);
        host.bad = Some(bad);
        let counts = f.node_counts_u128().unwrap();
        let probed = f.probe_nodes(&encoding, ProbeOrder::BottomUp { counts: &counts }, 1, Witness::None, &mut host);
        assert_eq!(probed.err(), Some(OperationError::InvalidLiteral(bad)));
        assert_eq!(host.calls, [Call::Poll(ProbePoint::Begin), Call::Fresh(bad)]);
    }
}

#[test]
#[should_panic(expected = "the encoding does not have the probed diagram's shape")]
fn an_encoding_of_another_diagram_panics() {
    let (f, _) = chain();
    let (g, _) = inline_marginal();
    let (encoding, mut host) = Host::over(&g, &[], None);
    let counts = f.node_counts_u128().unwrap();
    let _ = f.probe_nodes(&encoding, ProbeOrder::BottomUp { counts: &counts }, 0, Witness::None, &mut host);
}

#[test]
#[should_panic(expected = "the counts do not have the probed diagram's shape")]
fn counts_of_another_diagram_panic() {
    let (f, _) = chain();
    let (g, _) = inline_marginal();
    let (encoding, mut host) = Host::over(&f, &[], None);
    let counts = g.node_counts_u128().unwrap();
    let _ = f.probe_nodes(&encoding, ProbeOrder::BottomUp { counts: &counts }, 0, Witness::None, &mut host);
}
