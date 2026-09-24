//! Tests of the probe: its exact calls on hand-built diagrams, bonsai's
//! fixtures, its agreement with exhaustive search, and its stops.

use std::ops::ControlFlow;
use std::sync::Arc;

use super::*;
use crate::cnf::{ClauseSink, CnfScheme, EncodePoint};
use crate::diagram::NodeIdx;
use crate::test_helpers::*;
use crate::vtree::{VarId, Vtree, VtreeIdx};

mod fixtures;
mod golden;
mod soundness;
mod stops;

/// One call the probe made on its host.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Fresh(i32),
    Add(Vec<i32>),
    Solve(Vec<i32>, SolveCall, SolveStatus),
    Value(i32, bool),
    Poll(ProbePoint),
}

/// A [`ClauseStore`] that records the calls made after the encoding, stops
/// the encoding before level `levels` of the bottom-up walk, answers `Break`
/// to probe poll number `stop_at`, counting from zero, and hands out `bad`
/// as a fresh variable when it is set.
struct Host {
    store: ClauseStore,
    calls: Vec<Call>,
    levels: Option<usize>,
    polls: usize,
    stop_at: Option<usize>,
    bad: Option<i32>,
}

impl ClauseSink for Host {
    fn fresh_var(&mut self) -> i32 {
        let var = self.bad.unwrap_or_else(|| self.store.fresh_var());
        self.calls.push(Call::Fresh(var));
        var
    }

    fn leaf_literal(&mut self, var: VarId) -> i32 {
        self.store.leaf_literal(var)
    }

    fn clause(&mut self, body: &[i32]) {
        self.store.clause(body);
    }

    fn poll(&mut self, at: EncodePoint) -> ControlFlow<()> {
        match (at, self.levels) {
            (EncodePoint::Level { complete, .. }, Some(levels)) if complete >= levels => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    }
}

impl SatOracle for Host {
    fn add_clause(&mut self, clause: &[i32]) {
        self.calls.push(Call::Add(clause.to_vec()));
        self.store.add_clause(clause);
    }

    fn solve(&mut self, assumptions: &[i32], call: SolveCall) -> SolveStatus {
        let status = SatOracle::solve(&mut self.store, assumptions, call);
        self.calls.push(Call::Solve(assumptions.to_vec(), call, status));
        status
    }

    fn value(&mut self, literal: i32) -> bool {
        let value = self.store.value(literal);
        self.calls.push(Call::Value(literal, value));
        value
    }
}

impl ProbePolicy for Host {
    fn poll(&mut self, at: ProbePoint, _: &ProbeStats) -> ControlFlow<()> {
        self.calls.push(Call::Poll(at));
        self.polls += 1;
        if self.stop_at == Some(self.polls - 1) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }
}

impl Host {
    /// A host whose store holds `phi` over `f`'s variables, then the
    /// activation literal, then `f`'s encoding cut before level `levels`,
    /// with no call recorded yet.
    fn over(f: &Tdd, phi: &[Vec<i32>], levels: Option<usize>) -> (CnfEncoding, Self) {
        let mut store = ClauseStore::new(f.vtree().num_vars());
        for clause in phi { store.add_clause(clause); }
        let activation = store.activate();
        let mut host = Host { store, calls: Vec::new(), levels, polls: 0, stop_at: None, bad: None };
        let encoding = f.encode_cnf(CnfScheme::Equivalence, activation, &mut host).unwrap();
        host.calls.clear();
        (encoding, host)
    }

    /// Probe `f` in bottom-up order under this host.
    fn probe(&mut self, f: &Tdd, encoding: &CnfEncoding, rounds: u32, witness: Witness) -> ProbeOutcome {
        let counts = f.node_counts_u128().unwrap();
        f.probe_nodes(encoding, ProbeOrder::BottomUp { counts: &counts }, rounds, witness, self).unwrap()
    }
}

/// Encode `f` under `phi` and probe it in full.
fn probe(f: &Tdd, phi: &[Vec<i32>], rounds: u32, witness: Witness) -> (ProbeOutcome, CnfEncoding, Host) {
    let (encoding, mut host) = Host::over(f, phi, None);
    let outcome = host.probe(f, &encoding, rounds, witness);
    (outcome, encoding, host)
}

/// The reachable structural internal nodes of `f`, bottom-up and by index.
fn internal_nodes(f: &Tdd) -> Vec<TddNodeId> {
    let reachable = f.reachable_nodes();
    f.vtree().internal_bottomup()
        .flat_map(|(t, _, _)| (0..f.levels[t.idx()].nodes().len()).map(move |i| TddNodeId { vtree: t, local: NodeIdx(i as u32) }))
        .filter(|id| reachable[id.vtree.idx()][id.local.idx()])
        .collect()
}

/// The dead nodes of an outcome, bottom-up and by index.
fn dead_nodes(f: &Tdd, outcome: &ProbeOutcome) -> Vec<TddNodeId> {
    internal_nodes(f).into_iter().filter(|&id| outcome.is_dead(id)).collect()
}
