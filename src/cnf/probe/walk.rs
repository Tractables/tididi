//! The probe itself: the nodes with a count of zero, the enumeration rounds,
//! then the levels bottom-up, each in the order of its dead sides.
//!
//! Every loop over the encoded nodes runs in encoding order, so the reads and
//! the clauses reach the oracle in the order [`Engine::probe_nodes`]
//! documents.

use crate::diagram::{ChildDecoder, EncodedChildRef, NodeIdx, Tdd, TddNodeId, ZERO};
use crate::limits::{Limits, OperationError, PollGate};
use crate::vtree::VtreeIdx;
use crate::Engine;

use super::super::encode::{check_literal, table, NONE, TRUE};
use super::super::CnfEncoding;
use super::{ProbeOrder, ProbeOutcome, ProbePoint, ProbePolicy, ProbeStats, SatOracle, SolveCall, SolveStatus, Witness};

/// Whether a table entry is a node's literal rather than a sentinel.
fn encoded(literal: i32) -> bool {
    literal != NONE && literal != TRUE
}

/// Panic unless `encoding` and `counts` have `f`'s shape.
fn assert_shape(f: &Tdd, encoding: &CnfEncoding, counts: &[Vec<u128>]) {
    let slots = |t: usize| f.reference_slot_count(VtreeIdx(t as u32));
    let encoded = encoding.literals.len() == f.levels.len()
        && encoding.literals.iter().zip(&encoding.reachable).enumerate().all(|(t, (row, reach))| row.len() == slots(t) && reach.len() == slots(t));
    assert!(encoded, "the encoding does not have the probed diagram's shape");
    let counted = counts.len() == f.levels.len()
        && (f.is_zero() || f.vtree.internal_bottomup().all(|(t, _, _)| {
            let level = &f.levels[t.idx()];
            level.is_marginal() || counts[t.idx()].len() == level.nodes().len()
        }));
    assert!(counted, "the counts do not have the probed diagram's shape");
}

/// Sort the candidate indices in `order` by score, most first, then by count,
/// smallest first. The sort is stable, so ties keep the index order.
pub(super) fn sort_candidates(order: &mut [usize], scores: &[usize], counts: &[u128]) {
    order.sort_by(|&a, &b| scores[b].cmp(&scores[a]).then_with(|| counts[a].cmp(&counts[b])));
}

/// The probe's state between two calls on the oracle.
struct Probe<'a> {
    f: &'a Tdd,
    encoding: &'a CnfEncoding,
    lim: &'a Limits,
    gate: PollGate<'a>,
    dead: Vec<Vec<bool>>,
    /// Nodes a model made true. Empty when no model is read.
    seen: Vec<Vec<bool>>,
    /// The blocking clause being built.
    clause: Vec<i32>,
    stats: ProbeStats,
}

/// [`Engine::probe_nodes`], after its operation scope is open.
pub(super) fn probe<H: SatOracle + ProbePolicy + ?Sized>(
    eng: &Engine, f: &Tdd, encoding: &CnfEncoding, order: ProbeOrder<'_>, rounds: u32, witness: Witness, host: &mut H,
) -> Result<ProbeOutcome, OperationError> {
    let ProbeOrder::BottomUp { counts } = order;
    assert_shape(f, encoding, counts);
    let lim = eng.limits();
    let reads = rounds > 0 || witness == Witness::NodeLiterals;
    let mut probe = Probe {
        f,
        encoding,
        lim,
        gate: lim.gate(),
        dead: table(lim, f, false)?,
        seen: if reads { table(lim, f, false)? } else { Vec::new() },
        clause: Vec::new(),
        stats: ProbeStats::default(),
    };
    probe.zero_counts(counts)?;
    if ProbePolicy::poll(host, ProbePoint::Begin, &probe.stats).is_continue() {
        if rounds > 0 { probe.enumerate(rounds, host)?; }
        probe.walk(counts, witness, host)?;
    }
    lim.discard(std::mem::take(&mut probe.seen));
    lim.discard(std::mem::take(&mut probe.clause));
    Ok(ProbeOutcome { dead: std::mem::take(&mut probe.dead), stats: probe.stats })
}

impl Probe<'_> {
    fn kill(&mut self, t: VtreeIdx, i: usize) {
        self.dead[t.idx()][i] = true;
        self.stats.dead += 1;
    }

    /// Mark every reachable structural internal node whose count is zero,
    /// encoded or not.
    fn zero_counts(&mut self, counts: &[Vec<u128>]) -> Result<(), OperationError> {
        let (f, encoding) = (self.f, self.encoding);
        for (t, _, _) in f.vtree.internal_bottomup() {
            for (i, node) in f.levels[t.idx()].nodes().iter().enumerate() {
                self.gate.poll(1)?;
                if encoding.reachable[t.idx()][i] && node.is_internal() && counts[t.idx()][i] == 0 {
                    self.kill(t, i);
                    self.stats.zero_count_dead += 1;
                }
            }
        }
        Ok(())
    }

    /// The enumeration rounds, under a fresh variable that the blocking
    /// clauses are gated by and that is retired after the last round.
    fn enumerate<H: SatOracle + ProbePolicy + ?Sized>(&mut self, rounds: u32, host: &mut H) -> Result<(), OperationError> {
        let (f, encoding) = (self.f, self.encoding);
        let block = check_literal(host.fresh_var())?;
        let assumptions = [encoding.activation, block];
        let mut exhausted = false;
        for round in 0..rounds {
            if ProbePolicy::poll(host, ProbePoint::Round { round }, &self.stats).is_break() { break; }
            let status = host.solve(&assumptions, SolveCall::Round { round });
            self.stats.rounds += 1;
            match status {
                SolveStatus::Sat => {
                    self.stats.rounds_sat += 1;
                    // When every model of the clauses satisfies the diagram,
                    // the model makes one node true at each encoded level, so
                    // the clause names at least one node.
                    self.clause.clear();
                    self.lim.try_push(&mut self.clause, -block)?;
                    for (t, _, _) in f.vtree.internal_bottomup() {
                        for i in 0..f.levels[t.idx()].nodes().len() {
                            self.gate.poll(1)?;
                            let y = encoding.literals[t.idx()][i];
                            if !encoded(y) { continue; }
                            self.stats.value_reads += 1;
                            if host.value(y) {
                                if !self.seen[t.idx()][i] {
                                    self.seen[t.idx()][i] = true;
                                    self.stats.round_witnesses += 1;
                                }
                                self.lim.try_push(&mut self.clause, -y)?;
                            }
                        }
                    }
                    host.add_clause(&self.clause);
                }
                SolveStatus::Unsat => {
                    self.stats.rounds_unsat += 1;
                    exhausted = true;
                    break;
                }
                SolveStatus::Unknown => {
                    self.stats.rounds_unknown += 1;
                    break;
                }
            }
        }
        host.add_clause(&[-block]);
        // Each model then makes exactly one node true at each encoded level,
        // and the certificate keeps that so with summed-out children read as
        // true. Two models' sets of true nodes are equal or incomparable, so
        // a blocking clause excludes the models with its own set and no
        // others, and an Unsat round means every set was seen: a node in
        // none of them has no model.
        if exhausted && encoding.erasure_certified {
            for (t, _, _) in f.vtree.internal_bottomup() {
                for i in 0..f.levels[t.idx()].nodes().len() {
                    self.gate.poll(1)?;
                    if encoded(encoding.literals[t.idx()][i]) && !self.seen[t.idx()][i] && !self.dead[t.idx()][i] {
                        self.kill(t, i);
                        self.stats.enumeration_dead += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a pair side into level `child` is dead: `ZERO`, or a
    /// structural node marked dead. A summed-out side never is.
    fn side_dead(&self, child: VtreeIdx, raw: EncodedChildRef, marginal: bool) -> bool {
        raw == ZERO.into() || !marginal && self.dead[child.idx()][ChildDecoder::structural().node(raw).idx()]
    }

    /// The levels bottom-up, each in the order of [`ProbeOrder::BottomUp`].
    fn walk<H: SatOracle + ProbePolicy + ?Sized>(&mut self, counts: &[Vec<u128>], witness: Witness, host: &mut H) -> Result<(), OperationError> {
        let (f, encoding) = (self.f, self.encoding);
        let reads = witness == Witness::NodeLiterals;
        let mut scores: Vec<usize> = Vec::new();
        let mut order: Vec<usize> = Vec::new();
        'levels: for (t, left, right) in f.vtree.internal_bottomup() {
            let level = &f.levels[t.idx()];
            let literals = &encoding.literals[t.idx()];
            let (left_marginal, right_marginal) = (f.levels[left.idx()].is_marginal(), f.levels[right.idx()].is_marginal());
            scores.clear();
            order.clear();
            self.lim.try_resize(&mut scores, level.nodes().len(), 0)?;
            let mut reordered = false;
            for i in 0..level.nodes().len() {
                self.gate.poll(1)?;
                if !encoded(literals[i]) { continue; }
                let score = level.pairs_of_idx(i).iter()
                    .filter(|pair| self.side_dead(left, pair.left, left_marginal) || self.side_dead(right, pair.right, right_marginal))
                    .count();
                scores[i] = score;
                reordered |= score > 0;
                self.lim.try_push(&mut order, i)?;
            }
            if reordered { self.stats.reordered_levels += 1; }
            // The dead children read here are final: the walk marks only this
            // level while it runs.
            sort_candidates(&mut order, &scores, &counts[t.idx()]);
            for &i in &order {
                self.gate.poll(1)?;
                if self.dead[t.idx()][i] { continue; }
                if scores[i] == level.pairs_of_idx(i).len() {
                    self.kill(t, i);
                    continue;
                }
                if reads && self.seen[t.idx()][i] {
                    self.stats.witness_skips += 1;
                    continue;
                }
                let node = TddNodeId { vtree: t, local: NodeIdx(i as u32) };
                if ProbePolicy::poll(host, ProbePoint::Solve { node }, &self.stats).is_break() { break 'levels; }
                let status = host.solve(&[encoding.activation, literals[i]], SolveCall::Probe { node });
                self.stats.probes += 1;
                match status {
                    SolveStatus::Unsat => {
                        self.stats.probes_unsat += 1;
                        self.kill(t, i);
                    }
                    SolveStatus::Sat if reads => self.read_model(host)?,
                    SolveStatus::Sat => {}
                    SolveStatus::Unknown => self.stats.probes_unknown += 1,
                }
                if ProbePolicy::poll(host, ProbePoint::Probed { node, status }, &self.stats).is_break() { break 'levels; }
            }
        }
        self.lim.discard(scores);
        self.lim.discard(order);
        Ok(())
    }

    /// Mark as seen every encoded node, neither dead nor seen yet, that the
    /// last model makes true. The probe assumed the activation literal, under
    /// which each node's variable is its function, so such a node has a model.
    fn read_model<H: SatOracle + ?Sized>(&mut self, host: &mut H) -> Result<(), OperationError> {
        let (f, encoding) = (self.f, self.encoding);
        for (t, _, _) in f.vtree.internal_bottomup() {
            for i in 0..f.levels[t.idx()].nodes().len() {
                self.gate.poll(1)?;
                let y = encoding.literals[t.idx()][i];
                if self.dead[t.idx()][i] || self.seen[t.idx()][i] || !encoded(y) { continue; }
                self.stats.value_reads += 1;
                if host.value(y) { self.seen[t.idx()][i] = true; }
            }
        }
        Ok(())
    }
}
