//! Finding the encoded nodes that an external SAT solver proves impossible.
//!
//! [`Engine::probe_nodes`] asks the solver that received an encoding, through
//! [`SatOracle`], which encoded nodes can be true together with its other
//! clauses, and marks the others dead. A [`ProbePolicy`] decides at fixed
//! points whether the probe goes on. The caller removes the dead nodes with
//! [`Engine::filter_nodes_with`](crate::Engine::filter_nodes_with).

use std::ops::ControlFlow;

use crate::diagram::{Tdd, TddNodeId};
use crate::limits::OperationError;
use crate::Engine;

use super::{ClauseSink, CnfEncoding};

mod walk;

/// A solver's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolveStatus {
    /// The clauses have a model in which every assumption is true, and
    /// [`SatOracle::value`] reads it.
    Sat,
    /// They have none.
    Unsat,
    /// The solver gave up, for example at a conflict limit. The probe learns
    /// nothing from it.
    Unknown,
}

/// Which solve the probe asks for, so the caller can choose its limits and
/// account for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SolveCall {
    /// Enumeration round `round`, counting from zero.
    Round {
        /// The round.
        round: u32,
    },
    /// Whether `node` can be true.
    Probe {
        /// The node asked about.
        node: TddNodeId,
    },
}

/// The solver that received an encoding, as [`Engine::probe_nodes`] uses it.
///
/// It is the [`ClauseSink`] the encoding was made through, so the encoding's
/// literals are its variables and [`ClauseSink::fresh_var`] numbers after
/// them.
pub trait SatOracle: ClauseSink {
    /// Add a clause that holds from now on, without the encoding's gate.
    fn add_clause(&mut self, clause: &[i32]);

    /// Whether the clauses have a model in which every literal of
    /// `assumptions` is true. The assumptions hold for this call only.
    fn solve(&mut self, assumptions: &[i32], call: SolveCall) -> SolveStatus;

    /// Whether `literal` is true in the model the last solve found. Asked only
    /// after a solve that returned [`SolveStatus::Sat`], before the next
    /// solve.
    fn value(&mut self, literal: i32) -> bool;
}

/// A point at which [`Engine::probe_nodes`] asks its policy whether to go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProbePoint {
    /// After the nodes with a count of zero are marked, before any call on
    /// the oracle. `Break` ends the probe.
    Begin,
    /// Before enumeration round `round`. `Break` ends the enumeration, and the
    /// walk still runs.
    Round {
        /// The round about to be solved.
        round: u32,
    },
    /// Just before the solve that asks whether `node` can be true. `Break`
    /// ends the walk.
    Solve {
        /// The node about to be asked about.
        node: TddNodeId,
    },
    /// After that solve, once its answer is recorded and its model read.
    /// `Break` ends the walk.
    Probed {
        /// The node asked about.
        node: TddNodeId,
        /// The solver's answer.
        status: SolveStatus,
    },
}

/// The caller's stopping rule for [`Engine::probe_nodes`].
pub trait ProbePolicy {
    /// Whether the probe goes on at `at`, given its counts so far. The nodes
    /// already marked dead stay dead whatever the answer. The default always
    /// continues.
    fn poll(&mut self, at: ProbePoint, stats: &ProbeStats) -> ControlFlow<()> {
        let _ = (at, stats);
        ControlFlow::Continue(())
    }
}

/// The order in which [`Engine::probe_nodes`] visits the encoded nodes.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ProbeOrder<'a> {
    /// Every reachable structural internal node whose count is zero is dead
    /// before any solve, encoded or not. Then the internal levels are walked
    /// in [`Vtree::internal_bottomup`](crate::Vtree::internal_bottomup) order.
    ///
    /// A level's candidates are its encoded nodes. A pair side is dead when it
    /// is `ZERO` or names a structural node already marked dead; a summed-out
    /// side never is, whatever its count. The candidates are sorted by how
    /// many of their stored pairs have a dead side, most first, then by count,
    /// smallest first, then by index. The level counts toward
    /// [`ProbeStats::reordered_levels`] when some candidate has such a pair.
    /// In that order, a candidate already dead is passed over; one whose every
    /// stored pair has a dead side is marked dead without a solve; under
    /// [`Witness::NodeLiterals`] one already seen true is skipped; the rest are
    /// probed.
    BottomUp {
        /// The diagram's node counts, as
        /// [`Tdd::node_counts_u128`](crate::Tdd::node_counts_u128) returns them.
        counts: &'a [Vec<u128>],
    },
}

/// What the walk of [`Engine::probe_nodes`] learns from the models it finds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Witness {
    /// Nothing: every candidate the walk reaches is probed.
    #[default]
    None,
    /// After each [`SolveStatus::Sat`] probe, read the literal of every
    /// encoded node that is neither dead nor already seen true. A node seen
    /// true, by a probe's model or an enumeration round's, is not probed.
    NodeLiterals,
}

/// What [`Engine::probe_nodes`] did.
///
/// A node found dead is counted once in [`dead`](Self::dead) and in at most
/// one of the finer counts; the dead nodes that are in none of them had every
/// stored pair dead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProbeStats {
    /// Solves of the walk, one per probed node.
    pub probes: u64,
    /// Probes answered [`SolveStatus::Unsat`], each marking its node dead.
    pub probes_unsat: u64,
    /// Probes answered [`SolveStatus::Unknown`].
    pub probes_unknown: u64,
    /// Enumeration rounds solved.
    pub rounds: u64,
    /// Rounds answered [`SolveStatus::Sat`].
    pub rounds_sat: u64,
    /// Rounds answered [`SolveStatus::Unsat`], which end the enumeration.
    pub rounds_unsat: u64,
    /// Rounds answered [`SolveStatus::Unknown`], which end the enumeration.
    pub rounds_unknown: u64,
    /// Nodes marked dead, for any reason.
    pub dead: u64,
    /// Nodes marked dead because their count is zero.
    pub zero_count_dead: u64,
    /// Nodes marked dead because the enumeration ran out of models without
    /// making them true.
    pub enumeration_dead: u64,
    /// Nodes an enumeration round's model made true for the first time.
    pub round_witnesses: u64,
    /// Candidates the walk skipped because a model had made them true.
    pub witness_skips: u64,
    /// Calls to [`SatOracle::value`].
    pub value_reads: u64,
    /// Levels at which some candidate had a stored pair with a dead side.
    pub reordered_levels: u64,
}

/// The nodes [`Engine::probe_nodes`] marked dead, and its counts.
#[derive(Clone, Debug)]
pub struct ProbeOutcome {
    dead: Vec<Vec<bool>>,
    stats: ProbeStats,
}

impl ProbeOutcome {
    /// Whether node `id` was marked dead; false for IDs outside the diagram.
    #[must_use]
    pub fn is_dead(&self, id: TddNodeId) -> bool {
        self.dead.get(id.vtree.idx()).and_then(|row| row.get(id.local.idx())).copied().unwrap_or(false)
    }

    /// The dead nodes, in the shape [`Tdd::reachable_nodes`] returns.
    #[must_use]
    pub fn dead(&self) -> &[Vec<bool>] {
        &self.dead
    }

    /// What the probe did.
    #[must_use]
    pub fn stats(&self) -> ProbeStats {
        self.stats
    }
}

impl Tdd {
    /// Mark the encoded nodes that cannot be true under `host`'s clauses as
    /// dead; see [`Engine::probe_nodes`]. Runs without limits.
    ///
    /// # Errors
    ///
    /// As [`Engine::probe_nodes`], without [`OperationError::Stopped`].
    ///
    /// # Panics
    ///
    /// As [`Engine::probe_nodes`].
    pub fn probe_nodes<H: SatOracle + ProbePolicy + ?Sized>(
        &self, encoding: &CnfEncoding, order: ProbeOrder<'_>, rounds: u32, witness: Witness, host: &mut H,
    ) -> Result<ProbeOutcome, OperationError> {
        self.context().run(|eng| eng.probe_nodes(self, encoding, order, rounds, witness, host))
    }
}

impl Engine {
    /// Mark the encoded nodes of `f` that cannot be true under `host`'s
    /// clauses as dead.
    ///
    /// `encoding` is `f`'s, made with [`CnfScheme::Equivalence`](super::CnfScheme::Equivalence)
    /// through `host`, whose activation literal `a` is still unretired. The
    /// encoded nodes are the ones [`CnfEncoding::literal`] gives a literal,
    /// taken in encoding order: internal levels in
    /// [`Vtree::internal_bottomup`](crate::Vtree::internal_bottomup) order and
    /// by index within a level. A node is marked dead when its count in
    /// `order` is zero; when an enumeration round is Unsat,
    /// [`CnfEncoding::erasure_certified`] holds, and no round's model made it
    /// true; when every pair it stores has a dead side; or when the solve for
    /// its literal is Unsat. No model of the solver's other clauses makes a
    /// dead node's function true. The enumeration's conclusion needs every
    /// such model to satisfy `f`, as when `f` conjoins some of those clauses;
    /// nothing else does. An Unknown answer, or a node never asked about,
    /// leaves the node alive.
    ///
    /// The calls on `host`, in order:
    ///
    /// 1. [`ProbePoint::Begin`], once the zero counts are marked.
    /// 2. If `rounds` is positive, a fresh variable `b`. For each round `r`
    ///    below `rounds`: [`ProbePoint::Round`], then a solve under `[a, b]`.
    ///    After a Sat round, the literal of every encoded node is read, dead
    ///    ones included, and the clause of `-b` followed by the negation of
    ///    each literal read true is added. An Unsat or Unknown round ends the
    ///    enumeration. Then the clause `[-b]`.
    /// 3. The walk in `order`. For each candidate it probes:
    ///    [`ProbePoint::Solve`], a solve under `[a, y]` with `y` the node's
    ///    literal, under `witness` the reads of the model, then
    ///    [`ProbePoint::Probed`].
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] if an allocation is refused,
    /// [`OperationError::Stopped`] on cancellation, and
    /// [`OperationError::InvalidLiteral`] if the fresh variable is `0` or
    /// `i32::MIN`. After an error, the clauses already added hold only while
    /// `b` is assumed, which the probe no longer does. The caller retires `a`
    /// whether or not the probe succeeds.
    ///
    /// # Panics
    ///
    /// Panics if `encoding`, or the counts in `order`, do not have `f`'s
    /// shape.
    pub fn probe_nodes<H: SatOracle + ProbePolicy + ?Sized>(
        &self, f: &Tdd, encoding: &CnfEncoding, order: ProbeOrder<'_>, rounds: u32, witness: Witness, host: &mut H,
    ) -> Result<ProbeOutcome, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        walk::probe(self, f, encoding, order, rounds, witness, host)
    }
}

#[cfg(test)]
mod tests;
