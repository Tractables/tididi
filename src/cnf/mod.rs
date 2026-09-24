//! Clauses that define a diagram's nodes, for an external SAT solver.
//!
//! [`Tdd::encode_cnf`] gives each reachable structural internal node a solver
//! variable and passes the clauses that define it to a [`ClauseSink`], the
//! caller's binding to its solver. The sink supplies every variable, receives
//! every clause and decides at each [`EncodePoint`] whether the encoding goes
//! on. [`Engine::probe_nodes`] then asks the same solver, as a [`SatOracle`],
//! which encoded nodes can be true, and marks the others dead for
//! [`Engine::filter_nodes_with`](crate::Engine::filter_nodes_with) to remove.
//! The library holds no solver and reads no clock.

use std::ops::ControlFlow;

use crate::diagram::{Tdd, TddNodeId};
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeIdx};
use crate::Engine;

mod encode;
mod probe;

pub use probe::{ProbeOrder, ProbeOutcome, ProbePoint, ProbePolicy, ProbeStats, SatOracle, SolveCall, SolveStatus, Witness};

/// The caller's solver, as an encoding uses it.
///
/// Every clause passed to [`clause`](Self::clause) belongs to the encoding
/// and must hold only while its activation literal is assumed. The sink adds
/// that gate itself, for example by appending the literal's negation to each
/// clause, so that the unit clause `-activation` later disables them all.
pub trait ClauseSink {
    /// A variable the solver has not used yet, as a literal.
    fn fresh_var(&mut self) -> i32;

    /// The solver literal that is true exactly when `var` is.
    fn leaf_literal(&mut self, var: VarId) -> i32;

    /// Add one clause of the encoding, before the sink's gate.
    fn clause(&mut self, body: &[i32]);

    /// Whether the encoding goes on at `at`. `Break` ends it there, and
    /// [`CnfEncoding::stop`] reports the point. The default always continues.
    fn poll(&mut self, at: EncodePoint) -> ControlFlow<()> {
        let _ = at;
        ControlFlow::Continue(())
    }
}

/// How an encoding defines the variables of the nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CnfScheme {
    /// Each encoded node's variable is equivalent to the node's function.
    ///
    /// The sink sees its calls in this order:
    ///
    /// 1. At the first structural leaf in
    ///    [`Vtree::leaf_bottomup`](crate::Vtree::leaf_bottomup) order, a fresh
    ///    variable `t` and the clause `[t]`. [`ClauseSink::leaf_literal`] is
    ///    asked about each structural leaf in the same order. A side that names
    ///    a leaf's constant-true node is `t`, and one that names the leaf's
    ///    variable or its negation is that literal or its negation.
    /// 2. A fresh variable for every reachable structural internal node, level
    ///    by level in [`Vtree::internal_bottomup`](crate::Vtree::internal_bottomup)
    ///    order and by index within a level.
    /// 3. In the same order, each node's definition. With `y` the node's
    ///    variable: `[-y]` if none of its pairs can be true; `y ↔ l ∧ r` if the
    ///    node stores exactly one pair; otherwise, for each pair that can be
    ///    true, a fresh variable `z` and `z ↔ l ∧ r`, then
    ///    `[-y, z₁, …, zₖ]` and `[y, -zᵢ]` for each `zᵢ`. `x ↔ l ∧ r` is
    ///    `[-x, l]`, `[-x, r]`, `[x, -l, -r]`. A side that reads as true is left
    ///    out, giving `[-x, w]`, `[x, -w]`, or `[x]` when both sides are true.
    ///
    /// A pair cannot be true when a side is `ZERO`, or a summed-out child with
    /// a count of zero. A summed-out child with a positive count reads as true.
    /// The polls come during step 3: [`EncodePoint::Level`] before each
    /// internal level, and [`EncodePoint::Node`] before the node it names.
    #[default]
    Equivalence,
}

/// A point at which an encoding asks its sink whether to go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodePoint {
    /// Before internal level `level`, when the `complete` internal levels
    /// before it in [`Vtree::internal_bottomup`](crate::Vtree::internal_bottomup)
    /// order are encoded in full. Summed-out levels count, though they hold no
    /// node to encode.
    Level {
        /// The level about to be encoded.
        level: VtreeIdx,
        /// How many internal levels are done.
        complete: usize,
    },
    /// Before stored node `slot` of structural internal level `level`,
    /// reachable or not: at slot 0 and at every 1,024th slot after it.
    Node {
        /// The level being encoded.
        level: VtreeIdx,
        /// The index of the next node in that level.
        slot: usize,
    },
}

/// The variables an encoding gave a diagram's nodes, and how far it got.
///
/// Returned by [`Tdd::encode_cnf`]. The encoding borrows nothing: it holds the
/// literal of each node, the diagram's reachability and what the sink stopped.
#[derive(Debug)]
pub struct CnfEncoding {
    activation: i32,
    /// One entry per reference slot of each level: a literal, [`encode::NONE`]
    /// or, at a summed-out slot that reads as true, [`encode::TRUE`].
    literals: Vec<Vec<i32>>,
    reachable: Vec<Vec<bool>>,
    stop: Option<EncodePoint>,
    encoded: u64,
    skipped: u64,
    erasure_certified: bool,
}

impl CnfEncoding {
    /// The literal the caller passed as the encoding's activation.
    #[must_use]
    pub fn activation(&self) -> i32 {
        self.activation
    }

    /// The solver literal equivalent to node `id`.
    ///
    /// For a reachable structural internal node before the stop point, the
    /// node's variable. The three nodes of a structural leaf have the true
    /// variable, the leaf's literal and its negation, reachable or not.
    /// `None` for any other node, including summed-out values, unreachable or
    /// deleted nodes, nodes from the stop point on, and IDs outside the
    /// diagram.
    #[must_use]
    pub fn literal(&self, id: TddNodeId) -> Option<i32> {
        let literal = *self.literals.get(id.vtree.idx())?.get(id.local.idx())?;
        (literal != encode::NONE && literal != encode::TRUE).then_some(literal)
    }

    /// Where the sink stopped the encoding, if it did.
    ///
    /// The nodes before this point are defined in full. From it on, every node
    /// is without a literal, although the variables allocated for them in
    /// step 2 of [`CnfScheme::Equivalence`] stay allocated.
    #[must_use]
    pub fn stop(&self) -> Option<EncodePoint> {
        self.stop
    }

    /// How many nodes are defined: the reachable structural internal nodes
    /// before the stop point.
    ///
    /// A stop before the first node leaves only the true variable's clause
    /// with the sink.
    #[must_use]
    pub fn encoded_nodes(&self) -> u64 {
        self.encoded
    }

    /// How many entries the stop removed: the reachable structural internal
    /// nodes from the stop point on, and the reachable summed-out values with
    /// a positive count at internal levels from that point on. Zero when the
    /// encoding ran to the end.
    #[must_use]
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Whether reading summed-out children as true kept the nodes of each
    /// level disjoint.
    ///
    /// The nodes of one structural level have disjoint functions. Once each
    /// summed-out child reads as true, two of them overlap when a pair whose
    /// sides are both summed out can be true, or when the same structural child
    /// appears, beside a summed-out side that can be true, in pairs of two
    /// different nodes of one level. This is false exactly when one of those
    /// occurs at some reachable node, encoded or not. It is true for a diagram
    /// without summed-out levels.
    #[must_use]
    pub fn erasure_certified(&self) -> bool {
        self.erasure_certified
    }

    /// Which nodes the diagram's output reaches, in the shape
    /// [`Tdd::reachable_nodes`] returns.
    #[must_use]
    pub fn reachable_nodes(&self) -> &[Vec<bool>] {
        &self.reachable
    }
}

impl Tdd {
    /// Give each reachable structural internal node a solver variable, and pass
    /// the clauses that define it to `sink`.
    ///
    /// `sink` stands for the caller's SAT solver: it allocates every variable,
    /// receives every clause, and gates each clause with `activation`, a literal
    /// the caller allocated beforehand. Assuming `activation` then makes each
    /// node's variable equivalent to its function, and the unit clause
    /// `-activation` retires the encoding; the caller adds that unit. The
    /// diagram's variables enter through [`ClauseSink::leaf_literal`]; `scheme`
    /// fixes the definitions and their order.
    ///
    /// A summed-out child with a positive count reads as true, and one with a
    /// count of zero as false. A node's variable is then equivalent to its
    /// function with the summed-out variables quantified existentially, so a
    /// node whose variable cannot be true has no model.
    /// [`CnfEncoding::erasure_certified`] says whether the nodes of each level
    /// are still disjoint under this reading.
    ///
    /// The sink is polled at every [`EncodePoint`]. `Break` ends the encoding
    /// with the nodes before that point defined in full; see
    /// [`CnfEncoding::stop`]. The diagram is borrowed and unchanged.
    ///
    /// # Errors
    ///
    /// [`OperationError::InvalidLiteral`] if `activation`, a fresh variable or
    /// a leaf literal is `0` or `i32::MIN`, and
    /// [`OperationError::MarginalLevel`] for a level that holds summed-out
    /// weights rather than counts. A refused allocation returns
    /// [`OperationError::OverBudget`]. `activation` and the levels are checked
    /// before the sink is called; after any other error, the clauses already
    /// given to the sink stay there, and the caller retires `activation`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::cnf::{ClauseSink, CnfScheme};
    /// use tididi::vtree::VarId;
    ///
    /// /// Clauses gated by appending `-activation`, over variables numbered
    /// /// after the diagram's.
    /// struct Clauses { last: i32, activation: i32, clauses: Vec<Vec<i32>> }
    ///
    /// impl ClauseSink for Clauses {
    ///     fn fresh_var(&mut self) -> i32 { self.last += 1; self.last }
    ///     fn leaf_literal(&mut self, var: VarId) -> i32 { var.0 as i32 }
    ///     fn clause(&mut self, body: &[i32]) {
    ///         self.clauses.push(body.iter().copied().chain([-self.activation]).collect());
    ///     }
    /// }
    ///
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let f = Tdd::cube(&vtree, [1, -2])?;
    /// // Variables 1 and 2 are the diagram's, and 3 is the activation literal.
    /// let mut sink = Clauses { last: 3, activation: 3, clauses: Vec::new() };
    /// let encoding = f.encode_cnf(CnfScheme::Equivalence, 3, &mut sink)?;
    /// assert_eq!(encoding.literal(f.output()), Some(5));
    /// // The true variable 4, then 5 ↔ x1 ∧ ¬x2.
    /// assert_eq!(sink.clauses, [vec![4, -3], vec![-5, 1, -3], vec![-5, -2, -3], vec![5, -1, 2, -3]]);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn encode_cnf<K: ClauseSink + ?Sized>(&self, scheme: CnfScheme, activation: i32, sink: &mut K) -> Result<CnfEncoding, OperationError> {
        self.context().run(|eng| eng.encode_cnf(self, scheme, activation, sink))
    }
}

impl Engine {
    /// Run [`Tdd::encode_cnf`] with this engine's allocation and cancellation
    /// limits. The sink's own work is not bounded by them.
    ///
    /// # Errors
    ///
    /// As [`Tdd::encode_cnf`], and [`OperationError::Stopped`] on
    /// cancellation, which can come after clauses were given to the sink.
    pub fn encode_cnf<K: ClauseSink + ?Sized>(&self, f: &Tdd, scheme: CnfScheme, activation: i32, sink: &mut K) -> Result<CnfEncoding, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        encode::check_literal(activation)?;
        if let Some(level) = f.levels.iter().position(|level| level.is_weight_marginal()) {
            return Err(OperationError::MarginalLevel(VtreeIdx(level as u32)));
        }
        match scheme {
            CnfScheme::Equivalence => encode::equivalence(self, f, activation, sink),
        }
    }
}

#[cfg(test)]
mod tests;
