//! Clauses kept in memory, with satisfiability decided by exhaustive search,
//! for testing CNF encodings and the probe without a solver.

use std::collections::BTreeSet;

use crate::cnf::{ClauseSink, SatOracle, SolveCall, SolveStatus};
use crate::vtree::VarId;

/// A clause list over numbered variables that is also a [`ClauseSink`] and a
/// [`SatOracle`].
///
/// Variables `1..=n` of [`new(n)`](Self::new) are the caller's, and a diagram's
/// variable `VarId(v)` is the literal `v`. While an activation literal is
/// active, every clause the sink receives gets its negation appended, as a
/// solver binding gates an encoding. [`satisfiable`](Self::satisfiable) and
/// [`SatOracle::solve`] are exact and exponential in the worst case: small
/// instances only. A solve ignores its [`SolveCall`] unless
/// [`answer_unknown`](Self::answer_unknown) picked it.
#[derive(Clone, Debug, Default)]
pub struct ClauseStore {
    vars: u32,
    gate: Option<i32>,
    clauses: Vec<Vec<i32>>,
    /// The model the last solve found, indexed by variable.
    last: Option<Vec<bool>>,
    /// How many solves ran.
    solves: usize,
    /// The solves, by number, that answer Unknown.
    unknown: BTreeSet<usize>,
}

impl ClauseStore {
    /// A store without clauses whose first `vars` variables are taken.
    pub fn new(vars: u32) -> Self {
        ClauseStore { vars, ..ClauseStore::default() }
    }

    /// Allocate an activation literal and gate every later sink clause with it.
    pub fn activate(&mut self) -> i32 {
        let activation = self.fresh_var();
        self.gate = Some(activation);
        activation
    }

    /// Add the unit clause that disables the gated clauses, and stop gating.
    pub fn retire(&mut self) {
        if let Some(activation) = self.gate.take() { self.clauses.push(vec![-activation]); }
    }

    /// Add `literals` as a clause, without the gate.
    pub fn add_clause(&mut self, literals: &[i32]) {
        self.clauses.push(literals.to_vec());
    }

    /// The clauses in the order they were added, gates included.
    pub fn clauses(&self) -> &[Vec<i32>] {
        &self.clauses
    }

    /// The number of variables allocated so far.
    pub fn num_vars(&self) -> u32 {
        self.vars
    }

    /// Make solve number `solve`, counting this store's solves from zero,
    /// answer [`SolveStatus::Unknown`].
    pub fn answer_unknown(&mut self, solve: usize) {
        self.unknown.insert(solve);
    }

    /// How many solves have run.
    pub fn solves(&self) -> usize {
        self.solves
    }

    /// Whether some assignment satisfies every clause and every literal of
    /// `assumptions`.
    ///
    /// # Panics
    ///
    /// Panics if a literal names a variable the store has not allocated.
    pub fn satisfiable(&self, assumptions: &[i32]) -> bool {
        self.model(assumptions).is_some()
    }

    /// An assignment, indexed by variable, that satisfies every clause and
    /// every literal of `assumptions`, if there is one.
    ///
    /// Backtracks over the variables with unit propagation, trying false and
    /// then true for each variable it branches on, so the answer is exact and
    /// the same on every call. A variable the search leaves open is false.
    ///
    /// # Panics
    ///
    /// Panics if a literal names a variable the store has not allocated.
    pub fn model(&self, assumptions: &[i32]) -> Option<Vec<bool>> {
        let mut values = vec![0i8; self.vars as usize + 1];
        for &literal in assumptions.iter().chain(self.clauses.iter().flatten()) {
            assert!(literal != 0 && literal.unsigned_abs() <= self.vars, "literal {literal} is not over the store's variables");
        }
        for &literal in assumptions {
            if value(&values, literal) == -1 { return None; }
            assign(&mut values, literal);
        }
        search(&self.clauses, values).map(|values| values.iter().map(|&v| v == 1).collect())
    }
}

impl ClauseSink for ClauseStore {
    fn fresh_var(&mut self) -> i32 {
        self.vars += 1;
        i32::try_from(self.vars).expect("the store numbers variables as positive literals")
    }

    fn leaf_literal(&mut self, var: VarId) -> i32 {
        i32::try_from(var.0).expect("the store numbers variables as positive literals")
    }

    fn clause(&mut self, body: &[i32]) {
        self.clauses.push(body.iter().copied().chain(self.gate.map(|activation| -activation)).collect());
    }
}

impl SatOracle for ClauseStore {
    fn add_clause(&mut self, clause: &[i32]) {
        ClauseStore::add_clause(self, clause);
    }

    fn solve(&mut self, assumptions: &[i32], _call: SolveCall) -> SolveStatus {
        let solve = self.solves;
        self.solves += 1;
        self.last = None;
        if self.unknown.contains(&solve) { return SolveStatus::Unknown; }
        self.last = self.model(assumptions);
        if self.last.is_some() { SolveStatus::Sat } else { SolveStatus::Unsat }
    }

    fn value(&mut self, literal: i32) -> bool {
        let model = self.last.as_ref().expect("a value is read only after a Sat answer");
        model[literal.unsigned_abs() as usize] == (literal > 0)
    }
}

/// `1` if `literal` is true under `values`, `-1` if false, `0` if unassigned.
fn value(values: &[i8], literal: i32) -> i8 {
    let v = values[literal.unsigned_abs() as usize];
    if literal > 0 { v } else { -v }
}

fn assign(values: &mut [i8], literal: i32) {
    values[literal.unsigned_abs() as usize] = if literal > 0 { 1 } else { -1 };
}

/// Propagate units to a fixed point, then branch on an unassigned variable of
/// the first clause not yet satisfied, false first. The satisfying values, if
/// any, with 0 for the variables left open.
fn search(clauses: &[Vec<i32>], mut values: Vec<i8>) -> Option<Vec<i8>> {
    loop {
        let mut assigned = false;
        for clause in clauses {
            let mut open = None;
            let mut opens = 0;
            if clause.iter().any(|&literal| value(&values, literal) == 1) { continue; }
            for &literal in clause {
                if value(&values, literal) == 0 {
                    opens += 1;
                    open = Some(literal);
                }
            }
            match (opens, open) {
                (0, _) => return None,
                (1, Some(literal)) => {
                    assign(&mut values, literal);
                    assigned = true;
                }
                _ => {}
            }
        }
        if !assigned { break; }
    }
    let branch = clauses.iter()
        .find(|clause| !clause.iter().any(|&literal| value(&values, literal) == 1))
        .and_then(|clause| clause.iter().copied().find(|&literal| value(&values, literal) == 0));
    let Some(literal) = branch else { return Some(values) };
    let var = literal.abs();
    [-var, var].into_iter().find_map(|choice| {
        let mut next = values.clone();
        assign(&mut next, choice);
        search(clauses, next)
    })
}
