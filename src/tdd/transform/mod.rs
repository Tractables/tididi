//! TDD → TDD transformations.
//!
//! Operations that take one or two TDDs and produce a new TDD. Split by arity:
//!
//! - **pairwise** — two-TDD ops: conjunction (`conjoin`, `conjoin_clause`,
//!   `leaf`, `grid`) and disjunction (`disjoin`).
//! - **unary** — one-TDD transforms: `negate`, `condition`, `project`,
//!   `restrict`, `demarginalize`, `marginalize`.
//!
//! Read-only inspections (model counting, satisfiability, invariant checkers)
//! live in `tdd::query`, not here.

pub mod pairwise;
pub mod unary;
