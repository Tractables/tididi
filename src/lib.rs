//! Tree Decision Diagrams (TDDs): Boolean functions as canonical decision
//! diagrams shaped by a vtree.
//!
//! A [`tdd::Tdd`] represents a Boolean function over an `Arc<`[`vtree::Vtree`]`>`.
//! Diagrams combine by conjunction, disjunction, and negation, transform by
//! conditioning, quantification, restriction, and grafting, reduce to a
//! canonical form with `minimize`, and answer model-counting, weighted, and
//! semiring queries. The stored encoding is the public traversal contract,
//! documented in [`tdd::types`]. The crate reads no environment variables
//! and installs no process-wide state; limits and memory probes are
//! installed per thread through [`tdd::limits::apply_limits`].
//!
//! Module map:
//!
//! - [`vtree`]: the variable tree, its constructors, the `.vtree` text
//!   format, and rotations.
//! - [`tdd::types`]: the diagram's storage types and the traversal contract.
//! - [`tdd::build`]: constants and clauses.
//! - [`tdd::transform`]: pairwise conjunction and disjunction; unary
//!   negation, conditioning, projection, restriction, and marginalization.
//! - [`tdd::minimize`]: reduction to canonical form.
//! - [`tdd::restructure`]: rotation search and graft over a compiled diagram.
//! - [`tdd::query`]: model counting, satisfiability, semiring evaluation,
//!   implied literals, and size metrics.
//! - [`tdd::weight_store`]: per-node semiring values for weighted marginal
//!   levels.
//! - [`tdd::limits`]: deadlines, budgets, caps, memory probes, and meters.
//! - [`tdd::io`]: the `.tdd` text format and Graphviz rendering.
//!
//! `docs/api-guide.md` has one section per capability and `docs/tdd.md`
//! describes the data model.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use num_bigint::BigUint;
//! use tididi::tdd::Tdd;
//! use tididi::vtree::Vtree;
//!
//! // (x1 ∧ x2) ∨ x3 over a three-variable vtree; integers are DIMACS literals.
//! let vtree = Arc::new(Vtree::balanced(3));
//! let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
//! assert_eq!(f.model_count(), BigUint::from(5u32));
//! ```

// Guards the public-release doc surface: an undocumented public item warns.
#![warn(missing_docs)]

// Tier-0 invariant assertion: O(1) cost, compiled into every build including
// `--release`. Unlike `debug_assert!` this fires in optimized binaries, so
// reserve it for O(1) checks whose value justifies a hot-path branch.
//
// Crate-internal: it guards the crate's own invariants, so it is deliberately
// not `#[macro_export]`ed. `macro_rules!` textual scoping makes it visible to
// every module declared below.
macro_rules! cheap_assert {
    ($($arg:tt)*) => { ::std::assert!($($arg)*) };
}

pub mod vtree; // Variable tree (vtree): structure that governs TDD decomposition
pub mod tdd;   // Tree Decision Diagram: nodes, apply, minimize, query
