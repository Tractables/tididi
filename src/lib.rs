//! Tree Decision Diagrams (TDDs): Boolean functions as canonical decision
//! diagrams shaped by a vtree.
//!
//! A [`Tdd`] represents a Boolean function over an `Arc<`[`vtree::Vtree`]`>`.
//! Diagrams combine by conjunction, disjunction, and negation, transform by
//! conditioning, quantification, restriction, and grafting, reduce to a
//! canonical form with `minimize`, and answer model-counting, weighted, and
//! semiring queries. The stored encoding is the public traversal contract,
//! documented in [`diagram`]. The crate reads no environment variables and
//! holds no state of its own: the limits an operation runs under and the
//! scratch it reuses live on an [`engine::Engine`] the caller owns.
//!
//! Module map:
//!
//! - [`vtree`]: the variable tree, its constructors, the `.vtree` text
//!   format, and rotations.
//! - [`diagram`]: the diagram's storage types and the traversal contract.
//! - [`build`]: constants and clauses.
//! - [`apply`]: pairwise conjunction and disjunction; unary negation,
//!   conditioning, projection, and restriction.
//! - [`marginal`]: summing vtree levels out into per-node counts or weights.
//! - [`reduce`]: reduction to canonical form.
//! - [`restructure`]: rotation search and graft over a compiled diagram.
//! - [`query`]: model counting, satisfiability, semiring evaluation,
//!   implied literals, and size metrics.
//! - [`weight_store`]: per-node semiring values for weighted marginal
//!   levels.
//! - [`engine`]: the session object — limits, memory probes, meters, and the
//!   scratch operations reuse.
//! - [`error`]: [`ApplyError`], the one error a fallible operation returns.
//! - [`io`]: reading and writing the `.tdd` text format, and Graphviz rendering.
//!
//! `docs/api-guide.md` has one section per capability and `docs/tdd.md`
//! describes the data model.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use num_bigint::BigUint;
//! use tididi::Tdd;
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

pub mod vtree;      // The variable tree that shapes every diagram
pub mod diagram;    // The diagram's storage types and the traversal contract
pub mod build;      // Constants and clauses
pub mod apply;      // Conjunction, disjunction, negation, conditioning, projection, restriction
pub mod marginal;   // Summing vtree levels out into per-node counts or weights
pub mod reduce;     // Reduction to canonical form
pub mod restructure;// Rotation search and graft over a compiled diagram
pub mod query;      // Model counting, satisfiability, algebra evaluation, size metrics
pub mod io;         // The `.tdd` text format, both directions, and Graphviz rendering
pub mod engine;     // The session object: limits, memory probes, meters, scratch
pub mod error;      // ApplyError
pub mod weight_store; // Per-node semiring values for weighted marginal levels
pub mod ops;        // Operator sugar for diagrams
#[doc(hidden)]
pub mod check;      // Invariant checkers
#[doc(hidden)]
pub mod internals;  // The seam the CNF compiler compiles against

pub(crate) mod counts;
pub(crate) mod marg_slots;
pub(crate) mod utils;
pub(crate) mod scoped;

pub use diagram::{Literal, Tdd};
pub use vtree::Vtree;
pub use error::ApplyError;
pub use engine::Engine;
pub use apply::negate;

#[cfg(test)]
pub(crate) mod test_helpers;
