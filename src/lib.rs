//! Tree Decision Diagrams (diagrams): Boolean functions as canonical decision
//! diagrams shaped by a vtree.
//!
//! A [`Tdd`] represents a Boolean function over an `Arc<`[`vtree::Vtree`]`>`.
//! Diagrams combine by conjunction, disjunction, and negation, transform by
//! conditioning, quantification, restriction, and grafting, reduce to a
//! canonical form with `minimize`, and answer model-counting, weighted, and
//! algebraic queries. The stored encoding is the public traversal contract,
//! documented in [`diagram`]. The crate reads no environment variables and
//! holds no state of its own: the limits an operation runs under and the
//! scratch it reuses live on an [`engine::Engine`] the caller owns.
//!
//! Module map:
//!
//! - [`vtree`]: the variable tree, its constructors, the `.vtree` text
//!   format, and rotations.
//! - [`diagram`]: the diagram's storage types, the traversal contract, and
//!   the algebra a marginal level's values are drawn from.
//! - [`build`]: constants and clauses.
//! - [`apply`]: pairwise conjunction and disjunction; unary negation,
//!   conditioning, projection, and restriction, and the `&`, `|`, `!` impls.
//! - [`marginal`]: summing vtree levels out into per-node counts or weights.
//! - [`reduce`]: reduction to canonical form.
//! - [`restructure`]: rotation search and graft over a compiled diagram.
//! - [`query`]: model counting, satisfiability, algebra evaluation, a weighted
//!   diagram's value, implied literals, and size metrics.
//! - [`engine`]: the session object — limits, memory probes, meters, and the
//!   scratch operations reuse.
//! - [`error`]: [`ApplyError`], the one error a fallible operation returns.
//! - [`io`]: reading and writing the `.tdd` text format, and Graphviz rendering.
//!
//! Two modules are hidden from this reference. `check` holds the invariant
//! checkers and is compiled only under `cfg(test)` or `debug_assertions`, since
//! every checker walks the whole diagram; `compiler_seam` holds every entry
//! point a clause-by-clause driver reaches the crate through, outside the
//! compatibility promise.
//!
//! `docs/architecture.md` states the model, the numbered invariants, and the
//! boundary — one row per module, saying what it owns and what it may not
//! touch. Each module's own documentation repeats neither. `docs/api-guide.md`
//! has one section per capability and `docs/tdd.md` describes the data model.
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
#![warn(missing_debug_implementations)]

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
pub(crate) mod value;  // The value kernel: counts, the fold walk, the domains, the slot vocabulary
pub mod build;      // Constants and clauses
pub mod apply;      // Conjunction, disjunction, negation, conditioning, projection, restriction
pub mod marginal;   // Summing vtree levels out into per-node counts or weights
pub mod reduce;     // Reduction to canonical form
pub mod restructure;// Rotation search and graft over a compiled diagram
pub mod query;      // Model counting, satisfiability, algebra evaluation, size metrics
pub mod io;         // The `.tdd` text format, both directions, and Graphviz rendering
pub mod engine;     // The session object: limits, memory probes, meters, scratch
pub mod error;      // ApplyError
#[cfg(any(test, debug_assertions))]
#[doc(hidden)]
pub mod check;
#[doc(hidden)]
pub mod compiler_seam;  // The entry points a clause-by-clause driver uses

pub mod guide;      // The prose guides of `docs/`, compiled with the crate

#[doc = include_str!("../README.md")]
#[doc(hidden)]
pub mod readme {}

pub use diagram::{Literal, Tdd};
pub use vtree::Vtree;
pub use error::ApplyError;
pub use engine::Engine;

// The oracles and generators the crate's own tests run on, published so the
// randomized differential suite in `tests/` reaches the same ones rather than
// growing a second copy. Hidden from the documented API and outside the
// compatibility promise. The members that read `check` are compiled only where
// `check` is; `assert_canonical` degrades to a no-op elsewhere, which is why
// the differential suite is run in both configurations.
#[doc(hidden)]
pub mod test_helpers;

/// Declared last: the one struct that names every module's scratch.
mod session;
