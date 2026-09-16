//! Represent Boolean functions with Tree Decision Diagrams.
//!
//! A [`Vtree`] groups the variables and a [`Tdd`] owns a function on that tree.
//! Combine diagrams with [`and`] and [`or`], then reuse the result for
//! model counting, Boolean queries, or evaluation under different literal weights.
//!
//! # A first function
//!
//! For `(x ∧ y) ∨ z`, four assignments have `z` true and one more has both
//! `x` and `y` true while `z` is false:
//!
//! ```
//! use std::sync::Arc;
//! use tididi::{and, literal, or, Vtree};
//!
//! let vtree = Arc::new(Vtree::balanced(3));
//! let x = literal(&vtree, 1)?;
//! let y = literal(&vtree, 2)?;
//! let z = literal(&vtree, 3)?;
//! let f = or(and(x, y)?, z)?;
//! let count = u64::try_from(f.model_count()?)?;
//! assert_eq!(count, 5);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Integer literals are signed and one-based. Reuse the same `Arc<Vtree>` for
//! operands you will combine.
//! Queries borrow diagrams; transformations taking `Tdd` consume them, so clone
//! an operand first if it must be kept.
//! Operations reuse the context attached to the shared vtree. Named operations
//! such as [`and`], [`literal`] and [`Tdd::model_count`] return errors.
//! [`Context::with_limits`] lends an engine for an explicitly bounded batch. Diagrams own their results independently of that temporary checkout.
//!
//! # Where to go next
//!
//! Start with the [configuration walkthrough](guide::examples::configurations):
//! encode backup rules, count valid configurations, find one solution, and
//! narrow the choices with an observation.
//! The [worked examples](guide::examples) also explain probability queries,
//! reachable states and custom traversal in small steps.
//! Follow the [task guide](guide::api) for construction, queries, transformations,
//! probabilities, limits, and persistence. [`Tdd`] explains ownership and
//! checked operations; [`Context`] explains batches, and [`Vtree`] explains the
//! variable universe.
//!
//! The [data model](guide::model) introduces levels, pairs, and determinism.
//! For work on the implementation, use the [architecture reference](guide::architecture).

// Guards the public-release doc surface: an undocumented public item warns.
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

// An assertion that fires in `--release` too; reserve it for O(1) checks worth
// a hot-path branch. Not exported: `macro_rules!` textual scoping makes it
// visible to every module declared below.
macro_rules! cheap_assert {
    ($($arg:tt)*) => { ::std::assert!($($arg)*) };
}

// ── ground ──────────────────────────────────────────────────────────────────
// What everything else reads. A ground module names no operation; the one
// exception is the session type, which every layer may name.
pub mod vtree;      // The variable tree that shapes every diagram
pub mod diagram;    // The diagram's storage types and the traversal contract
pub mod limits;     // What an operation runs under and what it parks between calls
pub(crate) mod value;  // The value kernel: counts, the fold walk, the domains, the slot vocabulary

// ── operations ──────────────────────────────────────────────────────────────
// Operations share the storage types and may compose one another; the
// architecture reference records their responsibilities.
mod build;      // Constants and cubes
pub mod apply;      // Conjunction (of diagrams and of clauses), disjunction, negation, conditioning, projection, restriction
mod marginal;   // Summing vtree levels out into per-node counts or weights
pub mod reduce;     // Reduction to canonical form
pub mod restructure;// Rotation search and graft over a compiled diagram
pub mod query;      // Model counting, satisfiability, algebra evaluation

// ── session ─────────────────────────────────────────────────────────────────
// The hub. It holds the scratch of every operation, and every operation is a
// method on it, so it is the one module the layers point at in both directions.
pub mod engine;     // The session hub: the scratch every operation reuses

// ── edges ───────────────────────────────────────────────────────────────────
// Readers and writers of finished diagrams, and the compiled prose guides.
pub mod io;         // The `.tdd` text format, both directions, and Graphviz rendering
pub mod guide;      // The prose guides of `docs/`, compiled with the crate

#[doc = include_str!("../README.md")]
#[doc(hidden)]
pub mod readme {}

// ── seams ───────────────────────────────────────────────────────────────────
// Not a layer: the two ways into the crate from outside it. Both are hidden
// from the documented API and outside the compatibility promise.
#[doc(hidden)]
pub mod compiler_seam;  // The entry points a clause-by-clause driver uses

// Oracles, generators and invariant checkers; the checkers exist only under
// `cfg(test)` or `debug_assertions`, and `assert_canonical` is a no-op elsewhere.
#[doc(hidden)]
pub mod test_helpers;

pub use diagram::{Literal, Tdd};
pub use vtree::Vtree;
pub use limits::OperationError;
pub use engine::{Context, Engine};

pub use apply::{and, or, xor, ite, and_exists, and_exists_with_strategy};

pub use build::literal;
