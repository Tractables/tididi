//! Build and query Boolean functions with Tree Decision Diagrams (TDDs).
//!
//! A [`Tdd`] owns its diagram and shares a [`Vtree`] describing its variables.
//! An [`Engine`] provides checked operations, reusable scratch and resource limits.
//!
//! # Build a Boolean function
//!
//! ```
//! use std::sync::Arc;
//! use tididi::{Engine, Vtree};
//!
//! let engine = Engine::new();
//! let tree = Arc::new(Vtree::balanced(3));
//! let x = engine.literal(&tree, 1)?;
//! let y = engine.literal(&tree, 2)?;
//! let z = engine.literal(&tree, 3)?;
//! let f = engine.ite(x, y, z)?; // if x then y, otherwise z
//! assert_eq!(engine.model_count(&f)?, 4u32.into());
//! # Ok::<(), tididi::OperationError>(())
//! ```
//!
//! # Choose your next task
//!
//! The [task guide](guide::api) links examples for building functions, finding
//! assignments, comparing functions, changing variables, evaluating probabilities,
//! limiting resource use and saving diagrams.
//!
//! Keep a diagram for several operations: [`Tdd`].
//! Minimize under its current vtree: [`minimize`](reduce::minimize).
//! Understand vtrees and determinism: the [data model](guide::model).
//! Extend the implementation: the [architecture reference](guide::architecture).
//!
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
// The verbs. Each reads the ground layer, and where one names another it does
// so in a single direction, so no two operations depend on each other.
pub mod build;      // Constants and cubes
pub mod apply;      // Conjunction (of diagrams and of clauses), disjunction, negation, conditioning, projection, restriction
pub mod marginal;   // Summing vtree levels out into per-node counts or weights
pub mod reduce;     // Reduction to canonical form
pub mod restructure;// Rotation search and graft over a compiled diagram
pub mod query;      // Model counting, satisfiability, algebra evaluation

// ── session ─────────────────────────────────────────────────────────────────
// The hub. It holds the scratch of every operation, and every operation is a
// method on it, so it is the one module the layers point at in both directions.
pub mod engine;     // The session hub: the scratch every operation reuses

// ── edges ───────────────────────────────────────────────────────────────────
// Readers of a finished diagram, and the prose that ships compiled with the
// crate. Nothing below this band is named by anything above it.
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
pub use engine::Engine;
