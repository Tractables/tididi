//! Tree Decision Diagrams (TDDs): Boolean functions as canonical decision
//! diagrams shaped by a vtree.
//!
//! A [`Tdd`] represents a Boolean function over an `Arc<`[`vtree::Vtree`]`>`.
//! Diagrams combine by conjunction, disjunction, and negation, transform by
//! conditioning, quantification, restriction, and grafting, reduce to a
//! canonical form with [`minimize`](reduce::minimize), and answer model-counting, weighted, and
//! algebraic queries. The stored encoding is the public traversal contract,
//! documented in [`diagram`]. The crate reads no environment variables and
//! holds no state of its own: the limits an operation runs under and the
//! scratch it reuses live on an [`engine::Engine`] the caller owns.
//!
//! Module map. The modules form four layers, and `lib.rs` declares them in
//! this order under a banner per layer.
//!
//! Ground — what every other module reads:
//!
//! - [`vtree`]: the variable tree, its constructors, the `.vtree` text
//!   format, and rotations.
//! - [`diagram`]: the diagram's storage types, the traversal contract, and
//!   the algebra a marginal level's values are drawn from.
//! - [`limits`]: what an operation runs under — the budget, the caps, the stop
//!   axis, the meters — and [`ApplyError`], the one error a fallible operation
//!   returns.
//!
//! Operations — the verbs, each of them a method on the session:
//!
//! - [`build`]: constants and clauses.
//! - [`apply`]: pairwise conjunction and disjunction; unary negation,
//!   conditioning, projection, and restriction, and the `&`, `|`, `!` impls.
//! - [`marginal`]: summing vtree levels out into per-node counts or weights.
//! - [`reduce`]: reduction to canonical form.
//! - [`restructure`]: rotation search and graft over a compiled diagram.
//! - [`query`]: model counting, satisfiability, algebra evaluation, a weighted
//!   diagram's value, implied literals, and size metrics.
//!
//! Session — the hub every operation hangs its methods on:
//!
//! - [`engine`]: the session object — the scratch operations reuse and the
//!   limits armed on them.
//!
//! Edges — reading a finished diagram:
//!
//! - [`io`]: reading and writing the `.tdd` text format, and Graphviz rendering.
//!
//! Three modules are hidden from this reference. `check` holds the invariant
//! checkers and is compiled only under `cfg(test)` or `debug_assertions`, since
//! every checker walks the whole diagram. `compiler_seam` holds every entry
//! point a clause-by-clause driver reaches the crate through. `test_helpers`
//! holds the generators and oracles the crate's own tests decide a diagram by.
//! The last two are seams rather than layers, and neither is covered by the
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
pub mod build;      // Constants and clauses
pub mod apply;      // Conjunction, disjunction, negation, conditioning, projection, restriction
pub mod marginal;   // Summing vtree levels out into per-node counts or weights
pub mod reduce;     // Reduction to canonical form
pub mod restructure;// Rotation search and graft over a compiled diagram
pub mod query;      // Model counting, satisfiability, algebra evaluation, size metrics

// ── session ─────────────────────────────────────────────────────────────────
// The hub. It holds the scratch of every operation, and every operation is a
// method on it, so it is the one module the layers point at in both directions.
pub mod engine;     // The session hub: the scratch every operation reuses

// ── edges ───────────────────────────────────────────────────────────────────
// Readers of a finished diagram, and the prose that ships compiled with the
// crate. Nothing below this band is named by anything above it.
pub mod io;         // The `.tdd` text format, both directions, and Graphviz rendering
#[cfg(any(test, debug_assertions))]
#[doc(hidden)]
pub mod check;      // The invariant checkers, one per numbered invariant
pub mod guide;      // The prose guides of `docs/`, compiled with the crate

#[doc = include_str!("../README.md")]
#[doc(hidden)]
pub mod readme {}

// ── seams ───────────────────────────────────────────────────────────────────
// Not a layer: the two ways into the crate from outside it. Both are hidden
// from the documented API and outside the compatibility promise.
#[doc(hidden)]
pub mod compiler_seam;  // The entry points a clause-by-clause driver uses

// The oracles and generators the crate's own tests run on, published so the
// randomized differential suite in `tests/` reaches the same ones rather than
// growing a second copy. The members that read `check` are compiled only where
// `check` is; `assert_canonical` degrades to a no-op elsewhere, which is why
// the differential suite is run in both configurations.
#[doc(hidden)]
pub mod test_helpers;

pub use diagram::{Literal, Tdd};
pub use vtree::Vtree;
pub use limits::ApplyError;
pub use engine::Engine;
