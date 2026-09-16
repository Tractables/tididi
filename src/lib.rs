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

pub mod vtree;
pub mod diagram;
pub mod limits;
pub(crate) mod value;

mod build;
pub mod apply;
mod marginal;
pub mod reduce;
pub mod restructure;
pub mod query;
pub mod execution;
pub mod io;
pub mod guide;

#[doc = include_str!("../README.md")]
#[doc(hidden)]
pub mod readme {}

// Oracles, generators and invariant checkers; the checkers exist only under
// `cfg(test)` or `debug_assertions`, and `assert_canonical` is a no-op elsewhere.
#[doc(hidden)]
pub mod test_helpers;

pub use diagram::{Literal, Tdd};
pub use vtree::Vtree;
pub use limits::OperationError;
pub use execution::{Context, Engine};

pub use apply::{and, or, xor, ite, and_exists, and_exists_with_strategy};

pub use build::literal;
