//! Represent Boolean functions with Tree Decision Diagrams.
//!
//! A [`Vtree`] groups the variables and a [`Tdd`] owns a function on that vtree.
//! Combine diagrams with [`and`] and [`or`], then reuse the result for
//! model counting, Boolean queries, or evaluation under different literal weights.
//! For a fixed vtree, a minimized diagram is canonical: equivalent functions
//! have the same diagram.
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
//! # Applications
//!
//! The [configuration example](guide::examples::configurations) builds constraints,
//! counts solutions, and narrows the choices after an observation.
//! The [reachability example](guide::examples::reachability) combines Boolean
//! operations, quantification, and renaming to explore a transition system.
//! The [probability example](guide::examples::probability) evaluates events under
//! changing weights to answer conditional-probability queries.
//!
//! The [API overview](guide::api) groups circuit operations and links to their
//! specifications. Further [worked examples](guide::examples) cover persistence,
//! execution limits, variable grouping, and custom traversal.
//! The [data model](guide::model) explains the representation, and the
//! [architecture reference](guide::architecture) describes the implementation.

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
pub mod maintain;
mod marginal;
pub mod reduce;
pub mod restructure;
pub mod query;
pub mod execution;
pub mod io;
pub mod guide;

// Oracles, generators and invariant checkers. Compiled when the `testing`
// feature is on, and in a debug build regardless, because the `debug_assert`
// sites in `reduce`, `marginal` and `restructure` call the checkers. A
// consumer's release build has none of it.
#[cfg(any(test, debug_assertions, feature = "testing"))]
#[doc(hidden)]
pub mod test_helpers;

pub use diagram::{Literal, LiteralInput, Tdd};
pub use vtree::Vtree;
pub use limits::OperationError;
pub use execution::{Context, Engine};

pub use apply::{and, or, xor, ite, and_exists, Quantification};

pub use build::literal;
