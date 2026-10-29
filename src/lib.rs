// scenario: docs/scenarios.md#first-circuit

//! Tree Decision Diagrams for Rust.
//!
//! tididi represents Boolean functions as circuits called Tree Decision
//! Diagrams (TDDs), so you can work with their satisfying assignments without
//! listing them. Build a circuit once, then combine it with other circuits or
//! query it as your inputs change. Like binary decision diagrams, minimized
//! TDDs are canonical and combine efficiently under Boolean operations; in
//! practice they can be much smaller. The same library is available from
//! [Python](https://tractables.github.io/tididi/python/) and
//! [C](https://tractables.github.io/tididi/c/).
//!
//! # A first function
//!
//! This example builds `(x ∧ y) ∨ z` and counts its satisfying assignments:
//!
//! ```
//! use std::sync::Arc;
//! use tididi::{literal, Vtree};
//!
//! let vtree = Arc::new(Vtree::balanced(3));
//! let x = literal(&vtree, 1)?;
//! let y = literal(&vtree, 2)?;
//! let z = literal(&vtree, 3)?;
//! let f = (x & y) | z;
//! println!("Satisfying assignments: {}", f.model_count()?);
//! # assert_eq!(f.model_count()?, 5u32.into());
//! # Ok::<(), tididi::OperationError>(())
//! ```
//!
//! Output:
//!
//! ```text
//! Satisfying assignments: 5
//! ```
//!
//! Four assignments have `z` true, and one more has `z` false with `x` and `y`
//! true.
//!
//! A [`Vtree`] is a binary tree that groups the variables; diagrams you
//! combine must share the same `Arc<Vtree>`. Literals are signed variable
//! numbers: `1` is `x` and `-2` is `¬y`. Boolean operations consume their
//! operands, so clone an operand first to keep it; queries such as
//! [`Tdd::model_count`] borrow the diagram.
//!
//! # Worked examples
//!
//! The [worked examples](guide::examples) build up from basic circuit
//! operations to custom calculations. Each solves one problem, shows its
//! output and links a program you can run:
//!
//! 1. [Configurations](guide::examples::configurations) — Build rules, count solutions, and update a user's choices.
//! 2. [What are we counting?](guide::examples::counting) — Distinguish observations, substitution, and projection.
//! 3. [Probabilities](guide::examples::probability) — Evaluate events under changing probabilities.
//! 4. [Reachability](guide::examples::reachability) — Explore the states reachable through a transition system.
//! 5. [Tables and updates](guide::examples::tables) — Build a circuit from rows and maintain it as the table changes.
//! 6. [Saving and loading](guide::examples::persistence) — Save circuits and restore them for later use.
//! 7. [Variable grouping](guide::examples::vtrees) — Compare circuit sizes under different vtrees.
//! 8. [Reusable components](guide::examples::composition) — Place a compiled circuit in a larger model and connect its copies.
//! 9. [Simplify under assumptions](guide::examples::care) — Reduce a circuit where a care set is guaranteed to hold.
//! 10. [Execution limits](guide::examples::execution) — Bound work and release retained scratch buffers.
//! 11. [Minimum costs](guide::examples::optimization) — Define a custom evaluation algebra.
//! 12. [Keep values instead of structure](guide::examples::marginalization) — Preserve counts or fixed weighted values while discarding a subtree.
//! 13. [Circuit statistics](guide::examples::statistics) — Traverse nodes and pairs for a custom measurement.
//!
//! # Reference
//!
//! The [API overview](guide::api) finds the operation for a task, and each
//! item below gives its contract. The [data model](guide::model) explains the
//! representation; contributors can start with the
//! [architecture reference](guide::architecture).

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
mod sort;
mod isa;
pub(crate) mod skip;

mod build;
pub mod apply;
pub mod maintain;
mod marginal;
pub mod reduce;
pub mod restructure;
pub mod query;
pub mod cnf;
pub mod execution;
pub mod io;
pub mod guide;

// Invariant checkers, generators and oracles. The checkers compile in a debug
// build regardless of the feature, because the `debug_assertions` sites in
// `reduce` call them; the rest needs `testing`.
// A consumer's release build has none of it.
#[cfg(any(test, debug_assertions, feature = "testing"))]
#[doc(hidden)]
pub mod test_helpers;

pub use diagram::{Literal, LiteralInput, Tdd};
pub use skip::api::SkipTdd;
pub use vtree::Vtree;
pub use limits::OperationError;
pub use execution::{Context, Engine};

pub use apply::{add, and, and_not, nor_many, or, or_many, xor, ite, and_exists, Quantification};

pub use build::literal;
