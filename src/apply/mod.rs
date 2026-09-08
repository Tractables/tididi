//! Operations that take one or two diagrams and produce a new diagram.
//!
//! - [`conjoin`] — conjunction of two diagrams, by the compacting product
//!   construction; [`conjoin_clause`] is the specialized diagram-by-clause form,
//!   and [`leaf`] and [`grid`] are the tables and descriptors both use.
//! - [`disjoin`] — disjunction, and [`negate`] — complement.
//! - [`condition`], [`project`], [`restrict`] — the unary applies: fixing a
//!   literal, summing a variable out of the structure, and restricting a
//!   diagram to a region of interest.
//!
//! Marginalization — summing vtree levels out into per-node counts or weights —
//! is its own module, [`crate::marginal`]. Read-only inspection lives in
//! [`crate::query`].

pub(crate) mod conjoin;
pub(crate) mod conjoin_clause;
pub(crate) mod leaf;
mod grid;
pub(crate) mod disjoin;
pub(crate) mod negate;
pub(crate) mod condition;
pub(crate) mod project;
pub(crate) mod restrict;

pub use conjoin::{apply_and, BatchMerge, RebuiltMax, Spine};
pub use conjoin_clause::apply_and_clause;
pub use disjoin::apply_or;
pub use negate::negate;
pub use condition::{condition_var, condition_vars, Polarity};
pub use project::{project_var, project_vars};
pub use restrict::{restrict, CareCanonical, Restricted};

#[cfg(test)]
#[path = "unary_tests.rs"]
mod unary_tests;
