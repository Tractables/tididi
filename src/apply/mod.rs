//! Conjunction, disjunction, negation, conditioning, projection, restriction.
//!
//! Every operation here takes one or two diagrams over a shared vtree and
//! produces a new one. Summing vtree levels out into per-node values is
//! [`crate::marginal`]; making a result canonical afterwards is
//! [`crate::reduce`]; reading a finished diagram is [`crate::query`].
//!
//! Entry points, each an [`Engine`](crate::Engine) method that runs under the
//! caller's limits, most with a panicking one-line sugar beside it:
//!
//! - Binary: [`Engine::and`](crate::Engine::and) and the `&` operator,
//!   [`Engine::or`](crate::Engine::or) and `|`.
//!   [`Engine::and_clause`](crate::Engine::and_clause) conjoins one clause
//!   without building it as a diagram, and
//!   [`Engine::and_batch`](crate::Engine::and_batch) merges a small diagram into
//!   a large one by visiting only the levels it can have changed.
//! - Unary: [`negate`] and `!`; [`condition_var`] and [`condition_vars`] fix
//!   literals; [`project_var`] and [`project_vars`] sum a variable out of the
//!   structure; [`restrict`] shrinks a diagram to a region of interest.
//!
//! The submodules are an implementation layout: `conjoin` holds the compacting
//! product construction, `conjoin_clause` its diagram-by-clause form, and
//! `leaf` and `grid` the tables and descriptors both use.

pub(crate) mod conjoin;
pub(crate) mod scoped_flags;
pub(crate) mod conjoin_clause;
pub(crate) mod leaf;
mod grid;
pub(crate) mod disjoin;
pub(crate) mod negate;
pub(crate) mod condition;
pub(crate) mod project;
pub(crate) mod restrict;

pub(crate) use conjoin::apply_and;
pub use conjoin::{BatchMergeOutcome, MergeScope};
pub use conjoin_clause::apply_and_clause;
pub(crate) use disjoin::apply_or;
pub use negate::negate;
pub use condition::{condition_var, condition_vars};
pub use project::{project_var, project_vars, Projection};
pub use restrict::{restrict, CareCanonical, Restricted};

#[cfg(test)]
mod tests;
