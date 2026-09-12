//! Conjunction, disjunction, negation, conditioning, projection, restriction.
//!
//! Every operation here takes one or two diagrams over a shared vtree and
//! produces a new one. Summing vtree levels out into per-node values is
//! [`crate::marginal`]; making a result canonical afterwards is
//! [`crate::reduce`]; reading a finished diagram is [`crate::query`].
//!
//! Entry points. Each but negation is an [`Engine`](crate::Engine) method
//! that runs under the caller's limits, with a free function or operator
//! beside it that runs on a transient engine with nothing armed and panics
//! where the method would return an error:
//!
//! - Binary: [`Engine::and`](crate::Engine::and) and the `&` operator,
//!   [`Engine::or`](crate::Engine::or) and `|`.
//!   [`Engine::and_clause`](crate::Engine::and_clause) and [`apply_and_clause`]
//!   conjoin one clause without building it as a diagram.
//! - Unary: [`negate()`] and `!`, which have no engine form;
//!   [`condition_var`] and [`condition_vars`] fix literals; [`exists_var`]
//!   and [`exists_vars`] sum a variable out of the structure; [`restrict_to_care()`]
//!   shrinks a diagram to a region of interest.
//!
//! The `&`, `|` and `!` impls on [`Tdd`](crate::Tdd) live in `operators`, each a
//! one-line forward to the operation beside it.
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
pub(crate) mod restrict_to_care;
mod operators;

pub(crate) use conjoin::apply_and;
pub use conjoin_clause::apply_and_clause;
pub(crate) use disjoin::apply_or;
pub use negate::negate;
pub use condition::{condition_var, condition_vars};
pub use project::{exists_var, exists_vars, QuantificationStrategy};
pub use restrict_to_care::{restrict_to_care, RestrictionOutcome};

#[cfg(test)]
mod tests;
