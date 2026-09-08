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

pub mod conjoin;
pub mod conjoin_clause;
pub mod leaf;
mod grid;
pub mod disjoin;
pub mod negate;
pub mod condition;
pub mod project;
pub mod restrict;

#[cfg(test)]
#[path = "unary_tests.rs"]
mod unary_tests;
