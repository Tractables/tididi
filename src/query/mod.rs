//! Model counting, satisfiability, algebra evaluation, size metrics.
//!
//! Everything here reads a finished diagram and returns a number, a verdict or a
//! set; nothing mutates. Building and transforming a diagram is [`crate::apply`],
//! [`crate::marginal`] and [`crate::reduce`].
//!
//! Entry points:
//!
//! - Counting: [`Tdd::model_count`](crate::Tdd::model_count), and
//!   [`IncrementalCounter`] for a count under a partial assignment that updates
//!   when pins change. [`Engine::model_count`](crate::Engine::model_count) is the
//!   same count under the caller's limits.
//! - Satisfiability and support: [`is_sat_minimized`], [`implied_literals`].
//! - Algebra: [`evaluate`] folds any
//!   [`EvalAlgebra`](crate::diagram::EvalAlgebra) bottom-up;
//!   [`RationalWeights`](crate::diagram::RationalWeights) and
//!   [`SignedLog`](crate::diagram::SignedLog) are the two supplied domains.
//! - Size: [`Tdd::size`](crate::Tdd::size) and its neighbours on the diagram,
//!   and [`reduced_size`] for the size a non-smooth reduction would reach.
//!
//! Every query is spelled `query::name`; the submodules are an
//! implementation layout, not a namespace.

pub(crate) mod count;
pub(crate) mod fold;
pub(crate) mod sat;
pub(crate) mod semiring;
pub(crate) mod reduction;
pub(crate) mod support;

pub use count::{
    node_counts, node_counts_u128, KeepAllColumns, ColumnRetention, Evaluated, CounterState,
    Unevaluated, KeepFrontier, IncrementalCounter, Retention, SeedConvention,
};
pub(crate) use count::model_count;
#[cfg(test)]
pub(crate) use count::pinned_counts;
pub use sat::is_sat_minimized;
pub use semiring::evaluate;
pub use reduction::{reduced_size, ReductionRule};

pub use support::implied_literals;

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "traverse_tests.rs"]
mod traverse_tests;
