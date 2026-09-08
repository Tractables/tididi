//! Read-only queries on a compiled TDD: model counting, satisfiability,
//! semiring evaluation, variable support, and reduction-size metrics.
//!
//! Every query is spelled `query::name`; the submodules are an
//! implementation layout, not a namespace.

pub(crate) mod count;
pub(crate) mod fold;
pub(crate) mod sat;
pub(crate) mod semiring;
pub(crate) mod reduction;
pub(crate) mod support;

pub use count::{model_count, ColumnRetention, IncrementalPinnedCounter, SeedConvention};
pub(crate) use count::compute_node_counts;
#[cfg(test)]
pub(crate) use count::pinned_counts;
pub use sat::{is_sat_minimized, is_sat_structural};
pub use semiring::{evaluate, RationalWeights, EvalAlgebra, SignedLog, WeightVal};
pub use reduction::{reduced_size, ReductionRule};

pub use support::implied_literals;

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "traverse_tests.rs"]
mod traverse_tests;
