//! Read-only queries on a compiled TDD: model counting, satisfiability,
//! semiring evaluation, variable support, and reduction-size metrics.
//!
//! Every query is spelled `query::name`; the submodules are an
//! implementation layout, not a namespace.

pub(crate) mod count;
pub(crate) mod sat;
pub(crate) mod semiring;
pub(crate) mod reduction;
pub(crate) mod support;

pub use count::{model_count, ColumnRetention, IncrementalPinnedCounter};
pub(crate) use count::compute_node_counts;
#[cfg(test)]
pub(crate) use count::{model_count_pinned_bigint, model_count_pinned_fix};
pub use sat::is_sat;
pub use semiring::{evaluate, RationalSemiring, Semiring, SignedLog, WeightVal};
pub use reduction::reduced_tdd_size;

pub use support::implied_literals;

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;
