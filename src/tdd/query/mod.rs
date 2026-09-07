//! Read-only queries and structural analyses on compiled TDDs.
//!
//! `query` groups the operations that inspect a finished TDD without
//! transforming it:
//! - **count** — model counting (`model_count*`, `compute_node_counts*`,
//!   `IncrementalPinnedCounter`, hybrid u128/BigUint evaluation)
//! - **sat** — structural satisfiability (`is_sat`, `output_is_satisfiable`)
//! - **semiring** — generic bottom-up evaluation parameterized by a semiring
//! - **reduction** — SDD-style reduction size metrics (r1SDD / r2TDD)
//! - **support** — variable support / implied literals / reachable-pair size

pub mod count;
pub mod sat;
pub mod semiring;
pub mod reduction;
pub mod support;

// Re-export count/sat/reduction items so callers reach all query-related
// functionality through `tdd::query::*` (the historical public surface).
pub use count::*;
pub use sat::*;
pub use reduction::*;

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;

