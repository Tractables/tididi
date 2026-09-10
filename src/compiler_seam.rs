//! Hooks for a driver that builds a diagram clause by clause.
//!
//! Clause-spine marking, and the clustering a mid-compile rotation drives.
//! Neither is covered by the crate's compatibility promise, and both are hidden
//! from the documented API — a caller outside that driver wants the modules in
//! the module map instead.
//!
//! An item here is not reachable through the documented modules.

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;
