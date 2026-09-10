//! Clause-spine marking and mid-compile clustering, for a driver that builds a
//! diagram clause by clause.
//!
//! These are hooks, not operations: the operations they steer are
//! [`crate::apply`] and [`crate::restructure`], and a caller that is not that
//! driver wants those modules instead.
//!
//! Entry points: the spine marking a clause conjunction is planned from, and the
//! clustering a rotation mid-compile drives. Neither is covered by the crate's
//! compatibility promise, both are hidden from the documented API, and no item
//! here is reachable through the documented modules.

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;
