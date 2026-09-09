//! Hooks for a driver that builds a diagram clause by clause.
//!
//! The level pool, clause-spine marking, the mid-compile rotation primitives
//! and the clustering that drives them, and a test threshold. None of it is
//! covered by the crate's compatibility promise, and it is hidden from the
//! documented API — a caller outside that driver wants the modules in the
//! module map instead.
//!
//! An item here is not reachable through the documented modules.
//!
//! The rotation items — [`RestructureScratch`], the two bounded relevel
//! entries, and [`rotate_left`] / [`rotate_right`] — have no caller in the
//! driver today; they are the seam the crate's own rotation tests drive, kept
//! here rather than made public because a rotation mid-build is not a
//! capability of a finished diagram.

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::diagram::pool::{return_levels, take_levels, PoolSlot};
pub use crate::restructure::relevel::{
    relevel_after_left_rotation, relevel_after_right_rotation,
};
pub use crate::restructure::scratch::RestructureScratch;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;
pub use crate::vtree::rotate::{rotate_left, rotate_right};

#[cfg(any(test, debug_assertions))]
pub use crate::diagram::marginal_ref::set_marginal_inline_max;
