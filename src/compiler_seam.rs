//! Hooks for a driver that builds a diagram clause by clause.
//!
//! Clause-spine marking, the mid-compile rotation primitives and the
//! clustering that drives them. None of it is covered by the crate's
//! compatibility promise, and it is hidden from the documented API — a caller
//! outside that driver wants the modules in the module map instead.
//!
//! An item here is not reachable through the documented modules.
//!
//! The rotation items — [`RestructureScratch`], [`seat_rotated_vtree`], the
//! two bounded relevel entries, and [`rotate_left`] / [`rotate_right`] — have
//! no caller in the driver today; they are the seam the crate's own rotation
//! tests drive, kept here rather than made public because a rotation mid-build
//! is not a capability of a finished diagram. [`seat_rotated_vtree`] in
//! particular is the only way in to a diagram's vtree that does not preserve
//! the tree, and it is sound only paired with the relevel that follows it.

pub use crate::apply::conjoin_clause::mark_clause_levels;
pub use crate::restructure::relevel::{
    relevel_after_left_rotation, relevel_after_right_rotation, seat_rotated_vtree,
};
pub use crate::restructure::scratch::RestructureScratch;
pub use crate::restructure::search::cluster::rotate_marginal_cluster;
pub use crate::vtree::rotate::{rotate_left, rotate_right};
