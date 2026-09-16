//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`]. [`Tdd::graft`](crate::Tdd::graft)
//! joins diagrams over disjoint variables on a grafted vtree.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod graft;

pub use graft::GraftError;
