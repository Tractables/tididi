//! Vtree-rotation search over a compiled TDD.
//!
//! Two entries, both built on the same rotate → restructure → minimize
//! primitives:
//! - the public objective-generic greedy search (`local`) — the clean library
//!   form of "improve a compiled diagram's vtree by rotating it";
//! - the mid-compile marginal-clustering pass (`cluster`) — a size-driven
//!   specialization for a diagram still being built, which regroups two
//!   already-marginal levels under one parent so `marginalize_closure` can
//!   collapse a whole structural level out of it.
//!
//! Module map:
//! - `core`    — rotation-kind dispatch, per-level size helper, marginal-level
//!               guard, subtree allow-mask (shared by both entries).
//! - `local`   — the public greedy [`rotation_search`] / [`search_to_local_min`]
//!               and the [`RotationObjective`] trait.
//! - `cluster` — the mid-compile marginal-clustering pass.

pub(crate) mod cluster;
mod core;
mod local;

pub use local::{
    rotation_search, search_to_local_min, RotationObjective, RotationSearchConfig,
    RotationSearchStats, SizeDelta,
};

pub(crate) use local::rotation_search_on;
