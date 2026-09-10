//! Vtree-rotation search over a compiled diagram.
//!
//! Two entries, both running the same probe — rotate, guard, rebuild the two
//! affected levels under a bound, re-minimize, keep or restore:
//! - the public objective-generic greedy search (`local`) — the clean library
//!   form of "improve a compiled diagram's vtree by rotating it";
//! - the mid-compile marginal-clustering pass (`cluster`) — a size-driven
//!   specialization for a diagram still being built, which regroups two
//!   already-marginal levels under one parent so `marginalize_closure` can
//!   collapse a whole structural level out of it.
//!
//! Module map:
//! - `core`    — the shared rotation probe and what it is built from:
//!   rotation-kind dispatch, the per-level size helper, the marginal-level
//!   guard, and the subtree allow-mask.
//! - `local`   — the public greedy [`rotation_search`] / [`search_to_local_min`]
//!   and the [`RotationObjective`] trait.
//! - `cluster` — the mid-compile marginal-clustering pass.

pub(crate) mod cluster;
mod core;
mod local;

pub use local::{
    rotation_search, search_to_local_min, RotationObjective, RotationSearchConfig,
    RotationSearchStats,
};

pub(crate) use local::rotation_search_on;

use crate::diagram::Tdd;
use crate::error::ApplyError;

/// The rotation-search entry point on a caller's engine.
impl crate::engine::Engine {
    /// Descend the diagram's vtree greedily by rotation, keeping every probe
    /// the objective scores as an improvement.
    ///
    /// Rotations are pure variable reorders, so the model count is preserved
    /// under any objective. The armed stop is polled once per pivot, which is
    /// what lets a caller bound a search that would otherwise run to a local
    /// minimum.
    ///
    /// # Errors
    ///
    /// [`ApplyError::Deadline`] when the armed deadline passes or a stop
    /// decision concludes the search should end. The diagram is left canonical
    /// and count-correct at whatever local point the search had reached.
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, ApplyError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }
}
