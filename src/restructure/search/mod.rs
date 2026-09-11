//! Vtree-rotation search over a compiled diagram.
//!
//! Two entries, both running the same probe — rotate, guard, rebuild the two
//! affected levels under a bound, re-minimize, keep or restore:
//! - the public objective-generic greedy search (`local`);
//! - the mid-compile marginal-clustering pass (`cluster`) — a size-driven
//!   specialization for a diagram still being built, which regroups two
//!   already-marginal levels under one parent so `marginalize_closure` can
//!   collapse a whole structural level out of it.
//!
//! Module map:
//! - `probe`   — the shared rotation probe and what it is built from:
//!   rotation-kind dispatch, the per-level size helper, the marginal-level
//!   guard, and the subtree allow-mask.
//! - `local`   — the greedy search behind
//!   [`Engine::rotation_search`](crate::Engine::rotation_search) and the
//!   [`RotationObjective`] trait.
//! - `cluster` — the mid-compile marginal-clustering pass.

pub(crate) mod cluster;
mod probe;
pub(crate) mod local;

#[cfg(test)]
mod tests;

pub use local::{RotationObjective, RotationSearchConfig, RotationSearchStats};

pub(crate) use local::rotation_search_on;

use crate::diagram::Tdd;
use crate::limits::ApplyError;

/// The rotation-search entry point on a caller's engine.
impl crate::engine::Engine {
    /// Descend the diagram's vtree greedily by rotation, keeping every probe
    /// the objective scores as an improvement.
    ///
    /// Sweeps every internal vtree node, probing a left and a right rotation
    /// at each, accepting a move whenever the objective strictly improves
    /// ([`RotationObjective::delta`] `< 0`), and re-minimizing after each
    /// accept. Sweeps repeat until one accepts nothing (a local minimum) or
    /// `config.max_sweeps` is hit; the returned [`RotationSearchStats`] holds
    /// the probe, accept and sweep tallies. Rotations are pure variable
    /// reorders, so the model count is preserved under any objective, on
    /// marginal diagrams too. The armed stop is polled once per pivot, which
    /// is what lets a caller bound a search that would otherwise run to a
    /// local minimum.
    ///
    /// # Errors
    ///
    /// [`ApplyError::Deadline`] when the armed deadline passes or a stop
    /// decision concludes the search should end. The diagram is left canonical
    /// and count-correct at whatever local point the search had reached.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{ApplyError, Engine, Tdd};
    /// use tididi::diagram::TddLevel;
    /// use tididi::limits::LimitSet;
    /// use tididi::restructure::search::{RotationObjective, RotationSearchConfig};
    /// use tididi::vtree::Vtree;
    ///
    /// struct MinSize;
    /// impl RotationObjective for MinSize {
    ///     fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
    ///         (a.0.width() + a.1.width()) as i64 - (b.0.width() + b.1.width()) as i64
    ///     }
    /// }
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [3, 4]);
    /// let before = f.model_count();
    /// engine.rotation_search(&mut f, &mut MinSize, &RotationSearchConfig::default()).unwrap();
    /// assert_eq!(f.model_count(), before);
    ///
    /// // A byte budget of zero refuses the first rotation's reservation. The
    /// // diagram is left canonical and counting the same either way.
    /// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
    /// match engine.rotation_search(&mut f, &mut MinSize, &RotationSearchConfig::default()) {
    ///     Ok(_) => {}
    ///     Err(e) => assert!(matches!(e, ApplyError::OverBudget | ApplyError::Deadline)),
    /// }
    /// assert_eq!(f.model_count(), before);
    /// ```
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, ApplyError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }
}
