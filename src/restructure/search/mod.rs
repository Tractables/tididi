//! Search for a better vtree shape while preserving a compiled function.
//!
//! [`Engine::rotation_search`](crate::Engine::rotation_search) tries local tree
//! rotations and keeps improvements according to a caller-supplied
//! [`RotationObjective`]. [`RotationSearchConfig`] bounds the number of sweeps,
//! and [`RotationSearchStats`] reports the work performed.
//!
//! A changed vtree belongs to the resulting diagram. Build subsequent operands
//! using that diagram's [`Tdd::vtree`] so they share its allocation. The search
//! method's example shows both the objective and continued use of the result.

pub(crate) mod cluster;
mod probe;
pub(crate) mod local;

#[cfg(test)]
mod tests;

pub use local::{RotationObjective, RotationSearchConfig, RotationSearchStats};

pub(crate) use local::rotation_search_on;

use crate::diagram::Tdd;
use crate::limits::OperationError;

/// The rotation-search entry point on a caller's engine.
impl crate::engine::Engine {
    /// Descend the diagram's vtree greedily by rotation, keeping every probe
    /// the objective scores as an improvement.
    ///
    /// Sweeps every internal vtree node, probing a left and a right rotation
    /// at each and accepting a move whenever the objective strictly improves
    /// ([`RotationObjective::delta`] `< 0`). Sweeps repeat until one accepts
    /// nothing (a local minimum) or `config.max_sweeps` is hit; the returned
    /// [`RotationSearchStats`] holds the probe, accept and sweep tallies.
    /// Rotations regroup subtrees without changing variable ids, so the count is preserved
    /// under any objective, on marginal diagrams too; a rotation that would
    /// touch a marginal level is not probed. The diagram's vtree is rotated
    /// with it: when its `Arc<Vtree>` is shared, the diagram gets a private
    /// copy, so it no longer shares a vtree with the diagrams built beside it.
    ///
    /// A diagram with no marginal level is minimized first using this engine;
    /// a diagram with a marginal level must arrive canonical. An accepted
    /// rotation keeps the diagram canonical. Trial storage is outside the byte
    /// budget and output cap; the armed stop is polled once per pivot.
    ///
    /// # Errors
    ///
    /// [`OperationError::Stopped`] when the armed deadline passes or a stop
    /// decision concludes the search should end. The diagram is left canonical
    /// and count-correct at whatever local point the search had reached.
    /// The opening reduction can also refuse an allocation; its partial-result
    /// contract is stated on [`crate::reduce::try_reduce`].
    ///
    /// # Panics
    ///
    /// If the objective panics, the current trial is rolled back before the
    /// panic unwinds to the caller; earlier accepted rotations remain committed.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd};
    /// use tididi::diagram::TddLevel;
    /// use tididi::restructure::search::{RotationObjective, RotationSearchConfig};
    /// use tididi::vtree::Vtree;
    ///
    /// struct MinSize;
    /// impl RotationObjective for MinSize {
    ///     fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
    ///         (a.0.slot_count() + a.1.slot_count()) as i64 - (b.0.slot_count() + b.1.slot_count()) as i64
    ///     }
    /// }
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [3, 4]);
    /// let before = engine.model_count(&f)?;
    /// engine.rotation_search(&mut f, &mut MinSize, &RotationSearchConfig::default())?;
    /// assert_eq!(engine.model_count(&f)?, before);
    ///
    /// // Continue in the result's domain, which may use a different vtree.
    /// let extra = engine.literal(f.vtree(), 1)?;
    /// let constrained = engine.and(f, extra)?;
    /// assert!(engine.is_sat(&constrained)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, OperationError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }
}
