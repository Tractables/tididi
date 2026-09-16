//! Search for a better vtree shape while preserving a compiled function.
//!
//! [`Tdd::rotation_search`] tries local tree
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
impl crate::Engine {
    /// Run [`Tdd::rotation_search`](crate::Tdd::rotation_search) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Trial storage is outside the byte budget and output cap. Stops are polled
    /// once per pivot; cancellation retains accepted rotations and leaves the
    /// diagram canonical and count-correct. An opening reduction failure leaves
    /// the valid partial result described by [`crate::Engine::minimize`].
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, OperationError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }
}
