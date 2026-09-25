//! Ownership of an unfinished operation result.

use std::{ops::DerefMut, sync::Arc};

use crate::{Engine, OperationError};
use crate::vtree::{Vtree, VtreeIdx};
use super::{Dirty, Tdd, TddBuilder, TddLevel, TddNodeId, WeightStore};

/// An operation's output, returned to the level pool if construction fails.
///
/// Kernels may fill the level arenas directly; builder-based algorithms use
/// the same owner through `Deref`. Finishing transfers the levels and weights
/// together and seeds the result's reduction worklists.
pub(crate) struct Assembly<'a> {
    engine: &'a Engine,
    builder: Option<TddBuilder>,
}

impl<'a> Assembly<'a> {
    /// Allocate empty output levels through the caller's limits.
    #[inline]
    pub(crate) fn new(engine: &'a Engine, vtree: &Arc<Vtree>) -> Result<Self, OperationError> {
        Ok(Self { engine, builder: Some(Tdd::builder(engine, vtree)?) })
    }

    /// Take ownership of arenas already allocated or moved from an operand.
    #[inline]
    pub(crate) fn from_levels(
        engine: &'a Engine, vtree: Arc<Vtree>, levels: Vec<TddLevel>, weights: Option<WeightStore>,
    ) -> Self {
        Self { engine, builder: Some(TddBuilder::from_levels(vtree, levels, weights)) }
    }

    /// Disjoint access to arenas and weighted columns during a kernel sweep.
    #[inline]
    pub(crate) fn parts_mut(&mut self) -> (&mut Vec<TddLevel>, &mut Option<WeightStore>) {
        self.deref_mut().parts_mut()
    }

    /// Validate storage using the public builder's checks, then
    /// [`finish`](Self::finish) it.
    ///
    /// # Errors
    ///
    /// A failed check, as [`OperationError::InvalidDiagram`], or a refused
    /// worklist growth.
    #[inline]
    pub(crate) fn finish_checked(self, output: TddNodeId) -> Result<Tdd, OperationError> {
        self.check(output)?;
        self.finish(output)
    }

    /// Seat kernel-built storage and charge its reduction worklists, which
    /// start with every internal level.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the worklist growth is refused;
    /// the levels go back to the pool.
    #[inline]
    pub(crate) fn finish(mut self, output: TddNodeId) -> Result<Tdd, OperationError> {
        let dirty = self.seed_worklists(Dirty::default(), None, Some(self.engine))?;
        Ok(self.builder.take().expect("unfinished assembly").seat(output, dirty))
    }

    /// [`finish`](Self::finish) with the worklists supplied by the caller
    /// instead of every internal level.
    ///
    /// A level absent from a worklist is taken to be at its contraction
    /// fixpoint ([`Dirty`]), so the caller owes two things:
    ///
    /// 1. Every level whose pair list this operation changed is in `rebuilt`;
    /// 2. `carried` is the input diagram's own [`Dirty`], so nothing the input
    ///    had outstanding is dropped.
    ///
    /// Seeding only the rewritten levels makes the following contraction cost
    /// proportional to them rather than to the vtree.
    ///
    /// # Errors
    ///
    /// As [`finish`](Self::finish).
    pub(crate) fn finish_with(
        mut self, output: TddNodeId, carried: Dirty, rebuilt: &[VtreeIdx],
    ) -> Result<Tdd, OperationError> {
        let dirty = self.seed_worklists(carried, Some(rebuilt), Some(self.engine))?;
        Ok(self.builder.take().expect("unfinished assembly").seat(output, dirty))
    }
}

impl std::ops::Deref for Assembly<'_> {
    type Target = TddBuilder;
    fn deref(&self) -> &TddBuilder { self.builder.as_ref().expect("unfinished assembly") }
}

impl std::ops::DerefMut for Assembly<'_> {
    fn deref_mut(&mut self) -> &mut TddBuilder { self.builder.as_mut().expect("unfinished assembly") }
}

impl Drop for Assembly<'_> {
    fn drop(&mut self) {
        let Some(builder) = self.builder.as_mut() else { return };
        let (levels, _) = builder.parts_mut();
        if !levels.is_empty() && !std::thread::panicking() {
            super::return_levels(self.engine, super::PoolSlot::First, std::mem::take(levels));
        }
    }
}
