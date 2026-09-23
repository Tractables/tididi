//! Ownership of an unfinished operation result.

use std::{ops::DerefMut, sync::Arc};

use crate::{Engine, OperationError};
use crate::vtree::Vtree;
use super::{Tdd, TddBuilder, TddBuildError, TddLevel, TddNodeId, WeightStore};

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

    /// Validate storage using the public builder's checks before seating it.
    #[inline]
    pub(crate) fn finish_checked(self, output: TddNodeId) -> Result<Tdd, TddBuildError> {
        self.check(output)?;
        Ok(self.finish_untracked(output))
    }

    /// Seat kernel-built storage and charge its reduction worklists.
    #[inline]
    pub(crate) fn finish(mut self, output: TddNodeId) -> Result<Tdd, OperationError> {
        let dirty = self.seed_worklists(Some(self.engine))?;
        Ok(self.builder.take().expect("unfinished assembly").seat(output, dirty))
    }

    /// Seat kernel-built storage using the untracked assembly contract.
    #[inline]
    pub(crate) fn finish_untracked(mut self, output: TddNodeId) -> Tdd {
        let dirty = self.seed_worklists(None).expect("untracked worklists cannot be refused");
        self.builder.take().expect("unfinished assembly").seat(output, dirty)
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
