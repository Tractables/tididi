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

    /// [`finish`](Self::finish) storage whose construction established the
    /// public builder's storage invariants, as a copy of a valid diagram's
    /// levels does; debug builds still run the builder's checks.
    ///
    /// # Errors
    ///
    /// As [`finish`](Self::finish).
    #[inline]
    pub(crate) fn finish_asserted(self, output: TddNodeId) -> Result<Tdd, OperationError> {
        debug_assert!(
            self.check(output).is_ok(),
            "a copy built storage the checked seat would refuse",
        );
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
    pub(crate) fn finish(self, output: TddNodeId) -> Result<Tdd, OperationError> {
        self.finish_or_return(output).map_err(|(e, _)| e)
    }

    /// [`finish`](Self::finish) that hands the assembly back, levels intact,
    /// when the worklist growth is refused.
    #[inline]
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub(crate) fn finish_or_return(self, output: TddNodeId) -> Result<Tdd, (OperationError, Self)> {
        self.seed_or_return(output, Dirty::default(), None, None)
    }

    /// [`finish`](Self::finish) with the worklists supplied by the caller
    /// instead of every internal level.
    ///
    /// A level absent from a worklist is taken to be at its contraction
    /// fixpoint ([`Dirty`]), so the caller owes two things:
    ///
    /// 1. Every changed level not known to be at its contraction fixpoint is
    ///    in `rebuilt`;
    /// 2. `carried` is the input diagram's own [`Dirty`], so nothing the input
    ///    had outstanding is dropped.
    ///
    /// Seeding only the rewritten levels makes the following contraction cost
    /// proportional to them rather than to the vtree. `changed`, when given,
    /// holds every level the operation built or changed, the only ones the
    /// seat closes (`TddLevel::close`); every other level has to be as the
    /// end of an earlier operation left it. `None` closes every level.
    ///
    /// # Errors
    ///
    /// As [`finish`](Self::finish).
    pub(crate) fn finish_with(
        self, output: TddNodeId, carried: Dirty, rebuilt: &[VtreeIdx], changed: Option<&[VtreeIdx]>,
    ) -> Result<Tdd, OperationError> {
        self.finish_with_or_return(output, carried, rebuilt, changed).map_err(|(e, _)| e)
    }

    /// [`finish_with`](Self::finish_with) that hands the assembly back,
    /// levels intact, when the worklist growth is refused.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub(crate) fn finish_with_or_return(
        self, output: TddNodeId, carried: Dirty, rebuilt: &[VtreeIdx], changed: Option<&[VtreeIdx]>,
    ) -> Result<Tdd, (OperationError, Self)> {
        self.seed_or_return(output, carried, Some(rebuilt), changed)
    }

    /// Seed the worklists, `rebuilt` or every internal level on top of
    /// `carried`, and seat the result, closing `changed` or every level; the
    /// assembly back when refused.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    fn seed_or_return(
        mut self, output: TddNodeId, carried: Dirty, rebuilt: Option<&[VtreeIdx]>, changed: Option<&[VtreeIdx]>,
    ) -> Result<Tdd, (OperationError, Self)> {
        match self.seed_worklists(carried, rebuilt, Some(self.engine)) {
            Ok(dirty) => Ok(self.builder.take().expect("unfinished assembly").seat(output, dirty, changed)),
            Err(e) => Err((e, self)),
        }
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
