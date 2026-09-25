//! Build marginal output columns during conjunction.
//!
//! Each live cell is folded into a value before its temporary pairs are
//! discarded, bounding pair storage by the largest cell. Integer and weighted
//! routes share the same driver through [`ValueDomain`]. The output is still
//! a slice of levels, so column installation does not require a finished `Tdd`.

use crate::diagram::WeightValue;
use crate::diagram::WeightStore;
use crate::Engine;
use super::{OperationError, TddLevel, Sides};

use crate::value::{
    Retention, CountVec, FoldInput, IntFold, StreamChild,
    ValueDomain, WeightFold,
};

use crate::value::StreamCache;
use crate::marginal::transition::{InternalLevel, MarginalDomain, cascade, install_streamed};
mod level;
pub(crate) use level::*;

/// Per-level streaming state for one value kind, live only for the row loop:
/// Both child views plus a mutable borrow of the level's output column.
///
/// The output column itself is owned by the driver loop's [`StreamLevelState`]
/// so it outlives the child borrows — the level tail retakes `&mut levels` to
/// commit it, which it could not do while a view into `levels` was alive.
pub(crate) struct StreamState<'a, F: ValueDomain> {
    pub(crate) left: StreamChild<'a, F>,
    pub(crate) right: StreamChild<'a, F>,
    pub(crate) counts: &'a mut F::Col,
    /// The domain's own state: the diagram's weight store while
    /// `F = WeightFold`, and nothing at all in integer mode.
    pub(crate) store: &'a F::Store,
}

/// The level's in-flight output column, indexed by alive-cell position and
/// handed to [`MarginalDomain::commit_in_flight`] on commit. Holds no borrow: it
/// is carried across the cell-build route dispatch to the commit, so it must
/// not pin `levels`.
pub(crate) enum StreamLevelState {
    Int(CountVec),
    /// Weighted: exact `BigRational` semiring values carried into the
    /// external `WeightStore`.
    Weighted(Vec<WeightValue>),
}
