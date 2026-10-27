//! Build marginal output columns during conjunction.
//!
//! Each live cell is folded into a value before its temporary pairs are
//! discarded, bounding pair storage by the largest cell. Integer and weighted
//! routes share the same driver through [`ValueDomain`]. The output is still
//! a slice of levels, so column installation does not require a finished `Tdd`.

use crate::diagram::{Arithmetic, WeightStore, WeightValue};
use crate::Engine;
use super::{OperationError, TddLevel, Sides};

use crate::value::{
    Retention, CountVec, FloatCol, FloatFold, FoldInput, IntFold, StreamChild,
    ValueDomain, WeightFold,
};

use crate::value::{StreamCache, StreamKind};
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
    /// Weighted: semiring values carried into the external `WeightStore`.
    Weighted(Vec<WeightValue>),
    /// Scaled floats, kept by the apply until the level's parent has read
    /// them or the sweep ends.
    Float(FloatCol),
}

/// The value kind an apply streams in: integer without a weight store, bare
/// scaled floats under [`Arithmetic::ScaledFloat`], weight values otherwise.
pub(crate) fn stream_kind(ws: Option<&WeightStore>) -> StreamKind {
    match ws.map(WeightStore::arithmetic) {
        None => StreamKind::Int,
        Some(Arithmetic::ScaledFloat) => StreamKind::Float,
        Some(_) => StreamKind::Weighted,
    }
}
