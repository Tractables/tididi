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
use crate::vtree::VtreeIdx;

use crate::value::StreamCache;
use crate::marginal::transition::{MarginalDomain, install_streamed};
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

/// Cascade marginalization through every explicit non-leaf descendant of
/// `left_idx`, bottom-up. Each level's column must already be populated in
/// `computed` (call [`ValueDomain::ensure`] first). Mirrors
/// the marginalization cascade in `marginal::fold` but operates on the in-flight `levels`
/// slice during apply rather than a finished diagram.
///
/// Soundness: bottom-up order satisfies `assert_can_make_marginal` at each
/// call site (when installing `left_idx`, both children of `left_idx` are
/// already marginal or leaves). The caller is responsible for ensuring `left_idx`
/// itself is a sound streaming target (no future references) — within the
/// apply gate this is guaranteed because `marginalize_targets[left_idx] == true` flags
/// any descendant we'd touch, and the schedule monotonicity (a parent's
/// streaming step is at least as late as any descendant's) covers descendants
/// that aren't in this apply call's target set but were targets of an earlier
/// sub-batch and only stayed explicit because of the width gate.
pub(crate) fn cascade_marginalize_in_apply<F: MarginalDomain>(
    left_idx: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col>],
    store: &mut F::Store,
) {
    if vtree.node(VtreeIdx(left_idx as u32)).is_leaf() || levels[left_idx].is_marginal() {
        return;
    }
    let (left, right) = vtree.children(VtreeIdx(left_idx as u32));
    cascade_marginalize_in_apply::<F>(left.idx(), vtree, levels, computed, store);
    cascade_marginalize_in_apply::<F>(right.idx(), vtree, levels, computed, store);
    let Some(col) = computed[left_idx].take() else {
        // No cached column: the domain's `ensure` did not visit this branch
        // (cells structurally unreachable from the target's pair lists). Bail
        // rather than fabricate values.
        return;
    };
    install_streamed::<F>(levels, vtree, VtreeIdx(left_idx as u32), col, store);
}
