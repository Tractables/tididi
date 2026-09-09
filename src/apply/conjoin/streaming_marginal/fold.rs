//! The streaming fold contract and the per-level scratch it drives.

use super::*;

/// The fixed conceptual slots of a LEAF child's column, ordered
/// `{One=2, Pos=1, Neg=1}` per `LeafLabel::from_idx` (LEAF_WIDTH = 3). A
/// `static` so a leaf view borrows it rather than minting a `vec![2, 1, 1]`
/// per level.
pub(crate) static LEAF_COUNTS: [u128; 3] = [2, 1, 1];

/// The integer instantiation, spelled out because it is the one the hot path
/// and the overflow validation tests name directly.
pub(crate) type StreamChildCounts<'a> = StreamChild<'a, IntFold>;

/// Per-level streaming state for one value kind, live only for the row loop:
/// Both child views plus a mutable borrow of the level's output column.
///
/// The output column itself is owned by the driver loop's [`StreamLevelState`]
/// so it outlives the child borrows — the level tail retakes `&mut levels` to
/// commit it, which it could not do while a view into `levels` was alive.
pub(crate) struct StreamState<'a, F: ValueDomain> {
    pub(crate) left: StreamChild<'a, F>,
    pub(crate) right: StreamChild<'a, F>,
    pub(crate) counts: &'a mut F::Col<ApplyBudget>,
    /// The domain's own state: the diagram's weight store while
    /// `F = WeightFold`, and nothing at all in integer mode.
    pub(crate) store: &'a F::Store,
}

/// The level's in-flight output column, indexed by alive-cell position and
/// handed to [`ValueDomain::commit_in_flight`] on commit. The runtime value-kind
/// choice is made once per level in [`build_stream_state`]; everything
/// downstream of the two `match` arms (here and in
/// `cell::run_level_rows_stream_count`) is statically monomorphized — there is
/// no dynamic dispatch and no inert second payload.
///
/// Deliberately BORROW-FREE: it is carried across the whole cell-build route
/// dispatch to the commit, so it must not pin a borrow of `levels`.
pub(crate) enum StreamLevelState {
    Int(CountVec<ApplyBudget>),
    /// Weighted: exact `BigRational` semiring values carried into the
    /// external `WeightStore`.
    Weighted(Vec<WeightVal>),
}

/// Cascade marginalization through every explicit non-leaf descendant of
/// `left_idx`, bottom-up. Each level's column must already be populated in
/// `computed` (call [`ValueDomain::ensure`] first). Mirrors
/// the marginalize cascade in `marginal::fold` but operates on the in-flight `levels`
/// slice during apply rather than a finished diagram.
///
/// Soundness: bottom-up order satisfies `assert_can_make_marginal` at each
/// call site (by the time we marginalize `left_idx`, both children of `left_idx` are
/// already marginal or leaves). The caller is responsible for ensuring `left_idx`
/// itself is a sound streaming target (no future references) — within the
/// apply gate this is guaranteed because `marginalize_targets[left_idx] == true` flags
/// any descendant we'd touch, and the schedule monotonicity (a parent's
/// streaming step is at least as late as any descendant's) covers descendants
/// that aren't in this apply call's target set but were targets of an earlier
/// sub-batch and only stayed explicit because of the width gate.
pub(crate) fn cascade_marginalize_in_apply<F: ValueDomain>(
    left_idx: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    store: &mut F::Store,
) {
    if vtree.node(VtreeIdx(left_idx as u32)).is_leaf() || levels[left_idx].is_marginal() {
        return;
    }
    let (left, right) = vtree.children(VtreeIdx(left_idx as u32));
    cascade_marginalize_in_apply::<F>(left.idx(), vtree, levels, computed, store);
    cascade_marginalize_in_apply::<F>(right.idx(), vtree, levels, computed, store);
    let Some(col) = computed[left_idx].take() else {
        // No cached column: `ensure_level_counts` did not visit this branch
        // (cells structurally unreachable from the target's pair lists). Bail
        // rather than fabricate values.
        return;
    };
    diagram::assert_can_make_marginal(levels, vtree, VtreeIdx(left_idx as u32));
    F::commit_in_flight::<ApplyBudget>(levels, left_idx, col, store);
}
