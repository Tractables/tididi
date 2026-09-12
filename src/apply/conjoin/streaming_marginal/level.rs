//! Opening, attaching and committing one level of a streaming fold.

use super::*;
use crate::apply::conjoin::targets::MarginalTargets;

/// Prepare the children and open the [`StreamLevelState`] output column when
/// level `t_idx` is a streaming target; `None` otherwise. Weighted when `ws`
/// is given, integer otherwise; both arms run [`open_stream_output`].
///
/// Needs the whole `levels` slice, since the cascade re-marginalizes any
/// descendant, and returns nothing that borrows it; the child columns are
/// attached per row loop by [`attach_children`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_stream_state(
    eng: &Engine,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    left_width: usize,
    right_width: usize,
    marginalize_targets: MarginalTargets<'_>,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    cache: &mut StreamCache,
    ws: Option<&mut WeightStore>,
) -> Result<Option<StreamLevelState>, ApplyError> {
    if !marginalize_targets.is_target(t_idx) {
        return Ok(None);
    }
    if let Some(ws) = ws {
        Ok(Some(StreamLevelState::Weighted(open_stream_output::<WeightFold>(
            eng,
            left_idx, right_idx, left_width, right_width, vtree, levels, cache.weighted_mut(), ws,
        )?)))
    } else {
        Ok(Some(StreamLevelState::Int(open_stream_output::<IntFold>(
            eng,
            left_idx, right_idx, left_width, right_width, vtree, levels, cache.int_mut(), &mut (),
        )?)))
    }
}

/// The value-kind-generic streaming setup: compute both children's columns,
/// cascade-marginalize any still-explicit non-leaf descendant, and open the
/// output column.
///
/// The cascade makes every descendant marginal or a leaf, which streaming
/// this level requires. The output column is opened at capacity
/// `left_width.max(right_width)` and grows; the reservation is fallible.
///
/// # Errors
///
/// [`ApplyError::OverBudget`] when a column reservation is refused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_stream_output<F: ValueDomain>(
    eng: &Engine,
    left_idx: usize,
    right_idx: usize,
    left_width: usize,
    right_width: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    store: &mut F::Store,
) -> Result<F::Col<ApplyBudget>, ApplyError> {
    // 1. Compute the fold column for every non-leaf non-marginal descendant.
    //
    // Step 2 `take`s the column of every level in the walked subtree to install
    // it as that level's marginal store, so retention is `All`.
    let marginal = |i: usize| levels[i].is_marginal();
    F::ensure::<ApplyBudget>(
        eng, left_idx, vtree, levels, computed, store, &marginal, ColumnRetention::All,
    )?;
    F::ensure::<ApplyBudget>(
        eng, right_idx, vtree, levels, computed, store, &marginal, ColumnRetention::All,
    )?;
    // 2. Cascade-marginalize any still-explicit non-leaf descendant.
    cascade_marginalize_in_apply::<F>(left_idx, vtree, levels, computed, store);
    cascade_marginalize_in_apply::<F>(right_idx, vtree, levels, computed, store);
    F::try_with_capacity::<ApplyBudget>(eng, left_width.max(right_width))
}

/// Phase: streaming row loop (per value kind, per route).
///
/// Binds the two child column views to this level's in-flight output column for
/// the duration of one row loop. The views borrow `left_level`/`right_level`
/// directly, which is why the driver loop splits `levels[t_idx]` and the two
/// child slots apart up front: `t` and its two vtree children are three
/// distinct nodes of a tree, so the split is total and the output level stays
/// exclusively borrowed while the children are read.
///
/// The returned state must not outlive the row loop — the level tail retakes
/// `&mut levels` to commit [`StreamLevelState`], which owns the column this
/// only borrows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attach_children<'a, F: ValueDomain>(
    eng: &Engine,
    left_idx: usize,
    right_idx: usize,
    vtree: &crate::vtree::Vtree,
    left_level: &'a TddLevel,
    right_level: &'a TddLevel,
    computed: &'a [Option<F::Col<ApplyBudget>>],
    counts: &'a mut F::Col<ApplyBudget>,
    store: &'a F::Store,
) -> Result<StreamState<'a, F>, ApplyError> {
    Ok(StreamState {
        left: F::child_view(eng, left_idx, vtree, left_level, computed, store)?,
        right: F::child_view(eng, right_idx, vtree, right_level, computed, store)?,
        counts,
        store,
    })
}

/// Convert the completed [`StreamLevelState`] into level `t`'s marginal store.
/// Both children of `t` must already be marginal or leaves, which the
/// bottom-up sweep guarantees for a scheduled target. Values are not deduped
/// here (see [`ValueDomain::commit_in_flight`]); the slot prune establishes
/// slot uniqueness (`test_helpers::check::check_slot_count_uniqueness`).
#[inline(always)]
pub(crate) fn commit_stream_state(
    st: StreamLevelState,
    t: VtreeIdx,
    t_idx: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    ws: Option<&mut WeightStore>,
) {
    diagram::assert_can_make_marginal(levels, vtree, t);
    let ws = match st {
        StreamLevelState::Int(counts) => {
            IntFold::commit_in_flight::<ApplyBudget>(levels, t_idx, counts, &mut ());
            ws
        }
        StreamLevelState::Weighted(counts) => {
            let ws = ws.expect("a weighted column is only ever built with a store attached");
            WeightFold::commit_in_flight::<ApplyBudget>(levels, t_idx, counts, &mut *ws);
            Some(ws)
        }
    };
    // `t` now subsumes its children: their stores are dead.
    crate::marginal::free_subsumed_marginal_children(levels, vtree, t, ws);
}
