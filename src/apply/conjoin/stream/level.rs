//! Opening, attaching and committing one level of a streaming fold.

use super::*;

/// Single source of truth for the streaming-eligibility gate: a level streams
/// its marginal iff it is a marginalize target AND the streaming gate is on
/// (off ⇒ don't stream, materialize + post-apply `marginalize_batch`). Consulted per level by the emit-growth mode
/// decision (the `stream_marginal` local in the driver loop) and by
/// [`build_stream_state`]'s setup; the commit then keys off `stream_state` being
/// `Some` rather than re-reading the predicate. Do NOT re-inline the predicate
/// at a call site — it is cheap, and the cold per-level path can afford the
/// call.
#[inline]
pub(crate) fn stream_marginal_eligible(marginalize_targets: Option<&[bool]>, t_idx: usize) -> bool {
    marginalize_targets.is_some_and(|t| t[t_idx])
        && bothmarg_collapse_enabled()
}

/// Phase: streaming-marginal setup (inside the `for (t, left, right) in vtree.internal_bottomup()` loop).
///
/// Prepares the children and opens the [`StreamLevelState`] output column if
/// this level is a streaming target. Called after the NxM dead-pair pre-filter
/// block, before the dedicated marginal-child dispatch. This is the ONE place
/// the value kind is chosen at runtime; both arms run the same generic
/// [`open_stream_output`].
///
/// Runs while the whole `levels` slice is still mutably available — the
/// cascade re-marginalizes arbitrary descendants, not just the two children —
/// and returns nothing that borrows it. The child columns are attached later,
/// per row loop, by [`attach_children`].
///
/// In weighted mode streaming carries BigRational values into the external
/// [`WeightStore`]; not weighted → integer streaming.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_stream_state(
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    k1: usize,
    k2: usize,
    marginalize_targets: Option<&[bool]>,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    stream_computed: &mut Vec<Option<CountVec<ApplyBudget>>>,
    stream_computed_weights: &mut Vec<Option<Vec<WeightVal>>>,
    ws: Option<&mut WeightStore>,
) -> Result<Option<StreamLevelState>, ApplyError> {
    if !stream_marginal_eligible(marginalize_targets, t_idx) {
        return Ok(None);
    }
    if ws.is_some() {
        Ok(Some(StreamLevelState::Weighted(open_stream_output::<WeightFold>(
            left_idx, right_idx, k1, k2, vtree, levels, stream_computed_weights, ws,
        )?)))
    } else {
        Ok(Some(StreamLevelState::Int(open_stream_output::<IntFold>(
            left_idx, right_idx, k1, k2, vtree, levels, stream_computed, None,
        )?)))
    }
}

/// The value-kind-generic streaming setup: compute both children's columns,
/// cascade-marginalize any still-explicit non-leaf descendant, and open the
/// output column.
///
/// Step 2 (the cascade) restores the structural contract that streaming this
/// level requires its descendants to be marginal-or-leaf. The width gate at
/// lower levels may have left descendants explicit; those descendants would
/// have been targets of an earlier sub-batch (or this one), so marginalizing
/// them now is sound — no future clause references them.
///
/// The output column's initial capacity is bounded by alive cells (≤ k1*k2) but
/// typically far fewer — ask for `k1.max(k2)` and let it grow. That reservation
/// must be FALLIBLE: `k1.max(k2)` can reach ~1B on extreme widths, where an
/// an infallible `Vec::with_capacity` aborts the process on a single
/// over-large allocation. `?` propagates `OverBudget` so the caller can split
/// instead.
pub(crate) fn open_stream_output<F: StreamPayload>(
    left_idx: usize,
    right_idx: usize,
    k1: usize,
    k2: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut Vec<Option<F::Col<ApplyBudget>>>,
    mut ws: Option<&mut WeightStore>,
) -> Result<F::Col<ApplyBudget>, ApplyError> {
    // 1. Compute the fold column for every non-leaf non-marginal descendant.
    ensure_level_counts::<F>(left_idx, vtree, levels, computed, ws.as_deref())?;
    ensure_level_counts::<F>(right_idx, vtree, levels, computed, ws.as_deref())?;
    // 2. Cascade-marginalize any still-explicit non-leaf descendant.
    cascade_marginalize_in_apply::<F>(left_idx, vtree, levels, computed, ws.as_deref_mut());
    cascade_marginalize_in_apply::<F>(right_idx, vtree, levels, computed, ws.as_deref_mut());
    F::try_with_capacity::<ApplyBudget>(k1.max(k2))
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
pub(crate) fn attach_children<'a, F: StreamPayload>(
    left_idx: usize,
    right_idx: usize,
    vtree: &crate::vtree::Vtree,
    left_level: &'a TddLevel,
    right_level: &'a TddLevel,
    computed: &'a [Option<F::Col<ApplyBudget>>],
    counts: &'a mut F::Col<ApplyBudget>,
    ws: Option<&'a WeightStore>,
) -> Result<StreamState<'a, F>, ApplyError> {
    Ok(StreamState {
        left: F::child_view(left_idx, vtree, left_level, computed, ws)?,
        right: F::child_view(right_idx, vtree, right_level, computed, ws)?,
        counts,
        ws,
    })
}

/// Phase: streaming commit (inside the `for (t, left, right) in vtree.internal_bottomup()` loop).
///
/// Converts the completed [`StreamLevelState`] into a marginal level. The
/// caller keeps the `if let Some(st) = stream_state.take()` guard; this
/// function receives the unwrapped state. C3 is established later by
/// `prune_marg_slots` — emit-site dedup is forbidden, see
/// [`IntFold::store_level`].
///
/// Marginalization precondition (checked once, before the value-kind branch):
/// both children of `t` must already be marginal (or leaves). For
/// streaming-marginal during apply, the marginalize_schedule guarantees
/// descendants of `t` in the schedule are processed first (apply runs
/// bottom-up).
#[inline(always)]
pub(crate) fn commit_stream_state(
    st: StreamLevelState,
    t: VtreeIdx,
    t_idx: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    ws: Option<&mut WeightStore>,
) {
    diagram::assert_can_make_marginal(levels, vtree, t);
    match st {
        StreamLevelState::Int(counts) => IntFold::store_level(levels, t_idx, counts, None),
        StreamLevelState::Weighted(counts) => WeightFold::store_level(levels, t_idx, counts, ws),
    }
}
