//! Opening, attaching and committing one level of a streaming fold.

use super::*;
use crate::apply::conjoin::setup::LevelShape;
use crate::apply::conjoin::drive::Sweep;

/// What a streaming row loop reads beside its own level: the two child
/// indices, the vtree, the column cache and the weight store.
#[derive(Clone, Copy)]
pub(crate) struct StreamEnv<'a> {
    pub(crate) left_idx: usize,
    pub(crate) right_idx: usize,
    pub(crate) vtree: &'a crate::vtree::Vtree,
    pub(crate) cache: &'a StreamCache,
    pub(crate) ws: Option<&'a WeightStore>,
}

/// Prepare the children and open the [`StreamLevelState`] output column when
/// the level is a streaming target; `None` otherwise. Weighted when the sweep
/// carries a weight store, integer otherwise; both arms run
/// [`open_stream_output`].
///
/// Needs the whole `levels` slice, since the cascade re-marginalizes any
/// descendant, and returns nothing that borrows it; the child columns are
/// attached per row loop by [`attach_children`].
#[inline(always)]
pub(in crate::apply::conjoin) fn build_stream_state(
    eng: &Engine,
    shape: LevelShape,
    levels: &mut [TddLevel],
    cache: &mut StreamCache,
    sweep: &mut Sweep<'_>,
) -> Result<Option<StreamLevelState>, ApplyError> {
    if !sweep.targets.is_target(shape.t.idx()) {
        return Ok(None);
    }
    if let Some(ws) = sweep.ws.as_deref_mut() {
        Ok(Some(StreamLevelState::Weighted(open_stream_output::<WeightFold>(
            eng, shape, sweep.vtree, levels, cache.weighted_mut(), ws,
        )?)))
    } else {
        Ok(Some(StreamLevelState::Int(open_stream_output::<IntFold>(
            eng, shape, sweep.vtree, levels, cache.int_mut(), &mut (),
        )?)))
    }
}

/// The value-kind-generic streaming setup: compute both children's columns,
/// cascade-marginalize any still-explicit non-leaf descendant, and open the
/// output column.
///
/// The cascade makes every descendant marginal or a leaf, which streaming
/// this level requires. The output column is opened at the larger of the two
/// operands' widths and grows; the reservation is fallible.
///
/// # Errors
///
/// [`ApplyError::OverBudget`] when a column reservation is refused.
pub(in crate::apply::conjoin) fn open_stream_output<F: MarginalDomain>(
    eng: &Engine,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    store: &mut F::Store,
) -> Result<F::Col<ApplyBudget>, ApplyError> {
    let (left_idx, right_idx) = (shape.left.idx(), shape.right.idx());
    // 1. Compute the fold column for every non-leaf non-marginal descendant.
    //
    // Step 2 `take`s the column of every level in the walked subtree to install
    // it as that level's marginal store, so retention is `All`.
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput { vtree, levels, store };
    F::ensure::<ApplyBudget>(eng, shape.left, input, computed, &marginal, ColumnRetention::All)?;
    F::ensure::<ApplyBudget>(eng, shape.right, input, computed, &marginal, ColumnRetention::All)?;
    // 2. Cascade-marginalize any still-explicit non-leaf descendant.
    cascade_marginalize_in_apply::<F>(left_idx, vtree, levels, computed, store);
    cascade_marginalize_in_apply::<F>(right_idx, vtree, levels, computed, store);
    F::try_with_capacity::<ApplyBudget>(eng, shape.f.here.max(shape.g.here))
}

/// Phase: streaming row loop (per value kind, per route).
///
/// Binds the two child column views to this level's in-flight output column for
/// the duration of one row loop. The views borrow the child levels directly,
/// which is why the driver loop splits `levels[t_idx]` and the two child slots
/// apart up front: `t` and its two vtree children are three distinct nodes of
/// a tree, so the split is total and the output level stays exclusively
/// borrowed while the children are read.
///
/// The returned state must not outlive the row loop — the level tail retakes
/// `&mut levels` to commit [`StreamLevelState`], which owns the column this
/// only borrows.
pub(crate) fn attach_children<'a, F: ValueDomain>(
    eng: &Engine,
    env: StreamEnv<'a>,
    children: Sides<&'a TddLevel>,
    counts: &'a mut F::Col<ApplyBudget>,
) -> Result<StreamState<'a, F>, ApplyError> {
    let computed = F::stream_columns(env.cache);
    let store = F::store_of(env.ws);
    Ok(StreamState {
        left: F::child_view(eng, env.left_idx, env.vtree, children.left, computed, store)?,
        right: F::child_view(eng, env.right_idx, env.vtree, children.right, computed, store)?,
        counts,
        store,
    })
}

/// Convert the completed [`StreamLevelState`] into level `t`'s marginal store.
/// Both children of `t` must already be marginal or leaves, which the
/// bottom-up sweep guarantees for a scheduled target. Values are not deduped
/// here (see [`MarginalDomain::commit_in_flight`]); the slot prune establishes
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
    debug_assert_eq!(t.idx(), t_idx);
    match st {
        StreamLevelState::Int(counts) => {
            install_streamed::<IntFold, ApplyBudget>(levels, vtree, t, counts, &mut ());
        }
        StreamLevelState::Weighted(counts) => {
            let ws = ws.expect("a weighted column is only ever built with a store attached");
            install_streamed::<WeightFold, ApplyBudget>(levels, vtree, t, counts, ws);
        }
    }
}
