//! Writing a finished level of the output diagram.

use super::*;

/// Drop the operand-side `Vec`s of a dead operand-child level.
///
/// Called at the *start* of each iteration `t` in `apply_and_fallible`'s
/// bottom-up loop to release `c1.levels[left_idx/right_idx]` and
/// `c2.levels[left_idx/right_idx]`. Children of the current `t` are
/// guaranteed dead at this point: the post-order traversal already visited
/// them in earlier iterations, no future iteration walks into their
/// pairs/nodes/ext (only `c?_widths[child_idx]` is read, and that is a flat
/// usize array snapshot precomputed before the loop).
///
/// **Drop at the START of the iteration, never at the end.** The widest-level
/// output reserve (`budget_reserve_exact`, a single multi-GB allocation) fires
/// mid-iteration; freeing the children before it is what lets the allocator
/// recycle their slabs for the output grows. Dropping after it instead
/// recovers only a fraction of the peak.
///
/// **Unconditional, except where a free returns nothing.** The one gate is the
/// exact test below: it reads the three arenas' own `capacity()` and declines to
/// free a level whose arenas together fit in a single `Vec` minimum allocation
/// — a level where the motive above has nothing to release. Gating on a size
/// *estimate* instead loses more than the skipped drops save.
///
/// That case is the common one on a vtree with far more levels than the
/// operands' support touches: nearly every level is an identity pass-through
/// holding one node and one pair, and freeing those is a `free()` per level per
/// operand per merge that buys back nothing while stripping the level pool of
/// its warm arenas. Retention stays bounded — the skipped arenas are at `Vec`'s
/// minimum allocation, and only the two level arrays the pool parks survive an
/// apply.
///
/// `marginal_counts` / `marginal_counts_big` are left alone: they are small
/// relative to nodes/pairs/ext, and `is_marginal()` stays accurate, so the
/// marginal-schedule assert still functions on a dropped level.
#[inline]
pub(super) fn drop_dead_operand_level(level: &mut crate::diagram::TddLevel) {
    // Nothing worth releasing: the three arenas together hold no more than one
    // Vec minimum allocation (a 1-node / 1-pair identity pass-through level
    // rounds up to 4 slots of each = 64 B). Freeing that returns no slab the
    // output reserve can use, and costs a `free()` now plus a `malloc()` when
    // the pooled level is refilled. Exact capacity reads, not an estimate — see
    // the "Keep it unconditional and check-free" note above for why the
    // distinction is the whole point.
    let bytes = level.nodes.capacity() * std::mem::size_of::<TddNodeData>()
        + level.pairs.capacity() * std::mem::size_of::<InputPair>()
        + level.ext.capacity() * std::mem::size_of::<crate::diagram::ExtMulti>();
    // Leave the level completely untouched on this branch — including
    // `dead_pairs`, which stays consistent with the `pairs` arena it counts
    // garbage in. (The unconditional path can zero it only *because* it empties
    // `pairs` in the same breath.)
    if bytes <= 64 { return; }
    level.nodes = Vec::new();
    level.pairs = Vec::new();
    level.ext = Vec::new();
    // No arena left to sweep, so no garbage to remember.
    level.dead_pairs = 0;
}

/// Set `live_counts[t_idx] = v` while keeping the running `out_nodes_so_far`
/// (== `live_counts.iter().sum()`) in sync in O(1). Every write to
/// `live_counts` MUST go through here so the output-node-cap check can read the
/// counter instead of re-summing all levels each boundary (was O(levels²)). The
/// delta form (subtract the old value, add the new) is robust to re-writes; a
/// `debug_assert_eq!` at the cap check cross-validates against the full sum.
#[inline(always)]
pub(super) fn bump_live_count(
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    t_idx: usize,
    v: usize,
) {
    *out_nodes_so_far = *out_nodes_so_far - live_counts[t_idx] as u64 + v as u64;
    live_counts[t_idx] = v;
}

/// Shared post-output bookkeeping for a freshly-built sparse level's output:
/// refresh the live-node count, mark its product list as populated, and shrink
/// its now-final arrays. Common tail of the two sparse-output emit sites
/// (`apply_sparse_level` and `run_level_rows_marg_sparse`) in
/// `apply_and_fallible_inner`; each site's own pre-tail cleanup
/// (`release_sparse_ws_if_large` / `grid_free`) stays at the call site since
/// it isn't shared.
#[inline(always)]
pub(super) fn finish_sparse_output(
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    has_pl: &mut [bool],
    level: &mut TddLevel,
    t_idx: usize,
) {
    bump_live_count(live_counts, out_nodes_so_far, t_idx, level.nodes.len());
    has_pl[t_idx] = true;
    level.shrink_arrays();
}

/// Mark a level's pass-through inline-emit flags.
///
/// If this level was built via pass-through, its marginal-side pair fields hold
/// inline counts carried verbatim from the carrier operand (already emitted),
/// NOT fresh slots. Mark them so the end-of-apply tagger's emit arm skips
/// re-emitting (which would misread an inline count as a slot index →
/// miscount). Guarded on `!is_marginal()`: a level that became marginal during
/// its build had its markers reset by `make_marginal` and has no structural
/// pairs to describe. `left_passthrough`/`right_passthrough` are emit-gated, so
/// this is a no-op in baseline.
///
/// Shared by the general per-level tail (`finalize_level`) and the sparse
/// one-marginal-child route, which returns before that tail runs.
#[inline(always)]
pub(super) fn mark_passthrough_inlined(level: &mut TddLevel, left_passthrough: bool, right_passthrough: bool) {
    if (left_passthrough || right_passthrough) && !level.is_marginal() {
        if left_passthrough { level.set_marg_inlined_left(true); }
        if right_passthrough { level.set_marg_inlined_right(true); }
    }
}

/// Per-level tail after the cell-build route dispatch (extraction 5).
///
/// Covers: stream commit (`commit_stream_state`), `live_counts` update,
/// `grids[t_idx]` tagging, `shrink_arrays`, and the pass-through
/// inline-emit flags (`mark_passthrough_inlined`).
///
/// Grid reclamation ([`reclaim_child_grids`]) stays at the call site — the
/// three early-exit routes reclaim without running this tail at all, so it
/// cannot fold in here.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_level(
    lim: &Limits,
    stream_state: &mut Option<StreamLevelState>,
    t: VtreeIdx,
    t_idx: usize,
    t_base: usize,
    might_use_sparse: bool,
    left_passthrough: bool,
    right_passthrough: bool,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    grids: &mut Vec<LevelGrid>,
    live_counts: &mut Vec<usize>,
    out_nodes_so_far: &mut u64,
    ws: Option<&mut crate::weight_store::WeightStore>,
) {
    // Commit streaming-marginal emit: convert level to marginal_counts.
    // Must happen before the `levels[t_idx]` reborrows below; the local
    // `level: &mut TddLevel` borrow ends at last use above (in the j-loop).
    if let Some(st) = stream_state.take() {
        commit_stream_state(st, t, t_idx, vtree, levels, ws);
    }

    // Record live count for parent density checks (only when sparse mode possible).
    // Use `width()` so streaming-marginal levels (nodes.len() == 0 after
    // make_marginal) report their actual alive-cell count.
    if might_use_sparse {
        bump_live_count(live_counts, out_nodes_so_far, t_idx, levels[t_idx].width());
    }
    // Dense emit wrote node_idx in (i, j) row-major order keyed by
    // level.nodes.len() at each emission, so live cells are strictly
    // monotone → eligible for the H1 sort-skip at parent levels.
    grids[t_idx] = LevelGrid::DenseStrict { base: t_base };

    levels[t_idx].shrink_arrays();

    // Output-pair meter: this level is done, so its arena's capacity estimate
    // gives way to the pairs it actually holds.
    lim.level_settled(levels[t_idx].pairs.len() as u64);

    mark_passthrough_inlined(&mut levels[t_idx], left_passthrough, right_passthrough);
}
