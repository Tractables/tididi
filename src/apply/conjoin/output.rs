//! Writing a finished level of the output diagram.

use super::*;

/// Drop the `nodes`, `pairs` and `multi_pairs` arenas of an operand level
/// whose parent is about to be built.
///
/// The caller must hold that nothing reads the level's arenas again: the sweep
/// reads finished children only through the width snapshots taken at setup.
/// `marginal_counts` is kept, so `is_marginal()` stays accurate. Called before
/// the parent's output reserve so the allocator can reuse the freed slabs.
#[inline]
pub(super) fn drop_dead_operand_level(level: &mut crate::diagram::TddLevel) {
    // Arenas that together fit one `Vec` minimum allocation return no slab the
    // output reserve could use; exact capacities, so the test never misfires.
    let bytes = level.nodes.capacity() * std::mem::size_of::<EncodedNode>()
        + level.pairs.capacity() * std::mem::size_of::<ChildPair>()
        + level.multi_pairs.capacity() * std::mem::size_of::<crate::diagram::MultiPairRange>();
    // `dead_pairs` counts garbage in `pairs`, so it is zeroed only where
    // `pairs` is emptied.
    if bytes <= 64 { return; }
    level.nodes = Vec::new();
    level.pairs = Vec::new();
    level.multi_pairs = Vec::new();
    // No arena left to sweep, so no garbage to remember.
    level.dead_pairs = 0;
}

/// Mark a level's pass-through inline-emit flags.
///
/// If this level was built via pass-through, its marginal-side pair fields hold
/// inline counts carried verbatim from the carrier operand (already emitted),
/// not fresh slots. Mark them so the end-of-apply tagger's emit arm skips
/// re-emitting (which would misread an inline count as a slot index →
/// miscount). Guarded on `!is_marginal()`: a level that became marginal during
/// its build had its markers reset by `become_marginal` and has no structural
/// pairs to describe. `passthrough` is emit-gated.
///
/// Shared by the general per-level tail (`finalize_level`) and the sparse
/// one-marginal-child route, which returns before that tail runs.
pub(super) fn mark_passthrough_inlined(level: &mut TddLevel, passthrough: Sides<bool>) {
    if (passthrough.left || passthrough.right) && !level.is_marginal() {
        if passthrough.left { level.set_marginal_inlined(ChildSide::Left, true); }
        if passthrough.right { level.set_marginal_inlined(ChildSide::Right, true); }
    }
}

/// Per-level tail after the cell-build route dispatch: stream commit,
/// `live_counts` update, grid tagging, `shrink_arrays`, the output-pair meter,
/// and the pass-through inline-emit flags.
pub(super) fn finalize_level(
    eng: &Engine,
    stream_state: &mut Option<StreamLevelState>,
    shape: LevelShape,
    output_grid_base: GridBase,
    passthrough: Sides<bool>,
    run: &mut ApplyRun,
    sweep: &mut Sweep<'_>,
) {
    let lim = eng.limits();
    let (t, t_idx) = (shape.t, shape.t.idx());
    let ApplyRun { levels, products, .. } = run;
    // Commit streaming-marginal emit: convert the level to `marginal_counts`,
    // before the `levels[t_idx]` reborrows below.
    if let Some(st) = stream_state.take() {
        commit_stream_state(st, t, sweep.vtree, levels, sweep.ws.as_deref_mut());
    }

    // Record live count for parent density checks (only when sparse mode possible).
    // Use `slot_count()` so streaming-marginal levels (nodes.len() == 0 after
    // `become_marginal`) report their actual alive-cell count.
    if products.arena.is_bump() {
        products.record_live(t_idx, levels[t_idx].slot_count());
    }
    // Dense emit wrote `node_idx` in row-major order keyed by `nodes.len()` at
    // each emission, so live cells are strictly monotone.
    products.arena.set_dense(t_idx, output_grid_base);

    levels[t_idx].shrink_arrays();

    // Output-pair meter: this level is done, so its arena's capacity estimate
    // gives way to the pairs it actually holds.
    lim.level_settled(levels[t_idx].pairs.len() as u64);

    mark_passthrough_inlined(&mut levels[t_idx], passthrough);
}
