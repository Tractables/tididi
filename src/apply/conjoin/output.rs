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

/// Per-level live counts used to estimate sparse-grid density.
pub(super) struct LiveCounts {
    per_level: Vec<usize>,
}

impl LiveCounts {
    /// Take the pooled buffer and seed every level's count to zero.
    ///
    /// Zero is the correct "no output nodes built yet" seed for every level,
    /// and pooled reuse can retain stale entries (`resize` only appends or
    /// truncates, never clears the live prefix). A stale value would corrupt
    /// the parent density reads.
    pub(super) fn take(pool: &crate::limits::pool::Pool<Vec<usize>>, num_nodes: usize) -> Self {
        let mut per_level = pool.take();
        per_level.clear();
        per_level.resize(num_nodes, 0);
        LiveCounts { per_level }
    }

    /// Give the buffer back.
    pub(super) fn into_pool(self, pool: &crate::limits::pool::Pool<Vec<usize>>) {
        pool.put(self.per_level);
    }

    /// Record that level `t_idx` now holds `v` output nodes.
    #[inline(always)]
    pub(super) fn set(&mut self, t_idx: usize, v: usize) {
        self.per_level[t_idx] = v;
    }

    /// Level `t_idx`'s output-node count.
    #[inline(always)]
    pub(super) fn at(&self, t_idx: usize) -> usize {
        self.per_level[t_idx]
    }
}

/// Bookkeeping for a freshly built sparse-output level: refresh its live-node
/// count, mark its product list as populated, and shrink its final arrays.
#[inline(always)]
pub(super) fn finish_sparse_output(
    live_counts: &mut LiveCounts,
    has_pl: &mut [bool],
    level: &mut TddLevel,
    t_idx: usize,
) {
    live_counts.set(t_idx, level.nodes.len());
    has_pl[t_idx] = true;
    level.shrink_arrays();
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
#[inline(always)]
pub(super) fn mark_passthrough_inlined(level: &mut TddLevel, passthrough: Sides<bool>) {
    if (passthrough.left || passthrough.right) && !level.is_marginal() {
        if passthrough.left { level.set_marginal_inlined_left(true); }
        if passthrough.right { level.set_marginal_inlined_right(true); }
    }
}

/// Per-level tail after the cell-build route dispatch: stream commit,
/// `live_counts` update, grid tagging, `shrink_arrays`, the output-pair meter,
/// and the pass-through inline-emit flags.
#[inline(always)]
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
    let ApplyRun { levels, arena, live_counts, .. } = run;
    // Commit streaming-marginal emit: convert the level to `marginal_counts`,
    // before the `levels[t_idx]` reborrows below.
    if let Some(st) = stream_state.take() {
        commit_stream_state(st, t, sweep.vtree, levels, sweep.ws.as_deref_mut());
    }

    // Record live count for parent density checks (only when sparse mode possible).
    // Use `slot_count()` so streaming-marginal levels (nodes.len() == 0 after
    // `become_marginal`) report their actual alive-cell count.
    if arena.is_bump() {
        live_counts.set(t_idx, levels[t_idx].slot_count());
    }
    // Dense emit wrote `node_idx` in row-major order keyed by `nodes.len()` at
    // each emission, so live cells are strictly monotone.
    arena.set_dense(t_idx, output_grid_base);

    levels[t_idx].shrink_arrays();

    // Output-pair meter: this level is done, so its arena's capacity estimate
    // gives way to the pairs it actually holds.
    lim.level_settled(levels[t_idx].pairs.len() as u64);

    mark_passthrough_inlined(&mut levels[t_idx], passthrough);
}
