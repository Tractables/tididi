//! The sparse route's whole-level entry point.

use super::*;
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::Sides;

/// Run the scatter for one level: choose which side to iterate and how the
/// candidates are collected, then join. Returns whether the candidates were
/// collected flat, which is where the emit reads them from.
///
/// With both children non-leaf the direction comes from
/// `estimate_scatter_direction`, and its emit-step count decides between a
/// bucket per f parent and the flat list (`flat_candidates_win`). With a
/// leaf child the larger child grid is the outer side, into buckets; the leaf
/// arm runs only when the leaf is the inner side, otherwise the general arm
/// runs with the leaf as its outer child.
fn scatter_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    thresholds: SparseThresholds,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let t_idx = shape.t.idx();
    let leaves = Sides {
        left: f.vtree.node(shape.left).is_leaf(),
        right: f.vtree.node(shape.right).is_leaf(),
    };
    // Direction: the estimator (general path) sums what each direction walks
    // around the emit. Do not substitute a plain grid-size proxy — it ignores
    // selectivity and mispicks on wide×wide segment conjoins.
    let both_non_leaf = !leaves.left && !leaves.right;
    let (swap_direction, flat) = if both_non_leaf {
        let choice = estimate_scatter_direction(
            eng,
            &mut ws.est_counts,
            &f.levels[t_idx], &g.levels[t_idx], pl.left, pl.right,
            shape,
        )?;
        (choice.swapped, flat_candidates_win(thresholds, shape.f.here, choice.emit_steps))
    } else {
        (shape.f.left * shape.g.left > shape.f.right * shape.g.right, false)
    };

    if flat {
        ws.par_flat.clear();
    } else {
        ensure_buckets_cleared(eng, &mut ws.par_buckets, shape.f.here)?;
    }
    lim.try_resize(&mut ws.p2_map, shape.g.here, NO_PRODUCT)?;

    // The general arm carries no dead-probe inner loop; the leaf arm keeps
    // the leaf fast-path shape.
    if !swap_direction {
        scatter_join::<false>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, both_non_leaf, flat)?;
    } else {
        scatter_join::<true>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, both_non_leaf, flat)?;
    }
    if flat {
        sort_candidates(eng, ws, shape.f.here)?;
        // Sorted, the flat list is dead; its capacity stays for the next
        // level and the retention cap decides its fate at checkout's end.
        ws.par_flat.clear();
    }
    Ok(flat)
}

/// The sparse workspace, checked out for one level.
///
/// `p2_map` is lazily cleared — the emit pass restores only the entries it
/// wrote, listed in `p2_map_touched` — so a bail mid-level (an `OverBudget`
/// out of a `try_push` deep in the scatter) would leave stale product indices
/// behind, and the next level would read them as live and undercount; a stale
/// touched list would index a narrower level's map out of bounds. The guard
/// makes that impossible without any state surviving the call: the repair
/// runs in `Drop`, on the bail path only, because [`WsGuard::scatter_clean`]
/// disarms it once the level's own cleanup has finished.
///
/// Nothing else needs repair. Every other buffer is resized, filled or
/// cleared over its live range when the next level enters (`sides`,
/// `ensure_buckets_cleared`, `build_reverse_index`, the chunk steps), the
/// marking arrays are emptied by advancing their epoch, and `filtered`'s
/// touched list is cleared with it here for the same reason as `p2_map`'s.
struct WsGuard<'a> {
    ws: crate::execution::pool::PoolGuard<'a, SparseWorkspace>,
    repair: bool,
}

impl<'a> WsGuard<'a> {
    fn new(eng: &'a Engine) -> Self {
        WsGuard { ws: eng.scratch.sparse.checkout(eng.limits()), repair: true }
    }

    /// The lookup tables are all `NO_PRODUCT` again; nothing to repair.
    fn scatter_clean(&mut self) {
        self.repair = false;
    }
}

impl Drop for WsGuard<'_> {
    fn drop(&mut self) {
        if self.repair {
            self.ws.p2_map.fill(NO_PRODUCT);
            self.ws.p2_map_touched.clear();
            self.ws.filtered_touched.clear();
        }
    }
}

impl std::ops::Deref for WsGuard<'_> {
    type Target = SparseWorkspace;
    fn deref(&self) -> &SparseWorkspace {
        &self.ws
    }
}

impl std::ops::DerefMut for WsGuard<'_> {
    fn deref_mut(&mut self) -> &mut SparseWorkspace {
        &mut self.ws
    }
}

/// Build one internal level by the sparse scatter-filter-dedup pipeline:
/// reverse indices over the parent pairs, a scatter from the live child
/// products upward, then dedup and emit. Only alive products are touched.
///
/// Steps:
///   scatter:    fused scatter-filter by the outer child
///   dedup:      dedup parent products via `p2_map[g_parent]`; emit `ChildPair`s
///   node build: counting-sort pairs by parent product, create output nodes
///
/// Dedup and node build are chunked by f-parent index range when the projected transient
/// exceeds `thresholds.chunk_bytes`; each chunk's `par_buckets` rows are
/// dropped before the next chunk's `emit_pairs` grows. A level whose
/// candidates were collected flat holds them in one sorted list the chunks
/// read in place, so chunking bounds only its emit buffers.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a workspace or output reservation is refused.
pub(crate) fn apply_sparse_level(
    eng: &Engine,
    shape: LevelShape,
    f: &Tdd,
    g: &Tdd,
    levels: &mut [TddLevel],
    lists: ProductLists<'_>,
    thresholds: SparseThresholds,
) -> Result<(), OperationError> {
    let t_idx = shape.t.idx();
    let ProductLists { left, right, out: pl_output } = lists;
    let pl = Sides { left, right };

    assert_no_marginal_children(t_idx, shape.left, shape.right, f, g, levels);

    let mut guard = WsGuard::new(eng);
    let ws = &mut *guard;

    // Duplicate pairs in one node's list are legal once any level of the
    // diagram is marginal — pair lists are then multisets feeding a sum.
    // A duplicate here is *inherited*: an operand parent whose own list
    // holds the same pair twice produces the same product pair twice, which
    // is exactly the multiplicity the count recurrence needs. Only the
    // pure-Boolean case still guarantees set-ness, so that is where the
    // node build's check stays armed. `cfg!` is a compile-time constant, so the
    // level scan is dead code in release.
    let duplicates_legal = cfg!(debug_assertions)
        && (f.levels.iter().any(|l| l.is_marginal())
            || g.levels.iter().any(|l| l.is_marginal())
            || levels.iter().any(|l| l.is_marginal()));

    // ── Fused scatter-filter ──────────────────────────────────────
    //
    // Four-way join: parent(p1,p2) <- f(p1,a1,s1) /\ g(p2,a2,s2)
    //                                /\ left_alive(a1,a2) /\ right_alive(s1,s2)
    //
    // When the iterated child is a leaf, the reverse index for the
    // opposite operand is keyed by the non-leaf child for selectivity,
    // and `CONJOIN_GRID` supplies the leaf product directly.

    let flat = scatter_level(eng, ws, f, g, shape, pl, thresholds)?;

    // `plan_chunks` greedy-packs f-parent indices into dedup and node-build chunks
    // under the sparse chunk budget (`usize::MAX` disables). A level that
    // fits in one chunk is flushed once with `drop_consumed=false`,
    // preserving cross-apply par_buckets capacity reuse. Wider levels split
    // into several chunks with `drop_consumed=true`, releasing each consumed
    // range's `par_buckets[p1]` before the next chunk's `emit_pairs` grows.
    // Each chunk reuses `emit_pairs` and `pairs_by_parent`, so
    // the peak transient stays bounded by the chunk size; `pl_output` grows
    // across chunks, so `prod_idx` stays sequential over the level.
    let level = &mut levels[t_idx];
    let boundaries = if flat {
        plan_chunks(ws.par_sorted.offsets.windows(2).map(|w| (w[1] - w[0]) as usize), shape.f.here, thresholds.chunk_bytes)
    } else {
        plan_chunks(ws.par_buckets.iter().map(Vec::len), shape.f.here, thresholds.chunk_bytes)
    };
    let is_chunked = boundaries.len() > 2;
    for window in boundaries.windows(2) {
        let (p1_start, p1_end) = (window[0] as usize, window[1] as usize);
        let chunk_parent_start = pl_output.len() as u32;
        ws.emit_pairs.clear();
        dedup_chunk(eng, ws, pl_output, chunk_parent_start, p1_start, p1_end, flat, is_chunked)?;
        build_chunk_nodes(eng, ws, level, pl_output, chunk_parent_start, duplicates_legal)?;
    }

    #[cfg(debug_assertions)]
    debug_check_flushed_level(pl_output, &levels[t_idx]);

    guard.scatter_clean();
    Ok(())
}


/// Refuse a sparse apply whose children hold marginal levels.
///
/// Every marginal-parent level goes to the dedicated marginal-parent dispatch.
/// This matters because the reverse index buckets parents by the decoded child
/// coordinate: under inline encoding a marginal ref decodes to the count, not a
/// per-node index, collapsing equal-count children into one bucket and dropping
/// multiplicity. The operand-child checks are load-bearing — an inline marginal
/// ref can only exist on a marginal child level. Armed in every build, release
/// included, so a routing regression aborts loudly instead of miscounting.
fn assert_no_marginal_children(
    t_idx: usize,
    left: VtreeIdx,
    right: VtreeIdx,
    f: &Tdd,
    g: &Tdd,
    levels: &[TddLevel],
) {
    cheap_assert!(
        !f.levels[left.idx()].is_marginal() && !f.levels[right.idx()].is_marginal()
            && !g.levels[left.idx()].is_marginal() && !g.levels[right.idx()].is_marginal()
            && !levels[left.idx()].is_marginal() && !levels[right.idx()].is_marginal(),
        "apply_sparse_level reached with a marginal child (t={t_idx} l={} r={}): \
         the dedicated marginal-parent dispatch was bypassed",
        left.idx(), right.idx()
    );
}

/// Check the invariants a flushed level leaves behind: `pl_output` grew
/// monotonically with `prod_idx[i] == i`, and it has one entry per node.
///
/// `par_buckets` needs no check — single-chunk mode iterated it by reference and
/// multi-chunk mode replaced each consumed bucket, and either way the next
/// apply's `ensure_buckets_cleared` resets the lengths.
#[cfg(debug_assertions)]
fn debug_check_flushed_level(pl_output: &[ProductEntry], level: &TddLevel) {
    for (i, e) in pl_output.iter().enumerate() {
        debug_assert_eq!(e.prod_idx.0 as usize, i,
            "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
    }
    debug_assert!(level.nodes.len() == pl_output.len(),
        "level.nodes.len() {} != pl_output.len() {}", level.nodes.len(), pl_output.len());
}
