//! Whole-level entry points: the sparse route, leaf levels and the output index.

use super::*;
use crate::apply::conjoin::setup::{ApplyRun, LevelShape};
use crate::diagram::Sides;

/// The three product lists one sparse level reads and writes.
pub(crate) struct ProductLists<'a> {
    pub(crate) left: &'a [ProductEntry],
    pub(crate) right: &'a [ProductEntry],
    pub(crate) out: &'a mut Vec<ProductEntry>,
}

/// True when `f` and `g` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, ranges)` on every level is
/// sufficient. This is structural equality, not pointer identity — but it is
/// only sound when no level is marginal, since a marginal level hides its
/// content outside `nodes`/`pairs` where the structural test cannot see it.
pub(crate) fn is_self_conjunction(f: &Tdd, g: &Tdd) -> bool {
    // The shortcut lets `f ∧ g` return `f.clone()` when the operands are the
    // same function. It is a pure perf optimization, never needed for
    // correctness. A marginal level clears `nodes`/`pairs` (integer-marginal) or
    // `pairs` (weight-marginal) and moves its real content into
    // `marginal_counts`/the external weight store — which this structural test
    // does not compare. Two operands agreeing on every explicit level but
    // differing in marginal mass (or holding a marginal×marginal unsound
    // schedule the callers debug-assert against) would compare equal and
    // silently drop one side's content. Bail whenever either operand carries any
    // marginal level.
    if f.levels.iter().any(|l| l.is_marginal()) || g.levels.iter().any(|l| l.is_marginal()) {
        return false;
    }
    f.output == g.output
        && f.levels.iter().zip(g.levels.iter()).all(|(l1, l2)| {
            // `ranges` too: equal nodes+pairs with a differently-arranged `ranges` table
            // is a different function.
            l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ranges == l2.ranges
        })
}

/// Fill `pl` with the identity product mapping for a level where one diagram operand
/// is constant-true. Returns `true` if the level was identity (g-identity or
/// f-identity), `false` otherwise, leaving `pl` untouched for the caller to
/// fill some other way.
///
/// Identity means x ∧ 1 = x — the constant-true operand contributes a single
/// fixed index. The One label is at local index 0 on every level (leaf and
/// internal alike, since `LeafLabel::One = 0` and identity levels are width-1).
pub(crate) fn fill_identity_product_list(
    eng: &Engine,
    left_width: usize,
    right_width: usize,
    right_id: bool,
    left_id: bool,
    pl: &mut Vec<ProductEntry>,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    // The constant-true operand's One node is at index 0 regardless of leaf-ness
    // (`ONE_LEAF_IDX.0 == LeafLabel::One as u32 == 0`), so no leaf/internal split.
    const ID_IDX: u32 = 0;
    if right_id {
        lim.reserve(pl, left_width)?;
        for i in 0..left_width as u32 {
            // x ∧ 1 = x: output index equals f index (identity mapping).
            pl.push(ProductEntry { left_idx: LeftNodeIdx(i), right_idx: RightNodeIdx(ID_IDX), prod_idx: ProductNodeIdx(i) });
        }
        Ok(true)
    } else if left_id {
        lim.reserve(pl, right_width)?;
        for j in 0..right_width as u32 {
            // 1 ∧ x = x: output index equals g index (identity mapping).
            pl.push(ProductEntry { left_idx: LeftNodeIdx(ID_IDX), right_idx: RightNodeIdx(j), prod_idx: ProductNodeIdx(j) });
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Run the scatter for one level: choose which side to iterate and how the
/// candidates are collected, then join. Returns whether the candidates were
/// collected flat, which is where the emit reads them from.
///
/// With both children non-leaf the direction comes from
/// `estimate_scatter_direction`, and its emit-step count decides between a
/// bucket per f parent and the flat list (`flat_candidates_win`); with a
/// leaf child the larger grid is iterated into buckets.
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

    // Output-sensitive join: the one scatter engine, for both leaf and general
    // levels. The general arm carries no dead-probe inner loop; the leaf arm
    // keeps the leaf fast-path shape.
    if !swap_direction {
        scatter_outsens::<false>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, both_non_leaf, flat)?;
    } else {
        scatter_outsens::<true>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, both_non_leaf, flat)?;
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
/// `ensure_buckets_cleared`, `build_reverse_index`, the chunk phases), the
/// marking arrays are emptied by advancing their epoch, and `filtered`'s
/// touched list is cleared with it here for the same reason as `p2_map`'s.
struct WsGuard<'a> {
    ws: crate::limits::pool::PoolGuard<'a, SparseWorkspace>,
    repair: bool,
}

impl<'a> WsGuard<'a> {
    fn new(eng: &'a Engine) -> Self {
        WsGuard { ws: eng.sparse().checkout(eng.limits()), repair: true }
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
/// Phases:
///   A+C: fused scatter-filter by right sibling
///   E:   dedup parent products via `p2_map[p2]`; emit `ChildPair`s
///   F:   counting-sort pairs by parent product, create output nodes
///
/// Phases E+F are chunked by f-parent index range when the projected transient
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
    // Phase F check stays armed. `cfg!` is a compile-time constant, so the
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

    // `plan_e_f_chunks` greedy-packs f-parent indices into Phase E+F chunks
    // under the sparse chunk budget (`usize::MAX` disables). A level that
    // fits in one chunk is flushed once with `drop_consumed=false`,
    // preserving cross-apply par_buckets capacity reuse. Wider levels split
    // into several chunks with `drop_consumed=true`, releasing each consumed
    // range's `par_buckets[p1]` before the next chunk's `emit_pairs` grows.
    // Each chunk reuses `emit_pairs`, `pair_counts` and `sorted_pairs`, so
    // the peak transient stays bounded by the chunk size; `pl_output` grows
    // across chunks, so `prod_idx` stays sequential over the level.
    let level = &mut levels[t_idx];
    let boundaries = if flat {
        plan_e_f_chunks(ws.par_offsets.windows(2).map(|w| (w[1] - w[0]) as usize), shape.f.here, thresholds.chunk_bytes)
    } else {
        plan_e_f_chunks(ws.par_buckets.iter().map(Vec::len), shape.f.here, thresholds.chunk_bytes)
    };
    let is_chunked = boundaries.len() > 2;
    for window in boundaries.windows(2) {
        let (p1_start, p1_end) = (window[0] as usize, window[1] as usize);
        let chunk_parent_start = pl_output.len() as u32;
        ws.emit_pairs.clear();
        flush_chunk_phase_e(eng, ws, pl_output, chunk_parent_start, p1_start, p1_end, flat, is_chunked)?;
        flush_chunk_phase_f(eng, ws, level, pl_output, chunk_parent_start, duplicates_legal)?;
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

/// Fill grid entries at leaf vtree levels from the static `CONJOIN_GRID` table.
///
/// At leaf levels the conjunction is a constant 3×3 truth table (Pos, Neg, One),
/// so we just copy from `CONJOIN_GRID` into `node_idx`. When the arena bumps,
/// grid space is allocated as we go and live counts are recorded for parent
/// density checks; otherwise the grid offsets are pre-computed.
pub(crate) fn apply_leaf_levels(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    run: &mut ApplyRun,
) -> Result<(), OperationError> {
    let ApplyRun { left_widths, right_widths, products, .. } = run;
    for (t, _leaf_var) in vtree.leaf_bottomup() {
        let t_idx = t.idx();
        let left_width = left_widths[t_idx];
        let right_width = right_widths[t_idx];
        let base = products.arena.alloc(eng, t_idx, left_width * right_width)?;
        products.arena.set_dense(t_idx, base);
        let output_grid_base = base.idx();
        let slab = products.arena.slab_mut();
        let mut count = 0usize;
        for i in 0..left_width {
            for j in 0..right_width {
                let val = CONJOIN_GRID[i][j];
                slab[output_grid_base + i * right_width + j] = val;
                if val != NO_PRODUCT { count += 1; }
            }
        }
        if products.arena.is_bump() { products.record_live(t_idx, count); }
    }
    Ok(())
}

/// The conjunction's output local index, or `None` when the product is false.
///
/// Three cases by how the root level was processed:
/// - Dense grid: O(1) lookup in the slab.
/// - Ungridded with a product list: scan the list for the (left_out, right_out) entry.
/// - Ungridded identity: pass through the non-identity operand's output.
///
/// `None` when the grid says so (a `NO_PRODUCT` cell, or no product-list
/// entry) and also when the root level holds no slot at all, where the grid
/// branch reads a cell no producer wrote; the width test below rejects the
/// index in that case, so a `Some` always names an existing slot.
pub(crate) fn compute_apply_output(
    f: &Tdd,
    g: &Tdd,
    run: &ApplyRun,
    vtree: &crate::vtree::Vtree,
) -> Option<NodeIdx> {
    let out_ti = f.output.vtree.idx();
    let out_local = run.products.lookup(out_ti, f.output.local.0, g.output.local.0,
        run.right_widths[out_ti], run.right_identity[out_ti], run.left_identity[out_ti])?;
    // Mirror the later passes' indexing exactly: effective width is
    // `LEAF_WIDTH` for a leaf root and the level's own width otherwise — the
    // same quantity `prune`/`minimize` index their remap arena by.
    let eff_width = if vtree.node(crate::vtree::VtreeIdx(out_ti as u32)).is_leaf() {
        crate::diagram::LEAF_WIDTH
    } else {
        run.levels[out_ti].slot_count()
    };
    if out_local != ZERO && (out_local.0 as usize) >= eff_width {
        return None;
    }
    Some(out_local)
}
