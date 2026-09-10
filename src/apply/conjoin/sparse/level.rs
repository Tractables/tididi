//! Whole-level entry points: the sparse route, leaf levels and the output index.

use super::*;
use crate::apply::conjoin::setup::LevelShape;
use crate::apply::conjoin::marginal_plan::Sides;
use crate::apply::conjoin::grid_arena::GridArena;
use crate::apply::conjoin::output::LiveCounts;

/// True when `f` and `g` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, multi_pairs)` on every level is
/// sufficient. This is STRUCTURAL equality, not pointer identity — but it is
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
            // `multi_pairs` too: equal nodes+pairs with a differently-arranged `multi_pairs` table
            // is a different function.
            l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.multi_pairs == l2.multi_pairs
        })
}

/// Fill `pl` with the identity product mapping for a level where one diagram operand
/// is constant-true. Returns `true` if the level was identity (g-identity or
/// f-identity), `false` otherwise. On the identity path also sets
/// `*has_pl_slot = true` itself (co-located with the fill); on the non-identity
/// path `has_pl_slot` is left untouched for the caller to set once it fills `pl`
/// some other way.
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
    has_pl_slot: &mut bool,
) -> Result<bool, ApplyError> {
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
        *has_pl_slot = true;
        Ok(true)
    } else if left_id {
        lim.reserve(pl, right_width)?;
        for j in 0..right_width as u32 {
            // 1 ∧ x = x: output index equals g index (identity mapping).
            pl.push(ProductEntry { left_idx: LeftNodeIdx(ID_IDX), right_idx: RightNodeIdx(j), prod_idx: ProductNodeIdx(j) });
        }
        *has_pl_slot = true;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Process a single internal level using the sparse scatter-filter-dedup pipeline.
///
/// Instead of iterating all left_width*right_width cells, builds reverse indices from parent pairs
/// and scatters from live child products upward. Only alive products are touched.
///
/// The scatter and the sibling-liveness filter are fused: we iterate by
/// right-sibling s1 and scatter with inline s2 filtering, which avoids an
/// intermediate candidate buffer.
///
/// Phases:
///   A+C: Fused scatter-filter by right sibling
///   E:   Dedup parent products via p2_map[p2]; emit InputPairs
///   F:   Counting-sort pairs by parent product, create output nodes
///
/// Phases E+F are chunked by f-parent index range when the projected transient
/// cost exceeds the engine's sparse chunk budget — each chunk's
/// `par_buckets` rows are dropped before the next chunk's `emit_pairs` grows,
/// capping within-call peak on wide levels.
/// Run the scatter for one level: choose which side to iterate, then join.
///
/// With both children non-leaf the direction comes from a selectivity estimate
/// rather than a grid-size proxy, which mispicks on wide-by-wide conjunctions;
/// with a leaf child the larger grid is iterated.
fn scatter_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    leaves: Sides<bool>,
    pl: Sides<&[ProductEntry]>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let t_idx = shape.t.idx();
    // Direction: selectivity estimator (general path) picks the side with fewer
    // dead probes. Do not substitute a plain grid-size proxy — it ignores
    // selectivity and mispicks on wide×wide segment conjoins.
    let both_non_leaf = !leaves.left && !leaves.right;
    let swap_direction = if both_non_leaf {
        estimate_scatter_direction(
            eng,
            &mut ws.est_counts,
            &f.levels[t_idx], &g.levels[t_idx], pl.left, pl.right,
            shape,
        )?
    } else {
        shape.f.left * shape.g.left > shape.f.right * shape.g.right
    };

    ensure_buckets_cleared(eng, &mut ws.par_buckets, shape.f.here)?;
    lim.try_resize(&mut ws.p2_map, shape.g.here, NO_PRODUCT)?;

    // Output-sensitive join: THE scatter engine, for both leaf and general
    // levels. The general arm carries no dead-probe inner loop; the leaf arm
    // keeps the leaf fast-path shape. There is no alternative engine to
    // select.
    if !swap_direction {
        scatter_outsens::<false>(eng, ws, &f.levels[t_idx], &g.levels[t_idx],
            shape, pl, leaves.left)?;
    } else {
        scatter_outsens::<true>(eng, ws, &f.levels[t_idx], &g.levels[t_idx],
            shape, pl, leaves.right)?;
    }
    Ok(())
}

/// The sparse workspace, borrowed for one level.
///
/// `p2_map` is lazily cleared — the emit pass restores only the entries it
/// wrote — so a bail mid-level (an `OverBudget` out of a `try_push` deep in the
/// scatter) would leave stale product indices behind, and the next level would
/// read them as live and undercount. The guard makes that impossible without
/// any state surviving the call: the repair runs in `Drop`, on the bail path
/// only, because [`WsGuard::scatter_clean`] disarms it once the level's own
/// cleanup has finished.
struct WsGuard<'a> {
    ws: std::cell::RefMut<'a, SparseWorkspace>,
    repair: bool,
}

impl<'a> WsGuard<'a> {
    fn new(eng: &'a Engine) -> Self {
        WsGuard { ws: eng.sparse().borrow_mut(), repair: true }
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sparse_level(
    eng: &Engine,
    shape: LevelShape,
    f: &Tdd,
    g: &Tdd,
    levels: &mut [TddLevel],
    pl: Sides<&[ProductEntry]>,
    pl_output: &mut Vec<ProductEntry>,
    leaves: Sides<bool>,
) -> Result<(), ApplyError> {
    let t_idx = shape.t.idx();

    assert_no_marginal_children(t_idx, shape.left, shape.right, f, g, levels);

    let mut guard = WsGuard::new(eng);
    let ws = &mut *guard;

    // Duplicate pairs in one node's list are legal once any level of the
    // diagram is marginal — pair lists are then multisets feeding a sum
    // A duplicate here is *inherited*: an
    // operand parent whose own list holds the same pair twice produces the
    // same product pair twice, which is exactly the multiplicity the count
    // recurrence needs. Only the pure-Boolean case still guarantees
    // set-ness, so that is where the Phase F check stays armed. `cfg!` is a
    // compile-time constant, so the level scan is dead code in release.
    ws.duplicates_legal = cfg!(debug_assertions)
        && (f.levels.iter().any(|l| l.is_marginal())
            || g.levels.iter().any(|l| l.is_marginal())
            || levels.iter().any(|l| l.is_marginal()));

    // ── Fused scatter-filter ──────────────────────────────────────
    //
    // Four-way join: parent(p1,p2) <- f(p1,a1,s1) /\ g(p2,a2,s2)
    //                                /\ left_alive(a1,a2) /\ right_alive(s1,s2)
    //
    // Direction chosen by child grid size:
    //   left_grid <= right_grid: outer=s1 (normal)
    //   left_grid >  right_grid: outer=a1 (swapped)
    //
    // When the iterated child is a leaf, the reverse index for the
    // opposite operand is keyed by the non-leaf child for selectivity,
    // and CONJOIN_GRID supplies the leaf product directly.

    scatter_level(eng, ws, f, g, shape, leaves, pl)?;

    // `plan_e_f_chunks` greedy-packs f-parent indices into Phase E+F chunks
    // under the engine's sparse chunk budget (`usize::MAX` disables).
    // A level that fits in one chunk takes a single `flush_chunk` call with
    // `drop_consumed=false`, preserving cross-apply par_buckets capacity reuse.
    // Wider levels split into several chunks with `drop_consumed=true`,
    // releasing each consumed range's `par_buckets[p1]` before the next
    // chunk's `emit_pairs` grows.
    let level = &mut levels[t_idx];
    let boundaries = plan_e_f_chunks(&ws.par_buckets, shape.f.here, eng.tuning().sparse_chunk_bytes);
    let is_chunked = boundaries.len() > 2;
    for window in boundaries.windows(2) {
        flush_chunk(eng, ws, level, pl_output,
            window[0] as usize, window[1] as usize, is_chunked)?;
    }

    debug_check_flushed_level(pl_output, &levels[t_idx]);

    guard.scatter_clean();
    Ok(())
}


/// Refuse a sparse apply whose children hold marginal levels.
///
/// Every marginal-parent level goes to the dedicated marginal-parent dispatch.
/// This matters because the reverse index buckets parents by the decoded child
/// coordinate: under inline encoding a marginal ref decodes to the COUNT, not a
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
        debug_assert_eq!(e.prod_idx.idx(), i,
            "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
    }
    debug_assert!(level.nodes.len() == pl_output.len(),
        "level.nodes.len() {} != pl_output.len() {}", level.nodes.len(), pl_output.len());
}

#[cfg(not(debug_assertions))]
fn debug_check_flushed_level(_pl_output: &[ProductEntry], _level: &TddLevel) {}

/// Fill grid entries at leaf vtree levels from the static `CONJOIN_GRID` table.
///
/// At leaf levels the conjunction is a constant 3×3 truth table (Pos, Neg, One),
/// so we just copy from `CONJOIN_GRID` into `node_idx`. When `might_use_sparse`,
/// grid space is bump-allocated as we go and live counts are recorded for
/// parent density checks; otherwise the grid offsets are pre-computed.
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_leaf_levels(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    left_widths: &[usize],
    right_widths: &[usize],
    arena: &mut GridArena,
    live_counts: &mut LiveCounts,
    // MergeScope-bounded apply: the leaves that are children of a rebuilt level.
    // Every other leaf's grid is unreachable — its parent rides through
    // untouched — so building it would be pure waste. `None` = every leaf.
    only: Option<&[crate::vtree::VtreeIdx]>,
) -> Result<(), ApplyError> {
    let mut one_leaf = |t: crate::vtree::VtreeIdx| -> Result<(), ApplyError> {
        let t_idx = t.idx();
        let left_width = left_widths[t_idx];
        let right_width = right_widths[t_idx];
        let base = arena.alloc(eng, t_idx, left_width * right_width)?;
        arena.set_leaf(t_idx, base);
        let output_grid_base = base.idx();
        let slab = arena.slab_mut();
        let mut count = 0usize;
        for i in 0..left_width {
            for j in 0..right_width {
                let val = CONJOIN_GRID[i][j];
                slab[output_grid_base + i * right_width + j] = val;
                if val != NO_PRODUCT { count += 1; }
            }
        }
        if arena.is_bump() { live_counts.bump(t_idx, count); }
        Ok(())
    };
    match only {
        Some(l) => for &t in l { one_leaf(t)?; },
        None => for (t, _leaf_var) in vtree.leaf_bottomup() { one_leaf(t)?; },
    }
    Ok(())
}

/// The conjunction's output local index, or `None` when the product is FALSE.
///
/// Three cases by how the root level was processed:
/// - Dense grid: O(1) lookup in the slab.
/// - Ungridded with a product list: scan the list for the (left_out, right_out) entry.
/// - Ungridded identity: pass through the non-identity operand's output.
///
/// `None` covers both ways a conjunction comes out FALSE. The grid may say so
/// directly (a `NO_PRODUCT` cell, or no entry in the product list), or the root level
/// may hold no materialized slot at all — and then the grid branch reads a cell
/// no producer wrote and hands back an index past the level's slot count.
/// Either way the answer is the same, and the width test below is exactly the
/// one the later passes index by, so an index they could not use never leaves
/// this function. A true result always indexes an existing slot, and
/// constant-TRUE keeps the width at least one at every internal level, so it
/// never reaches the FALSE branch.
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_apply_output(
    f: &Tdd,
    g: &Tdd,
    arena: &GridArena,
    right_widths: &[usize],
    left_identity: &[bool],
    right_identity: &[bool],
    has_pl: &[bool],
    product_lists: &[Vec<ProductEntry>],
    levels: &[TddLevel],
    vtree: &crate::vtree::Vtree,
) -> Option<NodeIdx> {
    let out_ti = f.output.vtree.idx();
    let out_local = if let Some(out_base) = arena.materialized(out_ti) {
        let out_flat = out_base.idx()
            + f.output.local.idx() * right_widths[out_ti]
            + g.output.local.idx();
        let val = arena.slab()[out_flat];
        if val == NO_PRODUCT { return None; }
        NodeIdx(val)
    } else {
        let left_out = f.output.local.0;
        let right_out = g.output.local.0;
        if !has_pl[out_ti] {
            // Root is an identity level: pass through the non-identity operand's output.
            if right_identity[out_ti] {
                NodeIdx(left_out)
            } else if left_identity[out_ti] {
                NodeIdx(right_out)
            } else {
                // Invariant violation, not an UNSAT result: fabricating ZERO here
                // would silently miscount. Abort loudly in every build (A7).
                cheap_assert!(
                    false,
                    "compute_apply_output: root level t={out_ti} has no grid, no \
                     product list, and neither identity flag — apply-routing invariant \
                     violated (left_out={left_out} right_out={right_out})"
                );
                return None;
            }
        } else {
            let hit = product_lists[out_ti]
                .iter()
                .find(|e| e.left_idx == LeftNodeIdx(left_out) && e.right_idx == RightNodeIdx(right_out))?;
            NodeIdx(hit.prod_idx.0)
        }
    };
    // Mirror the later passes' indexing exactly: effective width is
    // `LEAF_WIDTH` for a leaf root and the level's own width otherwise — the
    // same quantity `prune`/`minimize` index their remap arena by.
    let eff_width = if vtree.node(crate::vtree::VtreeIdx(out_ti as u32)).is_leaf() {
        crate::diagram::LEAF_WIDTH
    } else {
        levels[out_ti].width()
    };
    if out_local != ZERO && (out_local.0 as usize) >= eff_width {
        return None;
    }
    Some(out_local)
}
