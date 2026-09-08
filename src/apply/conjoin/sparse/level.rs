//! Whole-level entry points: the sparse route, leaf levels and the output index.

use super::*;

/// True when `c1` and `c2` represent the same Boolean function, in which case
/// `apply_and` reduces to `f ∧ f = f` and we can short-circuit to a copy.
/// Canonicity means equal functions have identical *explicit* level structure —
/// so equal `output` plus equal `(nodes, pairs, ext)` on every level is
/// sufficient. This is STRUCTURAL equality, not pointer identity — but it is
/// only sound when no level is marginal, since a marginal level hides its
/// content outside `nodes`/`pairs` where the structural test cannot see it.
pub(crate) fn is_self_conjunction(c1: &Tdd, c2: &Tdd) -> bool {
    // The shortcut lets `c1 ∧ c2` return `c1.clone()` when the operands are the
    // same function. It is a pure perf optimization, never needed for
    // correctness. A marginal level clears `nodes`/`pairs` (integer-marginal) or
    // `pairs` (weight-marginal) and moves its real content into
    // `marginal_counts`/the external weight store — which this structural test
    // does NOT compare. Two operands agreeing on every explicit level but
    // differing in marginal mass (or holding a marginal×marginal unsound
    // schedule the callers debug-assert against) would compare equal and
    // silently drop one side's content. Bail whenever either operand carries any
    // marginal level.
    if c1.levels.iter().any(|l| l.is_marginal()) || c2.levels.iter().any(|l| l.is_marginal()) {
        return false;
    }
    c1.output == c2.output
        && c1.levels.iter().zip(c2.levels.iter()).all(|(l1, l2)| {
            // `ext` too: equal nodes+pairs with a differently-arranged `ext` table
            // is a different function.
            l1.nodes == l2.nodes && l1.pairs == l2.pairs && l1.ext == l2.ext
        })
}

/// Fill `pl` with the identity product mapping for a level where one TDD operand
/// is constant-true. Returns `true` if the level was identity (c2-identity or
/// c1-identity), `false` otherwise. On the identity path also sets
/// `*has_pl_slot = true` itself (co-located with the fill); on the non-identity
/// path `has_pl_slot` is left untouched for the caller to set once it fills `pl`
/// some other way.
///
/// Identity means x ∧ 1 = x — the constant-true operand contributes a single
/// fixed index. The One label is at local index 0 on every level (leaf and
/// internal alike, since `LeafLabel::One = 0` and identity levels are width-1).
pub(crate) fn fill_identity_product_list(
    eng: &Engine,
    k1: usize,
    k2: usize,
    c2_id: bool,
    c1_id: bool,
    pl: &mut Vec<ProductEntry>,
    has_pl_slot: &mut bool,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    // The constant-true operand's One node is at index 0 regardless of leaf-ness
    // (`ONE_LEAF_IDX.0 == LeafLabel::One as u32 == 0`), so no leaf/internal split.
    const ID_IDX: u32 = 0;
    if c2_id {
        lim.reserve(pl, k1)?;
        for i in 0..k1 as u32 {
            // x ∧ 1 = x: output index equals c1 index (identity mapping).
            pl.push(ProductEntry { c1_idx: C1NodeIdx(i), c2_idx: C2NodeIdx(ID_IDX), prod_idx: ProdNodeIdx(i) });
        }
        *has_pl_slot = true;
        Ok(true)
    } else if c1_id {
        lim.reserve(pl, k2)?;
        for j in 0..k2 as u32 {
            // 1 ∧ x = x: output index equals c2 index (identity mapping).
            pl.push(ProductEntry { c1_idx: C1NodeIdx(ID_IDX), c2_idx: C2NodeIdx(j), prod_idx: ProdNodeIdx(j) });
        }
        *has_pl_slot = true;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Process a single internal level using the sparse scatter-filter-dedup pipeline.
///
/// Instead of iterating all k1*k2 cells, builds reverse indices from parent pairs
/// and scatters from live child products upward. Only alive products are touched.
///
/// The scatter and sibling-liveness filter are fused: we iterate by
/// right-sibling s1, populate sib_lookup once per s1, then scatter with
/// inline s2 filtering — avoiding an intermediate candidate buffer.
///
/// Phases:
///   A+C: Fused scatter-filter by right sibling via sib_lookup[s2]
///   E:   Dedup parent products via p2_map[p2]; emit InputPairs
///   F:   Counting-sort pairs by parent product, create output nodes
///
/// Phases E+F are chunked by c1-parent index range when the projected transient
/// cost exceeds `sparse_chunk_bytes()` — each chunk's
/// `par_buckets` rows are dropped before the next chunk's `emit_pairs` grows,
/// capping within-call peak on wide levels.
/// Run the scatter for one level: choose which side to iterate, then join.
///
/// With both children non-leaf the direction comes from a selectivity estimate
/// rather than a grid-size proxy, which mispicks on wide-by-wide conjunctions;
/// with a leaf child the larger grid is iterated.
#[allow(clippy::too_many_arguments)]
fn scatter_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    c1: &Tdd,
    c2: &Tdd,
    t_idx: usize,
    k1: usize, k2: usize,
    k1_left: usize, k2_left: usize,
    k1_right: usize, k2_right: usize,
    left_is_leaf: bool,
    right_is_leaf: bool,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let left_grid = k1_left * k2_left;
    let right_grid = k1_right * k2_right;
    // Direction: selectivity estimator (general path) picks the side with fewer
    // dead probes. Do not substitute a plain grid-size proxy — it ignores
    // selectivity and mispicks on wide×wide segment conjoins.
    let both_non_leaf = !left_is_leaf && !right_is_leaf;
    let swap_direction = if both_non_leaf {
        estimate_scatter_direction(eng, 
            &mut ws.est_counts,
            &c1.levels[t_idx], &c2.levels[t_idx], pl_left, pl_right,
            k1_left, k2_left, k1_right, k2_right,
        )?
    } else {
        left_grid > right_grid
    };

    ensure_buckets_cleared(eng, &mut ws.par_buckets, k1)?;
    lim.try_resize(&mut ws.p2_map, k2, DEAD)?;

    // Output-sensitive join: THE scatter engine, for both leaf and general
    // levels. The general arm carries no dead-probe inner loop (that probe
    // ran 91-98% dead on dense segment conjoins); the leaf arm keeps the
    // leaf fast-path shape. There is no alternative engine to select.
    if !swap_direction {
        scatter_outsens::<false>(eng, ws, &c1.levels[t_idx], &c2.levels[t_idx],
            k1_left, k2_left, k1_right, k2_right,
            pl_left, pl_right, left_is_leaf)?;
    } else {
        scatter_outsens::<true>(eng, ws, &c1.levels[t_idx], &c2.levels[t_idx],
            k1_left, k2_left, k1_right, k2_right,
            pl_left, pl_right, right_is_leaf)?;
    }
    Ok(())
}

pub(crate) fn apply_sparse_level(
    eng: &Engine,
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    c1: &Tdd,
    c2: &Tdd,
    levels: &mut [TddLevel],
    c1_widths: &[usize],
    c2_widths: &[usize],
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    pl_output: &mut Vec<ProductEntry>,
    left_is_leaf: bool,
    right_is_leaf: bool,
    // Whether this level's vars are marginalized after (sparse-STREAM lever
    // eligibility). Used only by the `instrument` build's transient accounting.
    #[allow(unused_variables)] is_marg_target: bool,
) -> Result<(), ApplyError> {
    let t_idx = t.idx();
    let k1 = c1_widths[t_idx];
    let k2 = c2_widths[t_idx];
    let k1_left = c1_widths[left.idx()];
    let k2_left = c2_widths[left.idx()];
    let k1_right = c1_widths[right.idx()];
    let k2_right = c2_widths[right.idx()];

    // No-marginal-leakage guard (tier-0, every build incl. release). The sparse
    // path is never routed for a marginal child level — every marginal-parent
    // level goes to the dedicated marginal-parent dispatch. This
    // matters because the reverse-index below buckets parents by the decoded
    // child coordinate `decode_marg_coord(pair.left.0, …)`; under inline encoding
    // a marginal ref decodes to the COUNT, not a per-node index, collapsing
    // equal-count children into one bucket → dropped multiplicity (the mc007 ×4).
    // The operand-child (c1/c2) checks are load-bearing — an inline marginal ref
    // can only exist on a marginal child level. Always-on so a routing regression
    // aborts loudly instead of silently miscounting.
    cheap_assert!(
        !c1.levels[left.idx()].is_marginal() && !c1.levels[right.idx()].is_marginal()
            && !c2.levels[left.idx()].is_marginal() && !c2.levels[right.idx()].is_marginal()
            && !levels[left.idx()].is_marginal() && !levels[right.idx()].is_marginal(),
        "apply_sparse_level reached with a marginal child (t={t_idx} l={} r={}): \
         the dedicated marginal-parent dispatch was bypassed",
        left.idx(), right.idx()
    );

    let mut ws_guard = eng.sparse().borrow_mut();
    let ws = &mut *ws_guard;

    // Dirty-flag recovery: if the previous sparse apply bailed mid-iteration
    // (e.g. OverBudget from try_push inside the scatter loop), the lazy-cleared
    // lookup tables `sib_lookup`/`child_lookup`/`p2_map` may still hold
    // non-DEAD entries that the scatter-clean cleanup never restored.
    // `try_resize` below is a no-op when the table is already large enough,
    // so without this reset the new apply would read stale prod indices and
    // emit spurious pairs — a silent undercount.
    if ws.dirty {
        ws.sib_lookup.fill(DEAD);
        ws.child_lookup.fill(DEAD);
        ws.p2_map.fill(DEAD);
    }
    ws.dirty = true;

    // Duplicate pairs in one node's list are legal once any level of the
    // diagram is marginal — pair lists are then multisets feeding a sum
    // A duplicate here is *inherited*: an
    // operand parent whose own list holds the same pair twice produces the
    // same product pair twice, which is exactly the multiplicity the count
    // recurrence needs. Only the pure-Boolean case still guarantees
    // set-ness, so that is where the Phase F check stays armed. `cfg!` is a
    // compile-time constant, so the level scan is dead code in release.
    ws.dups_legal = cfg!(debug_assertions)
        && (c1.levels.iter().any(|l| l.is_marginal())
            || c2.levels.iter().any(|l| l.is_marginal())
            || levels.iter().any(|l| l.is_marginal()));

    // ── Fused scatter-filter ──────────────────────────────────────
    //
    // Four-way join: parent(p1,p2) <- c1(p1,a1,s1) /\ c2(p2,a2,s2)
    //                                /\ left_alive(a1,a2) /\ right_alive(s1,s2)
    //
    // Direction chosen by child grid size:
    //   left_grid <= right_grid: outer=s1, probe=sib_lookup (normal)
    //   left_grid >  right_grid: outer=a1, probe=child_lookup (swapped)
    //
    // When the iterated child is a leaf, the reverse index for the
    // opposite operand is keyed by the non-leaf child for selectivity,
    // and CONJOIN_GRID replaces the lookup table for the leaf product.

    scatter_level(eng, 
        ws, c1, c2, t_idx, k1, k2, k1_left, k2_left, k1_right, k2_right,
        left_is_leaf, right_is_leaf, pl_left, pl_right,
    )?;

    // `plan_e_f_chunks` greedy-packs c1-parent indices into Phase E+F chunks
    // under `sparse_chunk_bytes()` (default 256 MiB; `usize::MAX` disables).
    // A level that fits in one chunk takes a single `flush_chunk` call with
    // `drop_consumed=false`, preserving cross-apply par_buckets capacity reuse.
    // Wider levels split into several chunks with `drop_consumed=true`,
    // releasing each consumed range's `par_buckets[p1]` before the next
    // chunk's `emit_pairs` grows.
    let level = &mut levels[t_idx];
    let boundaries = plan_e_f_chunks(&ws.par_buckets, k1, sparse_chunk_bytes());
    let is_chunked = boundaries.len() > 2;
    for window in boundaries.windows(2) {
        flush_chunk(eng, ws, level, pl_output,
            window[0] as usize, window[1] as usize, is_chunked)?;
    }

    #[cfg(debug_assertions)]
    {
        // par_buckets contents are still present in single-chunk mode (we
        // iterated by reference) and replaced with Vec::new() in multi-chunk
        // mode. Either way they're "logically consumed" — the next apply's
        // ensure_buckets_cleared will reset length. No structural assertion
        // here; pl_output / level.nodes invariants below catch real bugs.
        //
        // pl_output grew monotonically and prod_idx[i] == i.
        for (i, e) in pl_output.iter().enumerate() {
            debug_assert_eq!(e.prod_idx.idx(), i,
                "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
        }
        debug_assert!(levels[t_idx].nodes.len() == pl_output.len(),
            "level.nodes.len() {} != pl_output.len() {}",
            levels[t_idx].nodes.len(), pl_output.len());
    }

    // Scatter-clean cleanup completed; lookup tables are all DEAD again.
    // The dirty-flag recovery at entry is unnecessary on the next call.
    ws.dirty = false;
    Ok(())
}

/// Fill grid entries at leaf vtree levels from the static `CONJOIN_GRID` table.
///
/// At leaf levels the conjunction is a constant 3×3 truth table (Pos, Neg, One),
/// so we just copy from `CONJOIN_GRID` into `node_idx`. When `might_use_sparse`,
/// grid space is bump-allocated as we go and live counts are recorded for
/// parent density checks; otherwise the grid offsets are pre-computed.
pub(crate) fn apply_leaf_levels(
    eng: &Engine,
    vtree: &crate::vtree::Vtree,
    c1_widths: &[usize],
    c2_widths: &[usize],
    grids: &mut [LevelGrid],
    node_idx: &mut Vec<u32>,
    grid_end: &mut usize,
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    might_use_sparse: bool,
    // Spine-bounded apply: the leaves that are children of a rebuilt level.
    // Every other leaf's grid is unreachable — its parent rides through
    // untouched — so building it would be pure waste. `None` = every leaf.
    only: Option<&[crate::vtree::VtreeIdx]>,
) -> Result<(), ApplyError> {
    let mut one_leaf = |t: crate::vtree::VtreeIdx| -> Result<(), ApplyError> {
        let t_idx = t.idx();
        let k1 = c1_widths[t_idx];
        let k2 = c2_widths[t_idx];
        let t_base = if might_use_sparse {
            let base = *grid_end;
            *grid_end += k1 * k2;
            try_resize_dead(eng, node_idx, *grid_end)?;
            base
        } else {
            grids[t_idx].base_unchecked()
        };
        grids[t_idx] = LevelGrid::Leaf { base: t_base };
        let mut count = 0usize;
        for i in 0..k1 {
            for j in 0..k2 {
                let val = CONJOIN_GRID[i][j];
                node_idx[t_base + i * k2 + j] = val;
                if val != DEAD { count += 1; }
            }
        }
        if might_use_sparse { bump_live_count(live_counts, out_nodes_so_far, t_idx, count); }
        Ok(())
    };
    match only {
        Some(l) => for &t in l { one_leaf(t)?; },
        None => for (t, _leaf_var) in vtree.leaf_bottomup() { one_leaf(t)?; },
    }
    Ok(())
}

/// Compute the output local index for the conjunction TDD.
///
/// Three cases by how the root level was processed:
/// - Dense grid: O(1) lookup at `node_idx[out_flat]`.
/// - Sparse with product list: scan list for the (c1_out, c2_out) entry.
/// - Sparse identity: pass through the non-identity operand's output.
///
/// Returns ZERO if the root conjunction is unsatisfiable.
pub(crate) fn compute_apply_output(
    c1: &Tdd,
    c2: &Tdd,
    grids: &[LevelGrid],
    node_idx: &[u32],
    c2_widths: &[usize],
    c1_identity: &[bool],
    c2_identity: &[bool],
    has_pl: &[bool],
    product_lists: &[Vec<ProductEntry>],
) -> LocalNodeIdx {
    let out_ti = c1.output.vtree.idx();
    if let Some(out_base) = grids[out_ti].base() {
        let out_flat = out_base
            + c1.output.local.idx() * c2_widths[out_ti]
            + c2.output.local.idx();
        let val = node_idx[out_flat];
        if val != DEAD { LocalNodeIdx(val) } else { ZERO }
    } else {
        let c1_out = c1.output.local.0;
        let c2_out = c2.output.local.0;
        if !has_pl[out_ti] {
            // Root is an identity level: pass through the non-identity operand's output.
            if c2_identity[out_ti] {
                LocalNodeIdx(c1_out)
            } else if c1_identity[out_ti] {
                LocalNodeIdx(c2_out)
            } else {
                // Invariant violation, not an UNSAT result: fabricating ZERO here
                // would silently miscount. Abort loudly in every build (A7).
                cheap_assert!(
                    false,
                    "compute_apply_output: root level t={out_ti} has no grid, no \
                     product list, and neither identity flag — apply-routing invariant \
                     violated (c1_out={c1_out} c2_out={c2_out})"
                );
                ZERO
            }
        } else {
            product_lists[out_ti]
                .iter()
                .find(|e| e.c1_idx == C1NodeIdx(c1_out) && e.c2_idx == C2NodeIdx(c2_out))
                .map(|e| LocalNodeIdx(e.prod_idx.0))
                .unwrap_or(ZERO)
        }
    }
}
