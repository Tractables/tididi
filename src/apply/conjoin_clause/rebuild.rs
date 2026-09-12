//! Rebuilding a spine level as the accumulator level conjoined with the clause.

use super::*;

/// Which sides of a spine level carry clause variables, and where their
/// `cd_map` blocks start.
#[derive(Clone, Copy)]
pub(super) struct SpineCtx {
    both_rel: bool,
    left_rel: bool,
    left_grid_base: usize,
    right_grid_base: usize,
    compute_dt: bool,
}

/// Conjoin one accumulator node with the clause's virtual `c_t` (and `d_t`
/// where the level needs it), writing the emitted node indices into `slot`.
///
/// `pair_mult` is this level's worst-case output pairs per input pair; the
/// caller has already put the level's growth mode in place.
#[allow(clippy::too_many_arguments)]
pub(super) fn conjoin_node_with_clause(
    eng: &Engine,
    inputs: &[InputPair],
    ctx: SpineCtx,
    pair_mult: usize,
    level: &mut TddLevel,
    cd_map: &mut [[u32; 2]],
    slot: usize,
    clause_t3_buf: &mut Vec<InputPair>,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
        // Reserve this node's whole worst case before emitting any of it, so
        // the direct c_t pushes stay infallible `Vec::push`es.
        reserve_pairs_for_emit(eng, level, pair_mult * inputs.len())?;

        // The virtual `c_t` depends on which children carry clause variables:
        //   only right relevant:  c_t = {(d_L, c_R)}
        //   only left relevant:   c_t = {(c_L, d_R)}
        //   both relevant:        c_t = {(c_L,c_R), (c_L,d_R), (d_L,c_R)}
        // In the single-pair cases the maps are monotone, so the output is
        // already sorted.
        if ctx.both_rel {
            let ct_start = level.pairs.len();
            clause_t3_buf.clear();
            if ctx.compute_dt { clause_dt_pairs.clear(); }
            build_both_rel_pairs(
                eng,
                inputs, ctx.left_grid_base, ctx.right_grid_base, ctx.compute_dt,
                cd_map, level, clause_t3_buf, clause_dt_pairs,
            )?;
            emit_clause_node_direct(level, ct_start, cd_map, 0, slot)?;
            // d_t after c_t: the parent's both-relevant pass relies on the
            // ct index being below the dt index.
            if ctx.compute_dt {
                emit_clause_node(
                    clause_dt_pairs, level, cd_map, 1, slot,
                )?;
            }
        } else {
            let ct_start = level.pairs.len();
            if ctx.compute_dt { clause_dt_pairs.clear(); }
            build_single_rel_pairs(
                eng,
                inputs, ctx.left_rel, ctx.left_grid_base, ctx.right_grid_base, ctx.compute_dt,
                cd_map, level, clause_dt_pairs,
            )?;
            emit_clause_node_direct(level, ct_start, cd_map, 0, slot)?;
            if ctx.compute_dt {
                emit_clause_node(
                    clause_dt_pairs, level, cd_map, 1, slot,
                )?;
            }
        }
    Ok(())
}

/// Rebuild one spine level as the conjunction of the accumulator's level with
/// the clause, filling this level's `cd_map` block.
#[allow(clippy::too_many_arguments)]
pub(super) fn rebuild_spine_level(
    eng: &Engine,
    t: VtreeIdx,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    cd_map: &mut [[u32; 2]],
    level_base: &[usize],
    need_dt: &[bool],
    on_spine: &[bool],
    clause_t3_buf: &mut Vec<InputPair>,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let t_idx = t.idx();
    let (left, right) = vtree.children(t);
    let left_idx = left.idx();
    let right_idx = right.idx();
    let base = level_base[t_idx];
    let compute_dt = need_dt[t_idx];
    let left_rel = on_spine[left_idx];
    let right_rel = on_spine[right_idx];
    let both_rel = left_rel && right_rel;
    let left_grid_base = level_base[left_idx];
    let right_grid_base = level_base[right_idx];

    // `old` holds the accumulator's pairs for this level; the emptied
    // `levels[t_idx]` receives the conjoined output.
    let old = std::mem::take(&mut levels[t_idx]);
    let k = old.width();
    let in_pairs = old.pairs.len();
    let level = &mut levels[t_idx];
    // At most a c_t and a d_t node per accumulator node.
    let node_cap = if compute_dt { 2 * k } else { k };
    level.nodes.try_reserve(node_cap).map_err(|_| ApplyError::OverBudget)?;
    // Worst-case output pairs per input pair: up to 3 c_t pairs for a
    // both-relevant node, plus 1 d_t pair when `compute_dt`. The arena is
    // sized at the input pair count and topped up per node, so the peak never
    // holds a whole-level worst case beside the still-live `old`.
    let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
    lim.begin_level(Some((in_pairs as u128).saturating_mul(pair_mult as u128)));
    level.pairs.try_reserve(in_pairs).map_err(|_| ApplyError::OverBudget)?;
    let ctx = SpineCtx { both_rel, left_rel, left_grid_base, right_grid_base, compute_dt };
    for i in 0..k {
        debug_assert!(old.nodes[i].is_internal()
            || old.nodes[i].b == u32::MAX,  // inline pair with right=ZERO (dead node)
            "expected internal node at internal vtree position: t={t:?} i={i}");
        let inputs = old.pairs_of_idx(i);
        if inputs.is_empty() {
            // Dead accumulator node: nothing emitted, and this is the one
            // write of its map entry.
            cd_map[base + i] = [NO_PRODUCT, NO_PRODUCT];
            continue;
        }
        conjoin_node_with_clause(
            eng,
            inputs, ctx, pair_mult, level, cd_map, base + i,
            clause_t3_buf, clause_dt_pairs,
        )?;
    }
    // Free `old` before `shrink_arrays` reallocates the rebuilt arenas, so
    // the two are not resident together at the peak.
    let old_inlined_sides = old.inlined_sides;
    drop(old);
    // Trim the slack the per-node top-up growth left behind.
    level.shrink_arrays();

    // The rebuilt level copied the irrelevant side's pair refs verbatim,
    // inline marginal counts included, but started with zero `inlined_sides`.
    // The relevant side is never marginal (`plan_cd_map_bases` panics), so
    // the input level's markers carry over exactly.
    level.inlined_sides = old_inlined_sides;
    Ok(())
}
