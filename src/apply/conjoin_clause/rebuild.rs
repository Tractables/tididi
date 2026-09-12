//! Rebuilding a spine level as the accumulator level conjoined with the clause.

use super::*;

/// The clause's per-level tables and the two pair buffers, threaded through
/// every spine level's rebuild.
pub(super) struct ClauseTables<'a> {
    /// Per accumulator node, the output index of its `c_t` and `d_t` node,
    /// in one block per spine level.
    pub(super) cd_map: &'a mut [[u32; 2]],
    /// Where each spine level's block starts in `cd_map`.
    pub(super) level_base: &'a [usize],
    /// Whether a level's complement conjunction is needed.
    pub(super) need_dt: &'a [bool],
    /// Whether a level's subtree holds a clause variable.
    pub(super) on_spine: &'a [bool],
    /// The both-relevant type-3 pairs of the node being rebuilt, held back
    /// until its left index changes so the pair list stays sorted.
    pub(super) t3_buf: &'a mut Vec<ChildPair>,
    /// The `d_t` pairs of the node being rebuilt.
    pub(super) dt_pairs: &'a mut Vec<ChildPair>,
}

/// Which sides of a spine level carry clause variables, where their
/// `cd_map` blocks start, and the level's worst-case output pairs per input
/// pair.
#[derive(Clone, Copy)]
pub(super) struct SpineCtx {
    pub(super) both_rel: bool,
    pub(super) left_rel: bool,
    pub(super) left_grid_base: usize,
    pub(super) right_grid_base: usize,
    pub(super) compute_dt: bool,
    pub(super) pair_mult: usize,
}

/// Conjoin one accumulator node with the clause's virtual `c_t` (and `d_t`
/// where the level needs it), writing the emitted node indices into `slot`.
///
/// The caller has already put the level's growth mode in place.
pub(super) fn conjoin_node_with_clause(
    eng: &Engine,
    inputs: &[ChildPair],
    ctx: SpineCtx,
    level: &mut TddLevel,
    slot: usize,
    tables: &mut ClauseTables<'_>,
) -> Result<(), OperationError> {
        // Reserve this node's whole worst case before emitting any of it, so
        // the direct c_t pushes stay infallible `Vec::push`es.
        reserve_pairs_for_emit(eng, level, ctx.pair_mult * inputs.len())?;

        // The virtual `c_t` depends on which children carry clause variables:
        //   only right relevant:  c_t = {(d_L, c_R)}
        //   only left relevant:   c_t = {(c_L, d_R)}
        //   both relevant:        c_t = {(c_L,c_R), (c_L,d_R), (d_L,c_R)}
        // In the single-pair cases the maps are monotone, so the output is
        // already sorted.
        if ctx.both_rel {
            let ct_start = level.pairs.len();
            tables.t3_buf.clear();
            if ctx.compute_dt { tables.dt_pairs.clear(); }
            build_both_rel_pairs(eng, inputs, ctx, level, tables)?;
            emit_clause_node_direct(level, ct_start, tables.cd_map, 0, slot)?;
            // d_t after c_t: the parent's both-relevant pass relies on the
            // ct index being below the dt index.
            if ctx.compute_dt {
                emit_clause_node(tables.dt_pairs, level, tables.cd_map, 1, slot)?;
            }
        } else {
            let ct_start = level.pairs.len();
            if ctx.compute_dt { tables.dt_pairs.clear(); }
            build_single_rel_pairs(eng, inputs, ctx, level, tables)?;
            emit_clause_node_direct(level, ct_start, tables.cd_map, 0, slot)?;
            if ctx.compute_dt {
                emit_clause_node(tables.dt_pairs, level, tables.cd_map, 1, slot)?;
            }
        }
    Ok(())
}

/// Rebuild one spine level as the conjunction of the accumulator's level with
/// the clause, filling this level's `cd_map` block.
pub(super) fn rebuild_spine_level(
    eng: &Engine,
    t: VtreeIdx,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    tables: &mut ClauseTables<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let t_idx = t.idx();
    let (left, right) = vtree.children(t);
    let left_idx = left.idx();
    let right_idx = right.idx();
    let base = tables.level_base[t_idx];
    let compute_dt = tables.need_dt[t_idx];
    let left_rel = tables.on_spine[left_idx];
    let right_rel = tables.on_spine[right_idx];
    let both_rel = left_rel && right_rel;
    let left_grid_base = tables.level_base[left_idx];
    let right_grid_base = tables.level_base[right_idx];

    // `old` holds the accumulator's pairs for this level; the emptied
    // `levels[t_idx]` receives the conjoined output.
    let old = std::mem::take(&mut levels[t_idx]);
    let k = old.slot_count();
    let in_pairs = old.pairs.len();
    let level = &mut levels[t_idx];
    // At most a c_t and a d_t node per accumulator node.
    let node_cap = if compute_dt { 2 * k } else { k };
    level.nodes.try_reserve(node_cap).map_err(|_| OperationError::OverBudget)?;
    // Worst-case output pairs per input pair: up to 3 c_t pairs for a
    // both-relevant node, plus 1 d_t pair when `compute_dt`. The arena is
    // sized at the input pair count and topped up per node, so the peak never
    // holds a whole-level worst case beside the still-live `old`.
    let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
    lim.begin_level(Some((in_pairs as u128).saturating_mul(pair_mult as u128)));
    level.pairs.try_reserve(in_pairs).map_err(|_| OperationError::OverBudget)?;
    let ctx = SpineCtx { both_rel, left_rel, left_grid_base, right_grid_base, compute_dt, pair_mult };
    for i in 0..k {
        debug_assert!(old.nodes[i].is_internal()
            || old.nodes[i].b == u32::MAX,  // inline pair with right=ZERO (dead node)
            "expected internal node at internal vtree position: t={t:?} i={i}");
        let inputs = old.pairs_of_idx(i);
        if inputs.is_empty() {
            // Dead accumulator node: nothing emitted, and this is the one
            // write of its map entry.
            tables.cd_map[base + i] = [NO_PRODUCT, NO_PRODUCT];
            continue;
        }
        conjoin_node_with_clause(eng, inputs, ctx, level, base + i, tables)?;
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
