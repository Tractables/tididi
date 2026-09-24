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
    /// Disjunction only: the `cd_map` slot of the accumulator's output node
    /// and the one extra pair naming the cube, appended to that node's `c_t`
    /// pair list. `None` for a conjunction; see [`cube`](super::cube).
    pub(super) output_cube_pair: Option<(usize, ChildPair)>,
}

/// The child-map offsets and worst-case output pairs per input pair for a spine level.
#[derive(Clone, Copy)]
pub(super) struct SpineCtx {
    pub(super) left_grid_base: usize,
    pub(super) right_grid_base: usize,
    pub(super) pair_mult: usize,
}

/// Conjoin one accumulator node with the clause's virtual `c_t` (and `d_t`
/// where the level needs it), writing the emitted node indices into `slot`.
///
/// The caller has already put the level's growth mode in place.
pub(super) fn conjoin_node_with_clause<const LEFT: bool, const RIGHT: bool, const DT: bool>(
    eng: &Engine,
    inputs: &[ChildPair],
    ctx: SpineCtx,
    level: &mut TddLevel,
    slot: usize,
    tables: &mut ClauseTables<'_>,
) -> Result<(), OperationError> {
    // The cube pair of a disjunction, if this is the node that carries it.
    let cube_pair = tables.output_cube_pair.and_then(|(s, p)| (s == slot).then_some(p));
    // Reserve this node's whole worst case before emitting any of it, so
    // the direct c_t pushes stay infallible `Vec::push`es.
    reserve_pairs_for_emit(eng, level, ctx.pair_mult * inputs.len() + usize::from(cube_pair.is_some()))?;
    let ct_start = level.pairs.len();
    if DT { tables.dt_pairs.clear(); }

    // The virtual c_t has three products when both children carry clause
    // variables, and one otherwise; each builder leaves its pairs sorted.
    if LEFT && RIGHT {
        tables.t3_buf.clear();
        build_both_rel_pairs::<DT>(eng, inputs, ctx, level, tables)?;
    } else {
        build_single_rel_pairs::<LEFT, DT>(eng, inputs, ctx, level, tables)?;
    }
    if let Some(pair) = cube_pair {
        // `c_t` emits only the (c,c), (c,d) and (d,c) cells, so the cube's
        // (d,d) cell is free and the union stays disjoint. Its child indices
        // are the cube chain's, unrelated to the accumulator's order, so the
        // node's pair list is re-sorted.
        level.pairs.push(pair);
        sort_pairs(&mut level.pairs[ct_start..]);
    }
    let ct = emit_from(level, ct_start)?;
    // The parent's both-relevant pass requires the c_t index below d_t. The
    // d_t pairs fit the reservation above: at most one per input pair, which
    // `pair_mult` counts.
    let dt = if DT {
        let dt_start = level.pairs.len();
        level.pairs.extend_from_slice(tables.dt_pairs);
        emit_from(level, dt_start)?
    } else {
        NO_PRODUCT
    };
    tables.cd_map[slot] = [ct, dt];
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
    let mut old = std::mem::take(&mut levels[t_idx]);
    let k = old.slot_count();
    let in_pairs = old.pairs.len();
    let level = &mut levels[t_idx];
    // Tiny levels keep their input descriptors on the stack so the output can reuse the node arena.
    let mut inline_nodes = [EncodedNode { a: 0, b: 0 }; 4];
    let nodes = if k <= inline_nodes.len() {
        inline_nodes[..k].copy_from_slice(&old.nodes);
        level.nodes = std::mem::take(&mut old.nodes);
        level.nodes.clear();
        &inline_nodes[..k]
    } else {
        old.nodes.as_slice()
    };
    // An empty input pair arena has no borrowed pairs and can retain emission capacity.
    if old.pairs.is_empty() {
        level.pairs = std::mem::take(&mut old.pairs);
    }
    // At most a c_t and a d_t node per accumulator node.
    let node_cap = if compute_dt { 2 * k } else { k };
    lim.reserve(&mut level.nodes, node_cap)?;
    // Worst-case output pairs per input pair: up to 3 c_t pairs for a
    // both-relevant node, plus 1 d_t pair when `compute_dt`. The arena is
    // sized at the input pair count and topped up per node, so the peak never
    // holds a whole-level worst case beside the still-live `old`.
    let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
    // One more for a disjunction's cube pair, which this level may carry.
    let level_pairs = (in_pairs as u128).saturating_mul(pair_mult as u128).saturating_add(1);
    lim.begin_level(Some(level_pairs));
    lim.reserve(&mut level.pairs, in_pairs)?;
    let ctx = SpineCtx { left_grid_base, right_grid_base, pair_mult };
    match (left_rel, right_rel, compute_dt) {
        (true, true, true) => rebuild_nodes::<true, true, true>(eng, &old, nodes, ctx, level, base, tables)?,
        (true, true, false) => rebuild_nodes::<true, true, false>(eng, &old, nodes, ctx, level, base, tables)?,
        (true, false, true) => rebuild_nodes::<true, false, true>(eng, &old, nodes, ctx, level, base, tables)?,
        (true, false, false) => rebuild_nodes::<true, false, false>(eng, &old, nodes, ctx, level, base, tables)?,
        (false, true, true) => rebuild_nodes::<false, true, true>(eng, &old, nodes, ctx, level, base, tables)?,
        (false, true, false) => rebuild_nodes::<false, true, false>(eng, &old, nodes, ctx, level, base, tables)?,
        (false, false, _) => unreachable!("a spine level has a relevant child"),
    }
    // Free `old` before `shrink_arrays` reallocates the rebuilt arenas, so
    // the two are not resident together at the peak.
    let old_inlined_sides = old.value_ref_sides;
    drop(old);
    // Trim the slack the per-node top-up growth left behind.
    level.shrink_arrays();

    // The rebuilt level copied the irrelevant side's pair refs verbatim,
    // inline marginal counts included, but started with zero `value_ref_sides`.
    // The relevant side is never marginal (`plan_cd_map_bases` rejects it), so
    // the input level's markers carry over exactly.
    level.value_ref_sides = old_inlined_sides;
    Ok(())
}

/// Rebuild the nodes with the level's relevant children and complement demand fixed.
/// The separate frame keeps level setup out of the specialized pair loops.
#[inline(never)]
fn rebuild_nodes<const LEFT: bool, const RIGHT: bool, const DT: bool>(
    eng: &Engine, old: &TddLevel, nodes: &[EncodedNode], ctx: SpineCtx, level: &mut TddLevel,
    base: usize, tables: &mut ClauseTables<'_>,
) -> Result<(), OperationError> {
    for (i, node) in nodes.iter().enumerate() {
        debug_assert!(node.is_internal()
            || node.b == u32::MAX,  // inline pair with right=ZERO (dead node)
            "expected internal node at internal vtree position: i={i}");
        let inputs = old.pairs_of(node);
        let carries_cube = tables.output_cube_pair.is_some_and(|(s, _)| s == base + i);
        if inputs.is_empty() && !carries_cube {
            // Dead accumulator node: nothing emitted, and this is the one
            // write of its map entry.
            tables.cd_map[base + i] = [NO_PRODUCT, NO_PRODUCT];
            continue;
        }
        conjoin_node_with_clause::<LEFT, RIGHT, DT>(eng, inputs, ctx, level, base + i, tables)?;
    }
    Ok(())
}

/// Emit the node whose pairs sit at `level.pairs[pair_start..]` and return
/// its index, or [`NO_PRODUCT`] when there are none. A single pair is popped
/// back off the arena and re-dispatched so its encoding is the inline one.
///
/// The pairs are not deduplicated: at a level whose subtree includes a
/// marginal child the accumulator's pair list may be a multiset, two equal
/// pairs each carrying one summed-out family's contribution, and dropping one
/// loses count.
fn emit_from(level: &mut TddLevel, pair_start: usize) -> Result<u32, OperationError> {
    let pair_len = level.pairs.len() - pair_start;
    if pair_len == 0 {
        return Ok(NO_PRODUCT);
    }
    let idx = level.nodes.len() as u32;
    if pair_len == 1 {
        let pair = level.pairs[pair_start];
        level.pairs.truncate(pair_start);
        level.push_node(&crate::diagram::Untracked, &[pair])?;
    } else {
        // `try_push_multi_by_range` requires `pair_len >= 2`, which the arm
        // above guarantees.
        level
            .try_push_multi_by_range(pair_start, pair_len)
            .map_err(|_| OperationError::OverBudget)?;
    }
    Ok(idx)
}
