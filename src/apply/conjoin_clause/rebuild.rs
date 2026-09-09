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
        // Guarantee this node's whole worst case before emitting any of it,
        // so the direct c_t pushes stay infallible `Vec::push`es and the d_t
        // extend inside `try_push_internal_node` cannot realloc mid-node.
        // A refused top-up surfaces as `OverBudget` — the same error class
        // every other reserve on this path returns.
        reserve_pairs_for_emit(eng, level, pair_mult * inputs.len())?;

        // ── Conjunction with c_t (clause node) ──
        //
        // The clause's virtual c_t represents "clause satisfied in this
        // subtree". Its shape depends on which children have clause
        // variables:
        //
        //   Only right relevant:  c_t = {(d_L, c_R)}                        — 1 virtual pair
        //   Only left relevant:   c_t = {(c_L, d_R)}                        — 1 virtual pair
        //   Both relevant:        c_t = {(c_L,c_R), (c_L,d_R), (d_L,c_R)}  — 3 virtual pairs
        //
        // The 3-pair case captures: "satisfied iff at least one side is
        // satisfied" = all combos except (d_L, d_R) = 1 − d_L·d_R.
        //
        // Single-pair cases: maps are monotone → output already sorted.
        if ctx.both_rel {
            // 3 virtual c_t pairs × N acc pairs, FUSED with the d_t
            // conjunction in one scan over the inputs. See `build_both_rel_pairs`
            // for the full algorithm description; the emit calls are caller-side
            // because they diverge between the allocating and in-place paths.
            let ct_start = level.pairs.len();
            clause_t3_buf.clear();
            if ctx.compute_dt { clause_dt_pairs.clear(); }
            build_both_rel_pairs(
                eng,
                inputs, ctx.left_grid_base, ctx.right_grid_base, ctx.compute_dt,
                cd_map, level, clause_t3_buf, clause_dt_pairs,
            )?;
            emit_clause_node_direct(level, ct_start, cd_map, 0, slot)?;
            // Emit d_t AFTER c_t so the ct lane < dt lane of cd_map — the
            // index ordering the parent level's both_rel pass relies on.
            if ctx.compute_dt {
                emit_clause_node(
                    clause_dt_pairs, level, cd_map, 1, slot,
                )?;
            }
        } else {
            // Single virtual pair: only one child is relevant. See
            // `build_single_rel_pairs` for the fused c_t/d_t algorithm;
            // emit calls are caller-side (diverge between allocating/in-place).
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

    // Swap the old (input) level out so we can rebuild in place. `old` holds
    // the accumulator's pairs for this level; the freshly emptied
    // `levels[t_idx]` receives the conjoined output. (`old` is dropped the
    // moment the emit loop ends — see the drop site below, which must
    // precede the rebuilt level's shrink.)
    let old = std::mem::take(&mut levels[t_idx]);
    let k = old.width();
    let in_pairs = old.pairs.len();
    let level = &mut levels[t_idx];
    // Pre-size the output node Vec to skip the per-node doubling reallocs in
    // the emit loop. The bound is cheap — no cross-product recount: <= 2k
    // nodes (a c_t and a d_t per acc node). try_reserve keeps the
    // OverBudget contract.
    let node_cap = if compute_dt { 2 * k } else { k };
    level.nodes.try_reserve(node_cap).map_err(|_| ApplyError::OverBudget)?;
    // Worst-case output pairs PER INPUT PAIR: a both_rel node emits up to 3
    // c_t pairs (type1/2/3), plus 1 d_t pair when compute_dt — so a level's
    // output can reach 4x its input. The slab is sized at 1x (the input-pair
    // count) and the per-node top-up in the loop grows it on demand, so the
    // peak never carries a whole-level worst case beside the still-live
    // `old`. The c_t emit pushes DIRECTLY onto level.pairs — the per-node
    // top-up, not a per-pair reserve, is what makes those pushes safe.
    let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
    // Same near-cap growth-mode decision the dense emit walk makes, fed the
    // same kind of sound emit bound, so the top-ups below take bounded
    // headroom-aware increments instead of doubling on a huge level.
    lim.begin_level(Some((in_pairs as u128).saturating_mul(pair_mult as u128)));
    level.pairs.try_reserve(in_pairs).map_err(|_| ApplyError::OverBudget)?;
    let ctx = SpineCtx { both_rel, left_rel, left_grid_base, right_grid_base, compute_dt };
    for i in 0..k {
        debug_assert!(old.nodes[i].is_internal()
            || old.nodes[i].b == u32::MAX,  // inline pair with right=ZERO (dead node)
            "expected internal node at internal vtree position: t={t:?} i={i}");
        let inputs = old.pairs_of_idx(i);
        if inputs.is_empty() {
            // Dead acc node — no c_t/d_t emitted. Write DEAD so this entry
            // is initialized (no separate bulk fill); a parent referencing
            // this idx must read DEAD.
            cd_map[base + i] = [DEAD, DEAD];
            continue;
        }
        conjoin_node_with_clause(
            eng,
            inputs, ctx, pair_mult, level, cd_map, base + i,
            clause_t3_buf, clause_dt_pairs,
        )?;
    }
    // The emit loop was the last reader of `old`; only its Copy marg flags
    // are still needed. Free the dead input level HERE, before
    // `shrink_arrays` — that shrink reallocs the rebuilt arenas (alloc +
    // copy + free), so anything still holding `old` pays both arenas plus
    // the realloc's destination copy at the peak.
    let old_inlined_sides = old.inlined_sides;
    drop(old);
    // Trim the slack the per-node top-up growth left behind.
    level.shrink_arrays();

    // The rebuilt level copied the irrelevant side's pair refs verbatim
    // — including inline marg counts (bit 30) toward a marginal sibling
    // child — but started from a fresh `TddLevel::new()` whose
    // `inlined_sides` are zero. Carry the markers over: the relevant side
    // is never marginal (gateway panic above), so its flags are false in
    // the input level and the wholesale copy is exact.
    level.inlined_sides = old_inlined_sides;
    Ok(())
}
