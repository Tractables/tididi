//! Turning a built pair list into an output node and its map entry.

use super::*;

/// If `pairs` is non-empty, emit as a new internal node in the level and
/// record its index in `result_map[base_plus_idx]`.
///
/// No `pairs.dedup()` here, by design.
///
/// The clause-emission paths construct `pairs` from the accumulator's
/// pair list under monotone-injective remaps (the `cd_map` lanes). For
/// non-marginal levels, the input pair list is canonical (sorted,
/// unique) and the output inherits that property — the pair-level
/// corollary of the no-compress proof covers this case.
///
/// For levels whose subtree includes a marginal child, the accumulator's
/// pair list may legitimately be a multiset: count-keyed slot sharing
/// (see `apply_p_fusion`) lets two
/// pairs `(L, R)` co-exist when each carries the `c(L)·c(R)` contribution
/// of one historical marginalization plan. Those duplicates survive
/// through clause apply and must be preserved here — deduplicating would
/// silently lose count.
///
/// **Do not add a defensive dedup here.** Duplicates in a marginal-aware
/// pair list are load-bearing for count correctness; collapsing them
/// reproduces a known count-corruption failure mode.
#[inline]
pub(super) fn emit_clause_node(
    pairs: &mut [InputPair],
    level: &mut TddLevel,
    result_map: &mut [[u32; 2]],
    lane: usize,
    base_plus_idx: usize,
) -> Result<(), ApplyError> {
    if !pairs.is_empty() {
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level.try_push_internal_node(pairs).map_err(|_| ApplyError::OverBudget)?;
    } else {
        // Empty c_t/d_t — no node emitted. Write DEAD here (rather than relying
        // on a separate bulk pre-fill) so every map entry in this level's block
        // is written exactly once, in the loop that already visits it. See the
        // "no bulk DEAD-fill" note at the map-sizing site.
        result_map[base_plus_idx][lane] = DEAD;
    }
    Ok(())
}

/// Direct-emission variant of `emit_clause_node` for the `c_t` lane: the pairs
/// were pushed straight onto `level.pairs` starting at `pair_start`, skipping
/// the scratch-buffer staging + `extend_from_slice` copy of the buffered path
/// (a measurable slice of the batch-1 apply loop). Finalizes the node —
/// re-dispatching single-pair lists through `try_push_internal_node` so the
/// inline/ext encodings stay byte-identical to the buffered path — or writes
/// DEAD when no pairs were produced. The same canonicity contract as
/// `emit_clause_node` applies (no defensive dedup — see above).
#[inline]
pub(super) fn emit_clause_node_direct(
    level: &mut TddLevel,
    pair_start: usize,
    result_map: &mut [[u32; 2]],
    lane: usize,
    base_plus_idx: usize,
) -> Result<(), ApplyError> {
    let pair_len = level.pairs.len() - pair_start;
    // No duplicate-pair assert here: like `emit_clause_node`, this path can
    // legitimately see multiset pair lists on levels whose subtree includes a
    // marginal child (count-keyed slot sharing) — see the multiset note above.
    if pair_len == 0 {
        result_map[base_plus_idx][lane] = DEAD;
    } else if pair_len == 1 {
        // Single pair: pop it back off the arena and re-dispatch so the
        // inline encoding (no arena slot) is preserved exactly.
        let pair = level.pairs[pair_start];
        level.pairs.truncate(pair_start);
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level.try_push_internal_node(&[pair]).map_err(|_| ApplyError::OverBudget)?;
    } else {
        // Invariant for `try_push_multi_by_range`: `pair_len >= 2` here — the
        // single-pair case is re-dispatched through `try_push_internal_node` in
        // the arm above. Its fast path only `debug_assert!`s this.
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level
            .try_push_multi_by_range(pair_start, pair_len)
            .map_err(|_| ApplyError::OverBudget)?;
    }
    Ok(())
}
