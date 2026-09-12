//! Turning a built pair list into an output node and its map entry.

use super::*;

/// If `pairs` is non-empty, emit as a new internal node in the level and
/// record its index in `result_map[base_plus_idx]`.
///
/// `pairs` is not deduplicated: at a level whose subtree includes a marginal
/// child the accumulator's pair list may be a multiset, two equal pairs each
/// carrying one summed-out family's contribution, and dropping one loses count.
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
        // Empty c_t/d_t: no node emitted, and this is the one write of the
        // map entry (there is no bulk `NO_PRODUCT` fill).
        result_map[base_plus_idx][lane] = NO_PRODUCT;
    }
    Ok(())
}

/// `emit_clause_node` for pairs already pushed onto `level.pairs` from
/// `pair_start` on: finalizes the node, re-dispatching a single pair through
/// `try_push_internal_node` so its encoding matches the buffered path, or
/// writes `NO_PRODUCT` when no pairs were produced. Like `emit_clause_node`,
/// it does not deduplicate.
#[inline]
pub(super) fn emit_clause_node_direct(
    level: &mut TddLevel,
    pair_start: usize,
    result_map: &mut [[u32; 2]],
    lane: usize,
    base_plus_idx: usize,
) -> Result<(), ApplyError> {
    let pair_len = level.pairs.len() - pair_start;
    if pair_len == 0 {
        result_map[base_plus_idx][lane] = NO_PRODUCT;
    } else if pair_len == 1 {
        // Single pair: pop it back off the arena and re-dispatch so the
        // inline encoding (no arena slot) is preserved exactly.
        let pair = level.pairs[pair_start];
        level.pairs.truncate(pair_start);
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level.try_push_internal_node(&[pair]).map_err(|_| ApplyError::OverBudget)?;
    } else {
        // `try_push_multi_by_range` requires `pair_len >= 2`, which the arm
        // above guarantees.
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level
            .try_push_multi_by_range(pair_start, pair_len)
            .map_err(|_| ApplyError::OverBudget)?;
    }
    Ok(())
}
