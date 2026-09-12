//! Installing one level's marginal column.

use crate::diagram::{TddLevel, WeightStore, WeightValue};
use crate::limits::ReservePolicy;
use crate::value::CountVec;

/// Commit a streamed integer column as level `left_idx`'s marginal store.
///
/// `CountVec` and `TddLevel` hold the same slot-keyed overflow table, so this
/// is a move. It does not dedup values: invariant 10 for an emit-born store is
/// established at the slot prune, after the tagger has inlined small counts,
/// because slots shared at birth would let twin merges produce duplicate pairs
/// that pair fusion then sums into new slots.
pub(crate) fn install_int_column<R: ReservePolicy>(
    levels: &mut [TddLevel],
    left_idx: usize,
    col: CountVec<R>,
) {
    let (fast, big) = col.into_parts();
    levels[left_idx].become_marginal(fast, big);
}

/// Commit a streamed weighted column as level `left_idx`'s marginal store:
/// the integer commit's mirror, except the payload goes to the shared
/// [`WeightStore`] and the level keeps only the slot count.
pub(crate) fn install_weight_column(
    levels: &mut [TddLevel],
    left_idx: usize,
    col: Vec<WeightValue>,
    ws: &mut WeightStore,
) {
    let slots = col.len() as u32;
    levels[left_idx].become_marginal_weighted(slots);
    ws.set_level(left_idx, col);
}
