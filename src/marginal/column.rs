//! Reading and installing one level's marginal column of values.
//!
//! A weighted column lives in the [`WeightStore`], which every diagram merged
//! into it shares, while marginality is a property of one diagram's level.
//! [`LevelColumns`] is the pairing of the two, so a reader cannot decode
//! its own node indices as slots of another diagram's column.

use crate::diagram::{TddLevel, WeightStore, WeightVal};
use crate::limits::ReservePolicy;
use crate::value::CountVec;

/// The weighted columns one diagram may read from a shared store.
pub(crate) struct LevelColumns<'a> {
    store: &'a WeightStore,
    owner: &'a [TddLevel],
}

impl<'a> LevelColumns<'a> {
    /// Pair `store` with the level slice of the diagram that reads it.
    pub(crate) fn new(store: &'a WeightStore, owner: &'a [TddLevel]) -> Self {
        LevelColumns { store, owner }
    }

    /// The store itself, for the reads that are not level columns — the
    /// semiring zero and the leaf bases.
    pub(crate) fn store(&self) -> &'a WeightStore {
        self.store
    }

    /// Level `t`'s column, or `None` when this diagram's level is structural
    /// and its node indices are therefore not slots of any column.
    pub(crate) fn get(&self, t: usize) -> Option<&'a [WeightVal]> {
        column_of(self.store, &self.owner[t], t)
    }
}

/// [`LevelColumns::get`] for a caller that holds the one level rather than the
/// whole slice — the apply, whose output levels are split apart for the level
/// it is building.
pub(crate) fn column_of<'a>(
    store: &'a WeightStore,
    level: &TddLevel,
    t: usize,
) -> Option<&'a [WeightVal]> {
    level.is_weight_marginal().then(|| store.level(t)).flatten()
}

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
    col: Vec<WeightVal>,
    ws: &mut WeightStore,
) {
    let slots = col.len() as u32;
    levels[left_idx].become_marginal_weighted(slots);
    ws.set_level(left_idx, col);
}
