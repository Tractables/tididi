//! The reusable scratch for rotation probes.

use crate::execution::pool::{Buffers, PooledScratch, Scratch};


use crate::diagram::ChildPair;
use super::relevel::{Prepared, ScoreScratch};

/// Reusable scratch for `rebuild_levels`, the score's tables, the pool
/// probe's tables and `restructure_to`, held across probes so the allocator
/// churn is paid once per search rather than once per probe. Each field is `clear()`-ed before use in
/// `rebuild_levels`, preserving the underlying capacity, and emptied again at
/// its last read. Every field grows through the engine's limits, so its growth
/// is charged to the operation's byte meter, and what the retention policy
/// drops is given back there.
#[derive(Default)]
pub(crate) struct RestructureScratch {
    // The outer level's pairs filed by old v-node, node `i`'s ending at
    // `outer_ends[i]`.
    pub(super) outer_ends: Vec<u32>,
    pub(super) outer_pairs: Vec<ChildPair>,
    pub(super) bucket: BucketScratch,
    // The rebuild's triples and their groups (see `Layout`). The sort in
    // `rebuild_levels` is the dominant cost of a probe; sorting the
    // packed words by a single integer key replaces a lexicographic compare
    // over the four fields.
    pub(super) prepared: Prepared,
    // The tables of a rotation's score (`score_levels`).
    pub(super) score: ScoreScratch,
    // A pool-search probe's tables, one entry per member.
    pub(super) probe: ProbeTables,
}

/// The tables a pool-search probe scores its members by: the pairs of each
/// member's two levels before and after the move, whether it is scored, the
/// least it can hold after the move, and the order it is scored in.
#[derive(Default)]
pub(super) struct ProbeTables {
    pub(super) before: Vec<usize>,
    pub(super) after: Vec<usize>,
    pub(super) scored: Vec<bool>,
    pub(super) floors: Vec<usize>,
    pub(super) order: Vec<usize>,
}

impl ProbeTables {
    pub(super) fn clear(&mut self) {
        self.before.clear();
        self.after.clear();
        self.scored.clear();
        self.floors.clear();
        self.order.clear();
    }
}

/// The clustering build of the inner level: the table of the first group of
/// each distinct cell list, and the inner pairs filed by the node they go
/// under, node `k`'s ending at `ends[k]`.
#[derive(Default)]
pub(super) struct BucketScratch {
    pub(super) table: Vec<u32>,
    pub(super) ends: Vec<u32>,
    pub(super) pairs: Vec<ChildPair>,
}

impl Buffers for RestructureScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.outer_ends);
        visit(&mut self.outer_pairs);
        self.prepared.buffers(visit);
        self.score.buffers(visit);
        visit(&mut self.bucket.table);
        visit(&mut self.bucket.ends);
        visit(&mut self.bucket.pairs);
        visit(&mut self.probe.before);
        visit(&mut self.probe.after);
        visit(&mut self.probe.scored);
        visit(&mut self.probe.floors);
        visit(&mut self.probe.order);
    }
}

impl PooledScratch for RestructureScratch {
    fn prepare(&mut self) {
        self.prepared.clear();
        self.score.clear();
        self.bucket.table.clear();
        self.bucket.ends.clear();
        self.bucket.pairs.clear();
        self.outer_ends.clear();
        self.outer_pairs.clear();
        self.probe.clear();
    }
}

#[cfg(test)]
#[path = "tests/scratch.rs"]
mod tests;
