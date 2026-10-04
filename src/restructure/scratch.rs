//! The reusable rotation-probe scratch and the engine's pool for it.

use crate::limits::Limits;
use crate::execution::pool::{Buffers, PooledScratch, Scratch};


use crate::diagram::ChildPair;

/// Reusable scratch for `rebuild_rotated_levels`. Threaded by
/// the rotation-search loops so the per-probe allocator churn is paid once per
/// search rather than once per probe. Each field is `clear()`-ed before use in
/// `rebuild_rotated_levels`, preserving the underlying capacity; a field is
/// then released at its last read within the call rather than held across the
/// successor-level builds (see `SCRATCH_RETAIN_ENTRIES`). Every field grows
/// through the engine's limits, so its growth is charged to the operation's
/// byte meter, and what the retention policy drops is given back there.
#[derive(Default)]
pub(crate) struct RestructureScratch {
    // The outer level's pairs filed by old v-node, node `i`'s ending at
    // `outer_ends[i]`.
    pub(super) outer_ends: Vec<u32>,
    pub(super) outer_pairs: Vec<ChildPair>,
    // The end of each axis value's cells, for a right rotation's filing.
    pub(super) axis_ends: Vec<u32>,
    pub(super) group_info: Vec<super::relevel::PairGroup>,
    pub(super) bucket: BucketScratch,
    // Search path triples, packed one per word (see `Layout`): `narrow` where
    // the four fields fit 64 bits between them, `wide` where they do not. The
    // sort in `rebuild_rotated_levels` is the dominant cost of a probe;
    // sorting the words by a single integer key replaces a lexicographic
    // compare over the four fields.
    pub(super) narrow: Vec<u64>,
    pub(super) wide: Vec<u128>,
    // A right rotation's cells in order of their axis.
    pub(super) by_axis: Vec<u64>,
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
        visit(&mut self.axis_ends);
        visit(&mut self.group_info);
        visit(&mut self.narrow);
        visit(&mut self.wide);
        visit(&mut self.by_axis);
        visit(&mut self.bucket.table);
        visit(&mut self.bucket.ends);
        visit(&mut self.bucket.pairs);
    }
}

impl PooledScratch for RestructureScratch {
    fn prepare(&mut self) {
        self.group_info.clear();
        self.narrow.clear();
        self.wide.clear();
        self.by_axis.clear();
        self.bucket.table.clear();
        self.bucket.ends.clear();
        self.bucket.pairs.clear();
        self.outer_ends.clear();
        self.outer_pairs.clear();
        self.axis_ends.clear();
    }

    /// A search whose triples outgrew their cap releases every buffer; any
    /// other keeps each buffer the byte cap allows.
    fn retain(&mut self, lim: &Limits) {
        let bytes = self.narrow.capacity() * size_of::<u64>() + self.wide.capacity() * size_of::<u128>();
        if bytes > RESTRUCTURE_TRIPLES_CAP_BYTES {
            self.release_all(lim);
        } else {
            self.release_oversized(lim);
        }
    }
}

/// Maximum retained capacity of the triples, in bytes (8M narrow triples or
/// 4M wide ones). A rare wide rotation search must not park its peak buffers
/// in the pool for the rest of the process; past this, the size-proportional
/// buffers are released and the next take starts from empty.
const RESTRUCTURE_TRIPLES_CAP_BYTES: usize = 64 << 20;

/// Scratch entries kept across probes. At its last read a buffer this size or
/// smaller is only `clear()`-ed, so the next probe reuses the allocation — the
/// churn-avoidance the scratch exists for. A larger one is released outright:
/// holding a high-water buffer across the successor-level builds, which
/// allocate their own copy of the same data, is what sets this function's peak,
/// and regrowing it costs one pass next to the sort that dominates a probe big
/// enough to be over the threshold. Bail-out probes return before any release
/// point, so the search's common path keeps full capacity either way.
pub(super) const SCRATCH_RETAIN_ENTRIES: usize = 1 << 16;

/// Retire a buffer at its last read within a probe: past
/// [`SCRATCH_RETAIN_ENTRIES`] entries of capacity the allocation goes back to
/// `lim`, otherwise the buffer is emptied and stays warm for the next probe.
///
/// A struct field read in place has to be emptied here, or the next probe
/// would see this one's contents; releasing leaves it empty as well.
pub(super) fn release_or_clear<T>(lim: &Limits, buf: &mut Vec<T>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES {
        lim.discard(std::mem::take(buf));
    } else {
        buf.clear();
    }
}

#[cfg(test)]
#[path = "tests/scratch.rs"]
mod tests;
