//! The reusable rotation-probe scratch and the engine's pool for it.

use crate::limits::pool::PooledScratch;

use rustc_hash::FxHashMap;

use crate::diagram::{ChildPair, NodeIdx};

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
    pub(super) inner_pair_to_idx: FxHashMap<ChildPair, NodeIdx>,
    // Per-v-node output pair lists; outer Vec grown with `resize_with`, inner
    // Vecs `clear()`-ed per call so their capacity survives across probes.
    pub(super) per_v_pairs: Vec<Vec<ChildPair>>,
    pub(super) group_info: Vec<super::relevel::PairGroup>,
    pub(super) bucket: BucketScratch,
    // Search path triples, packed one-per-u128 (see `pack_triple`). The sort in
    // `rebuild_rotated_levels` is the dominant cost of the joint next-merge-cost
    // probe on single-large-component pools; sorting a `Vec<u128>` by a single
    // integer key replaces the derived lexicographic compare over the
    // `(ChildPair, u32, NodeIdx)` tuple's four u32 fields.
    pub(super) packed: Vec<u128>,
}

/// The walk over one fingerprint bucket in the clustering build of the inner
/// level: which of the bucket's groups are already placed under a node, and
/// the pairs of the node being emitted.
#[derive(Default)]
pub(super) struct BucketScratch {
    pub(super) done: Vec<bool>,
    pub(super) pairs: Vec<ChildPair>,
}

impl PooledScratch for RestructureScratch {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::{capacity_bytes, nested_bytes};
        [
            capacity_bytes(&self.inner_pair_to_idx),
            nested_bytes(&self.per_v_pairs),
            capacity_bytes(&self.group_info),
            capacity_bytes(&self.packed),
            capacity_bytes(&self.bucket.done),
            capacity_bytes(&self.bucket.pairs),
        ].into_iter().sum()
    }

    fn prepare(&mut self) {
        self.inner_pair_to_idx.clear();
        self.group_info.clear();
        self.packed.clear();
        self.bucket.done.clear();
        self.bucket.pairs.clear();
        // Keep the outer Vec's length (bounded by `PER_V_PAIRS_RETAIN` on
        // return): `rebuild_rotated_levels` only `resize_with`s it upward and
        // clears the prefix it uses, so the inner Vecs' capacities are exactly
        // what we want to carry forward.
        for v in &mut self.per_v_pairs {
            v.clear();
        }
    }

    fn retain(&mut self, lim: &crate::limits::Limits) {
        self.per_v_pairs.truncate(PER_V_PAIRS_RETAIN);
        if self.packed.capacity() > RESTRUCTURE_PACKED_CAP_LIMIT {
            lim.discard(std::mem::take(&mut self.packed));
            lim.discard(std::mem::take(&mut self.group_info));
            lim.discard(std::mem::take(&mut self.inner_pair_to_idx));
            lim.discard(std::mem::take(&mut self.bucket.done));
            lim.discard(std::mem::take(&mut self.bucket.pairs));
            for lists in self.per_v_pairs.drain(..) {
                lim.discard(lists);
            }
            lim.discard(std::mem::take(&mut self.per_v_pairs));
        }
    }
}

/// Maximum retained `packed` capacity (4M triples × 16 B = 64 MB). A rare wide
/// rotation search must not park its peak buffers in the pool for the rest of
/// the process; past this, the size-proportional buffers are released and the
/// next take starts from empty.
const RESTRUCTURE_PACKED_CAP_LIMIT: usize = 4_000_000;

/// Maximum number of per-v-node pair lists carried across calls. The take-side
/// `clear()` walks the whole outer Vec, so an unbounded one would tax every
/// later (small) search with the widest level this engine ever saw — the pool
/// must not turn one wide rotation into a permanent per-call O(width) sweep.
/// Beyond this the tail is dropped; `rebuild_rotated_levels` re-grows it with
/// `resize_with` exactly as it does on a cold scratch.
const PER_V_PAIRS_RETAIN: usize = 1024;

/// Scratch entries kept across probes. At its last read a buffer this size or
/// smaller is only `clear()`-ed, so the next probe reuses the allocation — the
/// churn-avoidance the scratch exists for. A larger one is released outright:
/// holding a high-water buffer across the successor-level builds, which
/// allocate their own copy of the same data, is what sets this function's peak,
/// and regrowing it costs one pass next to the sort that dominates a probe big
/// enough to be over the threshold. Bail-out probes return before any release
/// point, so the search's common path keeps full capacity either way.
pub(super) const SCRATCH_RETAIN_ENTRIES: usize = 1 << 16;

#[cfg(test)]
#[path = "tests/scratch.rs"]
mod tests;
